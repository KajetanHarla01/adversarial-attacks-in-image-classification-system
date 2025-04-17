import torch
import timm
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, DeepFool, ZooAttack
from art.estimators.classification import PyTorchClassifier
from torchvision import transforms, datasets
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from collections import defaultdict
import csv

# Mapowanie nazw na modele timm
timm_models = {
    'resnet': 'resnet34.tv_in1k',
    'efficientnet': 'efficientnet_b0.ra_in1k',
    'densenet': 'densenet121.tv_in1k',
    'convnext': 'convnext_tiny.in12k_ft_in1k_384',
    'coatnet': 'coatnet_0_rw_224',
    'vit': 'vit_small_patch16_384'
}

def get_attack(attack_name, classifier, eps=0.03, targeted=False, max_iter=10, eps_step=0.01):
    if attack_name == 'FGSM':
        return FastGradientMethod(estimator=classifier, eps=eps, targeted=targeted)
    elif attack_name == 'PGD':
        return ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, targeted=targeted)
    elif attack_name == 'CW':
        return CarliniL2Method(classifier=classifier, targeted=targeted, max_iter=max_iter)
    elif attack_name == 'DeepFool':
        return DeepFool(classifier=classifier, max_iter=max_iter)
    elif attack_name == 'ZOO':
        return ZooAttack(classifier=classifier, max_iter=max_iter, nb_parallel=64, verbose=True)
    else:
        raise ValueError(f"Unsupported attack: {attack_name}")

def compute_ssim(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()

    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)

    h, w, _ = img1.shape
    win = min(7, h, w)

    if win % 2 == 0:
        win -= 1

    try:
        return ssim(
            img1,
            img2,
            data_range=1.0,
            channel_axis=2,
            win_size=win
        )
    except Exception as e:
        print(f"[!] SSIM failed: {e}")
        return np.nan

def compute_psnr(img1, img2, debug=True):
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).detach().cpu().numpy()

    # Debug — pokaż maksymalne różnice
    if debug:
        diff = img1 - img2
        print(f"  → Δ min: {diff.min():.10f}, max: {diff.max():.10f}, mean: {diff.mean():.10f}")
        print(f"  → Allclose: {np.allclose(img1, img2, atol=1e-8)}")
        print(f"  → Any diff: {(diff != 0).sum()} pixels changed")

    img1 = np.clip(img1, 0.0, 1.0)
    img2 = np.clip(img2, 0.0, 1.0)

    try:
        mse = np.mean((img1 - img2) ** 2)
        if debug:
            print(f"  → MSE: {mse:.10f}")
        if mse == 0:
            return float('inf')
        return 10 * np.log10(1.0 / mse)
    except Exception as e:
        print(f"[!] PSNR failed: {e}")
        return np.nan
    
def get_transform(model, input_size_override=384):
    global transform
    config = resolve_data_config({}, model=model)
    config['input_size'] = (3, input_size_override, input_size_override)
    transform = create_transform(**config)
    return transform

def get_file(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    if len(files) == 0:
        print("Error: \"" + dir + "\" directory is empty")
        exit()

    print("Choose file: ")
    for i, f in enumerate(files, start=1):
        print(f"{i} - {f}")

    file_choose = int(input("File number: "))
    if file_choose > len(files):
        print("Error: invalid file number")
        exit()

    return join(dir, files[file_choose - 1])

def load_model(model_name, path, num_classes=40):
    model = timm.create_model(timm_models[model_name], pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_image(image_tensor):
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(image_tensor)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image_tensor = inv_normalize(image_tensor)

    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    return image_tensor.permute(1, 2, 0).cpu().numpy()

def create_subset_per_class(dataset, samples_per_class=10):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
    selected_indices = [idx for indices in class_indices.values() for idx in indices]
    return torch.utils.data.Subset(dataset, selected_indices)

def run_attack_on_dataset(model, data_loader, attack_type='FGSM', targeted=False):
    device = torch.device("cuda" if torch.cuda.is_available() and attack_type != 'ZOO' else "cpu")
    model.to(device)

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        input_shape=(3, 384, 384),
        nb_classes=40,
        device_type="cpu" if attack_type == 'ZOO' else "gpu"
    )

    if attack_type in ['CW', 'DeepFool', 'ZOO']:
        iterations = [10, 50, 100] if attack_type != 'ZOO' else [1, 10, 100]
        for max_iter in iterations:
            print(f"\nTesting max_iter: {max_iter}")
            attack = get_attack(attack_type, classifier, max_iter=max_iter)
            run_single_attack(classifier, data_loader, attack, attack_type)
    elif attack_type == 'FGSM':
        epsilons = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]
        for eps in epsilons:
            print(f"\nTesting epsilon: {eps}")
            attack = get_attack(attack_type, classifier, eps=eps, targeted=targeted)
            run_single_attack(classifier, data_loader, attack, attack_type)
    else:
        attack = get_attack(attack_type, classifier, eps=0.1, targeted=targeted, eps_step=0.02, max_iter=100)
        run_single_attack(classifier, data_loader, attack, attack_type)

def run_single_attack(classifier, data_loader, attack, attack_type):
    all_preds_clean = []
    all_preds_adv = []
    all_labels = []
    all_ssim = []
    all_psnr = []
    sample_saved = False

    for i, (images, labels) in enumerate(data_loader):
        x_numpy = images.numpy()
        y_numpy = labels.numpy()
        x_adv = attack.generate(x=x_numpy)

        diff_mean = np.mean(np.abs(x_adv - x_numpy))
        diff_max = np.max(np.abs(x_adv - x_numpy))
        print(f"[DEBUG] Batch {i+1} — Mean Δ: {diff_mean:.6f}, Max Δ: {diff_max:.6f}")

        preds_clean = np.argmax(classifier.predict(x_numpy), axis=1)
        preds_adv = np.argmax(classifier.predict(x_adv), axis=1)

        all_preds_clean.extend(preds_clean)
        all_preds_adv.extend(preds_adv)
        all_labels.extend(y_numpy)

        for j in range(images.shape[0]):

            original = load_image(images[j])
            adversarial = load_image(x_adv[j])

            print(f"\n[DEBUG] Image {j+1} in batch {i+1}")
            print(f" - Original: min={x_numpy[j].min():.4f}, max={x_numpy[j].max():.4f}, mean={x_numpy[j].mean():.4f}")
            print(f" - Adv     : min={x_adv[j].min():.4f}, max={x_adv[j].max():.4f}, mean={x_adv[j].mean():.4f}")
            if preds_clean[j] != preds_adv[j]:
                print(f" - Changed prediction: {preds_clean[j]} → {preds_adv[j]}")
            else:
                print(f" - No change in prediction: {preds_clean[j]}")
            all_ssim.append(compute_ssim(original, adversarial))
            all_psnr.append(compute_psnr(original, adversarial))

            if not sample_saved:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(original)
                axes[0].set_title(f"Original\nPred: {preds_clean[j]}")
                axes[0].axis('off')

                axes[1].imshow(adversarial)
                axes[1].set_title(f"Adversarial\nPred: {preds_adv[j]}")
                axes[1].axis('off')

                plt.suptitle("Adversarial Example (Attack: " + attack_type + ")")
                plt.tight_layout()
                plt.savefig("original_vs_adv.png")
                plt.show()
                sample_saved = True

    acc_adv = accuracy_score(all_labels, all_preds_adv)
    f1_adv = f1_score(all_labels, all_preds_adv, average='weighted')
    precision_adv = precision_score(all_labels, all_preds_adv, average='weighted', zero_division=1)
    recall_adv = recall_score(all_labels, all_preds_adv, average='weighted', zero_division=1)

    print(f"Adversarial Accuracy: {acc_adv * 100:.2f}%, F1: {f1_adv * 100:.2f}%, Precision: {precision_adv * 100:.2f}%, Recall: {recall_adv * 100:.2f}%")
    print(f"Mean SSIM: {np.mean(all_ssim):.4f}, Mean PSNR: {np.mean(all_psnr):.2f} dB")

def auto_deepfool_test(samples_per_class=5):
    import csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    intensities = {
        'lagodny': 10,
        'umiarkowany': 50,
        'agresywny': 100
    }

    results = []
    os.makedirs("auto_examples", exist_ok=True)

    for model_key in timm_models.keys():
        model_filename = next((f for f in os.listdir("final-models") if f.startswith(model_key)), None)
        if not model_filename:
            print(f"[!] Brak modelu dla: {model_key}")
            continue

        print(f"\n=== MODEL: {model_key.upper()} ===")
        model = load_model(model_key, os.path.join("final-models", model_filename))
        model.to(device)

        transform = get_transform(model)
        dataset_full = datasets.ImageFolder(root='test', transform=transform)
        dataset = create_subset_per_class(dataset_full, samples_per_class=samples_per_class)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

        classifier = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
            input_shape=(3, 384, 384),
            nb_classes=40,
            device_type="gpu"
        )

        for level, max_iter in intensities.items():
            print(f"\n[>>] Tryb: {level}, iteracje: {max_iter}")

            attack = DeepFool(classifier=classifier, max_iter=max_iter)
            all_preds_clean = []
            all_preds_adv = []
            all_labels = []
            all_ssim = []
            all_psnr = []
            sample_saved = False

            for i, (images, labels) in enumerate(dataloader):
                x_numpy = images.numpy()
                y_numpy = labels.numpy()
                x_adv = attack.generate(x=x_numpy)

                preds_clean = np.argmax(classifier.predict(x_numpy), axis=1)
                preds_adv = np.argmax(classifier.predict(x_adv), axis=1)

                all_preds_clean.extend(preds_clean)
                all_preds_adv.extend(preds_adv)
                all_labels.extend(y_numpy)

                for j in range(images.shape[0]):
                    original = load_image(images[j])
                    adversarial = load_image(x_adv[j])
                    all_ssim.append(compute_ssim(original, adversarial))
                    all_psnr.append(compute_psnr(original, adversarial))

                    if not sample_saved:
                        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                        axes[0].imshow(original)
                        axes[0].set_title(f"Original\nPred: {preds_clean[j]}")
                        axes[0].axis('off')

                        axes[1].imshow(adversarial)
                        axes[1].set_title(f"Adversarial\nPred: {preds_adv[j]}")
                        axes[1].axis('off')

                        plt.suptitle(f"DeepFool – {model_key} ({level})")
                        plt.tight_layout()
                        path = f"auto_examples/original_vs_adv_{model_key}_{level}.png"
                        plt.savefig(path)
                        plt.close()
                        sample_saved = True

            acc_adv = accuracy_score(all_labels, all_preds_adv)
            f1_adv = f1_score(all_labels, all_preds_adv, average='weighted')
            precision_adv = precision_score(all_labels, all_preds_adv, average='weighted', zero_division=1)
            recall_adv = recall_score(all_labels, all_preds_adv, average='weighted', zero_division=1)
            mean_ssim = np.mean(all_ssim)
            mean_psnr = np.mean(all_psnr)

            print(f"Done. Accuracy: {acc_adv:.4f}, SSIM: {mean_ssim:.4f}, PSNR: {mean_psnr:.2f} dB")
            results.append([
                model_key,
                level,
                f"{acc_adv:.4f}",
                f"{f1_adv:.4f}",
                f"{precision_adv:.4f}",
                f"{recall_adv:.4f}",
                f"{mean_ssim:.4f}",
                f"{mean_psnr:.2f}"
            ])

    with open("deepfool_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Tryb", "Accuracy", "F1", "Precision", "Recall", "SSIM", "PSNR"])
        writer.writerows(results)

    print("\n✅ Wszystko gotowe. Wyniki zapisane w deepfool_results.csv oraz folderze auto_examples/")

if __name__ == '__main__':
    #auto_deepfool_test()
    print("Choose a model:")
    for i, key in enumerate(timm_models.keys()):
        print(f"{i + 1}. {key}")

    choice = int(input("Your choice (1-6): ")) - 1
    model_key = list(timm_models.keys())[choice]

    model_path = get_file("final-models")
    model = load_model(model_key, model_path)

    attack_options = ['FGSM', 'PGD', 'CW', 'DeepFool', 'ZOO']
    print("\nChoose an attack:")
    for i, name in enumerate(attack_options):
        print(f"{i + 1}. {name}")

    attack_choice = int(input("Your choice (1-5): ")) - 1
    attack_name = attack_options[attack_choice]

    samples_per_class = int(input("\nHow many samples per class? (e.g., 10): "))

    dataset_full = datasets.ImageFolder(root='test', transform=get_transform(model))
    dataset = create_subset_per_class(dataset_full, samples_per_class=samples_per_class)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    run_attack_on_dataset(model, dataloader, attack_type=attack_name, targeted=False)