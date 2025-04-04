import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import xml.etree.ElementTree as ET
from PIL import Image
import os
import time
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import random
from shutil import copyfile
from collections import defaultdict
import shutil
import signal
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def set_transform(model, input_size_override=384):
    global transform
    config = resolve_data_config({}, model=model)
    config['input_size'] = (3, input_size_override, input_size_override)
    transform = create_transform(**config)

class TimeoutExpired(Exception):
    pass


def input_with_timeout(prompt, timeout):
    def handler(signum, frame):
        raise TimeoutExpired

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        return input(prompt)
    except TimeoutExpired:
        print("(No input received — continuing training...)\n")
        return None
    finally:
        signal.alarm(0)


def get_model_by_type(type):
    model_names = {
        'resnet': 'resnet34.tv_in1k',
        'efficientnet': 'efficientnet_b0.ra_in1k',
        'densenet': 'densenet121.tv_in1k',
        'convnext': 'convnext_tiny.in12k_ft_in1k_384',
        'coatnet': 'coatnet_0_rw_224',
        'vit': 'vit_small_patch16_384'
    }
    if type in model_names:
        return lambda: timm.create_model(model_names[type], pretrained=True)
    return None

def save_confusion_matrix(cm, class_names, file_name='confusion_matrix.csv'):
    cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
    cm_df.to_csv(file_name)
    print(f"Confusion matrix saved to {file_name}")

def get_most_confused_classes(cm):
    errors_per_class = np.sum(cm, axis=1) - np.diag(cm)
    most_confused_indices = np.argsort(errors_per_class)[-5:]
    return most_confused_indices, errors_per_class

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

def get_xy_from_XML(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return xmin, xmax, ymin, ymax

def remove_all_folders(folder):
    for dirname in os.listdir(folder):
        dir_path = os.path.join(folder, dirname)
        if os.path.isdir(dir_path) and dirname != '.git':
            shutil.rmtree(dir_path)

def crop_image(image_path, xml_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    if not os.path.exists(xml_path):
        print(f"XML not found: {xml_path}")
        return None
    try:
        xmin, xmax, ymin, ymax = get_xy_from_XML(xml_path)
        img = Image.open(image_path).convert('RGB')
        cropped_image = img.crop((xmin, ymin, xmax, ymax))
        return cropped_image

    except Exception as e:
        print(f"Error cropping image: {e}")
        return None


def split_dataset(
    source_folder: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    crop: bool = True
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    train_folder = os.path.join(os.getcwd(), 'train')
    test_folder = os.path.join(os.getcwd(), 'test')
    val_folder = os.path.join(os.getcwd(), 'validate')

    remove_all_folders(train_folder)
    remove_all_folders(val_folder)
    remove_all_folders(test_folder)

    random.seed(seed)
    class_images = defaultdict(list)

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_name = os.path.splitext(os.path.basename(filename))[0][:-4]
            class_images[class_name].append(filename)

    for folder in [train_folder, val_folder, test_folder]:
        for class_name in class_images:
            os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    for class_name, images in class_images.items():
        random.shuffle(images)
        total = len(images)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        splits = {
            train_folder: images[:train_end],
            val_folder: images[train_end:val_end],
            test_folder: images[val_end:]
        }

        for dest_folder, image_list in splits.items():
            for image in image_list:
                src = os.path.join(source_folder, image)
                dst = os.path.join(dest_folder, class_name, image)
                if crop:
                    xml = os.path.splitext(image)[0] + ".xml"
                    xml_path = os.path.join("XMLAnnotations", xml)
                    cropped_image = crop_image(src, xml_path)
                    if cropped_image:
                        cropped_image.save(dst)
                else:
                    shutil.copyfile(src, dst)

def create_sets(train_images, test_images, images_folder, xml_folder, crop: bool = True):
    train_folder = os.path.join(os.getcwd(), 'train')
    test_folder = os.path.join(os.getcwd(), 'test')
    val_folder = os.path.join(os.getcwd(), 'validate')

    remove_all_folders(train_folder)
    remove_all_folders(val_folder)
    remove_all_folders(test_folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    with open(train_images, 'r') as train_file, open(test_images, 'r') as test_file:
        for image in train_file:
            image = image.strip()
            src_path = join(images_folder, image)
            dst_class = os.path.splitext(os.path.basename(image))[0][:-4]
            dst_folder = join(train_folder, dst_class)
            os.makedirs(dst_folder, exist_ok=True)

            dst_path = join(dst_folder, image)
            if crop:
                xml_path = join(xml_folder, os.path.splitext(image)[0] + ".xml")
                cropped_image = crop_image(src_path, xml_path)
                if cropped_image:
                    cropped_image.save(dst_path)
            else:
                shutil.copyfile(src_path, dst_path)

        for image in test_file:
            image = image.strip()
            src_path = join(images_folder, image)
            dst_class = os.path.splitext(os.path.basename(image))[0][:-4]
            dst_folder = join(test_folder, dst_class)
            os.makedirs(dst_folder, exist_ok=True)

            dst_path = join(dst_folder, image)
            if crop:
                xml_path = join(xml_folder, os.path.splitext(image)[0] + ".xml")
                cropped_image = crop_image(src_path, xml_path)
                if cropped_image:
                    cropped_image.save(dst_path)
            else:
                shutil.copyfile(src_path, dst_path)

def train_model(train_loader, val_loader, model, device):
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = int(input("Number of epochs: "))
    check_accuracy = input("Check accuracy every n pochs? [y/N]: ").strip().lower() == 'y'
    if check_accuracy:
        accurancy_check_epochs = input("How many n? : ")
    early_stop = input("Enable early stopping after validation? [y/N]: ").strip().lower() == 'y'

    model_name = model.__class__.__name__.lower()
    checkpoint_dir = os.path.join("checkpoints", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Clean old checkpoints
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            os.remove(os.path.join(checkpoint_dir, f))

    print(f"Starting training on model, epochs: {num_epochs}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch+1}")
        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s')

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if check_accuracy and (epoch + 1) % int(accurancy_check_epochs) == 0:
            val_accuracy, precision, recall, f1 = test_model(val_loader, model, device)
            print(f"\n[Validation] Accuracy: {val_accuracy * 100:.2f}%, Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%")
            if early_stop:
                stop = input_with_timeout("Stop training? [y/N]: ", 30)
                if stop and stop.strip().lower() == 'y':
                    print("Training stopped early by user.")
                    break
            model.train()

    print("Training completed!")

    # Let user pick a checkpoint to load
    checkpoints = sorted(os.listdir(checkpoint_dir))
    print("\nAvailable checkpoints:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"{i}. {ckpt}")

    choice = input("Choose checkpoint to load as final model [Enter number or press Enter to skip]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
        selected_checkpoint = checkpoints[int(choice) - 1]
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, selected_checkpoint)))
        print(f"Loaded model from {selected_checkpoint}")

def test_model(test_loader, model, device):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    return accuracy, precision, recall, f1

def classify_single_image(model, device, class_names):
    model.eval()
    model.to(device)

    image_path = get_file("test_single_image")

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    sorted_probs, sorted_classes = torch.sort(probabilities, descending=True)

    print(f"Predictions for {image_path}:")
    for i in range(5):
        class_index = sorted_classes[i].item()
        print(f"{class_names[class_index]}: {sorted_probs[i].item() * 100:.2f}%")

def show_test_output(accuracy, precision, recall, f1):
    print(f'Accuracy on test data: {accuracy * 100:.2f}%')
    print(f'Precision on test data: {precision * 100:.2f}%')
    print(f'Recall on test data: {recall * 100:.2f}%')
    print(f'F1-score on test data: {f1 * 100:.2f}%')

def get_class_names(dataset_path):
    dataset = datasets.ImageFolder(root=dataset_path)
    return dataset.classes

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

def load_model(model, filename, device, num_classes=40):
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist.")
        return None

    try:
        state_dict = torch.load(filename, map_location=device)

        if isinstance(model, nn.Module):
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
                else:
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif hasattr(model, 'heads'):
                model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

        model_state_dict = model.state_dict()
        
        for name, param in state_dict.items():
            if name in model_state_dict and param.shape == model_state_dict[name].shape:
                model_state_dict[name] = param

        model.load_state_dict(model_state_dict)

        print(f"Model loaded from {filename}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model

def choose_model_for_training():
    global input_size

    print("Choose a model for training:")
    print("1. ResNet-34")
    print("2. EfficientNet-b0")
    print("3. DenseNet-121")
    print("4. ConvNeXt-Small")
    print("5. CoAtNet-2")
    print("6. ViT Small Patch16")

    model_choice = int(input("Your choice (1-6): "))

    timm_models = {
        1: 'resnet34.tv_in1k',
        2: 'efficientnet_b0.tv_in1k',
        3: 'densenet121.tv_in1k',
        4: 'convnext_tiny.in12k_ft_in1k_384',
        5: 'coatnet_0_rw_224',
        6: 'vit_small_patch16_384'
    }

    if model_choice not in timm_models:
        print("Invalid choice.")
        exit()

    model_name = timm_models[model_choice]
    model = timm.create_model(model_name, pretrained=True, num_classes=40)

    set_transform(model)

    return model

def set_dataloaders():
    train_data = datasets.ImageFolder(root='train', transform=transform)
    test_data = datasets.ImageFolder(root='test', transform=transform)

    validate_path = 'validate'
    has_val_classes = any(os.path.isdir(os.path.join(validate_path, d)) for d in os.listdir(validate_path))
    if has_val_classes:
        val_data = datasets.ImageFolder(root=validate_path, transform=transform)
    else:
        print("No subfolders in 'validate' — using test set for validation.")
        val_data = test_data

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

    return train_loader, test_loader, val_loader

def start():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    option = int(input("Choose option (1 - use existing model, 2 - train new model): "))
    if option == 1:      
        model_filename = get_file("models")
        type = os.path.basename(model_filename).split('_')[0]
        model_class = get_model_by_type(type)
        if model_class:
            model = model_class()
            model = load_model(model, model_filename, device)
            set_transform(model)
            train_loader, test_loader, val_loader = set_dataloaders()
            train = input("Train this model more? [y/N]: ").strip().lower()
            if train == 'y':
                train_model(train_loader, val_loader, model, device)
        else:
            print("Unknown model type: ", type)
            return
    elif option == 2:
        model = choose_model_for_training()
        train_loader, test_loader, val_loader = set_dataloaders()
        train_model(train_loader, val_loader, model, device)
    else:
        print("Invalid option number")

    save = input("Save model to a file? [y/N]: ").strip().lower()
    if save == 'y':
        model_filename = input("Enter a filename to save the trained model: ")
        save_model(model, os.path.join("models", f"{model.__class__.__name__.lower()}_{model_filename}.pth"))

    test = input("Test model? [y/N]: ").strip().lower()
    if test == 'y':
        accuracy, precision, recall, f1 = test_model(test_loader, model, device)
        show_test_output(accuracy, precision, recall, f1)
    
    single_image_test = input("Classify a single image? [y/N]: ").strip().lower()
    if single_image_test == 'y':
        class_names = get_class_names('test')
        classify_single_image(model, device, class_names)

torch.cuda.empty_cache()
option = int(input("Choose option (1 - create training and testing image sets, 2 - test a model): "))
if option == 1:
    sub_option = input("Split using (1 - .txt files, 2 - automatic split): ").strip()
    crop = input("Crop images [Y/n]? ") != 'n'
    if sub_option == '1':
        create_sets('ImageSplits/train.txt', 'ImageSplits/test.txt', 'JPEGImages', 'XMLAnnotations', crop)
    elif sub_option == '2':
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        print(f"Splitting dataset automatically ({train_ratio*100}% train, {val_ratio*100}% val, {test_ratio*100}% test)...")
        split_dataset("JPEGImages", train_ratio, val_ratio, test_ratio, crop=crop)
    else:
        print("Invalid split option.")
elif option == 2:
    start()
else:
    print("Invalid option.")