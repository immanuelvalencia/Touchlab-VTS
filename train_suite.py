import os
import argparse
import time
import copy
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json
import gc

# Define suites for each architecture family
RESNET_SUITE = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
EFFICIENTNET_SUITE = [
    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
    'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
    'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
]
DENSENET_SUITE = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
MOBILENET_SUITE = ['mobilenet_v3_small', 'mobilenet_v3_large']
CONVNEXT_SUITE = ['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large']
SWIN_SUITE = ['swin_t', 'swin_s', 'swin_b', 'swin_v2_t', 'swin_v2_s', 'swin_v2_b']
VIT_SUITE = ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']
REGNET_SUITE = ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf', 'regnet_y_16gf', 'regnet_y_32gf']

# The comprehensive suite contains everything
DEFAULT_SUITE = (
    RESNET_SUITE + EFFICIENTNET_SUITE + DENSENET_SUITE + MOBILENET_SUITE + 
    CONVNEXT_SUITE + SWIN_SUITE + VIT_SUITE + REGNET_SUITE
)

def setup_argparse():
    parser = argparse.ArgumentParser(description="Unified Training Suite for Multiple Architectures and Weights.")
    parser.add_argument('--dataset_dir', type=str, default='ml_dataset', help='Path to the exported ML dataset (with train/val/test splits).')
    parser.add_argument('--models', nargs='+', default=[], 
                        help="List of model names to train (e.g., convnext_tiny swin_t).")
    parser.add_argument('--all', action='store_true', help='Train all models in the comprehensive predefined suite.')
    parser.add_argument('--resnet', action='store_true', help='Train all ResNet variants.')
    parser.add_argument('--efficientnet', action='store_true', help='Train all EfficientNet variants.')
    parser.add_argument('--densenet', action='store_true', help='Train all DenseNet variants.')
    parser.add_argument('--mobilenet', action='store_true', help='Train all MobileNet variants.')
    parser.add_argument('--convnext', action='store_true', help='Train all ConvNeXt variants.')
    parser.add_argument('--swin', action='store_true', help='Train all Swin variants.')
    parser.add_argument('--vit', action='store_true', help='Train all ViT variants.')
    parser.add_argument('--regnet', action='store_true', help='Train all RegNet variants.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train per model/weight variant.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--output_dir', type=str, default='train', help='Base directory to save the trained model weights.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    return parser.parse_args()

def replace_classifier(model, model_name, num_classes):
    """Dynamically replaces the final classification layer based on the architecture family."""
    if hasattr(model, 'fc'):
        # ResNet, GoogLeNet, Inception
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            # EfficientNet, MobileNetV3, VGG, SqueezeNet, AlexNet
            # Find the last linear layer in the sequential block
            last_linear_idx = -1
            for i, module in enumerate(model.classifier):
                if isinstance(module, nn.Linear):
                    last_linear_idx = i
            if last_linear_idx != -1:
                num_ftrs = model.classifier[last_linear_idx].in_features
                model.classifier[last_linear_idx] = nn.Linear(num_ftrs, num_classes)
            else:
                # SqueezeNet uses Conv2d as the final classifier layer
                last_conv_idx = -1
                for i, module in enumerate(model.classifier):
                    if isinstance(module, nn.Conv2d):
                        last_conv_idx = i
                if last_conv_idx != -1:
                    in_channels = model.classifier[last_conv_idx].in_channels
                    model.classifier[last_conv_idx] = nn.Conv2d(in_channels, num_classes, kernel_size=(1,1), stride=(1,1))
                else:
                    raise ValueError(f"Could not find Linear or Conv2d layer in Sequential classifier for {model_name}")
        elif isinstance(model.classifier, nn.Linear):
            # DenseNet
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unknown classifier type for {model_name}")
    elif hasattr(model, 'head'):
        # Swin Transformer
        if isinstance(model.head, nn.Linear):
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unknown head structure for {model_name}")
    elif hasattr(model, 'heads'):
        # Vision Transformer (ViT)
        if hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Linear(num_ftrs, num_classes)
        elif isinstance(model.heads, nn.Sequential):
            last_linear_idx = -1
            for i, module in enumerate(model.heads):
                if isinstance(module, nn.Linear):
                    last_linear_idx = i
            if last_linear_idx != -1:
                num_ftrs = model.heads[last_linear_idx].in_features
                model.heads[last_linear_idx] = nn.Linear(num_ftrs, num_classes)
            else:
                raise ValueError(f"Could not find Linear layer in Sequential heads for {model_name}")
        else:
            raise ValueError(f"Unknown heads structure for {model_name}")
    else:
        raise ValueError(f"Unsupported architecture {model_name}. Could not find final layer.")

def train_model(model_name, weight_name, weight_enum, args, dataloaders, dataset_sizes, class_names, device):
    """Handles the training loop for a single model and specific weight variant."""
    print(f"\n{'='*60}\nStarting Training for Architecture: {model_name} (Weight: {weight_name})\n{'='*60}")
    
    # Create specific run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{model_name}_{weight_name}_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Outputting run artifacts to: {run_dir}")

    # Save labels.txt
    labels_path = os.path.join(run_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    # Load architecture
    model_fn = getattr(models, model_name)
    try:
        if weight_enum is not None:
            model = model_fn(weights=weight_enum)
        else:
            model = model_fn(pretrained=True)
    except Exception as e:
        print(f"Error loading {model_name} with weights {weight_name}: {e}. Skipping...")
        return
    
    num_classes = len(class_names)
    try:
        replace_classifier(model, model_name, num_classes)
    except Exception as e:
        print(f"Error configuring classifier for {model_name}: {e}. Skipping...")
        return
        
    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Tracking
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 15)

        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            with tqdm(dataloaders[phase], desc=f"[{model_name}|{weight_name}] {phase.capitalize()}", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        
                        # Handle models that return multiple outputs (like GoogLeNet/Inception auxiliary outputs)
                        if isinstance(outputs, tuple) and phase == 'train':
                            loss = sum((criterion(o, labels) for o in outputs))
                            outputs = outputs[0]
                        else:
                            loss = criterion(outputs, labels)
                            
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Validation logic
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = os.path.join(run_dir, f"best_{model_name}_{weight_name}_model.pth")
                    torch.save(best_model_wts, best_model_path)
                    print(f"[*] New best {model_name} ({weight_name}) saved (Loss: {best_loss:.4f})")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"Early stopping counter: {epochs_no_improve} out of {args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered for {model_name} ({weight_name})!")
            break

    time_elapsed = time.time() - start_time
    print(f"\n{model_name} ({weight_name}) Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:4f}")

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val' in dataloaders:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} ({weight_name}) Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if 'val' in dataloaders:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} ({weight_name}) Accuracy over Epochs')
    plt.legend()

    plot_path = os.path.join(run_dir, "training_history.png")
    plt.savefig(plot_path)
    plt.close()

    # --- Test Evaluation ---
    if 'test' in dataloaders:
        print(f"\n--- Running Final Evaluation for {model_name} ({weight_name}) on Test Set ---")
        model.load_state_dict(best_model_wts)
        model.eval()
        
        test_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(dataloaders['test'], desc=f"[{model_name}|{weight_name}] Testing", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    test_corrects += torch.sum(preds == labels.data)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        test_acc = test_corrects.double() / dataset_sizes['test']
        print(f"{model_name} ({weight_name}) Final Test Accuracy: {test_acc:.4f}")
        
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        
        with open(os.path.join(run_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=4)
            
        with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Weight Variant: {weight_name}\n")
            f.write(f"Final Test Accuracy: {test_acc:.4f}\n\n")
            f.write(report_str)
            
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
        plt.close()
        
    # Free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    args = setup_argparse()

    # Determine models to train
    models_to_train = []
    if args.all or (args.models and 'all' in [m.lower() for m in args.models]):
        models_to_train.extend(DEFAULT_SUITE)
    if args.resnet:
        models_to_train.extend(RESNET_SUITE)
    if args.efficientnet:
        models_to_train.extend(EFFICIENTNET_SUITE)
    if args.densenet:
        models_to_train.extend(DENSENET_SUITE)
    if args.mobilenet:
        models_to_train.extend(MOBILENET_SUITE)
    if args.convnext:
        models_to_train.extend(CONVNEXT_SUITE)
    if args.swin:
        models_to_train.extend(SWIN_SUITE)
    if args.vit:
        models_to_train.extend(VIT_SUITE)
    if args.regnet:
        models_to_train.extend(REGNET_SUITE)
        
    if args.models and 'all' not in [m.lower() for m in args.models]:
        models_to_train.extend(args.models)
        
    # Remove duplicates while preserving order
    models_to_train = list(dict.fromkeys(models_to_train))

    if not models_to_train:
        print("No models specified. Use --models <model1>, --all, or family flags like --resnet.")
        return

    print(f"Queued {len(models_to_train)} models for training evaluation.")

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Define Data Transformations (using standard 224x224 which is compatible broadly)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(f"Loading dataset from {args.dataset_dir}...")
    
    image_datasets = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(args.dataset_dir, split)
        if os.path.exists(split_path):
            image_datasets[split] = datasets.ImageFolder(split_path, data_transforms[split])
        else:
            print(f"Warning: Split {split} not found in {args.dataset_dir}!")

    if 'train' not in image_datasets or len(image_datasets['train']) == 0:
        print("Error: Training dataset is empty or missing.")
        return

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=(x == 'train'), num_workers=4) 
                   for x in image_datasets.keys()}
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}
    class_names = image_datasets['train'].classes

    print(f"Classes found ({len(class_names)}): {class_names}")

    # Loop over models, then discover and loop over weights
    total_runs = 0
    for model_name in models_to_train:
        if not hasattr(models, model_name):
            print(f"Error: Model {model_name} not found in torchvision.models. Skipping...")
            continue
            
        weight_variants = []
        try:
            weights_enum_cls = models.get_model_weights(model_name)
            for w in weights_enum_cls:
                if w.name != 'DEFAULT':
                    weight_variants.append((w.name, w))
        except AttributeError:
            # Fallback if get_model_weights is missing for this model
            weight_variants.append(('DEFAULT', None))
            
        if not weight_variants:
            weight_variants.append(('DEFAULT', None))
            
        for weight_name, weight_enum in weight_variants:
            try:
                train_model(model_name, weight_name, weight_enum, args, dataloaders, dataset_sizes, class_names, device)
                total_runs += 1
            except Exception as e:
                print(f"Failed to train {model_name} with {weight_name}: {e}")

    print("\n" + "="*60)
    print(f"ALL MODELS PROCESSED! Completed {total_runs} specific architecture/weight combinations.")
    print("="*60)

if __name__ == "__main__":
    main()
