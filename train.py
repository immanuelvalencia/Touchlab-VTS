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

def setup_argparse():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on VisuoTactile Dataset.")
    parser.add_argument('--dataset_dir', type=str, default='ml_dataset', help='Path to the exported ML dataset (with train/val/test splits).')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the trained model weights.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (number of epochs with no improvement before stopping).')
    return parser.parse_args()

def main():
    args = setup_argparse()

    # Create run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Outputting run artifacts to: {run_dir}")

    # Device configuration (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Define Data Transformations
    # ResNet expects 224x224 images and specific normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(15),       # Randomly rotate +/- 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Slight color/lighting augmentation
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
    
    # Load Datasets
    image_datasets = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(args.dataset_dir, split)
        if os.path.exists(split_path):
            image_datasets[split] = datasets.ImageFolder(split_path, data_transforms[split])
        else:
            print(f"Warning: Split {split} not found in {args.dataset_dir}!")

    if 'train' not in image_datasets or len(image_datasets['train']) == 0:
        print("Error: Training dataset is empty or missing. Please run export_dataset.py first.")
        return

    # Create DataLoaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=(x == 'train'), num_workers=4) 
                   for x in image_datasets.keys()}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Classes found ({num_classes}): {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Load pre-trained ResNet-18
    print("Loading pre-trained ResNet-18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer to match our number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (reduces LR if validation loss plateaus)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize history tracking
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0

    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 15)

        for phase in ['train', 'val']:
            if phase not in dataloaders:
                continue

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data using tqdm for a progress bar
            with tqdm(dataloaders[phase], desc=phase.capitalize(), unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Update progress bar
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Early stopping and best model saving logic
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_path = os.path.join(run_dir, "best_resnet18_model.pth")
                    torch.save(best_model_wts, best_model_path)
                    print(f"[*] New best model saved to {best_model_path} (Loss: {best_loss:.4f})")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"Early stopping counter: {epochs_no_improve} out of {args.patience}")

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered! No improvement in validation loss for {args.patience} epochs.")
            break

    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:4f}")

    # Plot training history
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val' in dataloaders:
        plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    if 'val' in dataloaders:
        plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plot_path = os.path.join(run_dir, "training_history.png")
    plt.savefig(plot_path)
    print(f"Training plots saved to {plot_path}")

    # --- Test Evaluation ---
    if 'test' in dataloaders:
        print("\n--- Running Final Evaluation on Test Set ---")
        model.load_state_dict(best_model_wts)
        model.eval()
        
        test_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            with tqdm(dataloaders['test'], desc="Testing", unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    test_corrects += torch.sum(preds == labels.data)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        test_acc = test_corrects.double() / dataset_sizes['test']
        print(f"Final Test Accuracy: {test_acc:.4f}")
        
        # --- Generate Metrics Report ---
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        print("\nClassification Report:\n", report_str)
        
        # Save JSON Report
        report_path = os.path.join(run_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        # Save Text Report
        report_txt_path = os.path.join(run_dir, "classification_report.txt")
        with open(report_txt_path, "w") as f:
            f.write(f"Final Test Accuracy: {test_acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_str)
            
        print(f"Metrics saved to {report_path} and {report_txt_path}")
        
        # --- Generate Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.tight_layout()
        cm_path = os.path.join(run_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion Matrix saved to {cm_path}")

if __name__ == "__main__":
    main()
