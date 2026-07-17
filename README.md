# Training Guide (`train.py`)

This document explains how to use the `train.py` script to train a ResNet-18 image classification model on your exported VisuoTactile dataset.

## Overview
`train.py` handles the training, validation, and testing phases of the machine learning pipeline. It includes automatic early stopping, learning rate scheduling, and comprehensive metric reporting.

## Prerequisites
Before running `train.py`, ensure you have generated an ML-ready dataset using the dataset export tool (`export_dataset.py` or `export_dataset_gui.py`). The dataset should be structured in `train`, `val`, and `test` folders.

---

## Basic Usage

To run the training script with default settings, simply execute:
```bash
python train.py
```

By default, the script looks for the dataset in a folder named `ml_dataset` and saves the training outputs to a folder named `models`.

---

## Advanced Usage & Arguments

You can customize the training process using several command-line arguments:

| Argument        |  Type  |   Default    | Description                                                                                    |
| :-------------- | :----: | :----------: | :--------------------------------------------------------------------------------------------- |
| `--dataset_dir` | string | `ml_dataset` | Path to the exported ML dataset (must contain `train/`, `val/`, and `test/` subfolders).       |
| `--epochs`      |  int   |     `20`     | Maximum number of epochs to train.                                                             |
| `--batch_size`  |  int   |     `32`     | Batch size for training and validation. Lower this if you run out of GPU memory.               |
| `--lr`          | float  |   `0.001`    | Initial learning rate.                                                                         |
| `--output_dir`  | string |   `models`   | Base directory to save the trained model runs.                                                 |
| `--patience`    |  int   |     `5`      | Number of epochs to wait for validation loss improvement before triggering **early stopping**. |

### Example Command

```bash
python train.py --dataset_dir my_custom_dataset --epochs 50 --batch_size 16 --lr 0.0005 --patience 10
```

---

## Output Artifacts

Every time you run `train.py`, it generates a unique run folder inside your output directory, timestamped to keep your experiments organized (e.g., `models/run_20260717_123000/`).

Inside this run folder, the following files are automatically generated:

1. **`best_resnet18_model.pth`**
   - The PyTorch state dictionary (weights) of the model that achieved the best validation accuracy.
2. **`training_history.png`**
   - A plot showing the Training and Validation Loss and Accuracy over all epochs.
3. **`classification_report.txt`**
   - A text report containing the final overall Test Accuracy, as well as a per-class breakdown of Precision, Recall, and F1-score.
4. **`classification_report.json`**
   - The same classification metrics formatted as JSON, making it easy to parse for downstream analysis scripts.
5. **`confusion_matrix.png`**
   - A heatmap image of the Confusion Matrix, useful for diagnosing which specific classes the model frequently confuses.

---

## Data Augmentations
The script automatically applies standard augmentations during the training phase to improve model robustness, including:
- Random Resized Crops (224x224)
- Random Rotations (±15 degrees)
- Color Jittering (brightness and contrast)

Validation and testing phases only use center cropping. All images are normalized to ImageNet standards.
