"""Train an object classifier on sets of touches from a preprocess.py export.

Example: python train_attention.py --dataset_dir ml_dataset --model_name resnet18
Each class folder is an object class. Frames and exported augmentations from one
acquisition form one touch. Bags are synthetic same-class contact collections;
they do not establish that contacts came from the same physical object instance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from tools.dataset_contacts import AUGMENTATION, IMAGE_EXTENSIONS, contact_key


BACKBONES = (
    [f"resnet{n}" for n in (18, 34, 50, 101, 152)]
    + [f"efficientnet_b{n}" for n in range(8)]
    + [f"efficientnet_v2_{s}" for s in ("s", "m", "l")]
    + ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]
    + [f"densenet{n}" for n in (121, 161, 169, 201)]
    + [f"convnext_{s}" for s in ("tiny", "small", "base", "large")]
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scan_export(root):
    """Use existing splits; detect acquisition overlap without reading raw data."""
    root = Path(root).resolve()
    manifest_path = root / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("category_by") != "label":
            raise ValueError("This export uses local-feature labels. Export with category_by='label' first.")
        owners = {}
        for label, splits in manifest.get("classes", {}).items():
            for split, groups in splits.items():
                split = "val" if split in ("val", "valid", "validation") else split
                for group in groups:
                    owner = (label, split)
                    if group in owners and owners[group] != owner:
                        raise ValueError(f"Acquisition appears in different splits/classes: {group}")
                    owners[group] = owner

    val_dirs = [root / name for name in ("val", "valid", "validation") if (root / name).is_dir()]
    if len(val_dirs) != 1:
        raise ValueError("Expected exactly one validation directory: val, valid, or validation.")
    split_dirs = {"train": root / "train", "val": val_dirs[0], "test": root / "test"}
    contacts = {}
    class_names = None
    owners = {}
    for split, directory in split_dirs.items():
        if not directory.is_dir():
            raise ValueError(f"Missing split directory: {directory}")
        labels = sorted(p.name for p in directory.iterdir() if p.is_dir())
        if class_names is None:
            class_names = labels
        if not labels or labels != class_names:
            raise ValueError(f"{split}: class folders must match train and must not be empty.")
        contacts[split] = {}
        for label in labels:
            groups = defaultdict(list)
            for path in sorted((directory / label).rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                # The exporter is flat. Preserve a subfolder prefix for manually grouped exports.
                relative_parent = path.parent.relative_to(directory / label).as_posix()
                key = f"{relative_parent}/{contact_key(path)}"
                identity = (label, key)
                if identity in owners and owners[identity] != split:
                    raise ValueError(
                        f"Contact occurs in both {owners[identity]} and {split}: {label}/{key}. "
                        "Frames from one acquisition must stay in a single split. "
                        "This can result from an older frame-level export or stale files in a reused output folder. "
                        "Re-export the raw dataset with preprocess.py --include-video --category label "
                        "--output_dir <new_empty_directory>, then train with --dataset_dir <new_empty_directory>."
                    )
                owners[identity] = split
                groups[key].append(path)
            if not groups:
                raise ValueError(f"No images in {directory / label}")
            if split != "train" and any(AUGMENTATION.search(p.stem) for ps in groups.values() for p in ps):
                raise ValueError(f"{split}/{label} contains exported augmentations; only train may contain them.")
            contacts[split][label] = dict(groups)
    if len(class_names) < 2:
        raise ValueError("Object classification needs at least two class folders.")
    return class_names, contacts


def representative_frame(paths):
    """Export has no contact-area metadata: use the middle original frame."""
    originals = sorted(p for p in paths if not AUGMENTATION.search(p.stem))
    if not originals:
        raise ValueError(f"Contact has no original frame: {paths[0]}")
    return originals[len(originals) // 2]


class TouchBags(Dataset):
    def __init__(self, contacts, class_names, *, bags_per_class, min_touches,
                 max_touches, image_size=224, training=False, seed=42):
        self.contacts = contacts
        self.class_names = class_names
        self.bags_per_class = bags_per_class
        self.min_touches = min_touches
        self.max_touches = max_touches
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.keys = {label: sorted(contacts[label]) for label in class_names}
        for label, keys in self.keys.items():
            if len(keys) < max_touches:
                raise ValueError(f"{label}: need {max_touches} distinct contacts, found {len(keys)}. Lower touch count.")
        # Resize the complete sensor view so a small contact is not cropped away.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.class_names) * self.bags_per_class

    def sample(self, index):
        label_index = index % len(self.class_names)
        label = self.class_names[label_index]
        epoch = self.epoch if self.training else 0
        rng = random.Random(self.seed + epoch * len(self) + index)
        # Draw the full ordered bag first: evaluation at K touches uses its prefix.
        keys = rng.sample(self.keys[label], self.max_touches)
        length = rng.randint(self.min_touches, self.max_touches)
        paths = []
        for key in keys[:length]:
            candidates = self.contacts[label][key]
            paths.append(rng.choice(candidates) if self.training else representative_frame(candidates))
        return label_index, keys[:length], paths

    def __getitem__(self, index):
        label, _, paths = self.sample(index)
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        return torch.stack(images), label

    def manifest(self, root):
        result = []
        for index in range(len(self)):
            label, keys, paths = self.sample(index)
            result.append({"label": self.class_names[label], "contacts": keys,
                           "images": [p.relative_to(root).as_posix() for p in paths]})
        return result


def collate_bags(batch):
    max_touches = max(len(images) for images, _ in batch)
    images = batch[0][0].new_zeros((len(batch), max_touches, *batch[0][0].shape[1:]))
    mask = torch.zeros(len(batch), max_touches, dtype=torch.bool)
    for index, (touches, _) in enumerate(batch):
        images[index, :len(touches)] = touches
        mask[index, :len(touches)] = True
    return images, mask, torch.tensor([label for _, label in batch], dtype=torch.long)


def build_encoder(model_name, weights):
    if model_name not in BACKBONES:
        raise ValueError(f"Unsupported backbone: {model_name}")
    weight_enum = None if weights.upper() == "NONE" else models.get_model_weights(model_name)[weights]
    encoder = models.get_model(model_name, weights=weight_enum)
    if hasattr(encoder, "fc"):
        dimension = encoder.fc.in_features
        encoder.fc = nn.Identity()
    elif isinstance(encoder.classifier, nn.Linear):
        dimension = encoder.classifier.in_features
        encoder.classifier = nn.Identity()
    else:
        last_linear = max(i for i, layer in enumerate(encoder.classifier) if isinstance(layer, nn.Linear))
        dimension = encoder.classifier[last_linear].in_features
        encoder.classifier[last_linear] = nn.Identity()
    return encoder, dimension


class AttentionClassifier(nn.Module):
    def __init__(self, num_classes, model_name="resnet18", weights="DEFAULT",
                 attention_dim=128, dropout=0.25, pooling="attention", freeze_backbone=False):
        super().__init__()
        self.encoder, dimension = build_encoder(model_name, weights)
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling
        self.attention_v = nn.Sequential(nn.Linear(dimension, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(dimension, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dimension, num_classes))
        if freeze_backbone:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, images, mask):
        """Accept [batch, touches, 3, H, W], return logits and touch weights."""
        if mask.dtype != torch.bool or mask.shape != images.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("Each bag needs at least one valid touch and a matching boolean mask.")
        # Never encode padding: it must not affect BatchNorm or the pooled result.
        valid_embeddings = self.encoder(images[mask])
        embeddings = valid_embeddings.new_zeros((*mask.shape, valid_embeddings.shape[-1]))
        embeddings[mask] = valid_embeddings
        if self.pooling == "mean":
            attention = mask.to(embeddings.dtype) / mask.sum(dim=1, keepdim=True)
        else:
            scores = self.attention_w(self.attention_v(embeddings) * self.attention_u(embeddings)).squeeze(-1)
            attention = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        pooled = (embeddings * attention.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled), attention


def run_epoch(model, loader, device, optimizer=None, touches=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    targets, predictions = [], []
    for images, mask, labels in tqdm(loader, desc="train" if optimizer else "evaluate", leave=False):
        if touches is not None:
            images, mask = images[:, :touches], mask[:, :touches]
        images, mask, labels = images.to(device), mask.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits, _ = model(images, mask)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check input images and learning rates.")
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += loss.item() * len(labels)
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {"loss": total_loss / len(targets),
            "accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets)}, targets, predictions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=Path, default=Path("ml_dataset"), help="Export with train, val/valid, test object folders.")
    parser.add_argument("--model_name", choices=BACKBONES, default="resnet18", help="Shared touch-image backbone.")
    parser.add_argument("--weights", default="DEFAULT", help="Torchvision weight enum name, DEFAULT, or NONE (no download).")
    parser.add_argument("--pooling", choices=("attention", "mean"), default="attention", help="Touch aggregation method.")
    parser.add_argument("--attention_dim", type=int, default=128, help="Hidden dimension of gated attention.")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout before object classification.")
    parser.add_argument("--min_touches", type=int, default=1, help="Minimum training bag size.")
    parser.add_argument("--max_touches", type=int, default=8, help="Maximum training bag size.")
    parser.add_argument("--eval_touches", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="Test prefix lengths; unavailable counts are reported as skipped.")
    parser.add_argument("--train_bags_per_class", type=int, default=100, help="Synthetic bags per class per epoch.")
    parser.add_argument("--eval_bags_per_class", type=int, default=50, help="Fixed bags per class in each evaluation split.")
    parser.add_argument("--batch_size", type=int, default=4, help="Object bags per batch, not image count.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Attention and object-head learning rate.")
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Backbone fine-tuning learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Train pooling and object head only.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience on validation loss.")
    parser.add_argument("--image_size", type=int, default=224, help="Resize full images to this square size.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; 0 is convenient on Windows.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling and initialization seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Training device.")
    parser.add_argument("--output_dir", type=Path, default=Path("train"), help="Parent directory for timestamped runs.")
    parser.add_argument("--dry_run", action="store_true", help="Validate export and save bag manifests without loading a model.")
    args = parser.parse_args(argv)
    for name in ("attention_dim", "min_touches", "max_touches", "train_bags_per_class",
                 "eval_bags_per_class", "batch_size", "epochs", "patience"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.max_touches < args.min_touches or min(args.eval_touches) < 1:
        parser.error("Touch counts must be positive and max_touches >= min_touches")
    if args.image_size < 32 or args.num_workers < 0 or not 0 <= args.dropout < 1:
        parser.error("Require image_size >= 32, num_workers >= 0, and 0 <= dropout < 1")
    if args.lr <= 0 or args.backbone_lr <= 0 or args.weight_decay < 0:
        parser.error("Learning rates must be positive and weight_decay nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = args.dataset_dir.resolve()
    class_names, contacts = scan_export(root)
    print("Classes:", ", ".join(class_names), flush=True)
    counts = {split: {label: len(groups) for label, groups in by_class.items()}
              for split, by_class in contacts.items()}
    print("Distinct contacts per split:", json.dumps(counts, indent=2), flush=True)
    print("Bags combine same-class contacts; physical-object identity is not verified.", flush=True)
    val_max = min(args.max_touches, min(counts["val"].values()))
    if val_max < args.min_touches:
        raise ValueError("Validation has too few contacts for min_touches; lower it or adjust the export.")
    if val_max < args.max_touches:
        print(f"Validation bags use {args.min_touches}–{val_max} touches (limited by the smallest class).", flush=True)
    test_limit = min(counts["test"].values())
    eval_counts = sorted(set(k for k in args.eval_touches if k <= test_limit))
    skipped = sorted(set(args.eval_touches) - set(eval_counts))
    if not eval_counts:
        raise ValueError(f"No requested test count is supported; smallest test class has {test_limit} contacts.")
    if skipped:
        print(f"Skipping test counts {skipped}: smallest test class has {test_limit} distinct contacts.", flush=True)

    common = dict(class_names=class_names, image_size=args.image_size)
    datasets = {
        "train": TouchBags(contacts["train"], **common, bags_per_class=args.train_bags_per_class,
                           min_touches=args.min_touches, max_touches=args.max_touches, training=True, seed=args.seed),
        "val": TouchBags(contacts["val"], **common, bags_per_class=args.eval_bags_per_class,
                         min_touches=args.min_touches, max_touches=val_max, seed=args.seed + 100_000),
        "test": TouchBags(contacts["test"], **common, bags_per_class=args.eval_bags_per_class,
                          min_touches=max(eval_counts), max_touches=max(eval_counts), seed=args.seed + 200_000),
    }
    run_dir = args.output_dir.resolve() / f"{args.model_name}_{args.pooling}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(dataset_dir=str(root), class_names=class_names, contact_counts=counts,
                  supported_eval_touches=eval_counts, skipped_eval_touches=skipped,
                  validation_max_touches=val_max,
                  evaluation_frame_selection="middle_original_frame", bag_semantics="synthetic_same_class_contacts")
    write_json(run_dir / "config.json", config)
    (run_dir / "labels.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    write_json(run_dir / "contacts.json", {split: {label: {key: [p.relative_to(root).as_posix() for p in paths]
        for key, paths in groups.items()} for label, groups in by_class.items()} for split, by_class in contacts.items()})
    for split in ("val", "test"):
        write_json(run_dir / f"{split}_bags.json", datasets[split].manifest(root))
    print(f"Run artifacts: {run_dir}", flush=True)
    if args.dry_run:
        print("Dataset validation complete; no training performed.")
        return run_dir

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print(f"Device: {device}", flush=True)
    model_config = dict(num_classes=len(class_names), model_name=args.model_name,
                        attention_dim=args.attention_dim, dropout=args.dropout,
                        pooling=args.pooling, freeze_backbone=args.freeze_backbone)
    model = AttentionClassifier(**model_config, weights=args.weights).to(device)
    head_parameters = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    parameter_groups = [{"params": head_parameters, "lr": args.lr}]
    if not args.freeze_backbone:
        parameter_groups.append({"params": model.encoder.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
               num_workers=args.num_workers, collate_fn=collate_bags, pin_memory=device.type == "cuda",
               generator=torch.Generator().manual_seed(args.seed + offset))
               for offset, (split, dataset) in enumerate(datasets.items())}
    best_loss, stale_epochs, history = float("inf"), 0, []
    checkpoint_path = run_dir / "best_model.pth"
    for epoch in range(args.epochs):
        datasets["train"].epoch = epoch
        train_metrics, _, _ = run_epoch(model, loaders["train"], device, optimizer)
        val_metrics, _, _ = run_epoch(model, loaders["val"], device)
        scheduler.step(val_metrics["loss"])
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        write_json(run_dir / "history.json", history)
        print(f"Epoch {epoch + 1}/{args.epochs}: train loss={train_metrics['loss']:.4f} "
              f"acc={train_metrics['accuracy']:.3f}; val loss={val_metrics['loss']:.4f} "
              f"acc={val_metrics['accuracy']:.3f}", flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss, stale_epochs = val_metrics["loss"], 0
            torch.save({"format_version": 1, "model_state_dict": model.state_dict(),
                        "model_config": model_config, "class_names": class_names,
                        "image_size": args.image_size, "epoch": epoch + 1,
                        "validation": val_metrics, "training_config": config}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    from sklearn.metrics import classification_report, confusion_matrix
    reports = {"best_epoch": checkpoint["epoch"], "skipped_touch_counts": skipped, "by_touch_count": {}}
    for count in eval_counts:
        metrics, targets, predictions = run_epoch(model, loaders["test"], device, touches=count)
        reports["by_touch_count"][str(count)] = {
            **metrics, "classification_report": classification_report(targets, predictions,
                labels=list(range(len(class_names))), target_names=class_names, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(class_names)))).tolist(),
            "targets": targets, "predictions": predictions,
        }
        print(f"Test, {count} touches: accuracy={metrics['accuracy']:.3f}", flush=True)
        write_json(run_dir / "test_metrics.json", reports)
    print(f"Best checkpoint: {checkpoint_path}")
    return run_dir


if __name__ == "__main__":
    main()
