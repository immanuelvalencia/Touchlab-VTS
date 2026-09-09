"""Inspect an existing preprocess.py export without changing its images or splits.

python check_dataset.py --dataset_dir ml_dataset --report dataset_check.json
Add --hash-overlaps to compare bytes of same-named files across overlapping splits.
The default scan reads filenames and split_manifest.json only; no torch is needed.
Exit status: 0 = checks passed, 1 = dataset issues, 2 = command/read/write error.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from tools.dataset_contacts import AUGMENTATION, IMAGE_EXTENSIONS, contact_key


def inspect_manifest(root):
    path = root / "split_manifest.json"
    result = {"present": path.exists(), "overlaps": [], "errors": []}
    if not path.exists():
        return result
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("expected a JSON object")
        result["category_by"] = manifest.get("category_by")
        if result["category_by"] != "label":
            result["errors"].append("Attention training requires category_by='label'.")
        classes = manifest.get("classes")
        if not isinstance(classes, dict):
            raise ValueError("classes must be an object")
        owners = defaultdict(set)
        for label, splits in classes.items():
            if not isinstance(splits, dict):
                raise ValueError(f"{label}: splits must be an object")
            for split, groups in splits.items():
                if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
                    raise ValueError(f"{label}/{split}: acquisitions must be a list of strings")
                split = "val" if split in ("valid", "validation") else split
                for group in groups:
                    owners[group].add((label, split))
        result["overlaps"] = [
            {"acquisition": group, "owners": [
                {"label": label, "split": split} for label, split in sorted(locations)
            ]}
            for group, locations in sorted(owners.items()) if len(locations) > 1
        ]
    except (OSError, ValueError) as exc:
        result["errors"].append(f"Cannot validate split_manifest.json: {exc}")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_dataset(root, *, hash_overlaps=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    report = {
        "dataset_dir": str(root), "issues": [], "summary": {}, "overlaps": [],
        "evaluation_augmentations": [], "contacts_without_originals": [],
        "manifest": inspect_manifest(root), "hash_overlaps": hash_overlaps,
        "limitations": [
            "Contact identity is inferred from filenames and class-relative subfolders, as in train_attention.py.",
            "Reused sequence names from distinct acquisitions cannot be distinguished without source metadata.",
            "A group-only split manifest cannot prove which exported files belong to each source acquisition.",
            "Image decoding, renamed duplicates, and minimum touch counts are not checked.",
        ],
    }
    issues = report["issues"]
    val_names = [name for name in ("val", "valid", "validation") if (root / name).is_dir()]
    if len(val_names) != 1:
        issues.append("Expected exactly one validation directory: val, valid, or validation.")
    split_names = ["train", *val_names, "test"]
    all_contacts = defaultdict(dict)
    train_labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir()) if (root / "train").is_dir() else []
    if len(train_labels) < 2:
        issues.append("Object classification needs at least two train class folders.")
    for split in split_names:
        directory = root / split
        if not directory.is_dir():
            issues.append(f"Missing split directory: {split}")
            continue
        labels = sorted(p.name for p in directory.iterdir() if p.is_dir())
        if not labels or labels != train_labels:
            issues.append(f"{split}: class folders must match train and must not be empty.")
        report["summary"][split] = {}
        for label in labels:
            groups = defaultdict(list)
            label_dir = directory / label
            for path in sorted(label_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    key = f"{path.parent.relative_to(label_dir).as_posix()}/{contact_key(path)}"
                    groups[key].append(path)
            if not groups:
                issues.append(f"No images in {split}/{label}")
            augmentations = []
            for key, paths in groups.items():
                all_contacts[(label, key)][split] = paths
                augmented = [p for p in paths if AUGMENTATION.search(p.stem)]
                augmentations.extend(p.relative_to(root).as_posix() for p in augmented)
                if len(augmented) == len(paths):
                    report["contacts_without_originals"].append({"split": split, "label": label, "contact": key})
            if split != "train":
                report["evaluation_augmentations"].extend(augmentations)
            report["summary"][split][label] = {
                "images": sum(map(len, groups.values())), "contacts": len(groups),
                "augmentations": len(augmentations),
            }

    for (label, key), splits in sorted(all_contacts.items()):
        if len(splits) < 2:
            continue
        frame_owners = defaultdict(set)
        name_paths = defaultdict(list)
        for split, paths in splits.items():
            for path in paths:
                frame_owners[AUGMENTATION.sub("", path.stem)].add(split)
                name_paths[path.name].append(path)
        overlap = {
            "label": label, "contact": key,
            "splits": {split: [p.relative_to(root).as_posix() for p in paths] for split, paths in splits.items()},
            "shared_frame_stems": sorted(stem for stem, owners in frame_owners.items() if len(owners) > 1),
        }
        if hash_overlaps:
            comparisons = []
            for name, paths in sorted(name_paths.items()):
                if len(paths) < 2:
                    continue
                hashes = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
                comparisons.append({"filename": name, "sha256": hashes, "all_identical": len(set(hashes.values())) == 1})
            overlap["same_name_comparisons"] = comparisons
        report["overlaps"].append(overlap)

    if report["overlaps"]:
        issues.append(f"{len(report['overlaps'])} contact(s) occur in multiple splits.")
    if report["evaluation_augmentations"]:
        issues.append(f"{len(report['evaluation_augmentations'])} exported augmentation(s) found outside train.")
    if report["contacts_without_originals"]:
        issues.append(f"{len(report['contacts_without_originals'])} contact(s) have no original frame.")
    issues.extend(report["manifest"]["errors"])
    if report["manifest"]["overlaps"]:
        issues.append(f"{len(report['manifest']['overlaps'])} acquisition(s) have conflicting manifest owners.")
    report["ok"] = not issues
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", type=Path, default=Path("ml_dataset"))
    parser.add_argument("--report", type=Path, help="Write full findings and affected file paths to a new JSON file.")
    parser.add_argument("--hash-overlaps", action="store_true", help="Read only same-named files in overlapping contacts to compare SHA-256 hashes.")
    parser.add_argument("--examples", type=int, default=3, help="Example files per split per overlap (default: 3).")
    args = parser.parse_args(argv)
    if args.examples < 0:
        parser.error("--examples must be nonnegative")
    try:
        print(f"Scanning {args.dataset_dir.resolve()} (dataset files will not be changed)...", flush=True)
        report = check_dataset(args.dataset_dir, hash_overlaps=args.hash_overlaps)
        for split, labels in report["summary"].items():
            for label, counts in labels.items():
                print(f"{split}/{label}: {counts['images']} images, {counts['contacts']} contacts, {counts['augmentations']} augmentations")
        for overlap in report["overlaps"]:
            print(f"\nOVERLAP: {overlap['label']}/{overlap['contact']}")
            for split, paths in overlap["splits"].items():
                print(f"  {split}: {len(paths)} files")
                for path in paths[:args.examples]:
                    print(f"    {path}")
            shared = len(overlap["shared_frame_stems"])
            print(f"  Shared frame names (ignoring augmentation suffix): {shared}")
            if not shared:
                print("  Different frame names still share a contact identity; this also blocks training.")
            for comparison in overlap.get("same_name_comparisons", []):
                result = "identical bytes" if comparison["all_identical"] else "different bytes"
                print(f"  {comparison['filename']}: {result}")
        print("\nPASS (filename/manifest checks)" if report["ok"] else "\nFAIL")
        for issue in report["issues"]:
            print(f"  - {issue}")
        if not report["manifest"]["present"]:
            print("No split_manifest.json found; acquisition ownership was checked from filenames only.")
        print("Source acquisition identity and image validity are not verified. No dataset files were changed.")
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print(f"Full report: {args.report.resolve()}")
        return 0 if report["ok"] else 1
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
