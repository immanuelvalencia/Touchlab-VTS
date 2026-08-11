"""Metadata scanning and random Bayesian touch-sequence generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_FEATURE_ALIASES = {
    "multiface_vertex": "multi_face_vertex",
    "multi-face-vertex": "multi_face_vertex",
}

DEFAULT_SHAPE_ALIASES = {
    "pyramid": "square_pyramid",
}


@dataclass(frozen=True)
class DatasetFeatureScan:
    root: Path
    features_by_shape: Mapping[str, Tuple[str, ...]]
    records_by_shape: Mapping[str, int]
    metadata_files: int
    accepted_records: int
    ignored_records: int
    invalid_records: int
    unknown_shapes: Mapping[str, int]
    unknown_features: Mapping[str, int]


def _normalize(value: object, aliases: Mapping[str, str]) -> str:
    normalized = str(value or "").strip().lower()
    return aliases.get(normalized, normalized)


def scan_local_features(
    root: Path,
    *,
    valid_shapes: Optional[Iterable[str]] = None,
    valid_features: Optional[Iterable[str]] = None,
    shape_aliases: Mapping[str, str] = DEFAULT_SHAPE_ALIASES,
    feature_aliases: Mapping[str, str] = DEFAULT_FEATURE_ALIASES,
) -> DatasetFeatureScan:
    """Collects each unique metadata local_feature for every object class."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    allowed_shapes = set(valid_shapes) if valid_shapes is not None else None
    allowed_features = set(valid_features) if valid_features is not None else None
    features_by_shape: Dict[str, Set[str]] = defaultdict(set)
    records_by_shape: Counter[str] = Counter()
    unknown_shapes: Counter[str] = Counter()
    unknown_features: Counter[str] = Counter()
    metadata_files = 0
    accepted_records = 0
    ignored_records = 0
    invalid_records = 0

    for path in root.rglob("*.json"):
        if "metadata" not in path.stem.lower():
            continue
        metadata_files += 1
        try:
            with path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            invalid_records += 1
            continue

        if not isinstance(metadata, dict):
            invalid_records += 1
            continue
        custom_fields = metadata.get("custom_fields")
        if not isinstance(custom_fields, dict):
            ignored_records += 1
            continue

        raw_shape = metadata.get("label")
        raw_feature = custom_fields.get("local_feature")
        shape = _normalize(raw_shape, shape_aliases)
        feature = _normalize(raw_feature, feature_aliases)
        if not shape or not feature:
            ignored_records += 1
            continue
        if allowed_shapes is not None and shape not in allowed_shapes:
            unknown_shapes[shape] += 1
            continue
        if allowed_features is not None and feature not in allowed_features:
            unknown_features[feature] += 1
            continue

        features_by_shape[shape].add(feature)
        records_by_shape[shape] += 1
        accepted_records += 1

    return DatasetFeatureScan(
        root=root,
        features_by_shape={
            shape: tuple(sorted(features))
            for shape, features in sorted(features_by_shape.items())
        },
        records_by_shape=dict(sorted(records_by_shape.items())),
        metadata_files=metadata_files,
        accepted_records=accepted_records,
        ignored_records=ignored_records,
        invalid_records=invalid_records,
        unknown_shapes=dict(sorted(unknown_shapes.items())),
        unknown_features=dict(sorted(unknown_features.items())),
    )


def generate_random_sequences(
    features_by_shape: Mapping[str, Sequence[str]],
    *,
    sequences_per_shape: int,
    minimum_length: int,
    maximum_length: int,
    seed: int,
) -> List[dict]:
    """Samples feature labels with replacement from each class vocabulary."""
    if sequences_per_shape < 1:
        raise ValueError("sequences_per_shape must be at least 1")
    if minimum_length < 1:
        raise ValueError("minimum_length must be at least 1")
    if maximum_length < minimum_length:
        raise ValueError("maximum_length must be greater than or equal to minimum_length")

    randomizer = random.Random(seed)
    generated = []
    for shape in sorted(features_by_shape):
        vocabulary = tuple(sorted(set(features_by_shape[shape])))
        if not vocabulary:
            continue
        for _ in range(sequences_per_shape):
            length = randomizer.randint(minimum_length, maximum_length)
            sequence = [randomizer.choice(vocabulary) for _ in range(length)]
            generated.append(
                {
                    "shape": shape,
                    "sequence": sequence,
                }
            )
    return generated
