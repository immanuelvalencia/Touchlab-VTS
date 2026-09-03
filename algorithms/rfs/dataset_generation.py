"""Generation and leakage-resistant splitting of synthetic RFS trials."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Iterable, Mapping, Sequence

from algorithms.bayesian.dataset_sequences import generate_random_sequences


SPLIT_NAMES = ("train", "validation", "test")


def _split_counts(item_count: int, ratios: Sequence[float]) -> list[int]:
    raw = [item_count * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = item_count - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1

    active = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if item_count >= len(active):
        for empty in (index for index in active if counts[index] == 0):
            donor = max(active, key=counts.__getitem__)
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[empty] += 1
    return counts


def split_generated_sequences(
    trials: Iterable[Mapping],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Stratifies by shape and keeps duplicate sequences in one split."""
    ratios = (float(train_ratio), float(validation_ratio), float(test_ratio))
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Split ratios must sum to 1.0")
    active_split_count = sum(ratio > 0.0 for ratio in ratios)
    if active_split_count == 0:
        raise ValueError("At least one split ratio must be positive")

    grouped = defaultdict(lambda: defaultdict(list))
    for index, trial in enumerate(trials, start=1):
        if not isinstance(trial, Mapping):
            raise ValueError(f"Generated trial {index} must be an object")
        shape = str(trial.get("shape", "")).strip().lower()
        sequence = trial.get("sequence")
        if not shape:
            raise ValueError(f"Generated trial {index} shape cannot be empty")
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Generated trial {index} sequence must be non-empty")
        if any(not isinstance(feature, str) or not feature for feature in sequence):
            raise ValueError(f"Generated trial {index} contains an invalid feature")
        normalized = {
            "shape": shape,
            "sequence": [feature.strip().lower() for feature in sequence],
        }
        grouped[shape][tuple(normalized["sequence"])].append(normalized)
    if not grouped:
        raise ValueError("No generated trials were supplied")

    result = {name: [] for name in SPLIT_NAMES}
    randomizer = random.Random(seed)
    for shape in sorted(grouped):
        sequence_groups = list(grouped[shape].values())
        if len(sequence_groups) < active_split_count:
            raise ValueError(
                f"Shape {shape!r} has only {len(sequence_groups)} unique generated "
                f"sequences; at least {active_split_count} are required to populate "
                "every active split. Increase the length range or feature vocabulary."
            )
        randomizer.shuffle(sequence_groups)
        counts = _split_counts(len(sequence_groups), ratios)
        start = 0
        for split, count in zip(SPLIT_NAMES, counts):
            selected = sequence_groups[start : start + count]
            result[split].extend(
                trial
                for duplicate_group in selected
                for trial in duplicate_group
            )
            start += count

    for split in SPLIT_NAMES:
        randomizer.shuffle(result[split])
    return result


def generate_and_split_sequences(
    features_by_shape: Mapping[str, Sequence[str]],
    *,
    sequences_per_shape: int,
    minimum_length: int,
    maximum_length: int,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[list[dict], dict[str, list[dict]]]:
    generated = generate_random_sequences(
        features_by_shape,
        sequences_per_shape=sequences_per_shape,
        minimum_length=minimum_length,
        maximum_length=maximum_length,
        seed=seed,
    )
    splits = split_generated_sequences(
        generated,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return generated, splits
