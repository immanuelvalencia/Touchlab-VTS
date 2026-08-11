"""Leakage-resistant grouping and splitting for tactile image datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class DatasetRecord:
    source_file: Path
    sensor_prefix: str
    group_id: str


def metadata_group_id(metadata_path: Path, metadata: dict) -> str:
    """Keeps every frame from one video acquisition in the same split."""
    metadata_path = Path(metadata_path).resolve()
    if metadata.get("is_video_sequence", False):
        return f"video::{metadata_path.parent}"
    return f"capture::{metadata_path}"


def _split_group_counts(group_count: int, ratios: Sequence[float]) -> List[int]:
    raw_counts = [group_count * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw_counts]
    remainder = group_count - sum(counts)
    fractional_order = sorted(
        range(len(ratios)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1

    active = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if group_count >= len(active):
        for empty_index in (index for index in active if counts[index] == 0):
            donor = max(active, key=lambda index: counts[index])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[empty_index] += 1
    return counts


def grouped_split(
    records: Iterable[DatasetRecord],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[DatasetRecord]]:
    """Splits complete acquisition groups without frame-level leakage."""
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Split ratios must sum to 1.0")

    records_by_group = defaultdict(list)
    for record in records:
        records_by_group[record.group_id].append(record)

    group_ids = sorted(records_by_group)
    random.Random(seed).shuffle(group_ids)
    train_count, val_count, test_count = _split_group_counts(len(group_ids), ratios)
    boundaries = (train_count, train_count + val_count)
    group_splits = {
        "train": group_ids[: boundaries[0]],
        "val": group_ids[boundaries[0] : boundaries[1]],
        "test": group_ids[boundaries[1] : boundaries[1] + test_count],
    }
    return {
        split: [
            record
            for group_id in selected_groups
            for record in records_by_group[group_id]
        ]
        for split, selected_groups in group_splits.items()
    }
