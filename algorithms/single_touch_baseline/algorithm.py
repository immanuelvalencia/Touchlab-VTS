"""Stateless shape classification from the latest hard feature label."""

from __future__ import annotations

import math
from typing import Sequence

from .config import SingleTouchBaselineConfig


def _softmax(scores: Sequence[float], temperature: float) -> list[float]:
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    weights = [math.exp(score - maximum) for score in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


class SingleTouchBaselineClassifier:
    """Uses only the most recent touch and discards all previous evidence."""

    def __init__(self, config: SingleTouchBaselineConfig) -> None:
        self.config = config
        self.config_path = str(config.source_path) if config.source_path else ""
        self.features = config.features
        self.shapes = config.shapes
        self.aliases = config.aliases
        self.reset()

    def reset(self) -> None:
        self.touch_count = 0
        self.last_feature = None

    def update(self, predicted_feature: str) -> dict[str, float]:
        feature = self._normalize_feature(predicted_feature)
        self.touch_count += 1
        self.last_feature = feature
        scores = [
            math.log(self.config.feature_posteriors[feature][shape])
            for shape in self.shapes
        ]
        probabilities = _softmax(scores, self.config.score_temperature)
        return {shape: probabilities[index] for index, shape in enumerate(self.shapes)}

    def _normalize_feature(self, raw_label: str) -> str:
        if not isinstance(raw_label, str):
            raise TypeError("Single-touch baseline input must be one feature-label string")
        label = raw_label.strip().lower()
        label = self.aliases.get(label, label)
        if label not in self.config.feature_posteriors:
            raise ValueError(f"Unknown single-touch feature label: {raw_label!r}")
        return label

