"""Incremental Bag-of-Features classification using histogram prototypes."""

from __future__ import annotations

import math
from typing import Dict, Sequence

from .config import BagOfFeaturesConfig


def _softmax(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]

    def kl_divergence(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0
        )

    return 0.5 * kl_divergence(left) + 0.5 * kl_divergence(right)


class BagOfFeaturesClassifier:
    """Classifies the normalized histogram of deterministic feature labels."""

    def __init__(self, config: BagOfFeaturesConfig) -> None:
        self.config = config
        self.config_path = str(config.source_path) if config.source_path else ""
        self.features = config.features
        self.shapes = config.shapes
        self.aliases = config.aliases
        self.reset()

    def reset(self) -> None:
        self._counts: Dict[str, int] = {feature: 0 for feature in self.features}
        self.touch_count = 0

    def update(self, predicted_feature: str) -> dict[str, float]:
        feature = self._normalize_feature(predicted_feature)
        self._counts[feature] += 1
        self.touch_count += 1
        histogram = [self._counts[name] / self.touch_count for name in self.features]

        scores = []
        for shape in self.shapes:
            template = [
                self.config.histogram_templates[shape][name] for name in self.features
            ]
            distance = _jensen_shannon(histogram, template)
            scores.append(
                math.log(self.config.class_prior[shape])
                - distance / self.config.distance_temperature
            )
        probabilities = _softmax(scores)
        return {shape: probabilities[index] for index, shape in enumerate(self.shapes)}

    def _normalize_feature(self, raw_label: str) -> str:
        if not isinstance(raw_label, str):
            raise TypeError("Bag-of-Features input must be one feature-label string")
        label = raw_label.strip().lower()
        label = self.aliases.get(label, label)
        if label not in self._counts:
            raise ValueError(f"Unknown Bag-of-Features label: {raw_label!r}")
        return label
