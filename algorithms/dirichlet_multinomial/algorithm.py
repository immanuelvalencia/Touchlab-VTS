"""Incremental Dirichlet-Multinomial classifier for hard tactile labels."""

from __future__ import annotations

import math
from typing import Dict, Sequence

from .config import DirichletMultinomialConfig


def _softmax(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


class DirichletMultinomialClassifier:
    """Scores the complete unordered feature-count vector for every shape."""

    def __init__(self, config: DirichletMultinomialConfig) -> None:
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

        scores = [
            (
                math.log(self.config.class_prior[shape])
                + self._log_count_probability(shape)
            )
            / self.config.score_temperature
            for shape in self.shapes
        ]
        probabilities = _softmax(scores)
        return {shape: probabilities[index] for index, shape in enumerate(self.shapes)}

    def _log_count_probability(self, shape: str) -> float:
        alpha = self.config.alpha_templates[shape]
        alpha_zero = sum(alpha.values())
        score = math.lgamma(alpha_zero) - math.lgamma(self.touch_count + alpha_zero)
        for feature in self.features:
            score += math.lgamma(self._counts[feature] + alpha[feature])
            score -= math.lgamma(alpha[feature])
        # The multinomial coefficient depends only on the observed count vector,
        # so it cancels when class posteriors are normalized.
        return score

    def _normalize_feature(self, raw_label: str) -> str:
        if not isinstance(raw_label, str):
            raise TypeError("Dirichlet-Multinomial input must be one feature-label string")
        label = raw_label.strip().lower()
        label = self.aliases.get(label, label)
        if label not in self._counts:
            raise ValueError(f"Unknown Dirichlet-Multinomial label: {raw_label!r}")
        return label
