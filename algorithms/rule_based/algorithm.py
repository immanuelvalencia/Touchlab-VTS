"""Additive rule-based classification over observed hard feature labels."""

from __future__ import annotations

import math
from typing import Dict, Sequence

from .config import RuleBasedConfig


def _softmax(scores: Sequence[float], temperature: float) -> list[float]:
    scaled = [score / temperature for score in scores]
    maximum = max(scaled)
    weights = [math.exp(score - maximum) for score in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


class RuleBasedClassifier:
    """Scores each shape with explicit positive and negative feature rules."""

    def __init__(self, config: RuleBasedConfig) -> None:
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
        observed = {name for name, count in self._counts.items() if count > 0}

        scores = []
        for shape in self.shapes:
            rule = self.config.rules[shape]
            score = math.log(self.config.class_prior[shape])
            score += self.config.positive_weight * sum(
                rule["positive"][name] for name in observed
            )
            score -= self.config.negative_weight * sum(
                rule["negative"][name] for name in observed
            )
            score += self.config.repetition_weight * sum(
                max(0, count - 1) * rule["positive"][name]
                for name, count in self._counts.items()
            )
            scores.append(score)

        probabilities = _softmax(scores, self.config.score_temperature)
        return {shape: probabilities[index] for index, shape in enumerate(self.shapes)}

    def _normalize_feature(self, raw_label: str) -> str:
        if not isinstance(raw_label, str):
            raise TypeError("Rule-based input must be one feature-label string")
        label = raw_label.strip().lower()
        label = self.aliases.get(label, label)
        if label not in self._counts:
            raise ValueError(f"Unknown rule-based feature label: {raw_label!r}")
        return label

