"""Sequential Naive Bayes classifier for tactile feature observations."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .config import NaiveBayesConfig


def _softmax(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


class NaiveBayesClassifier:
    """Accumulates independent local-feature likelihood evidence."""

    def __init__(self, config: NaiveBayesConfig) -> None:
        self.config = config
        self.config_path = str(config.source_path) if config.source_path else ""
        self.features = config.features
        self.shapes = config.shapes
        self.aliases = config.aliases
        self.reset()

    def reset(self) -> None:
        self.touch_count = 0
        self._scores = {
            shape: math.log(self.config.class_prior[shape])
            for shape in self.shapes
        }

    def update(self, feature_evidence) -> dict[str, float]:
        probabilities = self._normalize_evidence(feature_evidence)
        for shape in self.shapes:
            self._scores[shape] += sum(
                probabilities[feature] * math.log(self.config.likelihoods[shape][feature])
                for feature in self.features
            )
        self.touch_count += 1
        scores = [
            self._scores[shape] / self.config.score_temperature
            for shape in self.shapes
        ]
        values = _softmax(scores)
        return {shape: values[index] for index, shape in enumerate(self.shapes)}

    def _normalize_evidence(self, feature_evidence) -> dict[str, float]:
        if isinstance(feature_evidence, str):
            feature = self._normalize_feature(feature_evidence)
            return {
                name: 1.0 if name == feature else 0.0
                for name in self.features
            }
        if not isinstance(feature_evidence, Mapping):
            raise TypeError("Naive Bayes input must be a feature label or probability map")

        values = {feature: 0.0 for feature in self.features}
        for raw_label, value in feature_evidence.items():
            feature = self._normalize_feature(str(raw_label))
            probability = float(value)
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError("Feature probabilities must be finite and nonnegative")
            values[feature] += probability
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("Feature probability map must contain positive mass")
        return {feature: value / total for feature, value in values.items()}

    def _normalize_feature(self, raw_label: str) -> str:
        label = raw_label.strip().lower()
        label = self.aliases.get(label, label)
        if label not in self.features:
            raise ValueError(f"Unknown Naive Bayes feature label: {raw_label!r}")
        return label

