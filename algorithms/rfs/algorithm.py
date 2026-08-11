"""Permutation-invariant hard-label set evidence for tactile shape recognition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .config import RFSConfig


def _normalize(values: Sequence[float], eps: float) -> List[float]:
    normalized = [float(value) for value in values]
    if not normalized or any(not math.isfinite(value) for value in normalized):
        raise ValueError("Probability values must be a non-empty finite vector")
    if any(value < 0.0 for value in normalized):
        raise ValueError("Probability values cannot be negative")
    total = sum(normalized)
    if total <= eps:
        raise ValueError("Probability values must have positive total mass")
    return [value / total for value in normalized]


def _softmax(scores: Sequence[float], temperature: float) -> List[float]:
    if not scores:
        raise ValueError("Cannot compute softmax of an empty score vector")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("score_temperature must be finite and greater than zero")
    scaled = [float(score) / temperature for score in scores]
    if any(not math.isfinite(score) for score in scaled):
        raise ValueError("Class scores must be finite")
    maximum = max(scaled)
    exponentials = [math.exp(score - maximum) for score in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


@dataclass
class EvidenceResult:
    prediction: str
    belief: Dict[str, float]
    scores: Dict[str, float]
    feature_coverage: Dict[str, float]
    entropy: float
    should_stop: bool
    is_uncertain: bool
    stopping_reason: Optional[str]
    touch_count: int


class SetEvidenceAccumulator:
    """Combines an unordered, variable-length collection of feature labels."""

    def __init__(self, config: RFSConfig) -> None:
        if not isinstance(config, RFSConfig):
            raise TypeError("config must be an RFSConfig")
        self.config = config
        self.config_path = config.source_path
        self.features = config.features
        self.shapes = config.shapes
        self.aliases = dict(config.aliases)
        self.theta_templates = config.theta_templates
        self.presence_templates = config.presence_templates
        self.class_prior = config.class_prior
        for name, value in config.parameters.items():
            setattr(self, name, value)
        self._validate_parameters()
        self._theta = self._build_theta()
        self._presence = self._build_presence()
        self._prior = self._build_prior()
        self.reset()

    def _validate_parameters(self) -> None:
        for name in ("lambda_evidence", "lambda_coverage", "unexpected_penalty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            setattr(self, name, value)

        self.eps = float(self.eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and greater than zero")
        self.score_temperature = float(self.score_temperature)
        if not math.isfinite(self.score_temperature) or self.score_temperature <= 0.0:
            raise ValueError("score_temperature must be finite and greater than zero")
        self.stop_threshold = float(self.stop_threshold)
        if not math.isfinite(self.stop_threshold) or not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError("stop_threshold must lie in [0, 1]")
        self.entropy_threshold = float(self.entropy_threshold)
        maximum_entropy = math.log(len(self.shapes))
        if (
            not math.isfinite(self.entropy_threshold)
            or not 0.0 <= self.entropy_threshold <= maximum_entropy
        ):
            raise ValueError(
                "entropy_threshold must lie in [0, log(number of shapes)]"
            )
        if not isinstance(self.min_touches, int) or self.min_touches < 1:
            raise ValueError("min_touches must be a positive integer")
        if not isinstance(self.stability_window, int) or self.stability_window < 1:
            raise ValueError("stability_window must be a positive integer")
        if not isinstance(self.max_touches, int) or self.max_touches < 1:
            raise ValueError("max_touches must be a positive integer")

    def reset(self) -> None:
        self._touches: List[str] = []
        self._prediction_history: List[str] = []

    def add_touch(self, predicted_feature: str) -> EvidenceResult:
        label = self._normalize_label(predicted_feature)
        if label not in self.features:
            raise ValueError(f"Unknown feature label: {predicted_feature!r}")
        self._touches.append(label)
        result = self.evaluate()
        self._prediction_history.append(result.prediction)
        return result

    def evaluate_trial(
        self,
        predictions: Iterable[str],
        *,
        reset: bool = True,
    ) -> EvidenceResult:
        if reset:
            self.reset()
        result: Optional[EvidenceResult] = None
        for prediction in predictions:
            result = self.add_touch(prediction)
        if result is None:
            raise ValueError("Cannot evaluate an empty trial")
        return result

    def evaluate(self) -> EvidenceResult:
        if not self._touches:
            raise ValueError("No touches have been added")
        compatibility = self._compatibility_scores()
        coverage = self._feature_coverage()
        coverage_scores = self._coverage_scores(coverage)
        scores = [
            math.log(self._prior[index] + self.eps)
            + self.lambda_evidence * compatibility[index]
            + self.lambda_coverage * coverage_scores[index]
            for index in range(len(self.shapes))
        ]
        beliefs = _softmax(scores, self.score_temperature)
        entropy = _entropy(beliefs)
        best_index = max(range(len(beliefs)), key=beliefs.__getitem__)
        prediction = self.shapes[best_index]
        stopping_reason = self._stopping_reason(prediction, beliefs, entropy)
        return EvidenceResult(
            prediction=prediction,
            belief={shape: beliefs[index] for index, shape in enumerate(self.shapes)},
            scores={shape: scores[index] for index, shape in enumerate(self.shapes)},
            feature_coverage={
                feature: coverage[index]
                for index, feature in enumerate(self.features)
            },
            entropy=entropy,
            should_stop=stopping_reason is not None,
            is_uncertain=stopping_reason == "max_touches",
            stopping_reason=stopping_reason,
            touch_count=len(self._touches),
        )

    def _normalize_label(self, raw_label: str) -> str:
        if not isinstance(raw_label, str):
            raise TypeError("RFS touch input must be one feature-label string")
        label = raw_label.strip().lower()
        return self.aliases.get(label, label)

    def _compatibility_scores(self) -> List[float]:
        touch_count = len(self._touches)
        return [
            sum(
                math.log(self._theta[shape_index][self.features.index(feature)] + self.eps)
                for feature in self._touches
            )
            / touch_count
            for shape_index in range(len(self.shapes))
        ]

    def _feature_coverage(self) -> List[float]:
        observed = set(self._touches)
        return [1.0 if feature in observed else 0.0 for feature in self.features]

    def _coverage_scores(self, coverage: Sequence[float]) -> List[float]:
        scores = []
        for shape_index in range(len(self.shapes)):
            positive = sum(
                self._presence[shape_index][feature_index] * coverage[feature_index]
                for feature_index in range(len(self.features))
            )
            unexpected = sum(
                (1.0 - self._presence[shape_index][feature_index])
                * coverage[feature_index]
                for feature_index in range(len(self.features))
            )
            scores.append(positive - self.unexpected_penalty * unexpected)
        return scores

    def _build_theta(self) -> List[List[float]]:
        self._validate_template_shapes(self.theta_templates, "theta_templates")
        rows = []
        for shape in self.shapes:
            row = self.theta_templates[shape]
            self._validate_template_features(row, f"theta template for {shape!r}")
            rows.append(_normalize([row[feature] for feature in self.features], self.eps))
        return rows

    def _build_presence(self) -> List[List[float]]:
        self._validate_template_shapes(self.presence_templates, "presence_templates")
        rows = []
        for shape in self.shapes:
            template = self.presence_templates[shape]
            self._validate_template_features(template, f"presence template for {shape!r}")
            row = [float(template[feature]) for feature in self.features]
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in row):
                raise ValueError(f"Presence values for {shape!r} must lie in [0, 1]")
            rows.append(row)
        return rows

    def _build_prior(self) -> List[float]:
        if self.class_prior is None:
            return [1.0 / len(self.shapes)] * len(self.shapes)
        missing = sorted(set(self.shapes) - set(self.class_prior))
        unknown = sorted(set(self.class_prior) - set(self.shapes))
        if missing or unknown:
            raise ValueError(
                f"class_prior mismatch; missing={missing}, unknown={unknown}"
            )
        return _normalize([self.class_prior[shape] for shape in self.shapes], self.eps)

    def _validate_template_shapes(
        self,
        templates: Mapping[str, Mapping[str, float]],
        name: str,
    ) -> None:
        missing = sorted(set(self.shapes) - set(templates))
        unknown = sorted(set(templates) - set(self.shapes))
        if missing or unknown:
            raise ValueError(f"{name} mismatch; missing={missing}, unknown={unknown}")

    def _validate_template_features(
        self,
        values: Mapping[str, float],
        name: str,
    ) -> None:
        missing = sorted(set(self.features) - set(values))
        unknown = sorted(set(values) - set(self.features))
        if missing or unknown:
            raise ValueError(f"{name} mismatch; missing={missing}, unknown={unknown}")

    def _stopping_reason(
        self,
        prediction: str,
        belief: Sequence[float],
        entropy: float,
    ) -> Optional[str]:
        confidence_ready = (
            len(self._touches) >= self.min_touches
            and max(belief) >= self.stop_threshold
            and entropy <= self.entropy_threshold
        )
        stability_ready = self.stability_window <= 1
        if not stability_ready:
            recent = self._prediction_history[-(self.stability_window - 1) :]
            stability_ready = (
                len(recent) == self.stability_window - 1
                and all(previous == prediction for previous in recent)
            )
        if confidence_ready and stability_ready:
            return "confidence"
        if len(self._touches) >= self.max_touches:
            return "max_touches"
        return None
