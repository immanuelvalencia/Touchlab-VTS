"""Central registration and dispatch for shape-classification algorithms.

To add an algorithm, create its folder under ``algorithms`` and add one
``register_algorithm`` call at the bottom of this file. ``predict.py`` reads
this registry to construct selectors, comparison controls, and instances.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


VALID_INPUT_TYPES = {"hard_label", "probabilities"}


@dataclass(frozen=True)
class AlgorithmSpec:
    """Import and input-adapter metadata for one registered algorithm."""

    key: str
    display_name: str
    module_path: str
    update_method: str
    input_type: str
    description: str = ""
    class_name: str = ""
    factory_name: str = ""
    model_config_suffix: str = ""

    def config_path_for_model(self, model_path: str = "") -> str:
        if not model_path or not self.model_config_suffix:
            return ""
        candidate = Path(model_path).with_suffix(self.model_config_suffix)
        return str(candidate) if candidate.is_file() else ""

    def create(self, model_path: str = "", config_path: str = "") -> Any:
        module = importlib.import_module(self.module_path)
        if self.factory_name:
            factory = getattr(module, self.factory_name)
            selected_config = config_path or self.config_path_for_model(model_path)
            instance = factory(config_path=selected_config or None)
        else:
            algorithm_class = getattr(module, self.class_name)
            instance = algorithm_class()
        if not callable(getattr(instance, self.update_method, None)):
            raise TypeError(
                f"Registered algorithm {self.key!r} does not provide callable "
                f"method {self.update_method!r}"
            )
        return instance

    def update(
        self,
        instance: Any,
        feature_probabilities: Mapping[str, float],
        top_feature: str,
    ) -> Any:
        payload = (
            top_feature
            if self.input_type == "hard_label"
            else dict(feature_probabilities)
        )
        return getattr(instance, self.update_method)(payload)


_ALGORITHMS: Dict[str, AlgorithmSpec] = {}


def register_algorithm(
    *,
    key: str,
    display_name: str,
    module_path: str,
    update_method: str,
    input_type: str,
    description: str = "",
    class_name: str = "",
    factory_name: str = "",
    model_config_suffix: str = "",
) -> AlgorithmSpec:
    """Registers one algorithm and rejects ambiguous registry entries."""
    normalized_key = key.strip().lower()
    normalized_name = display_name.strip()
    if not normalized_key:
        raise ValueError("Algorithm key cannot be empty")
    if not normalized_name:
        raise ValueError("Algorithm display_name cannot be empty")
    if input_type not in VALID_INPUT_TYPES:
        raise ValueError(
            f"input_type must be one of {sorted(VALID_INPUT_TYPES)}, got {input_type!r}"
        )
    if bool(class_name) == bool(factory_name):
        raise ValueError("Register exactly one of class_name or factory_name")
    if normalized_key in _ALGORITHMS:
        raise ValueError(f"Algorithm key {normalized_key!r} is already registered")
    if any(spec.display_name == normalized_name for spec in _ALGORITHMS.values()):
        raise ValueError(f"Algorithm display name {normalized_name!r} is already registered")

    spec = AlgorithmSpec(
        key=normalized_key,
        display_name=normalized_name,
        module_path=module_path,
        update_method=update_method,
        input_type=input_type,
        description=description.strip(),
        class_name=class_name.strip(),
        factory_name=factory_name.strip(),
        model_config_suffix=model_config_suffix.strip(),
    )
    _ALGORITHMS[normalized_key] = spec
    return spec


def get_algorithm_specs() -> Tuple[AlgorithmSpec, ...]:
    """Returns algorithms in their registration and UI display order."""
    return tuple(_ALGORITHMS.values())


def get_algorithm(key: str) -> AlgorithmSpec:
    try:
        return _ALGORITHMS[key.strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown algorithm key: {key!r}") from exc


def get_algorithm_by_display_name(display_name: str) -> AlgorithmSpec:
    for spec in _ALGORITHMS.values():
        if spec.display_name == display_name:
            return spec
    raise KeyError(f"Unknown algorithm display name: {display_name!r}")


# Add future algorithms here after creating their folder under algorithms/.
register_algorithm(
    key="bayesian",
    display_name="Bayesian",
    module_path="algorithms.bayesian.factory",
    factory_name="create_algorithm",
    model_config_suffix=".bayesian.json",
    update_method="update",
    input_type="hard_label",
    description="Sequential Bayesian update using the highest-probability feature label.",
)

register_algorithm(
    key="naive_bayes",
    display_name="Naive Bayes",
    module_path="algorithms.naive_bayes.factory",
    factory_name="create_algorithm",
    model_config_suffix=".nb.json",
    update_method="update",
    input_type="probabilities",
    description="Baseline Naive Bayes accumulation of local tactile feature evidence.",
)

register_algorithm(
    key="modified_naive_bayes",
    display_name="Modified Naive Bayes",
    module_path="algorithms.modified_naive_bayes.factory",
    factory_name="create_algorithm",
    model_config_suffix=".mnb.json",
    update_method="update",
    input_type="probabilities",
    description=(
        "Naive Bayes with repeated-touch damping, feature weighting, "
        "coverage evidence, and unexpected-feature penalties."
    ),
)

register_algorithm(
    key="single_touch_baseline",
    display_name="Single Touch Baseline",
    module_path="algorithms.single_touch_baseline.factory",
    factory_name="create_algorithm",
    model_config_suffix=".single_touch.json",
    update_method="update",
    input_type="hard_label",
    description="Latest-touch-only reference classifier for local-feature to shape mapping.",
)

register_algorithm(
    key="rule_based",
    display_name="Rule-Based",
    module_path="algorithms.rule_based.factory",
    factory_name="create_algorithm",
    model_config_suffix=".rules.json",
    update_method="update",
    input_type="hard_label",
    description="Transparent additive rules over observed positive and negative tactile features.",
)

register_algorithm(
    key="bag_of_features",
    display_name="Bag of Features",
    module_path="algorithms.bag_of_features.factory",
    factory_name="create_algorithm",
    model_config_suffix=".bof.json",
    update_method="update",
    input_type="hard_label",
    description="Permutation-invariant classification of tactile feature histograms.",
)

register_algorithm(
    key="dirichlet_multinomial",
    display_name="Dirichlet-Multinomial",
    module_path="algorithms.dirichlet_multinomial.factory",
    factory_name="create_algorithm",
    model_config_suffix=".dm.json",
    update_method="update",
    input_type="hard_label",
    description="Exchangeable count-vector classification with Dirichlet overdispersion.",
)

register_algorithm(
    key="rfs",
    display_name="Set Evidence",
    module_path="algorithms.rfs.factory",
    factory_name="create_algorithm",
    model_config_suffix=".rfs.json",
    update_method="add_touch",
    input_type="hard_label",
    description="RFS-inspired set accumulation of accurate local-feature labels.",
)
