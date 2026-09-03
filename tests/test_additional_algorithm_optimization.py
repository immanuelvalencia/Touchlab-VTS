import unittest

from algorithms.bag_of_features.config import BagOfFeaturesConfig
from algorithms.bag_of_features.optimization import optimize as optimize_bag_of_features
from algorithms.dirichlet_multinomial.config import DirichletMultinomialConfig
from algorithms.dirichlet_multinomial.optimization import optimize as optimize_dirichlet_multinomial
from algorithms.rule_based.config import RuleBasedConfig
from algorithms.rule_based.optimization import optimize as optimize_rule_based
from algorithms.single_touch_baseline.config import SingleTouchBaselineConfig
from algorithms.single_touch_baseline.optimization import optimize as optimize_single_touch


FEATURES = ["a", "b"]
SHAPES = ["left", "right"]
TRAIN = [
    {"shape": "left", "sequence": ["a", "a", "b"]},
    {"shape": "left", "sequence": ["a", "a"]},
    {"shape": "right", "sequence": ["b", "b", "a"]},
    {"shape": "right", "sequence": ["b", "b"]},
]
VALIDATION = [
    {"shape": "left", "sequence": ["a", "a"]},
    {"shape": "right", "sequence": ["b", "b"]},
]


def bag_config():
    return BagOfFeaturesConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "toy_bag",
            "features": FEATURES,
            "shapes": SHAPES,
            "aliases": {},
            "histogram_templates": {
                "left": {"a": 0.5, "b": 0.5},
                "right": {"a": 0.5, "b": 0.5},
            },
            "class_prior": {"left": 0.5, "right": 0.5},
            "parameters": {"distance_temperature": 1.0},
        }
    )


def dm_config():
    return DirichletMultinomialConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "toy_dm",
            "features": FEATURES,
            "shapes": SHAPES,
            "aliases": {},
            "alpha_templates": {
                "left": {"a": 1.0, "b": 1.0},
                "right": {"a": 1.0, "b": 1.0},
            },
            "class_prior": {"left": 0.5, "right": 0.5},
            "parameters": {"score_temperature": 1.0},
        }
    )


def single_touch_config():
    return SingleTouchBaselineConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "toy_single_touch",
            "features": FEATURES,
            "shapes": SHAPES,
            "aliases": {},
            "feature_posteriors": {
                "a": {"left": 0.5, "right": 0.5},
                "b": {"left": 0.5, "right": 0.5},
            },
            "parameters": {"score_temperature": 1.0},
        }
    )


def rule_based_config():
    empty = {feature: 0.0 for feature in FEATURES}
    return RuleBasedConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "toy_rules",
            "features": FEATURES,
            "shapes": SHAPES,
            "aliases": {},
            "rules": {
                "left": {"positive": dict(empty), "negative": dict(empty)},
                "right": {"positive": dict(empty), "negative": dict(empty)},
            },
            "class_prior": {"left": 0.5, "right": 0.5},
            "parameters": {
                "positive_weight": 1.0,
                "negative_weight": 1.0,
                "repetition_weight": 0.0,
                "score_temperature": 1.0,
            },
        }
    )


class AdditionalAlgorithmOptimizationTests(unittest.TestCase):
    def test_single_touch_optimizer_fits_feature_posteriors_and_temperature(self):
        optimized, report = optimize_single_touch(
            TRAIN,
            VALIDATION,
            single_touch_config(),
            smoothing_values=(0.1, 1.0),
            permutations=2,
            seed=7,
            name="optimized_single_touch",
        )

        parsed = SingleTouchBaselineConfig.from_mapping(optimized)
        self.assertGreater(
            parsed.feature_posteriors["a"]["left"],
            parsed.feature_posteriors["a"]["right"],
        )
        self.assertGreater(
            parsed.feature_posteriors["b"]["right"],
            parsed.feature_posteriors["b"]["left"],
        )
        self.assertGreater(parsed.score_temperature, 0.0)
        self.assertLess(report["validation_metrics"]["prefix"]["macro_nll"], 0.7)

    def test_rule_based_optimizer_fits_signed_feature_rules_and_temperature(self):
        optimized, report = optimize_rule_based(
            TRAIN,
            VALIDATION,
            rule_based_config(),
            smoothing_values=(0.1, 1.0),
            repetition_values=(0.0, 0.25),
            permutations=2,
            seed=7,
            name="optimized_rules",
        )

        parsed = RuleBasedConfig.from_mapping(optimized)
        self.assertGreater(parsed.rules["left"]["positive"]["a"], 0.0)
        self.assertGreater(parsed.rules["right"]["positive"]["b"], 0.0)
        self.assertGreater(parsed.rules["left"]["negative"]["b"], 0.0)
        self.assertGreater(parsed.score_temperature, 0.0)
        self.assertLess(report["validation_metrics"]["prefix"]["macro_nll"], 0.7)

    def test_bag_optimizer_fits_templates_and_selects_temperature(self):
        optimized, report = optimize_bag_of_features(
            TRAIN,
            VALIDATION,
            bag_config(),
            smoothing_values=(0.1, 1.0),
            permutations=2,
            seed=7,
            name="optimized_bag",
        )

        parsed = BagOfFeaturesConfig.from_mapping(optimized)
        self.assertGreater(
            parsed.histogram_templates["left"]["a"],
            parsed.histogram_templates["left"]["b"],
        )
        self.assertGreater(
            parsed.histogram_templates["right"]["b"],
            parsed.histogram_templates["right"]["a"],
        )
        self.assertGreater(parsed.distance_temperature, 0.0)
        self.assertLess(report["validation_metrics"]["prefix"]["macro_nll"], 0.7)

    def test_dm_optimizer_fits_positive_alpha_and_selects_temperature(self):
        optimized, report = optimize_dirichlet_multinomial(
            TRAIN,
            VALIDATION,
            dm_config(),
            permutations=2,
            seed=7,
            initial_concentration=5.0,
            max_iterations=100,
            name="optimized_dm",
        )

        parsed = DirichletMultinomialConfig.from_mapping(optimized)
        self.assertGreater(
            parsed.alpha_templates["left"]["a"],
            parsed.alpha_templates["left"]["b"],
        )
        self.assertGreater(
            parsed.alpha_templates["right"]["b"],
            parsed.alpha_templates["right"]["a"],
        )
        self.assertGreater(parsed.score_temperature, 0.0)
        self.assertLess(report["validation_metrics"]["prefix"]["macro_nll"], 0.7)


if __name__ == "__main__":
    unittest.main()
