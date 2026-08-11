import itertools
import math
import unittest

from algorithms.rfs.algorithm import SetEvidenceAccumulator
from algorithms.rfs.config import RFSConfig, load_rfs_config
from algorithms.rfs.factory import DEFAULT_CONFIG_PATH, create_algorithm


def make_accumulator(**overrides):
    parameters = {
        "lambda_evidence": 1.0,
        "lambda_coverage": 0.0,
        "unexpected_penalty": 0.25,
        "score_temperature": 1.0,
        "stop_threshold": 1.0,
        "entropy_threshold": 0.0,
        "min_touches": 3,
        "stability_window": 3,
        "max_touches": 8,
        "eps": 1e-8,
    }
    values = {
        "schema_version": 2,
        "name": "two_class_test",
        "features": ["a", "b"],
        "shapes": ["left", "right"],
        "aliases": {"alpha": "a"},
        "theta_templates": {
            "left": {"a": 0.8, "b": 0.2},
            "right": {"a": 0.1, "b": 0.9},
        },
        "presence_templates": {
            "left": {"a": 1.0, "b": 0.0},
            "right": {"a": 0.0, "b": 1.0},
        },
        "class_prior": {"left": 0.5, "right": 0.5},
        "parameters": parameters,
    }
    for name, value in overrides.items():
        if name in parameters:
            parameters[name] = value
        else:
            values[name] = value
    return SetEvidenceAccumulator(RFSConfig.from_mapping(values))


def two_class_softmax(left_score, right_score):
    maximum = max(left_score, right_score)
    left = math.exp(left_score - maximum)
    right = math.exp(right_score - maximum)
    return left / (left + right), right / (left + right)


class SetEvidenceEquationTests(unittest.TestCase):
    def test_default_vocabulary_and_templates_are_valid(self):
        config = load_rfs_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(len(config.features), 7)
        for template in config.theta_templates.values():
            self.assertAlmostEqual(sum(template.values()), 1.0, places=12)

    def test_hard_label_average_log_compatibility_matches_equation(self):
        accumulator = make_accumulator()
        result = accumulator.evaluate_trial(["a", "b"])

        left_compatibility = (
            math.log(0.8 + accumulator.eps) + math.log(0.2 + accumulator.eps)
        ) / 2.0
        right_compatibility = (
            math.log(0.1 + accumulator.eps) + math.log(0.9 + accumulator.eps)
        ) / 2.0
        expected_left = math.log(0.5 + accumulator.eps) + left_compatibility
        expected_right = math.log(0.5 + accumulator.eps) + right_compatibility
        expected_belief = two_class_softmax(expected_left, expected_right)

        self.assertAlmostEqual(result.scores["left"], expected_left, places=12)
        self.assertAlmostEqual(result.scores["right"], expected_right, places=12)
        self.assertAlmostEqual(result.belief["left"], expected_belief[0], places=12)

    def test_coverage_is_binary_and_repetition_does_not_inflate_it(self):
        once = make_accumulator(lambda_coverage=1.0).evaluate_trial(["a"])
        repeated = make_accumulator(lambda_coverage=1.0).evaluate_trial(["a"] * 20)

        self.assertEqual(once.feature_coverage, {"a": 1.0, "b": 0.0})
        self.assertEqual(repeated.feature_coverage, once.feature_coverage)
        self.assertEqual(repeated.belief, once.belief)

    def test_complementary_label_changes_coverage(self):
        result = make_accumulator().evaluate_trial(["a", "b"])
        self.assertEqual(result.feature_coverage, {"a": 1.0, "b": 1.0})

    def test_fixed_contact_set_is_permutation_invariant(self):
        touches = ["planar", "straight_edge", "multi_face_vertex"]
        beliefs = [
            create_algorithm().evaluate_trial(permutation).belief
            for permutation in itertools.permutations(touches)
        ]
        for belief in beliefs[1:]:
            for shape in belief:
                self.assertAlmostEqual(belief[shape], beliefs[0][shape], places=12)

    def test_belief_is_normalized_and_entropy_is_global(self):
        result = make_accumulator().add_touch("a")
        expected_entropy = -sum(
            probability * math.log(probability)
            for probability in result.belief.values()
        )
        self.assertAlmostEqual(sum(result.belief.values()), 1.0, places=12)
        self.assertAlmostEqual(result.entropy, expected_entropy, places=12)

    def test_alias_is_canonicalized(self):
        self.assertEqual(
            make_accumulator().add_touch(" ALPHA ").feature_coverage["a"], 1.0
        )

    def test_confident_stable_result_stops_as_accepted(self):
        accumulator = make_accumulator(
            theta_templates={
                "left": {"a": 0.999, "b": 0.001},
                "right": {"a": 0.001, "b": 0.999},
            },
            stop_threshold=0.9,
            entropy_threshold=0.5,
            min_touches=2,
            stability_window=2,
        )
        accumulator.add_touch("a")
        result = accumulator.add_touch("a")
        self.assertTrue(result.should_stop)
        self.assertFalse(result.is_uncertain)
        self.assertEqual(result.stopping_reason, "confidence")

    def test_max_touch_stop_is_explicitly_uncertain(self):
        accumulator = make_accumulator(max_touches=2, min_touches=3, stability_window=1)
        accumulator.add_touch("a")
        result = accumulator.add_touch("b")
        self.assertTrue(result.should_stop)
        self.assertTrue(result.is_uncertain)
        self.assertEqual(result.stopping_reason, "max_touches")

    def test_only_known_string_labels_are_accepted(self):
        accumulator = make_accumulator()
        with self.assertRaises(TypeError):
            accumulator.add_touch({"a": 1.0})
        with self.assertRaises(ValueError):
            accumulator.add_touch("unknown")


if __name__ == "__main__":
    unittest.main()
