import unittest

from algorithms.registry import (
    get_algorithm,
    get_algorithm_by_display_name,
    get_algorithm_specs,
)


class AlgorithmRegistryTests(unittest.TestCase):
    def test_expected_algorithms_are_registered_in_ui_order(self):
        specs = get_algorithm_specs()

        self.assertEqual([spec.key for spec in specs], ["bayesian", "rfs"])
        self.assertEqual(
            [spec.display_name for spec in specs],
            ["Bayesian", "Set Evidence"],
        )

    def test_lookup_supports_keys_and_display_names(self):
        self.assertIs(get_algorithm("BAYESIAN"), get_algorithm_by_display_name("Bayesian"))
        self.assertEqual(get_algorithm("rfs").module_path, "algorithms.rfs.factory")

    def test_bayesian_dispatches_the_top_feature(self):
        spec = get_algorithm("bayesian")
        algorithm = spec.create()

        result = spec.update(
            algorithm,
            feature_probabilities={"planar": 0.6, "double_curvature": 0.4},
            top_feature="planar",
        )

        self.assertAlmostEqual(sum(result.values()), 1.0)
        expected = max(
            algorithm.SHAPE_LIKELIHOODS["planar"],
            key=algorithm.SHAPE_LIKELIHOODS["planar"].get,
        )
        self.assertEqual(max(result, key=result.get), expected)

    def test_set_evidence_dispatches_only_the_top_label(self):
        spec = get_algorithm("rfs")
        algorithm = spec.create()

        result = spec.update(
            algorithm,
            feature_probabilities={"double_curvature": 0.9, "planar": 0.1},
            top_feature="double_curvature",
        )

        self.assertEqual(result.touch_count, 1)
        self.assertAlmostEqual(sum(result.belief.values()), 1.0)
        self.assertEqual(result.prediction, "sphere")
        self.assertEqual(spec.input_type, "hard_label")


if __name__ == "__main__":
    unittest.main()
