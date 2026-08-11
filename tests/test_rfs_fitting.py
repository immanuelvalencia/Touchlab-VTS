import unittest

from algorithms.rfs.config import load_rfs_config
from algorithms.rfs.factory import DEFAULT_CONFIG_PATH
from algorithms.rfs.fitting import fit_rfs_config


class RFSFittingTests(unittest.TestCase):
    def test_templates_are_learned_and_normalized_from_trials(self):
        base = load_rfs_config(DEFAULT_CONFIG_PATH)
        distinctive = {
            "cube": "planar",
            "sphere": "double_curvature",
            "cylinder": "circular_rim",
            "cone": "single_curvature",
            "square_pyramid": "sharp_apex",
        }
        trials = [
            {"shape": shape, "sequence": [feature, feature]}
            for shape, feature in distinctive.items()
            for _ in range(2)
        ]

        fitted = fit_rfs_config(trials, base, name="test", dirichlet_alpha=0.1)

        for shape, feature in distinctive.items():
            self.assertAlmostEqual(sum(fitted["theta_templates"][shape].values()), 1.0)
            self.assertGreater(fitted["presence_templates"][shape][feature], 0.5)
            self.assertEqual(
                max(
                    fitted["theta_templates"][shape],
                    key=fitted["theta_templates"][shape].get,
                ),
                feature,
            )

    def test_presence_is_a_smoothed_trial_level_probability(self):
        base = load_rfs_config(DEFAULT_CONFIG_PATH)
        trials = []
        for shape in base.shapes:
            trials.extend(
                [
                    {"shape": shape, "sequence": ["planar"]},
                    {"shape": shape, "sequence": ["straight_edge"]},
                ]
            )

        fitted = fit_rfs_config(trials, base, name="presence")

        self.assertAlmostEqual(
            fitted["presence_templates"]["cube"]["planar"], 0.5
        )
        self.assertAlmostEqual(
            fitted["presence_templates"]["cube"]["double_curvature"],
            1.0 / 6.0,
        )

    def test_probability_mappings_are_rejected(self):
        base = load_rfs_config(DEFAULT_CONFIG_PATH)
        trials = [
            {"shape": shape, "sequence": [{"planar": 1.0}]}
            for shape in base.shapes
        ]
        with self.assertRaisesRegex(ValueError, "one local-feature label string"):
            fit_rfs_config(trials, base, name="invalid")

    def test_every_shape_requires_trials(self):
        base = load_rfs_config(DEFAULT_CONFIG_PATH)
        with self.assertRaisesRegex(ValueError, "No trials supplied"):
            fit_rfs_config(
                [{"shape": "cube", "sequence": ["planar"]}],
                base,
                name="incomplete",
            )


if __name__ == "__main__":
    unittest.main()
