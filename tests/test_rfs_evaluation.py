import unittest

from algorithms.rfs.config import RFSConfig
from algorithms.rfs.evaluation import evaluate_rfs_trials


def config():
    return RFSConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "evaluation_test",
            "features": ["a", "b"],
            "shapes": ["left", "right"],
            "aliases": {},
            "theta_templates": {
                "left": {"a": 0.99, "b": 0.01},
                "right": {"a": 0.01, "b": 0.99},
            },
            "presence_templates": {
                "left": {"a": 1.0, "b": 0.0},
                "right": {"a": 0.0, "b": 1.0},
            },
            "class_prior": {"left": 0.5, "right": 0.5},
            "parameters": {
                "lambda_evidence": 1.0,
                "lambda_coverage": 0.5,
                "unexpected_penalty": 0.25,
                "score_temperature": 1.0,
                "stop_threshold": 0.8,
                "entropy_threshold": 0.5,
                "min_touches": 2,
                "stability_window": 2,
                "max_touches": 3,
                "eps": 1e-8,
            },
        }
    )


TRIALS = [
    {"shape": "left", "sequence": ["a", "a", "a"]},
    {"shape": "right", "sequence": ["b", "b", "b"]},
]


class RFSEvaluationTests(unittest.TestCase):
    def test_evaluation_reports_stopping_and_touch_curve(self):
        metrics = evaluate_rfs_trials(TRIALS, config(), permutations=3)
        self.assertEqual(metrics["run_count"], 6)
        self.assertEqual(metrics["overall_accuracy"], 1.0)
        self.assertEqual(metrics["accepted_accuracy"], 1.0)
        self.assertNotIn("mean_effective_touches", metrics)
        self.assertIn("1", metrics["accuracy_by_touch_count"])

    def test_probability_mapping_is_rejected(self):
        trials = [{"shape": "left", "sequence": [{"a": 1.0}]}]
        with self.assertRaises(TypeError):
            evaluate_rfs_trials(trials, config(), permutations=1)


if __name__ == "__main__":
    unittest.main()
