import json
from pathlib import Path
import tempfile
import unittest

from algorithms.rfs.config import RFSConfig
from algorithms.rfs.optimization import evaluate_frozen_config, optimize_rfs_config
from optimize import RunRequest, run_optimization


def base_config():
    return RFSConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "toy",
            "features": ["a", "b"],
            "shapes": ["left", "right"],
            "aliases": {},
            "theta_templates": {
                "left": {"a": 0.5, "b": 0.5},
                "right": {"a": 0.5, "b": 0.5},
            },
            "presence_templates": {
                "left": {"a": 0.5, "b": 0.5},
                "right": {"a": 0.5, "b": 0.5},
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


TRAIN = [
    {"shape": "left", "sequence": ["a", "a", "a"]},
    {"shape": "left", "sequence": ["a", "a"]},
    {"shape": "right", "sequence": ["b", "b", "b"]},
    {"shape": "right", "sequence": ["b", "b"]},
]

VALIDATION = [
    {"shape": "left", "sequence": ["a", "a", "a"]},
    {"shape": "right", "sequence": ["b", "b", "b"]},
]


def optimize_small():
    return optimize_rfs_config(
        TRAIN,
        VALIDATION,
        base_config(),
        name="toy_optimized",
        permutations=2,
        dirichlet_alphas=(0.1,),
        presence_priors=((0.5, 0.5),),
        evidence_weights=(0.5, 1.0),
        coverage_weights=(0.0, 0.5),
        unexpected_penalties=(0.0, 0.25),
        stop_thresholds=(0.7, 0.9),
        entropy_thresholds=(0.3, 0.6),
        min_touches_values=(1, 2),
        stability_windows=(1, 2),
        max_touches_values=(2, 3),
        target_accepted_accuracy=1.0,
        minimum_acceptance_rate=1.0,
    )


class RFSOptimizationTests(unittest.TestCase):
    def test_staged_optimizer_learns_hard_label_templates(self):
        optimized, report = optimize_small()

        self.assertGreater(
            optimized["theta_templates"]["left"]["a"],
            optimized["theta_templates"]["left"]["b"],
        )
        self.assertGreater(
            optimized["theta_templates"]["right"]["b"],
            optimized["theta_templates"]["right"]["a"],
        )
        self.assertEqual(
            optimized["optimization"]["input"],
            "deterministic_hard_local_feature_labels",
        )
        self.assertTrue(report["stopping_constraints_satisfied"])
        self.assertEqual(
            report["validation_stopping_metrics"]["accepted_accuracy"], 1.0
        )

    def test_frozen_evaluation_does_not_refit_the_config(self):
        optimized, _ = optimize_small()
        config = RFSConfig.from_mapping(optimized)

        metrics = evaluate_frozen_config(VALIDATION, config, permutations=2)

        self.assertEqual(metrics["overall_accuracy"], 1.0)
        self.assertEqual(metrics["trial_count"], 2)

    def test_unified_runner_saves_numbered_run_and_model_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "feature_model.pth"
            model.write_bytes(b"test weights placeholder")
            project = Path(__file__).parents[1]
            manifest = run_optimization(
                RunRequest(
                    model=model,
                    train=project / "trials" / "rfs_train.json",
                    validation=project / "trials" / "rfs_validation.json",
                    test=None,
                    algorithms=("bayesian",),
                    output_root=root / "optimize",
                    permutations=1,
                    test_permutations=1,
                    install_sidecars=True,
                )
            )

            run_dir = Path(manifest["run_directory"])
            sidecar = model.with_suffix(".bayesian.json")
            self.assertEqual(run_dir.name, "run_001")
            self.assertTrue((run_dir / "run_manifest.json").is_file())
            self.assertTrue((run_dir / "bayesian" / "evaluation_report.json").is_file())
            self.assertTrue(sidecar.is_file())
            self.assertIn("planar", json.loads(sidecar.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
