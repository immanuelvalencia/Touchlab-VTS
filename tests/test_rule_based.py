from pathlib import Path
import tempfile
import unittest

from algorithms.registry import get_algorithm
from algorithms.rule_based.factory import create_algorithm


class RuleBasedTests(unittest.TestCase):
    def test_accumulates_positive_feature_rules(self):
        algorithm = create_algorithm()
        algorithm.update("planar")
        result = algorithm.update("multi_face_vertex")

        self.assertAlmostEqual(sum(result.values()), 1.0, places=12)
        self.assertEqual(max(result, key=result.get), "cube")
        self.assertEqual(algorithm.touch_count, 2)

    def test_negative_rules_can_change_the_best_shape(self):
        algorithm = create_algorithm()
        algorithm.update("single_curvature")
        result = algorithm.update("sharp_apex")

        self.assertEqual(max(result, key=result.get), "cone")

    def test_reset_removes_previous_rule_state(self):
        algorithm = create_algorithm()
        algorithm.update("double_curvature")
        algorithm.reset()

        reset_result = algorithm.update("sharp_apex")
        fresh_result = create_algorithm().update("sharp_apex")

        self.assertEqual(reset_result, fresh_result)

    def test_registry_prefers_model_sidecar(self):
        spec = get_algorithm("rule_based")
        default_path = Path(__file__).parents[1] / "algorithms" / "rule_based" / "config.json"
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "features.pth"
            model_path.write_bytes(b"")
            sidecar_path = model_path.with_suffix(".rules.json")
            sidecar_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")

            algorithm = spec.create(model_path=str(model_path))

            self.assertEqual(Path(algorithm.config_path), sidecar_path.resolve())

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            create_algorithm().update("not_a_feature")


if __name__ == "__main__":
    unittest.main()

