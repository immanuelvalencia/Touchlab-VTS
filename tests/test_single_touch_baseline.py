from pathlib import Path
import tempfile
import unittest

from algorithms.registry import get_algorithm
from algorithms.single_touch_baseline.factory import create_algorithm


class SingleTouchBaselineTests(unittest.TestCase):
    def test_latest_touch_replaces_previous_evidence(self):
        algorithm = create_algorithm()
        algorithm.update("sharp_apex")
        after_second_touch = algorithm.update("double_curvature")
        fresh = create_algorithm().update("double_curvature")

        self.assertEqual(after_second_touch, fresh)
        self.assertEqual(algorithm.touch_count, 2)
        self.assertEqual(algorithm.last_feature, "double_curvature")

    def test_aliases_are_supported(self):
        result = create_algorithm().update("apex")

        self.assertAlmostEqual(sum(result.values()), 1.0, places=12)
        self.assertIn(max(result, key=result.get), {"cone", "square_pyramid"})

    def test_registry_prefers_model_sidecar(self):
        spec = get_algorithm("single_touch_baseline")
        default_path = (
            Path(__file__).parents[1]
            / "algorithms"
            / "single_touch_baseline"
            / "config.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "features.pth"
            model_path.write_bytes(b"")
            sidecar_path = model_path.with_suffix(".single_touch.json")
            sidecar_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")

            algorithm = spec.create(model_path=str(model_path))

            self.assertEqual(Path(algorithm.config_path), sidecar_path.resolve())

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            create_algorithm().update("not_a_feature")


if __name__ == "__main__":
    unittest.main()

