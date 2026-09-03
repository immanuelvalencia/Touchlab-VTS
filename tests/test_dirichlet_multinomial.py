from pathlib import Path
import tempfile
import unittest

from algorithms.dirichlet_multinomial.factory import create_algorithm
from algorithms.registry import get_algorithm


class DirichletMultinomialTests(unittest.TestCase):
    def test_probabilities_are_normalized(self):
        result = create_algorithm().update("double_curvature")

        self.assertAlmostEqual(sum(result.values()), 1.0, places=12)
        self.assertEqual(max(result, key=result.get), "sphere")

    def test_fixed_count_vector_is_permutation_invariant(self):
        sequence = ["planar", "sharp_apex", "straight_edge", "planar"]
        forward = create_algorithm()
        reverse = create_algorithm()

        for feature in sequence:
            forward_result = forward.update(feature)
        for feature in reversed(sequence):
            reverse_result = reverse.update(feature)

        for shape in forward_result:
            self.assertAlmostEqual(forward_result[shape], reverse_result[shape], places=12)

    def test_repetition_uses_the_count_vector(self):
        once = create_algorithm().update("double_curvature")
        repeated_algorithm = create_algorithm()
        repeated_algorithm.update("double_curvature")
        repeated = repeated_algorithm.update("double_curvature")

        self.assertGreater(repeated["sphere"], once["sphere"])
        self.assertLess(repeated["sphere"], 1.0)

    def test_reset_removes_previous_counts(self):
        algorithm = create_algorithm()
        algorithm.update("sharp_apex")
        algorithm.reset()

        reset_result = algorithm.update("double_curvature")
        fresh_result = create_algorithm().update("double_curvature")

        self.assertEqual(reset_result, fresh_result)

    def test_registry_prefers_model_sidecar(self):
        spec = get_algorithm("dirichlet_multinomial")
        default_path = (
            Path(__file__).parents[1]
            / "algorithms"
            / "dirichlet_multinomial"
            / "config.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "features.pth"
            model_path.write_bytes(b"")
            sidecar_path = model_path.with_suffix(".dm.json")
            sidecar_path.write_text(default_path.read_text(encoding="utf-8"), encoding="utf-8")

            algorithm = spec.create(model_path=str(model_path))

            self.assertEqual(Path(algorithm.config_path), sidecar_path.resolve())

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            create_algorithm().update("not_a_feature")


if __name__ == "__main__":
    unittest.main()
