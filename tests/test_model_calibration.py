import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools.model_calibration import (
    apply_temperature,
    calibration_path_for_model,
    classification_metrics,
    fit_temperature,
    load_encoder_temperature,
)


class ModelCalibrationTests(unittest.TestCase):
    def test_model_sidecar_path(self):
        self.assertEqual(
            calibration_path_for_model("models/features.pth"),
            Path("models/features.calibration.json"),
        )

    def test_missing_sidecar_uses_identity_temperature(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temperature, path = load_encoder_temperature(Path(temp_dir) / "model.pth")

        self.assertEqual(temperature, 1.0)
        self.assertEqual(path.name, "model.calibration.json")

    def test_sidecar_temperature_is_loaded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model.pth"
            sidecar = calibration_path_for_model(model)
            sidecar.write_text(json.dumps({"temperature": 2.5}), encoding="utf-8")

            temperature, _ = load_encoder_temperature(model)

        self.assertEqual(temperature, 2.5)

    def test_temperature_softens_probabilities(self):
        logits = torch.tensor([[4.0, 1.0]])
        original = torch.softmax(logits, dim=1).max().item()
        softened = torch.softmax(apply_temperature(logits, 2.0), dim=1).max().item()

        self.assertLess(softened, original)

    def test_fitted_temperature_reduces_overconfident_nll(self):
        logits = torch.tensor(
            [[8.0, 0.0], [8.0, 0.0], [8.0, 0.0], [8.0, 0.0]],
            dtype=torch.float64,
        )
        targets = torch.tensor([0, 0, 0, 1])
        before = classification_metrics(logits, targets)["nll"]

        temperature = fit_temperature(logits, targets)
        after = classification_metrics(logits / temperature, targets)["nll"]

        self.assertGreater(temperature, 1.0)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
