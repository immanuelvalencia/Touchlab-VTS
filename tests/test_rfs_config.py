import json
from pathlib import Path
import tempfile
import unittest

from algorithms.registry import get_algorithm
from algorithms.rfs.algorithm import SetEvidenceAccumulator
from algorithms.rfs.config import RFSConfig, load_rfs_config
from algorithms.rfs.factory import DEFAULT_CONFIG_PATH


def load_default_json():
    with Path(DEFAULT_CONFIG_PATH).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RFSConfigurationTests(unittest.TestCase):
    def test_default_config_constructs_the_algorithm(self):
        config = load_rfs_config(DEFAULT_CONFIG_PATH)
        accumulator = SetEvidenceAccumulator(config)

        self.assertEqual(accumulator.features, config.features)
        self.assertEqual(accumulator.shapes, config.shapes)
        self.assertEqual(accumulator.config_path, Path(DEFAULT_CONFIG_PATH).resolve())

    def test_registry_prefers_a_model_sidecar_config(self):
        raw = load_default_json()
        raw["name"] = "model_specific_config"

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "encoder_v2.pth"
            sidecar_path = Path(temp_dir) / "encoder_v2.rfs.json"
            sidecar_path.write_text(json.dumps(raw), encoding="utf-8")

            accumulator = get_algorithm("rfs").create(model_path=str(model_path))

        self.assertEqual(accumulator.config.name, "model_specific_config")
        self.assertEqual(accumulator.config_path, sidecar_path.resolve())

    def test_missing_required_config_key_is_rejected(self):
        raw = load_default_json()
        del raw["features"]

        with self.assertRaisesRegex(ValueError, "missing required keys: features"):
            RFSConfig.from_mapping(raw)

    def test_incomplete_template_is_rejected(self):
        raw = load_default_json()
        del raw["theta_templates"]["cube"]["planar"]
        config = RFSConfig.from_mapping(raw)

        with self.assertRaisesRegex(ValueError, "theta template.*mismatch"):
            SetEvidenceAccumulator(config)

    def test_unknown_parameter_is_rejected(self):
        raw = load_default_json()
        raw["parameters"]["invented_parameter"] = 1.0

        with self.assertRaisesRegex(ValueError, "unknown parameters"):
            RFSConfig.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
