import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from algorithms.bayesian.artifacts import (
    ACTIVE_CONFIG_FILE,
    BAYESIAN_DIR,
    DATASET_FILE,
    GENERATED_DATASETS_DIR,
    RUNS_DIR,
    build_generated_dataset_path,
    build_run_paths,
    system_timestamp,
    write_json,
)


class BayesianArtifactTests(unittest.TestCase):
    def test_stable_files_are_in_the_bayesian_folder(self):
        self.assertEqual(DATASET_FILE, BAYESIAN_DIR / "touch_sequences.json")
        self.assertEqual(ACTIVE_CONFIG_FILE, BAYESIAN_DIR / "config.json")
        self.assertEqual(RUNS_DIR, BAYESIAN_DIR / "runs")
        self.assertEqual(
            GENERATED_DATASETS_DIR,
            BAYESIAN_DIR / "generated_datasets",
        )

    def test_run_json_names_include_the_timestamp(self):
        timestamp = "2026-08-11_15-30-45"
        paths = build_run_paths(timestamp)

        self.assertEqual(paths.directory, RUNS_DIR / timestamp)
        self.assertEqual(
            paths.dataset_snapshot.name,
            f"touch_sequences_{timestamp}.json",
        )
        self.assertEqual(
            paths.optimized_config.name,
            f"bayesian_config_{timestamp}.json",
        )

    def test_generated_dataset_name_includes_the_timestamp(self):
        timestamp = "2026-08-11_15-30-45"
        path = build_generated_dataset_path(timestamp)

        self.assertEqual(path.parent, GENERATED_DATASETS_DIR)
        self.assertEqual(
            path.name,
            f"touch_sequences_random_{timestamp}.json",
        )

    def test_system_timestamp_uses_explicit_24_hour_time(self):
        local_timezone = datetime.now().astimezone().tzinfo
        afternoon = datetime(2026, 8, 11, 15, 30, 45, tzinfo=local_timezone)

        timestamp = system_timestamp(afternoon)

        self.assertEqual(timestamp, "2026-08-11_15-30-45")
        self.assertNotIn("PM", timestamp)

    def test_json_write_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "result.json"
            value = {"planar": {"cube": 0.75}}

            write_json(path, value)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
