import json
import tempfile
import unittest

from algorithms.rfs.trials import (
    append_trial,
    normalize_feature_label,
    trial_counts,
    trial_file,
)


FEATURES = ("planar", "multi_face_vertex")
ALIASES = {"multiface_vertex": "multi_face_vertex"}


class RFSTrialPersistenceTests(unittest.TestCase):
    def test_split_file_names_are_stable(self):
        self.assertEqual(trial_file("trials", "train").name, "rfs_train.json")
        self.assertEqual(
            trial_file("trials", "validation").name,
            "rfs_validation.json",
        )

    def test_model_alias_is_saved_as_canonical_feature(self):
        self.assertEqual(
            normalize_feature_label("multiface_vertex", FEATURES, ALIASES),
            "multi_face_vertex",
        )

    def test_unknown_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not configured"):
            normalize_feature_label("sharp_apex", FEATURES, ALIASES)

    def test_append_writes_hard_labels_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path, count = append_trial(
                temp_dir,
                split="train",
                shape="cube",
                sequence=["planar", "multiface_vertex"],
                features=FEATURES,
                aliases=ALIASES,
            )
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(set(data[0]), {"shape", "sequence"})
        self.assertEqual(data[0]["sequence"], ["planar", "multi_face_vertex"])

    def test_append_preserves_existing_trials_and_updates_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for shape in ("cube", "sphere"):
                append_trial(
                    temp_dir,
                    split="test",
                    shape=shape,
                    sequence=["planar"],
                    features=FEATURES,
                    aliases=ALIASES,
                )

            counts = trial_counts(temp_dir)

        self.assertEqual(counts, {"train": 0, "validation": 0, "test": 2})

if __name__ == "__main__":
    unittest.main()
