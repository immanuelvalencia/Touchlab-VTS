import json
from pathlib import Path
import tempfile
import unittest

from algorithms.bayesian.dataset_sequences import (
    generate_random_sequences,
    scan_local_features,
)


SHAPES = {"cube", "square_pyramid"}
FEATURES = {"planar", "straight_edge", "multi_face_vertex", "sharp_apex"}


def write_metadata(path, shape, feature):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": shape,
                "custom_fields": {"local_feature": feature},
            }
        ),
        encoding="utf-8",
    )


class BayesianDatasetSequenceTests(unittest.TestCase):
    def test_scan_tracks_unique_normalized_features_per_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_metadata(root / "cube" / "a_metadata.json", "cube", "planar")
            write_metadata(root / "cube" / "b_metadata.json", "cube", "planar")
            write_metadata(
                root / "cube" / "c_metadata.json",
                "cube",
                "multiface_vertex",
            )
            write_metadata(
                root / "pyramid" / "a_metadata.json",
                "pyramid",
                "sharp_apex",
            )
            (root / "other.json").write_text("{}", encoding="utf-8")

            result = scan_local_features(
                root,
                valid_shapes=SHAPES,
                valid_features=FEATURES,
            )

        self.assertEqual(
            result.features_by_shape["cube"],
            ("multi_face_vertex", "planar"),
        )
        self.assertEqual(result.features_by_shape["square_pyramid"], ("sharp_apex",))
        self.assertEqual(result.records_by_shape["cube"], 3)
        self.assertEqual(result.metadata_files, 4)
        self.assertEqual(result.accepted_records, 4)

    def test_random_generation_is_reproducible_and_uses_only_class_features(self):
        vocabularies = {
            "cube": ("planar", "straight_edge"),
            "square_pyramid": ("planar", "sharp_apex"),
        }
        arguments = {
            "sequences_per_shape": 5,
            "minimum_length": 2,
            "maximum_length": 4,
            "seed": 17,
        }

        first = generate_random_sequences(vocabularies, **arguments)
        second = generate_random_sequences(vocabularies, **arguments)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        for item in first:
            self.assertGreaterEqual(len(item["sequence"]), 2)
            self.assertLessEqual(len(item["sequence"]), 4)
            self.assertTrue(set(item["sequence"]).issubset(vocabularies[item["shape"]]))
            self.assertEqual(set(item), {"shape", "sequence"})

    def test_generation_rejects_invalid_lengths(self):
        with self.assertRaisesRegex(ValueError, "maximum_length"):
            generate_random_sequences(
                {"cube": ("planar",)},
                sequences_per_shape=1,
                minimum_length=5,
                maximum_length=3,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
