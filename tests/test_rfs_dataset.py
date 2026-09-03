import unittest

from algorithms.rfs.dataset_generation import (
    generate_and_split_sequences,
    split_generated_sequences,
)


class RFSDatasetGenerationTests(unittest.TestCase):
    def test_split_is_reproducible_stratified_and_complete(self):
        trials = []
        for shape, feature in (("cube", "planar"), ("sphere", "double_curvature")):
            for length in range(1, 7):
                trials.append({"shape": shape, "sequence": [feature] * length})

        first = split_generated_sequences(
            trials,
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=17,
        )
        second = split_generated_sequences(
            trials,
            train_ratio=0.5,
            validation_ratio=0.25,
            test_ratio=0.25,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(sum(len(values) for values in first.values()), len(trials))
        for split in ("train", "validation", "test"):
            self.assertEqual({trial["shape"] for trial in first[split]}, {"cube", "sphere"})

    def test_duplicate_shape_sequences_never_cross_splits(self):
        trials = []
        for length in (1, 2, 3):
            trials.extend(
                {"shape": "cube", "sequence": ["planar"] * length}
                for _ in range(4)
            )

        splits = split_generated_sequences(
            trials,
            train_ratio=1 / 3,
            validation_ratio=1 / 3,
            test_ratio=1 / 3,
            seed=4,
        )

        owner = {}
        for split, values in splits.items():
            for trial in values:
                key = (trial["shape"], tuple(trial["sequence"]))
                if key in owner:
                    self.assertEqual(owner[key], split)
                owner[key] = split

    def test_too_few_unique_sequences_for_active_splits_is_rejected(self):
        trials = [
            {"shape": "sphere", "sequence": ["double_curvature"]}
            for _ in range(10)
        ]

        with self.assertRaisesRegex(ValueError, "unique generated sequences"):
            split_generated_sequences(
                trials,
                train_ratio=0.7,
                validation_ratio=0.15,
                test_ratio=0.15,
            )

    def test_generation_wrapper_uses_class_vocabulary_and_splits(self):
        generated, splits = generate_and_split_sequences(
            {
                "cube": ("planar", "straight_edge"),
                "sphere": ("double_curvature",),
            },
            sequences_per_shape=20,
            minimum_length=1,
            maximum_length=5,
            seed=9,
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
        )

        self.assertEqual(len(generated), 40)
        self.assertEqual(sum(len(values) for values in splits.values()), 40)
        for trial in generated:
            if trial["shape"] == "sphere":
                self.assertEqual(set(trial["sequence"]), {"double_curvature"})


if __name__ == "__main__":
    unittest.main()
