from pathlib import Path
import unittest

from tools.dataset_splitting import DatasetRecord, grouped_split, metadata_group_id


class GroupedDatasetSplitTests(unittest.TestCase):
    def test_video_frames_share_the_parent_directory_group(self):
        first = Path("dataset/cone/video/sequence_001/frame_0001_metadata.json")
        second = Path("dataset/cone/video/sequence_001/frame_0002_metadata.json")

        self.assertEqual(
            metadata_group_id(first, {"is_video_sequence": True}),
            metadata_group_id(second, {"is_video_sequence": True}),
        )

    def test_non_video_captures_are_independent_groups(self):
        first = Path("dataset/cone/image_001_metadata.json")
        second = Path("dataset/cone/image_002_metadata.json")

        self.assertNotEqual(
            metadata_group_id(first, {"is_video_sequence": False}),
            metadata_group_id(second, {"is_video_sequence": False}),
        )

    def test_groups_never_cross_splits(self):
        records = [
            DatasetRecord(Path(f"group_{group}_frame_{frame}.png"), "GelSight", f"g{group}")
            for group in range(20)
            for frame in range(3)
        ]

        result = grouped_split(
            records,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        groups_by_split = {
            split: {record.group_id for record in split_records}
            for split, split_records in result.items()
        }
        self.assertFalse(groups_by_split["train"] & groups_by_split["val"])
        self.assertFalse(groups_by_split["train"] & groups_by_split["test"])
        self.assertFalse(groups_by_split["val"] & groups_by_split["test"])
        self.assertEqual(
            set().union(*groups_by_split.values()),
            {f"g{group}" for group in range(20)},
        )

    def test_small_dataset_keeps_each_requested_split_nonempty(self):
        records = [
            DatasetRecord(Path(f"g{group}.png"), "GelSight", f"g{group}")
            for group in range(3)
        ]

        result = grouped_split(
            records,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        self.assertTrue(all(result[split] for split in ("train", "val", "test")))


if __name__ == "__main__":
    unittest.main()
