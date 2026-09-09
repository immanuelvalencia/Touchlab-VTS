"""Read-only diagnostics for existing exports; these tests need only stdlib."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_dataset import check_dataset, main


def make_export(root, validation="val"):
    for split, sequence in (("train", 1), (validation, 2), ("test", 3)):
        for label in ("cube", "cylinder"):
            path = root / split / label / f"GelSight_sequence_{sequence:03d}_frame_0000_raw.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"image bytes are intentionally not decoded")


class DatasetCheckTests(unittest.TestCase):
    def test_clean_export_and_validation_aliases_without_image_reads(self):
        for validation in ("val", "valid", "validation"):
            with self.subTest(validation=validation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_export(root, validation)
                with patch("check_dataset.sha256", side_effect=AssertionError("unexpected hash")):
                    report = check_dataset(root)
                self.assertTrue(report["ok"], report["issues"])
                self.assertEqual(report["summary"][validation]["cylinder"]["contacts"], 1)

    def test_reports_all_overlaps_including_disjoint_frames_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            for label in ("cube", "cylinder"):
                (root / "test" / label / "GelSight_sequence_001_frame_9999_raw.png").write_bytes(b"other frame")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            report = check_dataset(root)
            self.assertFalse(report["ok"])
            self.assertEqual(len(report["overlaps"]), 2)
            for overlap in report["overlaps"]:
                self.assertEqual(set(overlap["splits"]), {"train", "test"})
                self.assertEqual(overlap["shared_frame_stems"], [])
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_hash_compares_only_same_named_files_in_overlapping_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            name = "GelSight_sequence_001_frame_0000_raw.png"
            (root / "test/cube" / name).write_bytes((root / "train/cube" / name).read_bytes())
            (root / "test/cylinder" / name).write_bytes(b"different")
            report = check_dataset(root, hash_overlaps=True)
            comparisons = {o["label"]: o["same_name_comparisons"] for o in report["overlaps"]}
            self.assertTrue(comparisons["cube"][0]["all_identical"])
            self.assertFalse(comparisons["cylinder"][0]["all_identical"])
            self.assertEqual(len(comparisons["cube"][0]["sha256"]), 2)

    def test_augmentations_are_grouped_and_evaluation_augmentation_is_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            (root / "train/cube/GelSight_sequence_001_frame_0000_raw_hflip.png").write_bytes(b"aug")
            (root / "val/cube/GelSight_sequence_002_frame_0000_raw_rot180.png").write_bytes(b"aug")
            (root / "test/cube/GelSight_sequence_004_frame_0000_raw_vflip.png").write_bytes(b"aug only")
            report = check_dataset(root)
            self.assertEqual(report["summary"]["train"]["cube"]["contacts"], 1)
            self.assertEqual(len(report["evaluation_augmentations"]), 2)
            self.assertEqual(len(report["contacts_without_originals"]), 1)

    def test_reports_manifest_overlap_and_malformed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            path = root / "split_manifest.json"
            path.write_text(json.dumps({"category_by": "label", "classes": {
                "cube": {"train": ["video::/raw/sequence_001"], "valid": ["video::/raw/sequence_001"]}}}))
            report = check_dataset(root)
            self.assertFalse(report["ok"])
            self.assertEqual(len(report["manifest"]["overlaps"]), 1)
            self.assertEqual(report["overlaps"], [])
            for malformed in ("broken JSON", "[]", '{"classes": {"cube": []}}'):
                path.write_text(malformed)
                report = check_dataset(root)
                self.assertFalse(report["ok"])
                self.assertTrue(report["manifest"]["errors"])

    def test_structure_errors_do_not_hide_other_findings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            (root / "valid").mkdir()
            (root / "test/extra").mkdir()
            report = check_dataset(root)
            self.assertTrue(any("exactly one validation" in issue for issue in report["issues"]))
            self.assertTrue(any("class folders must match" in issue for issue in report["issues"]))
            self.assertTrue(any("No images in test/extra" in issue for issue in report["issues"]))

    def test_cli_report_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            make_export(root)
            report_path = Path(directory) / "report.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--dataset_dir", str(root), "--report", str(report_path)]), 0)
                (root / "test/cube/GelSight_sequence_001_frame_9999_raw.png").write_bytes(b"overlap")
                self.assertEqual(main(["--dataset_dir", str(root)]), 1)
            self.assertTrue(json.loads(report_path.read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
