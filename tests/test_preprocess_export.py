"""Export destination checks prevent stale files from contaminating splits."""

from pathlib import Path
import tempfile
import unittest

from preprocess import export_records, validate_output_directory


class ExportDestinationTests(unittest.TestCase):
    def test_accepts_new_and_empty_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validate_output_directory(root / "new")
            validate_output_directory(root)

    def test_rejects_stale_export_before_writing_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / "train/cylinder/GelSight_sequence_007_frame_0000_raw.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"existing image")
            before = set(root.rglob("*"))
            with self.assertRaisesRegex(ValueError, "must be new or empty"):
                export_records({}, root)
            self.assertEqual(set(root.rglob("*")), before)
            self.assertEqual(stale.read_bytes(), b"existing image")

    def test_rejects_file_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("keep")
            with self.assertRaisesRegex(ValueError, "must be new or empty"):
                validate_output_directory(path)
            self.assertEqual(path.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
