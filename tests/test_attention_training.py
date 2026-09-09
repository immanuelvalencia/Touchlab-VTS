"""Checks for contact grouping, set pooling, and the complete training workflow."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from train_attention import (
    AttentionClassifier, TouchBags, build_encoder, collate_bags, main,
    representative_frame, scan_export,
)


def make_export(root, validation="val"):
    for split, sequence_ids in (("train", range(1, 4)), (validation, range(4, 6)), ("test", range(6, 8))):
        for label in ("cube", "sphere"):
            directory = root / split / label
            directory.mkdir(parents=True)
            for sequence in sequence_ids:
                for frame in range(3):
                    path = directory / f"GelSight_sequence_{sequence:03d}_frame_{frame:04d}_raw.png"
                    Image.new("RGB", (40, 40), (sequence * 20, frame * 40, 100)).save(path)
                    if split == "train":
                        Image.new("RGB", (40, 40), (sequence * 20, frame * 40, 100)).save(
                            path.with_stem(path.stem + "_hflip"))


class AttentionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_groups_frames_and_augmentations_and_accepts_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root, "valid")
            labels, contacts = scan_export(root)
            self.assertEqual(labels, ["cube", "sphere"])
            self.assertEqual(len(contacts["train"]["cube"]), 3)
            self.assertEqual(len(contacts["val"]["cube"]), 2)
            paths = next(iter(contacts["train"]["cube"].values()))
            self.assertEqual(len(paths), 6)
            self.assertTrue(representative_frame(paths).name.endswith("frame_0001_raw.png"))
            bags = TouchBags(contacts["train"], labels, bags_per_class=8,
                             min_touches=1, max_touches=3, training=True)
            original_samples = [bags.sample(i) for i in range(len(bags))]
            for _, keys, _ in original_samples:
                self.assertEqual(len(keys), len(set(keys)))
            bags.epoch = 1
            self.assertNotEqual(original_samples, [bags.sample(i) for i in range(len(bags))])

    def test_rejects_contact_leakage_even_with_different_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            Image.new("RGB", (40, 40)).save(root / "test/cube/GelSight_sequence_001_frame_9999_raw.png")
            with self.assertRaisesRegex(ValueError, "Contact occurs in both"):
                scan_export(root)

    def test_rejects_feature_export_and_manifest_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            path = root / "split_manifest.json"
            path.write_text(json.dumps({"category_by": "local_feature"}))
            with self.assertRaisesRegex(ValueError, "local-feature"):
                scan_export(root)
            path.write_text(json.dumps({"category_by": "label", "classes": {
                "cube": {"train": ["same_acquisition"], "test": ["same_acquisition"]}}}))
            with self.assertRaisesRegex(ValueError, "different splits/classes"):
                scan_export(root)

    def test_evaluation_bags_are_fixed_and_never_duplicate_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root)
            labels, contacts = scan_export(root)
            bags = TouchBags(contacts["test"], labels, bags_per_class=3,
                             min_touches=2, max_touches=2)
            before = bags.manifest(root)
            bags.epoch = 100
            self.assertEqual(before, bags.manifest(root))
            self.assertTrue(all(len(set(bag["contacts"])) == 2 for bag in before))
            with self.assertRaisesRegex(ValueError, "distinct contacts"):
                TouchBags(contacts["test"], labels, bags_per_class=1, min_touches=1, max_touches=3)

    def test_attention_is_permutation_invariant_and_ignores_padding(self):
        torch.manual_seed(5)
        model = AttentionClassifier(2, weights="NONE", dropout=0).eval()
        images = torch.randn(2, 3, 3, 64, 64)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        with torch.no_grad():
            logits, attention = model(images, mask)
            changed = images.clone()
            changed[~mask] = 1e6
            padded_logits, _ = model(changed, mask)
            permutation = [2, 0, 1]
            permuted_logits, permuted_attention = model(images[:, permutation], mask[:, permutation])
        torch.testing.assert_close(logits, padded_logits)
        torch.testing.assert_close(logits, permuted_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(attention[:, permutation], permuted_attention, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(2))
        self.assertEqual(attention[0, 2].item(), 0)
        with self.assertRaisesRegex(ValueError, "valid touch"):
            model(images, torch.zeros_like(mask))

    def test_backpropagation_reaches_backbone_and_attention(self):
        model = AttentionClassifier(2, weights="NONE", dropout=0).train()
        images, mask, labels = collate_bags([(torch.randn(2, 3, 64, 64), 0), (torch.randn(1, 3, 64, 64), 1)])
        logits, _ = model(images, mask)
        torch.nn.functional.cross_entropy(logits, labels).backward()
        for parameter in (model.encoder.conv1.weight, model.attention_w.weight, model.classifier[1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_supported_backbone_families_produce_embeddings(self):
        for name in ("resnet18", "efficientnet_b0", "mobilenet_v3_small", "densenet121", "convnext_tiny"):
            with self.subTest(name=name):
                encoder, dimension = build_encoder(name, "NONE")
                encoder.eval()
                with torch.no_grad():
                    output = encoder(torch.randn(1, 3, 64, 64))
                self.assertEqual(tuple(output.shape), (1, dimension))

    def test_training_checkpoint_reload_and_touch_count_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_export(root / "dataset")
            run = main(["--dataset_dir", str(root / "dataset"), "--output_dir", str(root / "runs"),
                        "--weights", "NONE", "--epochs", "1", "--max_touches", "2",
                        "--eval_touches", "1", "2", "8", "--train_bags_per_class", "1",
                        "--eval_bags_per_class", "1", "--batch_size", "2", "--image_size", "64",
                        "--freeze_backbone", "--device", "cpu"])
            report = json.loads((run / "test_metrics.json").read_text())
            self.assertEqual(report["skipped_touch_counts"], [8])
            self.assertEqual(set(report["by_touch_count"]), {"1", "2"})
            checkpoint = torch.load(run / "best_model.pth", map_location="cpu", weights_only=True)
            restored = AttentionClassifier(**checkpoint["model_config"], weights="NONE")
            restored.load_state_dict(checkpoint["model_state_dict"])
            restored.train()
            self.assertFalse(restored.encoder.training)
            self.assertEqual(checkpoint["class_names"], ["cube", "sphere"])
            self.assertTrue((run / "val_bags.json").exists())
            self.assertTrue((run / "test_bags.json").exists())


if __name__ == "__main__":
    unittest.main()
