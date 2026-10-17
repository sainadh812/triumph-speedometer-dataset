"""Small CPU contract checks for the speedometer training pipeline.

Run beside train_core.py with a PyTorch environment:
    python test_training.py

These check data isolation, coordinate conventions, and gradient flow. They do
not train a useful model or estimate its accuracy.
"""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from train_core import (
    DashboardDataset,
    SpeedometerModel,
    TrainConfig,
    heatmap_coordinates,
    select_group_subset,
    training_loss,
    verify_dataset,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "bw_dataset_500"


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.report = verify_dataset(DATA_DIR)

    def test_nested_complete_pose_groups_and_validation_isolation(self):
        training = [row for row in self.records if row["dataset_split"] == "train"]
        validation = [row for row in self.records if row["dataset_split"] == "validation"]
        self.assertEqual((len(training), len(validation)), (400, 100))
        validation_groups = {row["group_id"] for row in validation}
        self.assertEqual(len(validation_groups), 50)
        self.assertTrue(validation_groups.isdisjoint(row["group_id"] for row in training))
        self.assertTrue({row["speed_kmh"] for row in training}.isdisjoint(
            row["speed_kmh"] for row in validation))
        previous_ids = set()
        for count in (100, 250, 400):
            subset = select_group_subset(training, count, seed=42)
            selected_ids = {row["sample_id"] for row in subset}
            selected_groups = {row["group_id"] for row in subset}
            self.assertEqual(len(subset), count)
            self.assertTrue(previous_ids.issubset(selected_ids))
            # Any selected pose must retain every one of its image variants.
            self.assertEqual(selected_ids, {
                row["sample_id"] for row in training if row["group_id"] in selected_groups
            })
            previous_ids = selected_ids

    def test_native_pixels_and_labels_survive_preprocessing(self):
        validation = [row for row in self.records if row["dataset_split"] == "validation"]
        dataset = DashboardDataset(DATA_DIR, validation, input_size=512)
        first, repeat = dataset[0], dataset[0]
        self.assertTrue(torch.equal(first["image"], repeat["image"]))
        self.assertEqual(tuple(first["image"].shape), (3, 512, 512))
        row = validation[0]
        with Image.open(DATA_DIR / row["image"]) as image:
            expected = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        recovered = first["image"].numpy() * std + mean
        np.testing.assert_allclose(recovered, np.repeat(expected[None], 3, axis=0), atol=1e-7)
        expected_points = np.array([row["needle_position_px"][name] for name in ("pivot", "tip")])
        np.testing.assert_allclose(first["points"].numpy() * 511, expected_points, atol=3e-5)


class GeometryAndGradientTests(unittest.TestCase):
    def test_asymmetric_heatmap_preserves_xy_and_pixel_endpoints(self):
        # Non-square maps expose accidental x/y swaps and wrong denominators.
        logits = torch.full((1, 2, 3, 5), -100.0)
        logits[0, 0, 2, 0] = 100.0
        logits[0, 1, 1, 3] = 100.0
        actual = heatmap_coordinates(logits)
        expected = torch.tensor([[[0.0, 1.0], [0.75, 0.5]]])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(actual * 511, expected * 511, atol=1e-4, rtol=0)
        torch.testing.assert_close(heatmap_coordinates(torch.zeros(1, 2, 3, 5)),
                                   torch.full((1, 2, 2), 0.5), atol=1e-6, rtol=0)

    def test_both_prediction_heads_receive_finite_gradients(self):
        torch.manual_seed(7)
        model = SpeedometerModel(pretrained=False)
        model.configure_stage("heads")
        batch = {
            "image": torch.randn(2, 3, 128, 128),
            "speed": torch.tensor([30.0, 130.0]),
            "points": torch.tensor([[[0.48, 0.53], [0.21, 0.67]],
                                     [[0.49, 0.52], [0.28, 0.31]]]),
        }
        output = model(batch["image"])
        self.assertEqual(tuple(output["points"].shape), (2, 2, 2))
        self.assertEqual(tuple(output["speed"].shape), (2,))
        config = TrainConfig(data_dir=str(DATA_DIR), output_dir="unused")
        loss, _ = training_loss(output, batch, config)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for head in (model.speed_head, model.heatmap_head):
            gradients = [parameter.grad for parameter in head.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.features.parameters()))
        model.configure_stage("finetune")
        self.assertFalse(any(parameter.requires_grad for parameter in model.features[0].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.features[-1].parameters()))
        self.assertTrue(all(not module.training for module in model.features.modules()
                            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
