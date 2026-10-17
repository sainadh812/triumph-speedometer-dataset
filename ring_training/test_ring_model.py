"""Focused migration, freeze-state and supervision tests; no training run."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from ring_model import (RingSpeedometerModel, RingLossConfig, NEW_PARAMETER_PREFIXES,
                        load_legacy_checkpoint, training_loss, legacy)
from ring_data import RingDataset, verify_dataset

PROJECT = HERE.parent
SOURCE = PROJECT / "rotation_dataset/audits/20261017_100006_266720_f43a79f6_results/extracted/best_model.pt"
REVIEW = PROJECT / "ring_geometry_pilot/review20"


class RingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.payload = torch.load(SOURCE, map_location="cpu", weights_only=True)
        cls.data = RingDataset(REVIEW)
        cls.batch = {key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else torch.tensor([value])
                     for key, value in cls.data[3].items()}

    def model(self):
        model = RingSpeedometerModel(pretrained=False)
        metadata = load_legacy_checkpoint(model, SOURCE)
        self.assertEqual(metadata["source_epoch"], 19)
        self.assertFalse(metadata["optimizer_restored"])
        self.assertEqual(metadata["source_sha256"], "ba47ae2a3c07fd9a490342eebb8a01ac160dfe5e9dbca24dd13014ed68cac270")
        return model

    def test_migration_preserves_all_legacy_outputs_and_new_tensors(self):
        model = RingSpeedometerModel(pretrained=False)
        initial_new = {name: tensor.clone() for name, tensor in model.state_dict().items()
                       if name.startswith(NEW_PARAMETER_PREFIXES)}
        metadata = load_legacy_checkpoint(model, SOURCE)
        reference = legacy.SpeedometerModel(pretrained=False)
        reference.load_state_dict(self.payload["model_state_dict"], strict=True)
        reference.eval()
        model.eval()
        with torch.inference_mode():
            expected = reference(self.batch["image"])
            actual = model(self.batch["image"])
        for key in ("speed", "points", "heatmap_logits"):
            self.assertTrue(torch.equal(actual[key], expected[key]), key)
        for name, value in self.payload["model_state_dict"].items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        for name, value in initial_new.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        self.assertEqual(set(metadata["new_tensor_names"]), set(initial_new))
        self.assertEqual(actual["tick_points"].shape, (1, 6, 2))
        self.assertEqual(actual["tick_heatmap_logits"].shape, (1, 6, 128, 128))
        self.assertEqual(actual["tick_visibility_logits"].shape, (1, 6))
        self.assertEqual(actual["arc_logits"].shape, (1, 1, 512, 512))

    def test_new_head_update_preserves_old_parameters_and_bn_buffers(self):
        model = self.model()
        model.configure_stage("new_heads")
        before = {name: value.clone() for name, value in model.state_dict().items()}
        model.eval()
        with torch.inference_mode():
            old_predictions = model(self.batch["image"])
        model.train()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith(NEW_PARAMETER_PREFIXES), name)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.assertFalse(module.training)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss, components = training_loss(model(self.batch["image"]), self.batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for prefix in NEW_PARAMETER_PREFIXES:
            self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) > 0
                                for name, p in model.named_parameters() if name.startswith(prefix)), prefix)
        optimizer.step()
        state = model.state_dict()
        for name, value in before.items():
            if not name.startswith(NEW_PARAMETER_PREFIXES):
                self.assertTrue(torch.equal(state[name], value), name)
        self.assertTrue(any(not torch.equal(state[name], value) for name, value in before.items()
                            if name.startswith(NEW_PARAMETER_PREFIXES)))
        model.eval()
        with torch.inference_mode():
            after_predictions = model(self.batch["image"])
        for key in ("speed", "points", "heatmap_logits"):
            self.assertTrue(torch.equal(old_predictions[key], after_predictions[key]), key)
        self.assertIn("arc_dice_loss", components)

    def test_finetune_unfreezes_only_last_four_encoder_blocks(self):
        model = self.model()
        model.configure_stage("new_heads")
        model.configure_stage("finetune")
        model.train()
        for index, layer in enumerate(model.features):
            for parameter in layer.parameters():
                self.assertEqual(parameter.requires_grad, index >= len(model.features) - 4)
        for name, parameter in model.named_parameters():
            if not name.startswith("features."):
                self.assertTrue(parameter.requires_grad, name)
        for module in model.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.assertFalse(module.training)
        self.assertTrue(model.decode4.training)
        self.assertTrue(any(isinstance(module, nn.GroupNorm) for module in model.decode4.modules()))

    def test_migration_rejects_wrong_identity_or_any_old_tensor_discrepancy(self):
        changes = {
            "architecture": lambda p: p.update(architecture="different_model"),
            "preprocessing": lambda p: p.update(preprocessing="different_normalization"),
            "input": lambda p: p["config"].update(input_size=256),
            "missing": lambda p: p["model_state_dict"].pop("heatmap_head.bias"),
            "shape": lambda p: p["model_state_dict"].update({"heatmap_head.bias": torch.zeros(3)}),
            "nonfinite": lambda p: p["model_state_dict"].update({"heatmap_head.bias": torch.full((2,), float("nan"))}),
        }
        with tempfile.TemporaryDirectory(prefix="ring_migration_test_") as directory:
            for name, change in changes.items():
                with self.subTest(name=name):
                    altered = deepcopy(self.payload)
                    change(altered)
                    path = Path(directory) / f"{name}.pt"
                    torch.save(altered, path)
                    with self.assertRaises(ValueError):
                        load_legacy_checkpoint(RingSpeedometerModel(pretrained=False), path)

    def test_fresh_random_policy_can_train_all_encoder_parameters(self):
        model = RingSpeedometerModel(pretrained=False)
        model.configure_stage("new_heads", unfreeze_all_encoder=True)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        model.configure_stage("finetune", unfreeze_all_encoder=True)
        model.train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(not module.training for module in model.features.modules()
                            if isinstance(module, nn.modules.batchnorm._BatchNorm)))
        model.configure_stage("finetune")
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features[0].parameters()))

    @staticmethod
    def synthetic_loss_inputs():
        output = {
            "speed": torch.tensor([62.0], requires_grad=True),
            "points": torch.full((1, 2, 2), 0.5, requires_grad=True),
            "heatmap_logits": torch.zeros((1, 2, 8, 8), requires_grad=True),
            "tick_points": torch.full((1, 6, 2), 0.5, requires_grad=True),
            "tick_heatmap_logits": torch.zeros((1, 6, 8, 8), requires_grad=True),
            "tick_visibility_logits": torch.zeros((1, 6), requires_grad=True),
            "arc_logits": torch.zeros((1, 1, 512, 512), requires_grad=True),
        }
        batch = {"speed": torch.tensor([60.0]), "points": torch.full((1, 2, 2), 0.5),
                 "tick_points": torch.full((1, 6, 2), 0.6),
                 "tick_valid": torch.tensor([[0., 1., 1., 1., 1., 1.]]),
                 "arc_mask": torch.zeros((1, 1, 512, 512)),
                 "arc_valid": torch.ones((1, 1, 512, 512))}
        batch["arc_mask"][0, 0, 101, 101] = 1  # Single-pixel foreground must survive.
        batch["arc_valid"][0, 0, :8, :8] = 0
        return output, batch

    def test_ignored_targets_have_zero_loss_effect_and_zero_gradients(self):
        output, batch = self.synthetic_loss_inputs()
        baseline, _ = training_loss(output, batch)
        modified_output = {key: value.detach().clone().requires_grad_() for key, value in output.items()}
        modified_batch = {key: value.clone() for key, value in batch.items()}
        with torch.no_grad():
            modified_output["arc_logits"][0, 0, :8, :8] = 100
            modified_output["tick_heatmap_logits"][0, 0, :4, :4] = 100
            modified_output["tick_points"][0, 0] = 0.9
        modified_batch["arc_mask"][0, 0, :8, :8] = 1
        modified_batch["tick_points"][0, 0] = 0.1
        modified, _ = training_loss(modified_output, modified_batch)
        self.assertTrue(torch.equal(baseline, modified))
        modified.backward()
        self.assertEqual(torch.count_nonzero(modified_output["arc_logits"].grad[0, 0, :8, :8]), 0)
        self.assertEqual(torch.count_nonzero(modified_output["tick_heatmap_logits"].grad[0, 0]), 0)
        self.assertEqual(torch.count_nonzero(modified_output["tick_points"].grad[0, 0]), 0)
        self.assertLess(modified_output["arc_logits"].grad[0, 0, 101, 101].item(), 0)

    def test_empty_valid_masks_are_finite_and_zero_contribution(self):
        output, batch = self.synthetic_loss_inputs()
        batch["tick_valid"].zero_()
        batch["arc_valid"].zero_()
        loss, components = training_loss(output, batch)
        self.assertTrue(torch.isfinite(loss))
        for name in ("tick_heatmap_loss", "tick_coordinate_loss", "arc_bce_loss", "arc_dice_loss"):
            self.assertEqual(components[name], 0)
        loss.backward()
        self.assertEqual(torch.count_nonzero(output["arc_logits"].grad), 0)
        self.assertEqual(torch.count_nonzero(output["tick_heatmap_logits"].grad), 0)

    def test_real_review_dataset_guard_and_exact_preprocessing(self):
        rows, integrity = verify_dataset(REVIEW, allow_review=True)
        self.assertEqual(len(rows), 20)
        self.assertTrue(integrity["review_only"])
        self.assertEqual(integrity["split_counts"], {"review_only": 20})
        self.assertEqual(integrity["mask_files_verified"], 160)
        with self.assertRaises(ValueError):
            verify_dataset(REVIEW, allow_review=False)
        sample = self.data[0]
        self.assertTrue(torch.equal(sample["image"], legacy.preprocess_image(REVIEW / rows[0]["image"], 512)))
        altered = deepcopy(rows[0])
        altered["speed_kmh"] = 37
        with self.assertRaises(ValueError):
            RingDataset(REVIEW, records=[altered])


if __name__ == "__main__":
    unittest.main(verbosity=2)
