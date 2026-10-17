"""Small stochastic CPU fixtures test persistence, never speedometer accuracy."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

import ring_model
import ring_train as trainer


class TinyModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        assert not pretrained
        self.shared = nn.Linear(3, 4)
        self.tick_head = nn.Linear(4, 1)
        self.dropout = nn.Dropout(0.25)

    def configure_stage(self, stage, *, unfreeze_all_encoder=False):
        self.unfreeze_all_encoder = unfreeze_all_encoder
        for parameter in self.shared.parameters():
            parameter.requires_grad_(stage == "finetune")
        for parameter in self.tick_head.parameters():
            parameter.requires_grad_(True)

    def forward(self, image):
        return {"speed": self.tick_head(self.dropout(torch.tanh(self.shared(image)))).squeeze(-1)}


class TinyDataset(Dataset):
    def __init__(self, data_dir, records=None, input_size=512):
        self.records, self.root = records, Path(data_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        # Exercise Python, numpy, torch, dropout, and loader shuffle state.
        value = float(self.records[index]["value"])
        jitter = (random.random() + np.random.random() + float(torch.rand(()))) * 0.005
        return {"image": torch.tensor([value, value * .25, 1.]) + jitter,
                "speed": torch.tensor(value * .6 + .4), "index": index}


def tiny_loss(output, batch, loss_config=None):
    loss = (output["speed"] - batch["speed"]).square().mean()
    return loss, {"speed_loss": float(loss.detach())}


def fake_migration(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = model.state_dict()
    state.update(saved["shared"])
    model.load_state_dict(state)
    return {"source_sha256": trainer._sha(path), "source_epoch": saved["epoch"],
            "optimizer_restored": False, "copied_tensor_names": list(saved["shared"])}


def assert_tree_equal(test, first, second):
    if not isinstance(first, dict) or not isinstance(second, dict):
        test.assertIs(type(first), type(second))
    if isinstance(first, torch.Tensor):
        test.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_tree_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(first, second)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="rg_")
        self.root = Path(self.temporary.name)
        self.data = self.root / "d"
        self.data.mkdir()
        (self.data / "labels.json").write_text('{"fixture":1}')
        self.records = [{"sample_id": f"s{i}", "dataset_split": "review_only", "value": i / 8,
                         "source_pose_group": f"g{i}"} for i in range(8)]
        def verify(data_dir, allow_review=False):
            if not allow_review:
                raise ValueError("Review-only guard")
            return copy.deepcopy(self.records), {"review_only": True, "count": len(self.records)}
        self.patches = [patch.object(trainer.ring_model, "RingSpeedometerModel", TinyModel),
                        patch.object(trainer.ring_model, "training_loss", tiny_loss),
                        patch.object(trainer.ring_model, "load_legacy_checkpoint", fake_migration),
                        patch.object(trainer.ring_data, "RingDataset", TinyDataset),
                        patch.object(trainer.ring_data, "verify_dataset", verify)]
        for p in self.patches:
            p.start()
        self.config = trainer.RingTrainConfig(str(self.data), str(self.root / "runs"), batch_size=4,
                                              head_epochs=1, finetune_epochs=1, device="cpu", amp=False,
                                              review_smoke=True, head_lr=.02, finetune_lr=.01, backbone_lr=.005)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.temporary.cleanup()

    def payload(self, report):
        return torch.load(report["latest_checkpoint_path"], map_location="cpu", weights_only=True)

    def test_exact_resume_across_stage_boundary_restores_all_stochastic_state(self):
        uninterrupted = trainer.train(replace(self.config, output_dir=str(self.root / "full")), mode="fresh")
        interrupted = trainer.train(self.config, mode="fresh", stop_after_epochs=1)
        initial = torch.load(interrupted["initial_checkpoint_path"], weights_only=True)
        middle = self.payload(interrupted)
        rates = {group["name"]: group["lr"] for group in middle["optimizer_state_dict"]["param_groups"]}
        self.assertEqual(rates, {"encoder": self.config.backbone_lr, "shared": self.config.finetune_lr,
                                 "new": self.config.head_lr})
        for name in initial["model_state_dict"]:
            if name.startswith("shared."):
                self.assertTrue(torch.equal(initial["model_state_dict"][name], middle["model_state_dict"][name]))
        self.assertFalse(torch.equal(initial["model_state_dict"]["tick_head.weight"], middle["model_state_dict"]["tick_head.weight"]))
        resumed = trainer.train(self.config, mode="resume_latest", search_root=self.root / "runs")
        self.assertTrue(resumed["resume_metadata"]["optimizer_restored"])
        self.assertEqual(resumed["epochs_this_call"], 1)
        self.assertEqual(resumed["source_baseline_checkpoint_path"], interrupted["initial_checkpoint_path"])
        full, actual = self.payload(uninterrupted), self.payload(resumed)
        rates = {group["name"]: group["lr"] for group in actual["optimizer_state_dict"]["param_groups"]}
        self.assertEqual(rates, {"encoder": self.config.backbone_lr, "shared": self.config.finetune_lr,
                                 "new": self.config.new_head_finetune_lr})
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
                    "rng_state", "loader_generator_states", "training_state", "history", "best_state", "before_metrics"):
            assert_tree_equal(self, full[key], actual[key])
        hashes = {p: trainer._sha(p) for p in (self.root / "runs").rglob("epoch_*.pt")}
        completed = trainer.train(self.config, mode="resume_latest", search_root=self.root / "runs")
        self.assertEqual(completed["epochs_this_call"], 0)
        self.assertEqual(completed["target_epoch"], 2)
        self.assertNotEqual(completed["run_dir"], resumed["run_dir"])
        for path, checksum in hashes.items():
            self.assertEqual(trainer._sha(path), checksum)

    def test_migration_starts_epoch_zero_and_resets_optimizer(self):
        source_model = TinyModel()
        source = self.root / "legacy.pt"
        old = {name: torch.full_like(value, .3) for name, value in source_model.state_dict().items() if name.startswith("shared.")}
        torch.save({"shared": old, "epoch": 19}, source)
        checksum = trainer._sha(source)
        report = trainer.train(self.config, mode="transfer", source_checkpoint=source, stop_after_epochs=0)
        saved = self.payload(report)
        self.assertEqual(saved["training_state"]["epoch"], 0)
        self.assertEqual(saved["training_state"]["target_epoch"], 2)
        self.assertEqual(saved["initialization"]["source_epoch"], 19)
        self.assertEqual(saved["initialization"]["mode"], "transfer")
        self.assertFalse(saved["initialization"]["unfreeze_all_encoder"])
        self.assertEqual(saved["optimizer_state_dict"]["state"], {})
        self.assertEqual(report["evaluation_semantics"], "training_reconstruction_check")
        self.assertEqual(report["before_metrics"], report["after_metrics"])
        for name, value in old.items():
            self.assertTrue(torch.equal(saved["model_state_dict"][name], value))
        self.assertEqual(trainer._sha(source), checksum)
        loaded, payload = trainer.load_model(report["initial_checkpoint_path"])
        assert_tree_equal(self, loaded.state_dict(), saved["model_state_dict"])
        self.assertEqual(payload["training_state"]["epoch"], 0)

    def test_fresh_ignores_old_or_corrupt_files_and_requires_review_opt_in(self):
        corrupt = self.root / "old"
        corrupt.mkdir()
        (corrupt / "epoch_00001.complete.json").write_text("bad")
        report = trainer.train(self.config, mode="fresh", source_checkpoint=corrupt / "absent.pt", search_root=corrupt, stop_after_epochs=0)
        self.assertIsNone(report["resume_metadata"]["source_checkpoint_path"])
        self.assertEqual(report["initialization"]["mode"], "fresh")
        self.assertTrue(report["initialization"]["unfreeze_all_encoder"])
        self.assertEqual(report["identity"]["finetune_policy"]["fresh_random"], "all_encoder")
        self.assertIn("ring_train", report["identity"]["implementation_hashes"])
        self.assertIn(trainer.ring_model.legacy.__name__, report["identity"]["implementation_hashes"])
        with self.assertRaisesRegex(ValueError, "Review"):
            trainer.train(replace(self.config, review_smoke=False), mode="fresh")

    def test_resume_best_restores_selected_full_state(self):
        scores = iter([4., 1., 3.])
        def evaluation(*args):
            score = next(scores)
            return {"combined_loss": score, "samples": 8, "components": {"speed_loss": score}}
        with patch.object(trainer, "evaluate_loss", evaluation):
            first = trainer.train(self.config, mode="fresh")
        self.assertEqual(first["best_epoch"], 1)
        best = trainer.train(self.config, mode="resume_best", search_root=self.root / "runs", stop_after_epochs=0)
        self.assertEqual(best["completed_epoch"], 1)
        self.assertEqual(best["target_epoch"], 2)
        self.assertTrue(best["resume_metadata"]["full_state_restored"])
        source = torch.load(first["best_checkpoint_path"], weights_only=True)
        selected = torch.load(best["latest_checkpoint_path"], weights_only=True)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "loader_generator_states", "history"):
            assert_tree_equal(self, source[key], selected[key])

    def test_corruption_incompatibility_and_incomplete_state_fail_closed(self):
        report = trainer.train(self.config, mode="fresh", stop_after_epochs=0)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            trainer.train(replace(self.config, batch_size=2), mode="resume_latest", search_root=self.root / "runs")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            trainer.train(replace(self.config, finetune_epochs=2), mode="resume_latest", search_root=self.root / "runs")
        (self.data / "labels.json").write_text('{"fixture":2}')
        with self.assertRaisesRegex(ValueError, "incompatible"):
            trainer.train(self.config, mode="resume_latest", search_root=self.root / "runs")
        (self.data / "labels.json").write_text('{"fixture":1}')
        path = Path(report["latest_checkpoint_path"])
        with path.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            trainer.train(self.config, mode="resume_latest", search_root=self.root / "runs")
        empty = self.root / "empty"
        empty.mkdir()
        (empty / "epoch_00001_dead.pt").write_bytes(b"unpublished")
        with self.assertRaisesRegex(ValueError, "No verified"):
            trainer.train(self.config, mode="resume_latest", search_root=empty)

    def test_pose_group_leakage_rejected(self):
        records = [{"sample_id": "a", "dataset_split": "train", "source_pose_group": "same"},
                   {"sample_id": "b", "dataset_split": "validation", "source_pose_group": "same"}]
        with self.assertRaisesRegex(ValueError, "leakage"):
            trainer._partitions(records, False)

    def test_corrupt_latest_and_orphan_recover_previous_state_with_diagnostics(self):
        first = trainer.train(self.config, mode="fresh")
        latest = Path(first["latest_checkpoint_path"])
        original = latest.read_bytes()
        latest.write_bytes(original + b"corrupt")
        orphan = latest.parent / "epoch_00999_unpublished.pt"
        orphan.write_bytes(b"incomplete")
        with self.assertWarnsRegex(RuntimeWarning, "Skipped 2"):
            recovered = trainer.train(self.config, mode="resume_latest", search_root=self.root / "runs", stop_after_epochs=0)
        self.assertEqual(recovered["completed_epoch"], 1)
        self.assertEqual(len(recovered["resume_metadata"]["recovery_diagnostics"]), 2)
        self.assertEqual(latest.read_bytes(), original + b"corrupt")
        source = self.payload(recovered)
        self.assertEqual(source["training_state"]["target_epoch"], 2)
        self.assertEqual(recovered["initialization"]["unfreeze_all_encoder"], True)

    def test_incompatible_foreign_candidate_cannot_override_compatible_state(self):
        compatible = trainer.train(self.config, mode="fresh", stop_after_epochs=1)
        trainer.train(replace(self.config, batch_size=2), mode="fresh", stop_after_epochs=0)
        with self.assertWarnsRegex(RuntimeWarning, "incompatible"):
            recovered = trainer.train(self.config, mode="resume_latest", search_root=self.root / "runs", stop_after_epochs=0)
        self.assertEqual(recovered["latest_checkpoint_path"], compatible["latest_checkpoint_path"])
        self.assertEqual(recovered["completed_epoch"], 1)
        self.assertEqual(len(recovered["resume_metadata"]["recovery_diagnostics"]), 1)


class MaskedLossTests(unittest.TestCase):
    def test_ignored_arc_pixels_and_hidden_tick_coordinates_have_zero_effect(self):
        torch.manual_seed(3)
        output = {"speed": torch.tensor([30.], requires_grad=True), "points": torch.rand(1, 2, 2, requires_grad=True),
                  "heatmap_logits": torch.randn(1, 2, 8, 8, requires_grad=True),
                  "tick_points": torch.rand(1, 6, 2, requires_grad=True),
                  "tick_heatmap_logits": torch.randn(1, 6, 8, 8, requires_grad=True),
                  "arc_logits": torch.randn(1, 1, 8, 8, requires_grad=True),
                  "tick_visibility_logits": torch.randn(1, 6, requires_grad=True)}
        valid = torch.ones(1, 1, 8, 8)
        valid[:, :, :, :3] = 0
        batch = {"speed": torch.tensor([25.]), "points": torch.rand(1, 2, 2),
                 "tick_points": torch.rand(1, 6, 2), "tick_valid": torch.tensor([[1., 0, 1, 1, 1, 1]]),
                 "arc_mask": (torch.rand(1, 1, 8, 8) > .7).float(), "arc_valid": valid}
        first, _ = ring_model.training_loss(output, batch)
        first.backward()
        self.assertTrue(torch.equal(output["arc_logits"].grad[valid == 0], torch.zeros_like(output["arc_logits"].grad[valid == 0])))
        self.assertTrue(torch.equal(output["tick_points"].grad[:, 1], torch.zeros_like(output["tick_points"].grad[:, 1])))
        self.assertTrue(torch.equal(output["tick_heatmap_logits"].grad[:, 1], torch.zeros_like(output["tick_heatmap_logits"].grad[:, 1])))
        changed_output = {key: value.detach().clone() for key, value in output.items()}
        changed_batch = {key: value.clone() for key, value in batch.items()}
        changed_output["arc_logits"][valid == 0] = 100
        changed_output["tick_points"][:, 1] = .99
        changed_output["tick_heatmap_logits"][:, 1] = -99
        changed_batch["arc_mask"][valid == 0] = 1 - changed_batch["arc_mask"][valid == 0]
        changed_batch["tick_points"][:, 1] = .01
        second, _ = ring_model.training_loss(changed_output, changed_batch)
        self.assertEqual(float(first.detach()), float(second.detach()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
