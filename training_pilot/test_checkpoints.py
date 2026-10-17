"""CPU contract tests for persistent, restartable training.

Run: python test_checkpoints.py

The real persistent training loop, AdamW, loss, dataloaders, and original image
labels are exercised with a tiny stochastic model. This makes exact restart
checks inexpensive and isolates checkpoint behavior from model accuracy. No
network download, user checkpoint, or Google Drive content is modified.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import train_core as core
import notebook_workflow as workflow


DATA_DIR = Path(__file__).resolve().parents[1] / "bw_dataset_500"


class TinyStochasticModel(nn.Module):
    """Same output/stage contract; dropout and all three RNGs affect training."""

    initializations = []

    def __init__(self, pretrained=True):
        super().__init__()
        self.initializations.append(bool(pretrained))
        self.features = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 8))
        self.dropout = nn.Dropout(0.2)
        self.heatmap_head = nn.Linear(8, 2 * 8 * 8)
        self.speed_head = nn.Linear(8, 1)

    def configure_stage(self, stage):
        if stage not in {"heads", "finetune"}:
            raise ValueError(stage)
        self.features.requires_grad_(stage == "finetune")
        self.train()

    def forward(self, image):
        value = self.features(image.mean(dim=(-2, -1)))
        if self.training:
            value = value * (0.9 + 0.1 * random.random() + 0.1 * np.random.random())
        value = self.dropout(value)
        heatmaps = self.heatmap_head(value).reshape(-1, 2, 8, 8)
        return {
            "speed": self.speed_head(value).squeeze(1).sigmoid() * 180.0,
            "points": core.heatmap_coordinates(heatmaps),
            "heatmap_logits": heatmaps,
        }


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.records, cls.integrity = core.verify_dataset(DATA_DIR)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="speedometer_checkpoint_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        TinyStochasticModel.initializations = []
        self.model_patch = patch.object(core, "SpeedometerModel", TinyStochasticModel)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        # Verify the real 500-image dataset once above, then reuse that evidence.
        self.data_patch = patch.object(core, "verify_dataset", side_effect=lambda _: (
            copy.deepcopy(self.records), copy.deepcopy(self.integrity)))
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.plot_patch = patch.object(core, "save_evaluation_artifacts")
        self.plot_patch.start()
        self.addCleanup(self.plot_patch.stop)

    def config(self, name, **changes):
        default = core.TrainConfig(
            data_dir=str(DATA_DIR), output_dir=str(self.root / name), input_size=64,
            batch_size=2, head_epochs=1, finetune_epochs=3, patience=50,
            seed=13, pretrained=False, max_train_samples=8, max_val_samples=4,
            amp=False, device="cpu", num_workers=0,
        )
        return replace(default, **changes)

    def checkpoint(self, result):
        return torch.load(result["latest_checkpoint_path"], map_location="cpu", weights_only=True)

    def assertNestedEqual(self, left, right, trail="state"):
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor, trail)
            self.assertTrue(torch.equal(left, right), trail)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, err_msg=trail)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right), trail)
            for key in left:
                self.assertNestedEqual(left[key], right[key], f"{trail}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), trail)
            for index, (a, b) in enumerate(zip(left, right)):
                self.assertNestedEqual(a, b, f"{trail}[{index}]")
        else:
            self.assertEqual(left, right, trail)

    def assertRestartEquivalent(self, expected, actual):
        # Full optimizer moments/step counters and all RNG streams must match,
        # not merely predictions or the number of epochs shown in a report.
        for key in ("model_state_dict", "optimizer_state_dict", "scaler_state_dict",
                    "rng_state", "loader_generator_states", "history"):
            self.assertNestedEqual(expected[key], actual[key], key)
        for key in ("epoch", "stage", "stage_epoch", "stage_best", "stale", "target_epoch"):
            self.assertNestedEqual(expected["training_state"][key],
                                   actual["training_state"][key], f"training_state.{key}")

    def test_resume_latest_exact_across_stage_boundary_and_fixed_target(self):
        uninterrupted = core.train_persistent(self.config("uninterrupted"))
        interrupted = core.train_persistent(self.config("interrupted"), stop_after_epochs=1)
        first = self.checkpoint(interrupted)
        self.assertEqual(first["training_state"]["epoch"], 1)
        self.assertEqual(first["training_state"]["stage"], "heads")
        self.assertFalse(interrupted["training_complete"])
        resumed = core.train_persistent(
            self.config("resumed"), mode="resume_latest",
            search_root=str(self.root / "interrupted"))
        self.assertRestartEquivalent(self.checkpoint(uninterrupted), self.checkpoint(resumed))
        self.assertEqual(resumed["resume_metadata"]["source_epoch"], 1)
        self.assertEqual(resumed["target_epoch"], 4)
        self.assertEqual(resumed["epochs_this_call"], 3)
        self.assertTrue(resumed["training_complete"])
        self.assertEqual(len(core.discover_checkpoints(self.root / "interrupted")), 1)
        # Restarting a finished run cannot silently add another four epochs.
        completed = core.train_persistent(
            self.config("completed_again"), mode="resume_latest",
            search_root=str(self.root / "resumed"))
        self.assertEqual(completed["epochs_this_call"], 0)
        self.assertEqual(completed["target_epoch"], 4)
        self.assertRestartEquivalent(self.checkpoint(resumed), self.checkpoint(completed))

    def test_resume_latest_exact_within_stage(self):
        full = core.train_persistent(self.config("full"))
        partial = core.train_persistent(self.config("partial"), stop_after_epochs=2)
        self.assertEqual(self.checkpoint(partial)["training_state"]["stage"], "finetune")
        resumed = core.train_persistent(self.config("restart"), mode="resume_latest",
                                        search_root=str(self.root / "partial"))
        self.assertRestartEquivalent(self.checkpoint(full), self.checkpoint(resumed))

    def increasing_mae(self, already_evaluated=0):
        evaluate = core.evaluate_model
        calls = already_evaluated

        def controlled(model, loader, device):
            nonlocal calls
            metrics, rows = evaluate(model, loader, device)
            calls += 1
            metrics["mae_kmh"] = float(calls)
            return metrics, rows

        return patch.object(core, "evaluate_model", side_effect=controlled)

    def test_resume_best_rewinds_full_state_not_only_weights(self):
        config = self.config("source", head_epochs=0, finetune_epochs=3)
        with self.increasing_mae():
            source = core.train_persistent(config)
        selected = core.find_checkpoint(self.root / "source", "resume_best")
        self.assertEqual(selected["epoch"], 1)
        self.assertEqual(core.find_checkpoint(self.root / "source", "resume_latest")["epoch"], 3)
        with self.increasing_mae(already_evaluated=1):
            resumed = core.train_persistent(replace(config, output_dir=str(self.root / "best_branch")),
                                            mode="resume_best", search_root=str(self.root / "source"),
                                            stop_after_epochs=1)
        with self.increasing_mae():
            expected = core.train_persistent(replace(config, output_dir=str(self.root / "expected")),
                                             stop_after_epochs=2)
        self.assertEqual(resumed["resume_metadata"]["source_epoch"], 1)
        self.assertEqual(self.checkpoint(resumed)["training_state"]["epoch"], 2)
        self.assertRestartEquivalent(self.checkpoint(expected), self.checkpoint(resumed))
        self.assertEqual(self.checkpoint(source)["training_state"]["epoch"], 3)

    def test_early_stopping_counter_survives_resume(self):
        config = self.config("full", head_epochs=0, finetune_epochs=6, patience=2)
        with self.increasing_mae():
            full = core.train_persistent(config)
        with self.increasing_mae():
            partial = core.train_persistent(replace(config, output_dir=str(self.root / "partial")),
                                             stop_after_epochs=2)
        self.assertEqual(self.checkpoint(partial)["training_state"]["stale"], 1)
        with self.increasing_mae(already_evaluated=2):
            resumed = core.train_persistent(replace(config, output_dir=str(self.root / "resumed")),
                                            mode="resume_latest", search_root=str(self.root / "partial"))
        self.assertEqual(resumed["epochs_this_call"], 1)
        self.assertTrue(resumed["training_complete"])
        self.assertRestartEquivalent(self.checkpoint(full), self.checkpoint(resumed))

    def test_early_stopped_heads_do_not_increase_finetune_budget(self):
        config = self.config("full", head_epochs=5, finetune_epochs=2, patience=2)
        with self.increasing_mae():
            full = core.train_persistent(config)
        with self.increasing_mae():
            partial = core.train_persistent(replace(config, output_dir=str(self.root / "partial")),
                                             stop_after_epochs=3)
        boundary = self.checkpoint(partial)
        self.assertEqual(boundary["training_state"]["stage"], "heads")
        self.assertEqual(boundary["training_state"]["stale"], 2)
        with self.increasing_mae(already_evaluated=3):
            resumed = core.train_persistent(replace(config, output_dir=str(self.root / "resumed")),
                                            mode="resume_latest", search_root=str(self.root / "partial"))
        actual = self.checkpoint(resumed)
        self.assertEqual([row["stage"] for row in actual["history"]],
                         ["heads"] * 3 + ["finetune"] * 2)
        self.assertEqual(resumed["epochs_this_call"], 2)
        self.assertEqual(actual["training_state"]["epoch"], 5)
        self.assertTrue(resumed["training_complete"])
        self.assertEqual(resumed["completion_reason"], "stage_budget_reached")
        self.assertRestartEquivalent(self.checkpoint(full), actual)

    def test_fresh_random_ignores_saved_runs_and_refuses_overwrite(self):
        saved = core.train_persistent(self.config("saved"), stop_after_epochs=2)
        fresh = core.train_persistent(self.config("fresh"), mode="fresh",
                                      search_root=str(self.root / "saved"), stop_after_epochs=1)
        control = core.train_persistent(self.config("control"), mode="fresh", stop_after_epochs=1)
        self.assertRestartEquivalent(self.checkpoint(control), self.checkpoint(fresh))
        self.assertEqual(self.checkpoint(fresh)["training_state"]["epoch"], 1)
        self.assertTrue(all(value is False for value in TinyStochasticModel.initializations))
        digest_before = core._sha256(saved["latest_checkpoint_path"])
        with self.assertRaises(FileExistsError):
            core.train_persistent(self.config("saved"), mode="fresh")
        self.assertEqual(core._sha256(saved["latest_checkpoint_path"]), digest_before)

    def test_mismatched_or_missing_resume_never_silently_starts_fresh(self):
        core.train_persistent(self.config("source"), stop_after_epochs=1)
        for changed in ({"input_size": 128}, {"max_train_samples": 4}, {"head_lr": 0.03}):
            with self.subTest(changed=changed):
                with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                    core.train_persistent(self.config("incompatible_" + next(iter(changed)), **changed),
                                          mode="resume_latest",
                                          search_root=str(self.root / "source"))
        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            core.train_persistent(self.config("missing"), mode="resume_best",
                                  search_root=str(self.root / "does_not_exist"))

    def test_corrupt_and_uncommitted_checkpoints_are_ignored(self):
        result = core.train_persistent(self.config("source"), stop_after_epochs=2)
        latest_path = Path(result["latest_checkpoint_path"])
        known_good = self.checkpoint(result)
        # A half-written save and a fully written tensor file without a commit
        # marker must not supersede a previously verified checkpoint.
        (latest_path.parent / "epoch_999999.pt.tmp").write_bytes(b"partial")
        torch.save(known_good, latest_path.parent / "epoch_999998.pt")
        latest_path.write_bytes(b"corrupted after a simulated interrupted write")
        candidates = core.discover_checkpoints(self.root / "source")
        self.assertEqual([row["epoch"] for row in candidates], [1])
        selected = core.find_checkpoint(self.root / "source", "resume_latest")
        self.assertEqual(selected["epoch"], 1)
        resumed = core.train_persistent(self.config("fallback"), mode="resume_latest",
                                        search_root=str(self.root / "source"), stop_after_epochs=1)
        self.assertEqual(resumed["resume_metadata"]["source_epoch"], 1)
        Path(selected["checkpoint_path"]).write_bytes(b"also corrupted")
        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            core.train_persistent(self.config("all_corrupt"), mode="resume_latest",
                                  search_root=str(self.root / "source"))

    def test_legacy_warm_start_preserves_baseline_and_absolute_target(self):
        config = self.config("warm", head_epochs=5, finetune_epochs=25)
        core._seed_everything(config.seed)
        model = TinyStochasticModel(pretrained=False)
        validation = core.select_group_subset(
            [row for row in self.records if row["dataset_split"] == "validation"], 4, config.seed)
        loader = core._loader(core.DashboardDataset(DATA_DIR, validation, config.input_size), config)
        metrics, _ = core.evaluate_model(model, loader, torch.device("cpu"))
        metrics["mae_kmh"] = 0.001  # Deliberately unbeatable to detect lost baseline weights.
        baseline_weights = copy.deepcopy(model.state_dict())
        legacy = self.root / "legacy_epoch_29.pt"
        torch.save({"format_version": 1, "model_state_dict": baseline_weights,
                    "config": asdict(config), "best_epoch": 29, "validation_metrics": metrics,
                    "labels_sha256": self.integrity["labels_sha256"],
                    "preprocessing": core._PREPROCESSING_ID,
                    "keypoint_order": ["pivot", "tip"], "speed_range_kmh": [0, 180]}, legacy)
        result = core.train_persistent(config, mode="warm_start", warm_start_path=str(legacy),
                                       additional_epochs=20, stop_after_epochs=1)
        self.assertEqual(result["target_epoch"], 49)
        self.assertEqual(result["best_epoch"], 29)
        self.assertEqual(result["epochs_this_call"], 1)
        self.assertEqual(self.checkpoint(result)["training_state"]["epoch"], 30)
        best_model = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
        self.assertNestedEqual(best_model["model_state_dict"], baseline_weights)
        resumed = core.train_persistent(replace(config, output_dir=str(self.root / "warm_restart")),
                                        mode="resume_latest", search_root=str(self.root / "warm"),
                                        additional_epochs=20, stop_after_epochs=1)
        self.assertEqual(resumed["target_epoch"], 49)
        self.assertEqual(self.checkpoint(resumed)["training_state"]["epoch"], 31)
        self.assertEqual(resumed["best_epoch"], 29)

    def test_notebook_recovers_verified_state_from_interrupted_drive_save(self):
        drive = self.root / "drive"
        plan = workflow.prepare_persistent_run(
            core, data_dir=DATA_DIR, drive_root=drive, experiment_name="restart_test",
            requested_mode="fresh", run_mode="smoke", fresh_pretrained=False)
        first = core.train_persistent(plan["config"], **plan["kwargs"], stop_after_epochs=1)
        original_path = Path(first["latest_checkpoint_path"])
        original_hash = core._sha256(original_path)
        partial = original_path.parent / "epoch_00002_interrupted.pt.partial"
        partial.write_bytes(b"incomplete checkpoint upload")
        restarted = workflow.prepare_persistent_run(
            core, data_dir=DATA_DIR, drive_root=drive, experiment_name="restart_test",
            requested_mode="resume_latest", run_mode="smoke")
        self.assertEqual(restarted["actual_mode"], "resume_latest")
        self.assertNotEqual(restarted["run_dir"], plan["run_dir"])
        self.assertTrue(Path(restarted["run_dir"]).is_relative_to(drive))
        self.assertEqual(partial.read_bytes(), b"incomplete checkpoint upload")
        result = core.train_persistent(restarted["config"], **restarted["kwargs"])
        self.assertEqual(result["resume_metadata"]["source_epoch"], 1)
        self.assertEqual(result["epochs_this_call"], 1)
        self.assertEqual(result["target_epoch"], 2)
        self.assertEqual(core._sha256(original_path), original_hash)
        # No good epoch remaining means a hard failure, never a legacy reimport.
        for candidate in core.discover_checkpoints(plan["experiment_dir"]):
            Path(candidate["checkpoint_path"]).write_bytes(b"corrupt")
        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            workflow.prepare_persistent_run(
                core, data_dir=DATA_DIR, drive_root=drive, experiment_name="restart_test",
                requested_mode="resume_best", run_mode="smoke")

    def test_notebook_fresh_and_runtime_identity_guards(self):
        drive = self.root / "drive"
        with self.assertRaises(ValueError):
            workflow.prepare_persistent_run(
                core, data_dir=DATA_DIR, drive_root=drive, experiment_name="invalid_fresh",
                requested_mode="fresh", run_mode="smoke", extend_by_epochs=1)
        self.assertFalse((drive / "experiments" / "invalid_fresh" / "experiment_config.json").exists())
        plan = workflow.prepare_persistent_run(
            core, data_dir=DATA_DIR, drive_root=drive, experiment_name="fresh_test",
            requested_mode="fresh", run_mode="smoke", fresh_pretrained=False)
        self.assertFalse(plan["config"].pretrained)
        manifest = Path(plan["manifest_path"])
        original = manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            workflow.prepare_persistent_run(
                core, data_dir=DATA_DIR, drive_root=drive, experiment_name="fresh_test",
                requested_mode="fresh", run_mode="smoke")
        versions = workflow.runtime_versions()
        versions["numpy"] += "_changed"
        with patch.object(workflow, "runtime_versions", return_value=versions):
            with self.assertRaises(RuntimeError):
                workflow.prepare_persistent_run(
                    core, data_dir=DATA_DIR, drive_root=drive, experiment_name="fresh_test",
                    requested_mode="resume_latest", run_mode="smoke")
        self.assertEqual(manifest.read_bytes(), original)
        fresh_again = workflow.prepare_persistent_run(
            core, data_dir=DATA_DIR, drive_root=drive, experiment_name="fresh_new_name",
            requested_mode="fresh", run_mode="smoke", fresh_pretrained=False)
        self.assertEqual(fresh_again["actual_mode"], "fresh")
        self.assertNotEqual(fresh_again["experiment_dir"], plan["experiment_dir"])

    def test_notebook_legacy_import_is_hash_checked_and_fixes_twenty_epochs(self):
        drive = self.root / "drive"
        legacy = self.root / "legacy.pt"
        legacy.write_bytes(b"fixture for notebook import policy; tensor loading is tested separately")
        with patch.object(workflow, "LEGACY_SHA256", core._sha256(legacy)):
            plan = workflow.prepare_persistent_run(
                core, data_dir=DATA_DIR, drive_root=drive, experiment_name="continuation",
                requested_mode="resume_latest", run_mode="pilot", warm_start_path=legacy,
                approved_source_sha256=workflow.LEGACY_SHA256)
            self.assertEqual(plan["actual_mode"], "warm_start")
            self.assertEqual(plan["kwargs"]["additional_epochs"], 20)
            self.assertEqual((plan["config"].head_epochs, plan["config"].finetune_epochs), (0, 20))
            self.assertEqual(plan["config"].input_size, 512)
            self.assertEqual(plan["config"].max_train_samples, 400)
            self.assertIsNone(plan["config"].max_val_samples)
            manifest = json.loads(Path(plan["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["additional_epochs"], 20)
            self.assertEqual(manifest["initialization_kind"], "legacy_warm_start")
            legacy.write_bytes(b"changed or corrupted legacy file")
            with self.assertRaises(ValueError):
                workflow.prepare_persistent_run(
                    core, data_dir=DATA_DIR, drive_root=drive, experiment_name="continuation",
                    requested_mode="resume_best", run_mode="pilot", warm_start_path=legacy,
                    approved_source_sha256=workflow.LEGACY_SHA256)

    def transfer_source(self, config):
        core._seed_everything(config.seed)
        model = TinyStochasticModel(pretrained=False)
        payload = {
            "format_version": 1, "model_state_dict": copy.deepcopy(model.state_dict()),
            "config": asdict(config), "best_epoch": 47,
            "validation_metrics": {"mae_kmh": 0.001},
            "labels_sha256": self.integrity["labels_sha256"],
            "preprocessing": core._PREPROCESSING_ID, "architecture": core._ARCHITECTURE_ID,
            "keypoint_order": ["pivot", "tip"], "speed_range_kmh": [0, 180],
        }
        path = self.root / "source_epoch47.pt"
        torch.save(payload, path)
        return path, payload

    def transfer_target(self):
        records, integrity = copy.deepcopy(self.records), copy.deepcopy(self.integrity)
        # This fixture tests dataset-identity and report plumbing, not image
        # rotation rendering (which has its own geometric audit).
        integrity["labels_sha256"] = "a" * 64
        for index, row in enumerate(records):
            row["rotation_degrees"] = 0 if index % 2 == 0 else 90
            row["parent_sample_id"] = "parent_" + row["sample_id"]
        return patch.object(core, "verify_dataset", side_effect=lambda _: (
            copy.deepcopy(records), copy.deepcopy(integrity)))

    def test_transfer_recomputes_baseline_and_restores_epoch_zero_exactly(self):
        config = self.config("transfer", head_epochs=0, finetune_epochs=20)
        source, payload = self.transfer_source(config)
        with self.transfer_target():
            initial = core.train_persistent(config, mode="warm_start_transfer", warm_start_path=source,
                                            additional_epochs=20, stop_after_epochs=0)
            checkpoint = self.checkpoint(initial)
            self.assertEqual(checkpoint["training_state"]["epoch"], 0)
            self.assertEqual(checkpoint["training_state"]["target_epoch"], 20)
            self.assertEqual(checkpoint["optimizer_state_dict"]["state"], {})
            self.assertEqual(initial["resume_metadata"]["source_epoch"], 47)
            self.assertFalse(initial["resume_metadata"]["optimizer_restored"])
            self.assertFalse(initial["resume_metadata"]["full_state_restored"])
            self.assertEqual(checkpoint["identity"]["labels_sha256"], "a" * 64)
            self.assertNestedEqual(checkpoint["model_state_dict"], payload["model_state_dict"])
            self.assertNotAlmostEqual(checkpoint["validation_metrics"]["mae_kmh"], 0.001)
            self.assertEqual(checkpoint["best_state"]["validation_metrics"],
                             checkpoint["transfer_baseline"]["report"]["metrics"])
            self.assertEqual(initial["baseline_validation_report"]["training_updates_before_evaluation"], 0)
            self.assertEqual(set(initial["baseline_validation_report"]["metrics_by_rotation_degrees"]),
                             {"0", "90"})
            self.assertTrue(all(row["parent_sample_id"].startswith("parent_")
                                for row in checkpoint["transfer_baseline"]["rows"]))
            resumed = core.train_persistent(replace(config, output_dir=str(self.root / "resumed")),
                                            mode="resume_latest", search_root=self.root / "transfer",
                                            additional_epochs=99, stop_after_epochs=1)
            direct = core.train_persistent(replace(config, output_dir=str(self.root / "direct")),
                                           mode="warm_start_transfer", warm_start_path=source,
                                           additional_epochs=20, stop_after_epochs=1)
            self.assertEqual(resumed["target_epoch"], 20)
            self.assertEqual(resumed["epochs_run"], 1)
            self.assertRestartEquivalent(self.checkpoint(direct), self.checkpoint(resumed))
            self.assertNestedEqual(self.checkpoint(initial)["transfer_baseline"],
                                   self.checkpoint(resumed)["transfer_baseline"])

    def test_changed_dataset_requires_transfer_not_full_state_resume(self):
        config = self.config("old_data", head_epochs=0, finetune_epochs=20)
        core.train_persistent(config, stop_after_epochs=1)
        with self.transfer_target():
            with self.assertRaises(FileNotFoundError):
                core.train_persistent(replace(config, output_dir=str(self.root / "wrong_resume")),
                                      mode="resume_latest", search_root=self.root / "old_data")

    def test_transfer_rejects_incompatible_architecture_preprocessing_and_outputs(self):
        config = self.config("unused", head_epochs=0, finetune_epochs=20)
        _, payload = self.transfer_source(config)
        mismatches = [
            ("architecture", "different_model"), ("preprocessing", "different_crop"),
            ("keypoint_order", ["tip", "pivot"]), ("speed_range_kmh", [0, 240]),
            ("config", {**payload["config"], "input_size": 128}),
        ]
        with self.transfer_target():
            for field, value in mismatches:
                with self.subTest(field=field):
                    altered = {**payload, field: value}
                    path = self.root / ("bad_" + field + ".pt")
                    torch.save(altered, path)
                    with self.assertRaises(ValueError):
                        core.train_persistent(replace(config, output_dir=str(self.root / ("bad_" + field))),
                                              mode="warm_start_transfer", warm_start_path=path,
                                              stop_after_epochs=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
