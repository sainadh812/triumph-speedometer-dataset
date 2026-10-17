"""V2 grouped training, validation selection, and epoch-boundary persistence.

The test split is never loaded here. Transfer is a new experiment with a fresh
optimizer; exact resume only accepts this implementation's complete states.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
import warnings

import numpy as np
import PIL
import torch
import torchvision
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from ring_training_v2 import data as ring_data
from ring_training_v2 import model as ring_model
from ring_training import ring_eval

FORMAT_VERSION = 1
CHECKPOINT_KIND = "augmented_ring_full_training_state_v2"
FINETUNE_POLICY = {"fresh_random": "all_encoder", "fresh_pretrained": "last_four_encoder_blocks",
                   "transfer": "last_four_encoder_blocks"}


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    input_size: int = 512
    batch_size: int = 8
    head_epochs: int = 3
    finetune_epochs: int = 27
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    amp: bool = True
    pretrained: bool = False
    train_sample_limit: int | None = None
    geometry_penalty_px: float = 20.0
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    new_head_finetune_lr: float = 1e-3
    backbone_lr: float = 2e-5
    needle_lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    loss: dict = field(default_factory=dict)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.parent / (".tmp_" + uuid.uuid4().hex[:12] + ".partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"kind": n[0], "keys": n[1].tolist(),
            "position": n[2], "has_gauss": n[3], "cached_gaussian": n[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["kind"], np.array(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA random state cannot be restored on a CPU runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_payload(payload):
    required = {"format_version", "checkpoint_kind", "identity", "config", "model_state_dict",
                "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "rng_state",
                "loader_generator_states", "training_state", "history", "best_state",
                "before_metrics", "current_metrics", "initialization", "source_baseline_checkpoint_path"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Incomplete ring training state")
    if payload["format_version"] != FORMAT_VERSION or payload["checkpoint_kind"] != CHECKPOINT_KIND:
        raise ValueError("Unsupported checkpoint format or architecture family")
    state = payload["training_state"]
    epoch, target = state["epoch"], state["target_epoch"]
    if not isinstance(epoch, int) or not isinstance(target, int) or not 0 <= epoch <= target:
        raise ValueError("Invalid saved epoch/target")
    if target != payload["config"]["head_epochs"] + payload["config"]["finetune_epochs"]:
        raise ValueError("Saved target disagrees with the fixed stage budgets")
    expected_stage = "new_heads" if (epoch <= payload["config"]["head_epochs"] and payload["config"]["head_epochs"] > 0) else "finetune"
    if state["stage"] != expected_stage:
        raise ValueError("Saved stage disagrees with the completed epoch")
    if [row["epoch"] for row in payload["history"]] != list(range(1, epoch + 1)):
        raise ValueError("Saved history does not cover every completed epoch")
    if state["training_complete"] != (epoch == target):
        raise ValueError("Saved completion flag disagrees with fixed epoch target")
    if not (0 <= payload["best_state"]["epoch"] <= epoch) or not math.isfinite(payload["best_state"]["selection_loss"]):
        raise ValueError("Invalid best-state selection")
    for key in ("python", "numpy", "torch", "cuda"):
        if key not in payload["rng_state"]:
            raise ValueError("Missing random-generator state")
    if set(payload["loader_generator_states"]) != {"train", "evaluation"}:
        raise ValueError("Missing data-loader generator state")
    if payload["scheduler_state_dict"] is not None:
        raise ValueError("This trainer uses fixed stage learning rates, not a scheduler")
    if not isinstance(payload["model_state_dict"], dict) or not payload["model_state_dict"]:
        raise ValueError("Missing model tensors")
    if not {"state", "param_groups"}.issubset(payload["optimizer_state_dict"]):
        raise ValueError("Missing optimizer training state")
    if not isinstance(payload["source_baseline_checkpoint_path"], str):
        raise ValueError("Missing original zero-update checkpoint reference")
    if not math.isfinite(payload["current_metrics"]["selection_score"]):
        raise ValueError("Nonfinite checkpoint selection metric")
    if payload["best_state"]["selection_loss"] != payload["best_state"]["metrics"]["selection_score"]:
        raise ValueError("Inconsistent best-state metric")
    origin = payload["initialization"]
    expected_all = origin.get("mode") == "fresh" and not origin.get("pretrained", False)
    if origin.get("mode") not in {"fresh", "transfer"} or origin.get("unfreeze_all_encoder") is not expected_all:
        raise ValueError("Missing or inconsistent saved encoder fine-tuning policy")
    if origin.get("freeze_needle") is not (origin.get("mode") == "transfer"):
        raise ValueError("Inconsistent frozen needle policy")
    if payload["identity"].get("finetune_policy") != FINETUNE_POLICY:
        raise ValueError("Incompatible fine-tuning policy")


def _save_checkpoint(payload, directory, filename=None):
    _validate_payload(payload)
    directory = Path(directory)
    name = filename or f"epoch_{payload['training_state']['epoch']:05d}_{uuid.uuid4().hex[:10]}.pt"
    target = directory / name
    temporary = directory / (".tmp_" + uuid.uuid4().hex[:12] + ".partial")
    try:
        with temporary.open("xb") as handle:
            torch.save(_cpu(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_payload(torch.load(temporary, map_location="cpu", weights_only=True))
        checksum = _sha(temporary)
        os.replace(temporary, target)
        if _sha(target) != checksum:
            raise OSError("Published checkpoint failed its checksum")
        manifest = {"checkpoint": name, "sha256": checksum, "bytes": target.stat().st_size,
                    "epoch": payload["training_state"]["epoch"],
                    "selection_loss": payload["current_metrics"]["selection_score"],
                    "created_at_ns": time.time_ns(), "identity": payload["identity"],
                    "format_version": FORMAT_VERSION, "checkpoint_kind": CHECKPOINT_KIND}
        _json(target.with_suffix(".complete.json"), manifest)
        return {**manifest, "path": str(target)}
    finally:
        temporary.unlink(missing_ok=True)


def discover_checkpoints(search_root, identity=None, *, diagnostics=None):
    """Recover only from verified compatible states, reporting rejected artifacts.

    Corrupt, incompatible and interrupted saves are ineligible. A remaining valid
    state permits recovery with a warning. No valid states means an error, never
    a silent fresh/transfer initialization. diagnostics receives every rejection.
    """
    root = Path(search_root)
    candidates, rejected = [], []
    for sidecar in sorted(root.rglob("epoch_*.complete.json")):
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            name = manifest["checkpoint"]
            if Path(name).name != name or not name.startswith("epoch_") or not name.endswith(".pt"):
                raise ValueError("Unsafe checkpoint filename")
            path = sidecar.parent / name
            if path.with_suffix(".complete.json") != sidecar or _sha(path) != manifest["sha256"] or path.stat().st_size != manifest["bytes"]:
                raise ValueError("Checkpoint checksum/size mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            _validate_payload(payload)
            if payload["identity"] != manifest["identity"]:
                raise ValueError("Checkpoint identity disagrees with its completion marker")
            if identity is not None and payload["identity"] != identity:
                raise ValueError("Checkpoint is incompatible with this dataset, architecture, runtime, or configuration")
            if payload["training_state"]["epoch"] != manifest["epoch"] or payload["current_metrics"]["selection_score"] != manifest["selection_loss"]:
                raise ValueError("Checkpoint selection metadata is inconsistent")
            candidates.append({**manifest, "path": str(path)})
        except Exception as error:
            rejected.append({"path": str(sidecar), "reason": str(error)})
    for path in sorted(root.rglob("epoch_*.pt")):
        if not path.with_suffix(".complete.json").is_file():
            rejected.append({"path": str(path), "reason": "Incomplete save: completion marker is absent"})
    for path in sorted(root.rglob("*.partial")):
        rejected.append({"path": str(path), "reason": "Incomplete temporary save"})
    if diagnostics is not None:
        diagnostics.extend(rejected)
    if not candidates:
        reasons = "; ".join(row["reason"] for row in rejected[:3])
        raise ValueError("No verified complete ring checkpoint exists; refusing a fresh fallback" +
                         (": " + reasons if reasons else ""))
    if rejected:
        warnings.warn(f"Skipped {len(rejected)} invalid/incompatible/incomplete checkpoint artifacts; "
                      f"{len(candidates)} verified compatible states remain. " +
                      "; ".join(row["reason"] for row in rejected[:3]), RuntimeWarning, stacklevel=2)
    return candidates


def _select_checkpoint(search_root, mode, identity):
    rejected = []
    candidates = discover_checkpoints(search_root, identity, diagnostics=rejected)
    if mode == "resume_latest":
        selected = max(candidates, key=lambda row: row["created_at_ns"])
    else:
        selected = min(candidates, key=lambda row: (row["selection_loss"], -row["created_at_ns"]))
    return {**selected, "recovery_diagnostics": rejected, "verified_candidates": candidates}


def _partitions(records):
    splits = {row.get("dataset_split") for row in records}
    if splits != {"train", "validation", "test"}:
        raise ValueError("V2 requires explicit train, validation, and test splits")
    train_records = [row for row in records if row["dataset_split"] == "train"]
    evaluation_records = [row for row in records if row["dataset_split"] == "validation"]
    def groups(rows):
        values = [row.get("source_pose_group", row.get("group_id")) for row in rows]
        if any(value is None for value in values):
            raise ValueError("Training/validation records must declare their pose group")
        return set(values)
    all_groups = {split: groups([row for row in records if row["dataset_split"] == split]) for split in splits}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if all_groups[a] & all_groups[b]:
            raise ValueError("Pose-group leakage between " + a + " and " + b)
    return train_records, evaluation_records, "synthetic_validation"


def _batch_to(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate_loss(model, loader, device, loss_config):
    model.eval()
    total, samples, components = 0.0, 0, {}
    with torch.no_grad():
        for raw in loader:
            batch = _batch_to(raw, device)
            output = model(batch["image"])
            loss, terms = ring_model.training_loss(output, batch, loss_config)
            n = len(batch["image"])
            total += float(loss.detach()) * n
            samples += n
            for key, value in terms.items():
                components[key] = components.get(key, 0.0) + float(value) * n
    if not samples or not math.isfinite(total):
        raise ValueError("Empty or nonfinite loss evaluation")
    return {"combined_loss": total / samples, "samples": samples,
            "components": {key: value / samples for key, value in components.items()}}


def evaluate_validation(model, loader, device, loss_config, geometry_penalty_px=20.0):
    """Use predictions for geometry; labels are used only for scoring."""
    losses = evaluate_loss(model, loader, device, loss_config)
    metrics, rows = ring_eval.evaluate(model, loader, device)
    landmarks = metrics["landmarks"]
    counts = landmarks["visibility_confusion_counts"]
    visible = counts.get("visible_detected", 0) + counts.get("visible_missed", 0)
    hidden = counts.get("occluded_detected", 0) + counts.get("occluded_called_visible", 0)
    visible_recall = counts.get("visible_detected", 0) / visible if visible else None
    hidden_recall = counts.get("occluded_detected", 0) / hidden if hidden else None
    valid_recalls = [x for x in (visible_recall, hidden_recall) if x is not None]
    geometry_success = sum(row["geometry_accepted"] and row["geometry_error_kmh"] <= 2.0 for row in rows) / len(rows)
    tick_mean = landmarks["visible_tick_mean_error_px"]
    if tick_mean is None or not math.isfinite(tick_mean):
        raise ValueError("Validation requires finite visible tick targets")
    # Explicit heuristic in pixel-equivalent units. Rejected geometry cannot
    # improve selection by disappearing from an accepted-only denominator.
    score = tick_mean + geometry_penalty_px * (1.0 - geometry_success)
    return {**losses, "selection_score": float(score),
            "visible_tick_mean_error_px": tick_mean,
            "visible_tick_p95_error_px": landmarks["visible_tick_p95_error_px"],
            "needle_pivot_mean_error_px": landmarks["needle_pivot_mean_error_px"],
            "needle_tip_mean_error_px": landmarks["needle_tip_mean_error_px"],
            "arc_iou": metrics["arc"]["valid_pixel_iou"],
            "visibility_balanced_accuracy": float(np.mean(valid_recalls)),
            "visible_tick_recall": visible_recall, "hidden_tick_recall": hidden_recall,
            "visibility_confusion_counts": counts,
            "direct_mae_kmh": metrics["direct_speed"]["mae_kmh"],
            "geometry_coverage_percent": metrics["geometric_speed"]["coverage_percent"],
            "geometry_within_2_all_inputs_percent": 100.0 * geometry_success,
            "evaluation_semantics": "synthetic_validation", "test_samples_evaluated": 0}


def _export_inference(model_state, best, identity, config, initialization, output):
    path = Path(output) / "best_model.pt"
    payload = {"format_version": FORMAT_VERSION, "checkpoint_kind": "augmented_ring_inference_weights_v2",
               "architecture": ring_model.ARCHITECTURE_ID, "preprocessing": ring_model.PREPROCESSING_ID,
               "model_state_dict": _cpu(model_state), "best_epoch": best["epoch"],
               "metrics": best["metrics"], "identity": identity, "config": asdict(config),
               "initialization": initialization}
    temporary = path.parent / (".tmp_" + uuid.uuid4().hex[:12] + ".partial")
    try:
        torch.save(payload, temporary)
        torch.load(temporary, map_location="cpu", weights_only=True)
        os.replace(temporary, path)
        _json(path.with_suffix(".complete.json"), {"checkpoint": path.name, "sha256": _sha(path), "bytes": path.stat().st_size})
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def load_model(path, device="cpu"):
    path = Path(path)
    marker = json.loads(path.with_suffix(".complete.json").read_text(encoding="utf-8"))
    if marker["checkpoint"] != path.name or marker["sha256"] != _sha(path) or marker["bytes"] != path.stat().st_size:
        raise ValueError("Inference weights failed their integrity check")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported ring model format version")
    if payload.get("checkpoint_kind") == CHECKPOINT_KIND:
        _validate_payload(payload)
        architecture, preprocessing = payload["identity"]["architecture"], payload["identity"]["preprocessing"]
    elif payload.get("checkpoint_kind") == "augmented_ring_inference_weights_v2":
        architecture, preprocessing = payload.get("architecture"), payload.get("preprocessing")
    else:
        raise ValueError("Unsupported ring model file")
    if architecture != ring_model.ARCHITECTURE_ID or preprocessing != ring_model.PREPROCESSING_ID:
        raise ValueError("Incompatible ring inference model")
    model = ring_model.AugmentedRingModel(pretrained=False,
                freeze_needle=payload["initialization"]["freeze_needle"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def train(config, mode="transfer", ring_checkpoint=None, needle_checkpoint=None, search_root=None, stop_after_epochs=None):
    if isinstance(config, dict):
        config = TrainConfig(**config)
    if mode not in {"transfer", "fresh", "resume_latest", "resume_best"}:
        raise ValueError("Unknown initialization/resume mode")
    if config.num_workers != 0:
        raise ValueError("Epoch-exact persistence currently requires num_workers=0")
    if any(type(value) is not int for value in (config.head_epochs, config.finetune_epochs, config.batch_size, config.input_size, config.seed)):
        raise ValueError("Epoch, batch, image, and seed settings must be integers")
    if stop_after_epochs is not None and type(stop_after_epochs) is not int:
        raise ValueError("stop_after_epochs must be an integer or None")
    if config.head_epochs < 0 or config.finetune_epochs < 0 or config.head_epochs + config.finetune_epochs <= 0:
        raise ValueError("A positive fixed epoch budget is required")
    if config.batch_size <= 0 or config.input_size < 8 or stop_after_epochs is not None and stop_after_epochs < 0:
        raise ValueError("Invalid batch, image, or interruption settings")
    if any(not math.isfinite(value) or value <= 0 for value in [config.head_lr, config.finetune_lr, config.new_head_finetune_lr, config.backbone_lr, config.needle_lr, config.grad_clip]) or not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("Invalid optimizer configuration")
    if not math.isfinite(config.geometry_penalty_px) or config.geometry_penalty_px < 0:
        raise ValueError("geometry_penalty_px must be finite and nonnegative")
    if config.train_sample_limit is not None and (type(config.train_sample_limit) is not int or config.train_sample_limit <= 0):
        raise ValueError("train_sample_limit must be a positive integer or None")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device)
    records, integrity = ring_data.verify_dataset(config.data_dir)
    train_records, evaluation_records, semantics = _partitions(records)
    if config.train_sample_limit is not None:
        train_records = train_records[:config.train_sample_limit]
    visible_count = sum(bool(tick["heatmap_loss_valid"]) for row in train_records for tick in row["tick_landmarks"])
    hidden_count = sum(not bool(tick["heatmap_loss_valid"]) for row in train_records for tick in row["tick_landmarks"])
    negative_weight = min(20.0, max(1.0, visible_count / max(hidden_count, 1)))
    loss_config = ring_model.RingLossConfig(**{**config.loss, "visibility_negative_weight": negative_weight})
    settings = asdict(config)
    for key in ("data_dir", "output_dir", "device"):
        settings.pop(key)
    identity = {"architecture": ring_model.ARCHITECTURE_ID, "preprocessing": ring_model.PREPROCESSING_ID,
                "labels_sha256": _sha(Path(config.data_dir) / "labels.json"), "config": settings,
                "loss_config": asdict(loss_config), "torch_version": str(torch.__version__),
                "visibility_weight_basis": {"train_visible": visible_count, "train_hidden": hidden_count,
                                            "rule": "clip(train_visible/max(train_hidden,1),1,20)"},
                "torchvision_version": str(torchvision.__version__), "numpy_version": str(np.__version__),
                "pillow_version": str(PIL.__version__), "opencv_version": str(ring_data.cv2.__version__),
                "finetune_policy": FINETUNE_POLICY,
                "optimizer_policy": "encoder_backbone_lr__shared_decoder_finetune_lr__new_heads_independent_stage_lrs_v2",
                "device": device.type, "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "train_sample_ids": [row["sample_id"] for row in train_records],
                "evaluation_sample_ids": [row["sample_id"] for row in evaluation_records],
                "evaluation_semantics": semantics,
                "implementation_hashes": {"ring_training_v2/train.py": _sha(__file__), **{
                    relative: _sha(PROJECT / relative) for relative in (
                        "ring_training_v2/model.py", "ring_training_v2/data.py", "ring_training/ring_model.py",
                        "training_pilot/train_core.py", "ring_training/ring_eval.py", "ring_geometry_pilot/geometry_reader.py")}}}
    candidate = None
    if mode.startswith("resume_"):
        if search_root is None:
            raise ValueError("Exact resume requires an explicit checkpoint search_root")
        candidate = _select_checkpoint(search_root, mode, identity)
    if mode == "transfer" and any(path is None or not Path(path).is_file() for path in (ring_checkpoint, needle_checkpoint)):
        raise ValueError("Transfer requires both ring and needle checkpoints")
    _seed(config.seed)
    model = ring_model.AugmentedRingModel(pretrained=config.pretrained if mode == "fresh" else False).to(device)
    initialization = {"mode": "fresh", "source_epoch": None, "optimizer_restored": False,
                      "pretrained": bool(config.pretrained), "unfreeze_all_encoder": not config.pretrained,
                      "freeze_needle": False}
    if mode == "transfer":
        migration = ring_model.load_transfer_checkpoints(model, ring_checkpoint, needle_checkpoint)
        initialization = {**migration, "migration_kind": migration.get("migration_kind", migration.get("mode", "two_source_architecture_migration")),
                          "mode": "transfer", "pretrained": False, "unfreeze_all_encoder": False,
                          "freeze_needle": True, "source_epoch": migration.get("source_ring_epoch")}
    grouped = {name: [] for name in ("backbone", "shared", "new", "needle")}
    for name, parameter in model.named_parameters():
        group = ring_model.parameter_group(name)
        if group is not None:
            grouped[group].append(parameter)
    rates = {"backbone": config.backbone_lr, "shared": config.finetune_lr,
             "new": config.head_lr, "needle": config.needle_lr}
    optimizer = torch.optim.AdamW([{"params": parameters, "name": name, "lr": rates[name]}
                                   for name, parameters in grouped.items()], weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp and device.type == "cuda"))
    generators = {"train": torch.Generator().manual_seed(config.seed + 1),
                  "evaluation": torch.Generator().manual_seed(config.seed + 2)}
    train_loader = DataLoader(ring_data.RingDataset(config.data_dir, train_records, input_size=config.input_size),
                             batch_size=config.batch_size, shuffle=True, generator=generators["train"], num_workers=0)
    evaluation_loader = DataLoader(ring_data.RingDataset(config.data_dir, evaluation_records, input_size=config.input_size),
                                  batch_size=config.batch_size, shuffle=False, generator=generators["evaluation"], num_workers=0)
    target = config.head_epochs + config.finetune_epochs
    state = {"epoch": 0, "target_epoch": target, "stage": "new_heads" if config.head_epochs else "finetune", "training_complete": False}
    history, best, before = [], None, None
    current_metrics = None
    baseline_checkpoint_path = None
    resume_metadata = {"requested_mode": mode, "full_state_restored": False, "optimizer_restored": False,
                       "source_checkpoint_path": None, "source_epoch": None, "recovery_diagnostics": []}
    if candidate:
        payload = torch.load(candidate["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        state, history, best, before = payload["training_state"], payload["history"], payload["best_state"], payload["before_metrics"]
        initialization, current_metrics = payload["initialization"], payload["current_metrics"]
        model.freeze_needle = initialization["freeze_needle"]
        baseline_checkpoint_path = payload["source_baseline_checkpoint_path"]
        for key, generator in generators.items():
            generator.set_state(payload["loader_generator_states"][key])
        _restore_rng(payload["rng_state"])
        resume_metadata.update(full_state_restored=True, optimizer_restored=True,
                               source_checkpoint_path=candidate["path"], source_epoch=state["epoch"],
                               recovery_diagnostics=candidate["recovery_diagnostics"])
    else:
        model.configure_stage(state["stage"], unfreeze_all_encoder=initialization["unfreeze_all_encoder"])
        before = evaluate_validation(model, evaluation_loader, device, loss_config, config.geometry_penalty_px)
        current_metrics = before
        best = {"epoch": 0, "selection_loss": before["selection_score"], "metrics": before,
                "model_state_dict": _cpu(model.state_dict())}
        resume_metadata.update(source_checkpoint_path=str(ring_checkpoint) if mode == "transfer" else None,
                               source_epoch=initialization.get("source_epoch"))
    output = Path(config.output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "config.json", asdict(config))
    _json(output / "data_integrity.json", integrity)
    latest = {key: value for key, value in candidate.items() if key != "verified_candidates"} if candidate else None
    best_checkpoint_path = None

    def save(filename=None):
        payload = {"format_version": FORMAT_VERSION, "checkpoint_kind": CHECKPOINT_KIND,
                   "identity": identity, "config": asdict(config), "model_state_dict": _cpu(model.state_dict()),
                   "optimizer_state_dict": _cpu(optimizer.state_dict()), "scheduler_state_dict": None,
                   "scaler_state_dict": scaler.state_dict(), "rng_state": _rng_state(),
                   "loader_generator_states": {key: generator.get_state() for key, generator in generators.items()},
                   "training_state": dict(state), "history": list(history), "best_state": best,
                   "before_metrics": before, "current_metrics": current_metrics, "initialization": initialization,
                   "source_baseline_checkpoint_path": baseline_checkpoint_path}
        saved = _save_checkpoint(payload, output, filename)
        _json(output / "latest.json", saved)
        return saved

    if not candidate:
        initial_name = "epoch_00000_" + uuid.uuid4().hex[:10] + ".pt"
        baseline_checkpoint_path = str(output / initial_name)
        latest = save(initial_name)
        best_checkpoint_path = latest["path"]
    else:
        # Find the complete epoch corresponding to the restored best weights.
        matching = [row for row in candidate["verified_candidates"]
                    if row["epoch"] == best["epoch"] and row["selection_loss"] == best["selection_loss"]]
        best_checkpoint_path = max(matching, key=lambda row: row["created_at_ns"])["path"] if matching else None
    epochs_this_call = 0
    while state["epoch"] < state["target_epoch"] and (stop_after_epochs is None or epochs_this_call < stop_after_epochs):
        epoch = state["epoch"] + 1
        stage = "new_heads" if epoch <= config.head_epochs else "finetune"
        model.configure_stage(stage, unfreeze_all_encoder=initialization["unfreeze_all_encoder"])
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = {"backbone": config.backbone_lr, "shared": config.finetune_lr,
                           "new": config.head_lr if stage == "new_heads" else config.new_head_finetune_lr,
                           "needle": config.needle_lr}[group["name"]]
        total, count, terms_total = 0.0, 0, {}
        for raw in train_loader:
            batch = _batch_to(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(config.amp and device.type == "cuda")):
                loss, terms = ring_model.training_loss(model(batch["image"]), batch, loss_config)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; incomplete epoch was not published")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            n = len(batch["image"])
            total += float(loss.detach()) * n
            count += n
            for key, value in terms.items():
                terms_total[key] = terms_total.get(key, 0.0) + float(value) * n
        current_metrics = evaluate_validation(model, evaluation_loader, device, loss_config, config.geometry_penalty_px)
        history.append({"epoch": epoch, "stage": stage, "train_combined_loss": total / count,
                        "train_components": {key: value / count for key, value in terms_total.items()},
                        "selection_score": current_metrics["selection_score"], "validation_metrics": current_metrics})
        if current_metrics["selection_score"] < best["selection_loss"]:
            best = {"epoch": epoch, "selection_loss": current_metrics["selection_score"],
                    "metrics": current_metrics, "model_state_dict": _cpu(model.state_dict())}
        state.update(epoch=epoch, stage=stage, training_complete=epoch == state["target_epoch"])
        latest = save()
        if best["epoch"] == epoch:
            best_checkpoint_path = latest["path"]
        epochs_this_call += 1
        print(f"Epoch {epoch}/{target} {stage}: train={total/count:.5f}, val tick={current_metrics['visible_tick_mean_error_px']:.2f}px "
              f"p95={current_metrics['visible_tick_p95_error_px']:.2f}px, direct={current_metrics['direct_mae_kmh']:.2f}km/h, "
              f"arcIoU={current_metrics['arc_iou']:.3f}, visibility_balanced={current_metrics['visibility_balanced_accuracy']:.3f}, "
              f"geometry_coverage={current_metrics['geometry_coverage_percent']:.1f}%, score={current_metrics['selection_score']:.3f}", flush=True)
    _json(output / "latest.json", latest)
    _json(output / "best.json", {"checkpoint_path": best_checkpoint_path, "epoch": best["epoch"], "selection_loss": best["selection_loss"]})
    inference_path = _export_inference(best["model_state_dict"], best, identity, config, initialization, output)
    _json(output / "history.json", history)
    report = {"run_dir": str(output), "mode": mode, "training_samples": len(train_records),
              "evaluation_samples": len(evaluation_records), "evaluation_semantics": semantics,
              "model_selection": "minimum visible tick mean error px + geometry_penalty_px * (1 - geometry within2km/h fraction over ALL validation inputs); includes epoch-zero baseline",
              "before_metrics": before, "after_metrics": best["metrics"], "latest_metrics": current_metrics,
              "best_epoch": best["epoch"], "completed_epoch": state["epoch"], "target_epoch": target,
              "epochs_this_call": epochs_this_call, "training_complete": state["training_complete"],
              "latest_checkpoint_path": latest["path"], "best_checkpoint_path": best_checkpoint_path,
              "initial_checkpoint_path": baseline_checkpoint_path,
              "source_baseline_checkpoint_path": baseline_checkpoint_path,
              "best_model_path": inference_path, "history_path": str(output / "history.json"),
              "initialization": initialization, "resume_metadata": resume_metadata,
              "learning_rate_policy": "Encoder uses backbone_lr; legacy decoder and heads use finetune_lr; new heads use head_lr then new_head_finetune_lr. Frozen parameters do not update. Encoder batch normalization stays frozen, including random fresh runs.",
              "identity": identity, "config": asdict(config),
              "test_samples_evaluated": 0,
              "scope": "Synthetic validation for model selection; test split untouched; not real-camera accuracy"}
    _json(output / "training_report.json", report)
    return report
