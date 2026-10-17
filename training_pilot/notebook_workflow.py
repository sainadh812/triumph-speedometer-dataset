"""Drive-backed experiment selection for the self-contained Colab notebook."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import csv
import hashlib
import json
import os
import re
import uuid


LEGACY_SHA256 = "a2ed22a3feb4d4d5f8552bc31af6c5051b66503840ab8a22d3f361a5e6a1dd36"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions():
    return {name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}


def ensure_writable(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / (".write_probe_" + uuid.uuid4().hex)
    try:
        with probe.open("xb") as handle:
            handle.write(b"speedometer drive write verification")
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_bytes() != b"speedometer drive write verification":
            raise IOError("Drive write/read verification failed")
    finally:
        probe.unlink(missing_ok=True)


def _save_manifest(path, document):
    staged = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with staged.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(staged.read_text(encoding="utf-8")) != document:
        raise IOError("Experiment manifest verification failed")
    os.replace(staged, path)


def epoch_artifacts(experiment_dir):
    return sorted(path for path in Path(experiment_dir).rglob("epoch_*")
                  if path.is_file() and (path.suffix == ".pt"
                                        or path.name.endswith(".complete.json")
                                        or ".partial" in path.name))


def require_clean_checkpoint_inventory(training, experiment_dir, identity):
    """Use verified recovery states and preserve incomplete files for inspection."""
    artifacts = epoch_artifacts(experiment_dir)
    candidates = training.discover_checkpoints(experiment_dir, expected_identity=identity)
    verified = {Path(item[key]).resolve() for item in candidates
                for key in ("checkpoint_path", "manifest_path")}
    invalid = [path for path in artifacts if path.resolve() not in verified]
    if invalid and not candidates:
        names = "\n".join(str(path) for path in invalid[:6])
        raise RuntimeError("Incomplete, corrupt, or incompatible epoch files exist. "
                           "Training stopped; no fresh fallback or legacy reimport was attempted. "
                           "Preserve the files and inspect/recover them before resuming:\n" + names)
    if invalid:
        print(f"Warning: preserving and skipping {len(invalid)} incomplete/corrupt/incompatible epoch files.")
        for path in invalid[:6]:
            print("  Ignored:", path)
        print("Resuming from a checksum-verified completed epoch instead.")
    if artifacts and not candidates:
        raise RuntimeError("Checkpoint artifacts exist but no verified full-state checkpoint is usable")
    return candidates


def prepare_persistent_run(training, *, data_dir, drive_root, experiment_name,
                           requested_mode="resume_latest", run_mode="pilot",
                           fresh_pretrained=True, import_pilot_if_missing=True,
                           warm_start_path=None, extend_by_epochs=0,
                           core_sha256=None, first_import_mode="warm_start",
                           approved_source_sha256=None, source_epoch=29,
                           additional_epochs=20, train_sample_limit=400,
                           smoke_train_samples=8, smoke_val_samples=4,
                           dataset_name=None, allow_smoke_transfer=False):
    """Resolve a new immutable run branch and the exact train_persistent call."""
    approved_source_sha256 = LEGACY_SHA256 if approved_source_sha256 is None else approved_source_sha256
    if requested_mode not in {"fresh", "resume_latest", "resume_best"}:
        raise ValueError("Choose fresh, resume_latest, or resume_best")
    if first_import_mode not in {"warm_start", "warm_start_transfer"}:
        raise ValueError("The first import must be warm_start or warm_start_transfer")
    if additional_epochs < 1:
        raise ValueError("A warm start requires a positive fixed epoch budget")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", experiment_name):
        raise ValueError("Use a short experiment name containing letters, digits, _, - or .")
    if run_mode not in {"pilot", "smoke"} or extend_by_epochs < 0:
        raise ValueError("Invalid run mode or extension budget")
    if requested_mode == "fresh" and extend_by_epochs:
        raise ValueError("extend_by_epochs is only valid for explicit resumed-budget extensions")
    root = Path(drive_root).resolve()
    ensure_writable(root)
    experiment = root / "experiments" / experiment_name
    experiment.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment / "experiment_config.json"
    artifacts = epoch_artifacts(experiment)
    saved = None
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("manifest_version") != 1 or saved.get("experiment_name") != experiment_name:
            raise ValueError("Experiment configuration manifest is invalid")
    elif artifacts:
        raise RuntimeError("Epoch saves exist without their experiment configuration. "
                           "Restore the configuration manifest; do not create a new one over these saves.")
    if requested_mode == "fresh" and (saved is not None or artifacts):
        raise FileExistsError("Fresh training needs a new experiment name. Earlier runs were preserved. "
                              "For example, use speedometer_fresh_v2; later choose resume_latest for that name.")
    if saved is not None:
        if saved["run_mode"] != run_mode:
            raise ValueError("Smoke and pilot experiments must use different names")
        if saved["runtime_versions"] != runtime_versions():
            raise RuntimeError("Dependency versions differ from the saved experiment. Exact continuation "
                               "requires its torch, torchvision, numpy and Pillow versions. "
                               f"Saved: {saved['runtime_versions']}; current: {runtime_versions()}. "
                               "No new model was initialized.")
        configuration = dict(saved["config"])
    else:
        use_legacy = requested_mode != "fresh"
        smoke_transfer = (run_mode == "smoke" and allow_smoke_transfer
                          and first_import_mode == "warm_start_transfer" and warm_start_path is not None)
        if use_legacy and (not import_pilot_if_missing or (run_mode != "pilot" and not smoke_transfer)):
            raise FileNotFoundError("No saved experiment exists. Choose fresh explicitly with a new "
                                    "experiment name, or enable the verified legacy import for the pilot.")
        if use_legacy:
            legacy = Path(warm_start_path or (root / "pilot_best_epoch29.pt"))
            if not legacy.is_file() or file_sha256(legacy) != approved_source_sha256:
                raise ValueError(f"The approved epoch-{source_epoch} checkpoint is missing or has the wrong SHA256: "
                                 + str(legacy) + ". No fresh model was initialized.")
        configuration = asdict(training.TrainConfig(
            data_dir=str(data_dir), output_dir="resolved_per_run",
            input_size=128 if run_mode == "smoke" else 512,
            batch_size=2 if run_mode == "smoke" else 8,
            head_epochs=(0 if use_legacy else 1) if run_mode == "smoke" else (0 if use_legacy else 5),
            finetune_epochs=(additional_epochs if use_legacy else 1) if run_mode == "smoke" else (additional_epochs if use_legacy else 25),
            patience=7, seed=42,
            pretrained=False if run_mode == "smoke" or use_legacy else bool(fresh_pretrained),
            max_train_samples=smoke_train_samples if run_mode == "smoke" else train_sample_limit,
            max_val_samples=smoke_val_samples if run_mode == "smoke" else None,
            device="cpu" if run_mode == "smoke" else "auto",
        ))
        saved = {"manifest_version": 1, "experiment_name": experiment_name,
                 "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_mode": run_mode,
                 "config": configuration, "runtime_versions": runtime_versions(),
                 "initialization_kind": ("transfer_warm_start" if first_import_mode == "warm_start_transfer"
                                          else "legacy_warm_start") if use_legacy else "fresh",
                 "legacy_checkpoint": str(legacy) if use_legacy else None,
                 "legacy_sha256": approved_source_sha256 if use_legacy else None,
                 "first_import_mode": first_import_mode if use_legacy else None,
                 "source_epoch": int(source_epoch) if use_legacy else None,
                 "additional_epochs": int(additional_epochs) if use_legacy else 0,
                 "dataset_name": dataset_name,
                 "training_core_sha256_at_creation": core_sha256}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = experiment / "runs" / (stamp + "_" + uuid.uuid4().hex[:8])
    configuration.update(data_dir=str(data_dir), output_dir=str(run_dir))
    config = training.TrainConfig(**configuration)
    identity = training.checkpoint_identity(config)
    if saved.get("identity") is not None and saved["identity"] != identity:
        raise RuntimeError("The saved experiment identity does not match this dataset, runtime, "
                           "or training configuration. No weights were reset.")
    candidates = require_clean_checkpoint_inventory(training, experiment, identity)
    actual_mode = requested_mode
    kwargs = {"mode": requested_mode, "search_root": str(experiment),
              "additional_epochs": int(additional_epochs), "extend_by_epochs": int(extend_by_epochs)}
    if requested_mode.startswith("resume_") and not candidates:
        if saved["initialization_kind"] not in {"legacy_warm_start", "transfer_warm_start"} or not import_pilot_if_missing:
            raise FileNotFoundError("This experiment has no completed full-state checkpoint. "
                                    "Resume cannot silently restart it. Use a new name for a fresh experiment.")
        legacy = Path(saved["legacy_checkpoint"])
        if not legacy.is_file() or file_sha256(legacy) != saved["legacy_sha256"]:
            raise ValueError("The saved experiment's approved legacy checkpoint is missing or changed")
        if extend_by_epochs:
            raise ValueError("Do not extend the budget on the initial legacy import")
        actual_mode = saved.get("first_import_mode", "warm_start")
        kwargs.update(mode=actual_mode, warm_start_path=str(legacy), additional_epochs=saved["additional_epochs"])
    if not manifest_path.exists():
        saved["identity"] = identity
        _save_manifest(manifest_path, saved)
    # train_persistent creates this unique branch only after input validation.
    print("Experiment:", experiment)
    print("Requested mode:", requested_mode, "| effective mode:", actual_mode)
    if actual_mode == "warm_start":
        start_epoch = saved.get("source_epoch", 29)
        print(f"Verified legacy epoch {start_epoch} import: optimizer/RNG state is reset once.")
        print(f"The fixed target is epoch {start_epoch + saved['additional_epochs']}, subject to early stopping.")
    elif actual_mode == "warm_start_transfer":
        print(f"Verified source epoch {saved['source_epoch']} imported into the new dataset experiment.")
        print("Optimizer/RNG state resets once; source weights are evaluated before the first update.")
        print(f"The NEW experiment runs epoch 0 -> {saved['additional_epochs']}, subject to early stopping.")
    elif actual_mode.startswith("resume_"):
        selected = (candidates[-1] if actual_mode == "resume_latest" else
                    min(candidates, key=lambda item: (item["validation_mae_kmh"], -item["created_at_ns"])))
        print("Restoring complete state from:", selected["checkpoint_path"])
        print("The saved target is retained; explicit extension:", extend_by_epochs)
    else:
        print("Fresh initialization:", "ImageNet" if config.pretrained else "random")
    return {"config": config, "kwargs": kwargs, "experiment_dir": str(experiment),
            "run_dir": str(run_dir), "manifest_path": str(manifest_path),
            "requested_mode": requested_mode, "actual_mode": actual_mode}


def read_prediction_rows(path):
    """Read exported evaluation rows with numeric fields restored."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {"target_speed_kmh", "predicted_speed_kmh", "error_kmh", "absolute_error_kmh",
               "pivot_error_px", "tip_error_px", "rotation_degrees"}
    for row in rows:
        for key in numeric.intersection(row):
            row[key] = float(row[key])
        row.setdefault("rotation_degrees", 0.0)
    return rows


def rotation_breakdown(training, rows):
    """Metrics grouped by known image rotation, with no effect on selection."""
    result = {"all": training.metrics_from_rows(rows)}
    for label, predicate in (("original_zero", lambda angle: abs(angle) < 1e-8),
                             ("rotated", lambda angle: abs(angle) >= 1e-8)):
        members = [row for row in rows if predicate(float(row.get("rotation_degrees", 0)))]
        if members:
            result[label] = training.metrics_from_rows(members)
    for angle in sorted({float(row.get("rotation_degrees", 0)) for row in rows}):
        members = [row for row in rows if float(row.get("rotation_degrees", 0)) == angle]
        result[f"angle_{angle:g}"] = training.metrics_from_rows(members)
    return result


def evaluate_rotation_checkpoint(training, *, checkpoint_path, data_dir, output_dir,
                                 label_filename="labels.json", expected_labels_sha256=None,
                                 batch_size=8, max_samples=None, device="auto"):
    """Evaluate a checkpoint on a validation-only manifest, including robustness.

    This path never calls an optimizer, early stopping, or model selection.
    Image and label hashes are rechecked; cached results are bound to exact
    checkpoint/label/sample identities rather than reused by filename alone.
    """
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    root = Path(data_dir).resolve()
    labels = root / label_filename
    digest = file_sha256(labels)
    if expected_labels_sha256 and digest != expected_labels_sha256:
        raise ValueError("Evaluation label checksum mismatch")
    document = json.loads(labels.read_text(encoding="utf-8"))
    records = document["samples"]
    if any(row["dataset_split"] != "validation" for row in records):
        # A main-dataset manifest contains both splits; evaluate validation only.
        records = [row for row in records if row["dataset_split"] == "validation"]
    if not records:
        raise ValueError("No validation records in evaluation manifest")
    records = training.select_group_subset(records, max_samples, 42)
    provenance = {"checkpoint_sha256": file_sha256(checkpoint_path), "labels_sha256": digest,
                  "sample_ids": [row["sample_id"] for row in records],
                  "runtime_versions": runtime_versions()}
    out = Path(output_dir)
    report_path, rows_path = out / "evaluation.json", out / "predictions.csv"
    reusable_cache = None
    if report_path.exists() and rows_path.exists():
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        if cached.get("provenance") == provenance and cached.get("predictions_sha256") == file_sha256(rows_path):
            reusable_cache = cached
        else:
            raise ValueError("Existing evaluation cache has different provenance; use a new output directory")
    seen = set()
    for row in records:
        path = (root / row["image"]).resolve()
        if not path.is_relative_to(root) or row["sample_id"] in seen:
            raise ValueError("Invalid evaluation image path or duplicate sample ID")
        seen.add(row["sample_id"])
        if file_sha256(path) != row["image_sha256"]:
            raise ValueError(f"Evaluation image hash mismatch: {row['sample_id']}")
        with Image.open(path) as image:
            if image.mode != "L" or image.size != (row["width"], row["height"]):
                raise ValueError("Evaluation image dimensions or grayscale mode disagree with labels")
        for name in ("pivot", "tip"):
            xy = np.asarray(row["needle_position_px"][name], dtype=float)
            norm = np.asarray(row["needle_position_normalized"][name], dtype=float)
            limits = np.array([row["width"]-1, row["height"]-1])
            if (xy.shape != (2,) or not np.isfinite(xy).all() or not np.all((xy >= 0) & (xy <= limits))
                    or not np.allclose(xy/limits, norm, atol=1e-6)):
                raise ValueError("Invalid evaluation keypoint coordinates")
        if not np.isfinite(float(row.get("rotation_degrees", 0))):
            raise ValueError("Invalid evaluation rotation angle")
    if reusable_cache is not None:
        return reusable_cache
    model, checkpoint = training.load_model(checkpoint_path, device=device)
    dataset = training.DashboardDataset(root, records, checkpoint["config"]["input_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    metrics, rows = training.evaluate_model(model, loader, next(model.parameters()).device)
    out.mkdir(parents=True, exist_ok=True)
    training._save_csv(rows_path, rows)
    report = {"provenance": provenance, "checkpoint_path": str(checkpoint_path),
              "metrics": metrics, "breakdown": rotation_breakdown(training, rows),
              "predictions_path": str(rows_path), "predictions_sha256": file_sha256(rows_path),
              "scope": "Synthetic evaluation only; not independent real-camera testing"}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def save_rotation_comparison(training, *, before_predictions, after_predictions, output_path,
                             evaluation_name):
    before = read_prediction_rows(before_predictions)
    after = read_prediction_rows(after_predictions)
    if {row["sample_id"] for row in before} != {row["sample_id"] for row in after}:
        raise ValueError("Before/after evaluations must contain the same samples")
    baseline, selected = rotation_breakdown(training, before), rotation_breakdown(training, after)
    rows = []
    for group in baseline:
        b, a = baseline[group], selected[group]
        rows.append({"evaluation": evaluation_name, "group": group, "images": a["n"],
                     "pose_groups": a["pose_groups"], "before_mae_kmh": b["mae_kmh"],
                     "after_mae_kmh": a["mae_kmh"], "mae_change_kmh": a["mae_kmh"]-b["mae_kmh"],
                     "before_within_2_percent": b["within_2_kmh_percent"],
                     "after_within_2_percent": a["within_2_kmh_percent"],
                     "within_2_change_percentage_points": a["within_2_kmh_percent"]-b["within_2_kmh_percent"],
                     "before_tip_error_px": b["tip_mae_px"], "after_tip_error_px": a["tip_mae_px"]})
    training._save_csv(output_path, rows)
    return rows
