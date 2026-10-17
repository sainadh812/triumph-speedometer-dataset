"""Drive-backed experiment selection for the self-contained Colab notebook."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
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
                           core_sha256=None):
    """Resolve a new immutable run branch and the exact train_persistent call."""
    if requested_mode not in {"fresh", "resume_latest", "resume_best"}:
        raise ValueError("Choose fresh, resume_latest, or resume_best")
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
        if use_legacy and (not import_pilot_if_missing or run_mode != "pilot"):
            raise FileNotFoundError("No saved experiment exists. Choose fresh explicitly with a new "
                                    "experiment name, or enable the verified legacy import for the pilot.")
        if use_legacy:
            legacy = Path(warm_start_path or (root / "pilot_best_epoch29.pt"))
            if not legacy.is_file() or file_sha256(legacy) != LEGACY_SHA256:
                raise ValueError("The approved epoch-29 checkpoint is missing or has the wrong SHA256: "
                                 + str(legacy) + ". No fresh model was initialized.")
        configuration = asdict(training.TrainConfig(
            data_dir=str(data_dir), output_dir="resolved_per_run",
            input_size=128 if run_mode == "smoke" else 512,
            batch_size=2 if run_mode == "smoke" else 8,
            head_epochs=1 if run_mode == "smoke" else (0 if use_legacy else 5),
            finetune_epochs=1 if run_mode == "smoke" else (20 if use_legacy else 25),
            patience=7, seed=42,
            pretrained=False if run_mode == "smoke" or use_legacy else bool(fresh_pretrained),
            max_train_samples=8 if run_mode == "smoke" else 400,
            max_val_samples=4 if run_mode == "smoke" else None,
            device="cpu" if run_mode == "smoke" else "auto",
        ))
        saved = {"manifest_version": 1, "experiment_name": experiment_name,
                 "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_mode": run_mode,
                 "config": configuration, "runtime_versions": runtime_versions(),
                 "initialization_kind": "legacy_warm_start" if use_legacy else "fresh",
                 "legacy_checkpoint": str(legacy) if use_legacy else None,
                 "legacy_sha256": LEGACY_SHA256 if use_legacy else None,
                 "additional_epochs": 20 if use_legacy else 0,
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
              "additional_epochs": 20, "extend_by_epochs": int(extend_by_epochs)}
    if requested_mode.startswith("resume_") and not candidates:
        if saved["initialization_kind"] != "legacy_warm_start" or not import_pilot_if_missing:
            raise FileNotFoundError("This experiment has no completed full-state checkpoint. "
                                    "Resume cannot silently restart it. Use a new name for a fresh experiment.")
        legacy = Path(saved["legacy_checkpoint"])
        if not legacy.is_file() or file_sha256(legacy) != saved["legacy_sha256"]:
            raise ValueError("The saved experiment's approved legacy checkpoint is missing or changed")
        if extend_by_epochs:
            raise ValueError("Do not extend the budget on the initial legacy import")
        actual_mode = "warm_start"
        kwargs.update(mode="warm_start", warm_start_path=str(legacy), additional_epochs=saved["additional_epochs"])
    if not manifest_path.exists():
        saved["identity"] = identity
        _save_manifest(manifest_path, saved)
    # train_persistent creates this unique branch only after input validation.
    print("Experiment:", experiment)
    print("Requested mode:", requested_mode, "| effective mode:", actual_mode)
    if actual_mode == "warm_start":
        print("Verified legacy epoch29 import: optimizer/RNG state is unavailable and is reset once.")
        print("The fixed continuation target is epoch49, or earlier if early stopping applies.")
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
