"""Build the self-contained, Drive-persistent 20-image ring learning notebook."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import textwrap


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
NOTEBOOK = HERE / "Speedometer_Arc_Needle_Colab.ipynb"
SOURCE_FILES = (
    "ring_training/ring_model.py", "ring_training/ring_data.py",
    "ring_training/ring_train.py", "ring_training/ring_eval.py",
    "training_pilot/train_core.py", "ring_geometry_pilot/geometry_reader.py",
)


def build():
    cells = []

    def cell(kind, value, *, hidden=False):
        source = textwrap.dedent(value).strip() + "\n"
        metadata = {"cellView": "form"} if hidden else {}
        result = {"cell_type": kind, "metadata": metadata, "source": source.splitlines(keepends=True),
                  "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12]}
        if kind == "code":
            compile(source, f"notebook_cell_{len(cells)}", "exec")
            result.update(execution_count=None, outputs=[])
        cells.append(result)

    def md(value):
        cell("markdown", value)

    def code(value, **kwargs):
        cell("code", value, **kwargs)

    source_bytes = {name: (PROJECT / name).read_bytes() for name in SOURCE_FILES}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()}
    embedded = base64.b64encode(gzip.compress(json.dumps({
        name: base64.b64encode(value).decode("ascii") for name, value in source_bytes.items()
    }, sort_keys=True).encode("utf-8"), mtime=0)).decode("ascii")

    md("""
    # Triumph dashboard — arc and needle learning check

    This notebook adds six named tick landmarks, a visible scale-arc mask, and tick visibility to the existing needle/speed model. It starts from the verified rotation model's best epoch **19**, initializes only the new heads, and keeps the old speed and needle branches.

    **Scope: the same 20 approved review images are used to learn and inspect labels. These are correlated synthetic examples, not a held-out test set. No 98% accuracy or real-camera claim follows from this run.** This notebook does not expand the dataset or launch a 500-image experiment.

    1. Select **Runtime → Change runtime type → T4 GPU**, then **Run all**.
    2. Authorize Google Drive when asked. The public review dataset downloads automatically.
    3. Leave `resume_latest` selected. On the first-ever run of a new experiment it imports the epoch-19 weights; subsequent runs restore complete training state.

    The new experiment has a fixed **33-epoch budget: 3 epochs training new heads, then 30 fine-tuning the decoder and final encoder blocks**. The short local run checked software and checkpoint recovery; it did not establish successful landmark learning. This longer, bounded run tests whether the new labels can be learned before deciding whether to scale up. Every completed epoch, including epoch zero, is saved to Drive. Best selection uses combined reconstruction loss on these same review images. Final reports compare the untouched migrated model with the selected best model, including geometric rejection rate.

    [Open this notebook in Colab](https://colab.research.google.com/github/sainadh812/triumph-speedometer-dataset/blob/main/ring_training/Speedometer_Arc_Needle_Colab.ipynb)
    """)
    md("""
    ## 1. Runtime and checkpoint mode

    - `resume_latest`: restore the most recently saved complete epoch and continue to the saved target.
    - `resume_best`: restore the full state at the lowest reconstruction loss and continue toward the same target.
    - `transfer`: explicitly initialize a new, unused experiment from the verified epoch-19 model.
    - `fresh`: create a uniquely named experiment with random weights and no previous learned weights.

    Fresh random runs fine-tune the entire encoder in stage 2; transferred or pretrained runs fine-tune only its final four blocks. Encoder BatchNorm running statistics stay fixed for these small batches.

    During fine-tuning, the new heads use learning rate `1e-3`, the shared decoder and existing heads use `1e-4`, and the trainable encoder blocks use `2e-5`. These settings are saved with the experiment.

    Existing experiments restore their saved configuration. Changing the epoch settings below does not extend a resumed experiment. Corrupt candidates are reported and excluded; recovery requires a verified compatible checkpoint. If none remains, the run stops without a fresh fallback. Exact resume also requires matching source code, dependencies and device type/model. A mid-epoch interruption resumes from the last completed epoch.
    """)
    code('''
    #@title Training mode and new-experiment settings
    TRAIN_MODE = "resume_latest" #@param ["resume_latest", "resume_best", "transfer", "fresh"]
    EXPERIMENT_NAME = "speedometer_arc_v1" #@param {type:"string"}
    BATCH_SIZE = 8 #@param {type:"integer"}
    HEAD_EPOCHS = 3 #@param {type:"integer"}
    FINETUNE_EPOCHS = 30 #@param {type:"integer"}

    import os, sys, json, hashlib, importlib.util, shutil, uuid, re, stat, zipfile
    import base64, gzip, urllib.request
    from pathlib import Path, PurePosixPath, PureWindowsPath
    from datetime import datetime, timezone
    from dataclasses import asdict
    import numpy as np
    import torch, torchvision, PIL
    from torch.utils.data import DataLoader
    from IPython.display import display, Image as DisplayImage

    LOCAL_SMOKE = os.environ.get("RING_LOCAL_SMOKE", "0") == "1"
    IN_COLAB = bool(importlib.util.find_spec("google") and importlib.util.find_spec("google.colab"))
    if not IN_COLAB and not LOCAL_SMOKE:
        raise RuntimeError("Open in Colab, or explicitly enable the local verification harness.")
    TRAIN_MODE = os.environ.get("RING_TRAIN_MODE", TRAIN_MODE)
    EXPERIMENT_NAME = os.environ.get("RING_EXPERIMENT", EXPERIMENT_NAME)
    if TRAIN_MODE not in {"transfer", "fresh", "resume_latest", "resume_best"}:
        raise ValueError("Unknown training mode")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", EXPERIMENT_NAME):
        raise ValueError("Use a simple experiment name without path separators")
    WORK_DIR = Path(os.environ.get("RING_WORK_DIR", "/content/speedometer_arc_work" if IN_COLAB else str(Path.cwd() / "ring_notebook_work"))).resolve()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if LOCAL_SMOKE:
        torch.set_num_threads(4)
    print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU (slower)")
    print("PyTorch:", torch.__version__, "| Torchvision:", torchvision.__version__)
    print("20-image learning check only; no held-out accuracy claim.")
    ''')
    md("""
    ## 2. Mount Drive

    Checkpoints and reports stay under `MyDrive/SpeedometerTraining/experiments/`. The original rotation checkpoint is read from Drive and is not uploaded to GitHub. Fresh mode needs no old checkpoint. Drive authorization is handled by Colab.
    """)
    code('''
    if LOCAL_SMOKE:
        DRIVE_ROOT = Path(os.environ["RING_DRIVE_ROOT"]).resolve()
    else:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive")
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Google Drive is not mounted; training has not started")
        DRIVE_ROOT = Path("/content/drive/MyDrive/SpeedometerTraining")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    probe = DRIVE_ROOT / (".write_probe_" + uuid.uuid4().hex)
    with probe.open("xb") as handle:
        handle.write(b"drive writable")
        handle.flush()
        os.fsync(handle.fileno())
    probe.unlink()
    SOURCE_CHECKPOINT = Path(os.environ.get("RING_SOURCE_CHECKPOINT", str(
        DRIVE_ROOT / "experiments/speedometer_rotation_v2_from47/runs/20261017_100006_266720_f43a79f6/best_model.pt")))
    SOURCE_SHA256 = "ba47ae2a3c07fd9a490342eebb8a01ac160dfe5e9dbca24dd13014ed68cac270"
    print("Persistent root:", DRIVE_ROOT)
    ''')
    md("""
    ## 3. Restore the notebook's verified code

    The notebook contains its exact Python sources and verifies their SHA256 checksums before importing them. Code inside the downloaded dataset ZIP is not executed. Every experiment also retains a copy of these sources for reproducibility.
    """)
    code('''
    #@title Restore embedded implementation
    EMBEDDED_SOURCE_B64 = "__EMBEDDED__"
    SOURCE_HASHES = __HASHES__
    source_payload = json.loads(gzip.decompress(base64.b64decode(EMBEDDED_SOURCE_B64)).decode("utf-8"))
    if set(source_payload) != set(SOURCE_HASHES):
        raise ValueError("Embedded source inventory mismatch")
    CODE_ROOT = WORK_DIR / ("code_" + hashlib.sha256(json.dumps(SOURCE_HASHES, sort_keys=True).encode()).hexdigest()[:12])
    for relative, encoded in source_payload.items():
        contents = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(contents).hexdigest() != SOURCE_HASHES[relative]:
            raise ValueError("Embedded source checksum mismatch: " + relative)
        target = CODE_ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != contents:
            raise ValueError("Local implementation differs; restart with a clean work directory")
        if not target.exists():
            target.write_bytes(contents)
    for folder in (CODE_ROOT, CODE_ROOT / "ring_training", CODE_ROOT / "ring_geometry_pilot"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    import ring_data, ring_model, ring_train, ring_eval
    from ring_train import RingTrainConfig, train, load_model
    for module in (ring_data, ring_model, ring_train, ring_eval, ring_model.legacy,
                   sys.modules["geometry_reader"], sys.modules[ring_data.preprocess_image.__module__]):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(CODE_ROOT.resolve()):
            raise RuntimeError("A different implementation is already imported; restart this runtime")
    print("Verified", len(SOURCE_HASHES), "embedded source files")
    '''.replace("__EMBEDDED__", embedded).replace("__HASHES__", repr(hashes)), hidden=True)
    md("""
    ## 4. Download and verify the 20 review images

    Images and all eight masks per image are checked. Labels retain their `review_only` split. The loss uses the visible arc with a separate valid-pixel mask; uncertain source reconstruction, current needle occlusion, and border fill do not become arc supervision.
    """)
    code('''
    DATASET_URL = "https://raw.githubusercontent.com/sainadh812/triumph-speedometer-dataset/main/ring_training/arc_needle_review20.zip"
    ZIP_SHA256 = "651c9052d59f3652bcdff10b04d559f0858e9cfbb067c1eccb372f202b896dee"
    LABELS_SHA256 = "a7fb97a37edbb654ff46653d7cc4090b5dbc4e5f7e2e90eb51cc4729bc40c3d5"
    ZIP_PATH = Path(os.environ.get("RING_DATASET_ZIP", str(WORK_DIR / "arc_needle_review20.zip")))

    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if not ZIP_PATH.exists():
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial = ZIP_PATH.with_name(".tmp_" + uuid.uuid4().hex[:12] + ".partial")
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Speedometer-Arc-Colab"})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(partial) != ZIP_SHA256:
            raise ValueError("Downloaded archive checksum mismatch; training has not started")
        os.replace(partial, ZIP_PATH)
    if sha256(ZIP_PATH) != ZIP_SHA256:
        raise ValueError("Review archive checksum mismatch; refusing extraction")

    def safe_extract_review(archive_path, destination):
        destination = Path(destination).resolve()
        if destination.exists():
            raise ValueError("Extraction destination must be new")
        plans, seen = [], set()
        reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
        with zipfile.ZipFile(archive_path) as archive:
            items = archive.infolist()
            if len(items) > 1000 or sum(item.file_size for item in items) > 100 * 1024 * 1024:
                raise ValueError("Unexpected archive size")
            for item in items:
                name = item.orig_filename
                parts = PurePosixPath(name).parts
                win = PureWindowsPath(name)
                invalid = (not parts or name.startswith("/") or win.drive or "\\\\" in name
                           or any(part in {".", ".."} or part.endswith((".", " "))
                                  or any(character in part for character in '<>:"|?*\\x00')
                                  or part.split(".")[0].rstrip(" .").upper() in reserved for part in parts)
                           or name.casefold() in seen or item.flag_bits & 1
                           or item.file_size > 20 * 1024 * 1024)
                mode = item.external_attr >> 16
                if invalid or stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Unsafe archive entry: " + name)
                seen.add(name.casefold())
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Archive path escaped its destination")
                plans.append((item, target))
            destination.mkdir(parents=True, exist_ok=False)
            for item, target in plans:
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("xb") as handle:
                        shutil.copyfileobj(source, handle)

    # Extract into a new directory each invocation; an interrupted old extraction
    # is never mistaken for a complete dataset.
    EXTRACT_ROOT = WORK_DIR / ("review_bundle_" + uuid.uuid4().hex[:12])
    safe_extract_review(ZIP_PATH, EXTRACT_ROOT)
    DATA_DIR = EXTRACT_ROOT / "review20"
    if sha256(DATA_DIR / "labels.json") != LABELS_SHA256:
        raise ValueError("Review labels checksum mismatch")
    RECORDS, DATA_INTEGRITY = ring_data.verify_dataset(DATA_DIR, allow_review=True)
    if len(RECORDS) != 20 or not DATA_INTEGRITY["review_only"]:
        raise ValueError("This notebook is restricted to the 20 review images")
    print(json.dumps(DATA_INTEGRITY, indent=2))
    ''')
    md("""
    ## 5. Select or create the persistent experiment

    `fresh` automatically creates a unique experiment name and uses random weights. Save the printed name to resume that experiment later. Transfer requires the verified epoch-19 checkpoint. The initial migrated model is saved at epoch zero, before any optimizer update.
    """)
    code('''
    def atomic_json(path, value):
        path = Path(path)
        temporary = path.with_name(".tmp_" + uuid.uuid4().hex[:12] + ".partial")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    RUNTIME = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
               "numpy": str(np.__version__), "pillow": str(PIL.__version__),
               "device": DEVICE, "device_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"}
    REQUESTED_MODE = TRAIN_MODE
    if TRAIN_MODE == "fresh":
        EXPERIMENT_NAME = EXPERIMENT_NAME[:70] + "_fresh_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    EXPERIMENT_DIR = DRIVE_ROOT / "experiments" / EXPERIMENT_NAME
    MANIFEST_PATH = EXPERIMENT_DIR / "experiment_config.json"
    RUNS_DIR = EXPERIMENT_DIR / "runs"
    existed = EXPERIMENT_DIR.exists()
    if existed:
        if TRAIN_MODE in {"fresh", "transfer"}:
            raise ValueError("Initialization requires a new experiment; existing files are preserved")
        if not MANIFEST_PATH.is_file():
            raise ValueError("Existing experiment has no complete manifest; refusing initialization fallback")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = {"manifest_version": 1, "experiment_name": EXPERIMENT_NAME,
                    "dataset_labels_sha256": LABELS_SHA256, "dataset_zip_sha256": ZIP_SHA256,
                    "source_hashes": SOURCE_HASHES, "runtime": RUNTIME}
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError("Saved experiment differs in " + key + "; exact resume is unavailable")
        initial_mode = manifest.get("initial_mode")
        if initial_mode not in {"transfer", "fresh"}:
            raise ValueError("Saved experiment has invalid initialization provenance")
        if (manifest.get("source_checkpoint_sha256") != (SOURCE_SHA256 if initial_mode == "transfer" else None)
                or manifest.get("source_epoch") != (19 if initial_mode == "transfer" else None)):
            raise ValueError("Saved experiment source provenance differs")
        # Corrupt/incomplete states never trigger transfer. Recovery requires a
        # verified compatible candidate; the trainer reports rejected candidates.
        ring_train.discover_checkpoints(RUNS_DIR)
        settings = dict(manifest["config"])
        settings.update(data_dir=str(DATA_DIR), output_dir=str(RUNS_DIR), device=DEVICE)
        CONFIG = RingTrainConfig(**settings)
        if not CONFIG.review_smoke or CONFIG.input_size != 512 or CONFIG.pretrained:
            raise ValueError("Saved configuration is outside this notebook's review scope")
        EFFECTIVE_MODE = TRAIN_MODE
    else:
        if TRAIN_MODE == "resume_best":
            raise ValueError("A new experiment has no best checkpoint to resume")
        if (type(BATCH_SIZE) is not int or BATCH_SIZE <= 0
                or type(HEAD_EPOCHS) is not int or HEAD_EPOCHS < 0
                or type(FINETUNE_EPOCHS) is not int or FINETUNE_EPOCHS < 0
                or HEAD_EPOCHS + FINETUNE_EPOCHS <= 0):
            raise ValueError("Use a positive integer batch size and nonnegative integer stages with a positive total")
        EFFECTIVE_MODE = "transfer" if TRAIN_MODE == "resume_latest" else TRAIN_MODE
        if EFFECTIVE_MODE == "transfer":
            if not SOURCE_CHECKPOINT.is_file() or sha256(SOURCE_CHECKPOINT) != SOURCE_SHA256:
                raise ValueError("Verified rotation epoch-19 weights are missing or changed in Drive")
        CONFIG = RingTrainConfig(data_dir=str(DATA_DIR), output_dir=str(RUNS_DIR), input_size=512,
                                 batch_size=BATCH_SIZE, head_epochs=HEAD_EPOCHS,
                                 finetune_epochs=FINETUNE_EPOCHS, review_smoke=True,
                                 device=DEVICE, num_workers=0, pretrained=False)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
        RUNS_DIR.mkdir()
        manifest = {"manifest_version": 1, "experiment_name": EXPERIMENT_NAME,
                    "dataset_labels_sha256": LABELS_SHA256, "dataset_zip_sha256": ZIP_SHA256,
                    "source_hashes": SOURCE_HASHES, "runtime": RUNTIME, "config": asdict(CONFIG),
                    "initial_mode": EFFECTIVE_MODE,
                    "source_checkpoint_sha256": SOURCE_SHA256 if EFFECTIVE_MODE == "transfer" else None,
                    "source_epoch": 19 if EFFECTIVE_MODE == "transfer" else None,
                    "scope": "20-image training reconstruction check; no held-out accuracy claim"}
        atomic_json(MANIFEST_PATH, manifest)
    for relative, expected_hash in SOURCE_HASHES.items():
        saved = EXPERIMENT_DIR / "source" / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.exists() and sha256(saved) != expected_hash:
            raise ValueError("Saved experiment code checksum mismatch: " + relative)
        if not saved.exists():
            shutil.copy2(CODE_ROOT / relative, saved)
        if sha256(saved) != expected_hash:
            raise OSError("Drive source copy verification failed")
    print("Experiment:", EXPERIMENT_NAME)
    print("Requested / effective mode:", REQUESTED_MODE, "/", EFFECTIVE_MODE)
    print("Persistent folder:", EXPERIMENT_DIR)
    print("Fixed target:", CONFIG.head_epochs + CONFIG.finetune_epochs, "epochs; review images:", len(RECORDS))
    ''')
    md("""
    ## 6. Train and retain every completed checkpoint

    Full checkpoints include model, optimizer, mixed-precision scaler, random generators, data-loader state, history and the original epoch-zero baseline reference. Partial saves cannot be selected. Re-running after completion evaluates the saved best model without adding epochs.
    """)
    code('''
    stop_after = int(os.environ["RING_STOP_AFTER_EPOCHS"]) if LOCAL_SMOKE and "RING_STOP_AFTER_EPOCHS" in os.environ else None
    REPORT = train(CONFIG, mode=EFFECTIVE_MODE,
                   source_checkpoint=str(SOURCE_CHECKPOINT) if EFFECTIVE_MODE == "transfer" else None,
                   search_root=str(RUNS_DIR) if EFFECTIVE_MODE.startswith("resume_") else None,
                   stop_after_epochs=stop_after)
    SESSION_DIR = Path(REPORT["run_dir"])
    atomic_json(SESSION_DIR / "experiment_config_snapshot.json", manifest)
    atomic_json(SESSION_DIR / "drive_paths.json", {
        "experiment_dir": str(EXPERIMENT_DIR), "run_dir": str(SESSION_DIR),
        "latest_checkpoint_path": REPORT["latest_checkpoint_path"],
        "best_checkpoint_path": REPORT["best_checkpoint_path"],
        "source_baseline_checkpoint_path": REPORT["source_baseline_checkpoint_path"],
        "best_model_path": REPORT["best_model_path"], "source_hashes": SOURCE_HASHES,
        "dataset_labels_sha256": LABELS_SHA256, "dataset_zip_sha256": ZIP_SHA256,
        "archive_excludes": "All epoch_*.pt full checkpoints and their completion markers remain on Drive"})
    print(json.dumps({key: REPORT[key] for key in (
        "mode", "best_epoch", "completed_epoch", "target_epoch", "epochs_this_call", "training_complete",
        "evaluation_semantics", "resume_metadata")}, indent=2))
    ''')
    md("""
    ## 7. Inspect before and after

    Geometric speed uses predicted landmarks and predicted visibility only. Ground-truth tick coordinates and camera transforms are excluded from inference. Report both coverage and error: rejected readings count against the all-input success rate. Visibility scores are not calibrated confidence scores.
    """)
    code('''
    DATASET = ring_data.RingDataset(DATA_DIR)
    LOADER = DataLoader(DATASET, batch_size=CONFIG.batch_size, shuffle=False, num_workers=0)
    before_model, before_payload = load_model(REPORT["source_baseline_checkpoint_path"], device=DEVICE)
    BEFORE, before_rows = ring_eval.evaluate(before_model, LOADER, DEVICE)
    ring_eval.save_evaluation(SESSION_DIR, "before", BEFORE, before_rows)
    del before_model, before_payload
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    best_model, best_payload = load_model(REPORT["best_model_path"], device=DEVICE)
    AFTER, after_rows = ring_eval.evaluate(best_model, LOADER, DEVICE)
    ring_eval.save_evaluation(SESSION_DIR, "after", AFTER, after_rows)
    ring_eval.comparison_plot(BEFORE, AFTER, SESSION_DIR / "learning_check.png")
    ring_eval.prediction_sheet(best_model, DATASET, DEVICE, SESSION_DIR / "model_predictions.png")
    for name, metrics in (("Before", BEFORE), ("After selected best", AFTER)):
        print(name, json.dumps({"direct_speed": metrics["direct_speed"],
                               "geometry": metrics["geometric_speed"],
                               "landmarks": metrics["landmarks"], "arc": metrics["arc"]}, indent=2))
    display(DisplayImage(filename=str(SESSION_DIR / "learning_check.png")))
    display(DisplayImage(filename=str(SESSION_DIR / "model_predictions.png")))
    REPORT.update(detailed_before=BEFORE, detailed_after=AFTER)
    atomic_json(SESSION_DIR / "review_smoke_report.json", REPORT)
    atomic_json(EXPERIMENT_DIR / "latest_review_result.json", REPORT)
    ''')
    md("""
    ## 8. Save and download the compact results

    The ZIP contains metrics, predictions, plots, the selected best weights and reproducible source code. Full optimizer checkpoints remain in Drive and are deliberately excluded from this compact download. The ZIP contains private Drive paths and model weights; keep it private unless you choose to share it.
    """)
    code('''
    run_source = SESSION_DIR / "source"
    for relative in SOURCE_HASHES:
        destination = run_source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE_ROOT / relative, destination)
        if sha256(destination) != SOURCE_HASHES[relative]:
            raise OSError("Results source verification failed")
    atomic_json(SESSION_DIR / "source_hashes.json", SOURCE_HASHES)
    for name in ("source_annotations_snapshot.json", "calibration_snapshot.json"):
        shutil.copy2(DATA_DIR / name, SESSION_DIR / name)
    (SESSION_DIR / "RESULTS_README.txt").write_text(
        "20-image reconstruction check: the same correlated review examples were used for training and evaluation.\\n"
        "This is not held-out accuracy or real-camera testing. Best selection uses combined reconstruction loss.\\n"
        "The source baseline is epoch zero of the expanded architecture, before any optimizer update.\\n"
        "All full optimizer checkpoints remain on Google Drive; they are not in this compact ZIP.\\n"
        "This archive includes private paths and selected model weights.\\n", encoding="utf-8")
    exports = EXPERIMENT_DIR / "exports"
    exports.mkdir(exist_ok=True)
    archive_path = exports / (SESSION_DIR.name + "_results.zip")
    partial = archive_path.with_name(".tmp_" + uuid.uuid4().hex[:12] + ".partial")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(SESSION_DIR.rglob("*")):
            if path.is_file() and not path.name.startswith(("epoch_", ".tmp_")) and not path.name.endswith(".partial"):
                archive.write(path, path.relative_to(SESSION_DIR).as_posix())
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise OSError("Results ZIP failed its integrity check")
    os.replace(partial, archive_path)
    atomic_json(archive_path.with_suffix(".sha256.json"), {"file": archive_path.name, "sha256": sha256(archive_path),
                                                        "bytes": archive_path.stat().st_size})
    print("Results saved:", archive_path)
    print("All full checkpoints retained:", RUNS_DIR)
    print("To resume this experiment later, use EXPERIMENT_NAME =", repr(EXPERIMENT_NAME))
    if IN_COLAB and not LOCAL_SMOKE:
        from google.colab import files
        local_download = WORK_DIR / archive_path.name
        shutil.copy2(archive_path, local_download)
        files.download(str(local_download))
    ''')
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU",
                     "colab": {"name": NOTEBOOK.name, "provenance": []}},
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(NOTEBOOK), "code_cells": sum(c["cell_type"] == "code" for c in cells),
                      "source_hashes": hashes}, indent=2))
    return notebook


if __name__ == "__main__":
    build()
