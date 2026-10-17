"""Verified notebook setup helpers; no model import, training, or Drive crawling."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import urllib.request
import uuid
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value, length=64, label="SHA256"):
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{label} must be an exact published {length}-character hexadecimal value")
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.parent / (".tmp_" + uuid.uuid4().hex[:12] + ".partial")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download_verified(url, destination, expected_sha256):
    require_digest(expected_sha256)
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Existing download has an unexpected checksum: {destination.name}")
        return destination
    if not url.startswith("https://"):
        raise ValueError("Dataset download requires HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".download_" + uuid.uuid4().hex[:12] + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "SpeedometerNotebook/3"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if sha256(temporary) != expected_sha256:
            raise ValueError("Downloaded archive failed its approved checksum")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def safe_extract(archive, destination, expected_sha256, expected_root):
    """Extract a pinned data ZIP, validating paths, CRC and cached file contents."""
    require_digest(expected_sha256)
    archive, destination = Path(archive), Path(destination).resolve()
    if sha256(archive) != expected_sha256:
        raise ValueError("Archive checksum mismatch before extraction")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", expected_root) or expected_root in {".", ".."}:
        raise ValueError("Expected root must be one plain directory name")
    marker = destination / ".verified_archive.json"
    if destination.exists():
        if not marker.is_file():
            raise ValueError("Existing extraction is incomplete; use a new work directory")
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("archive_sha256") != expected_sha256 or saved.get("root") != expected_root:
            raise ValueError("Existing extraction belongs to a different archive")
        for name, expected in saved["files"].items():
            candidate = (destination / name).resolve()
            if not candidate.is_relative_to(destination) or not candidate.is_file() or sha256(candidate) != expected:
                raise ValueError(f"Extracted data changed or is missing: {name}")
        return destination / expected_root
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".extract_" + uuid.uuid4().hex[:12])
    if not temporary.resolve().is_relative_to(destination.parent):
        raise ValueError("Invalid extraction staging path")
    inventory = {}
    with zipfile.ZipFile(archive) as bundle:
        names = set()
        members = []
        total_bytes = 0
        for item in bundle.infolist():
            # ZipInfo normalizes backslashes on Windows; reject the stored
            # archive name before any host-dependent normalization.
            name = item.orig_filename
            posix, windows = PurePosixPath(name), PureWindowsPath(name)
            mode = (item.external_attr >> 16) & 0xFFFF
            if (not name or "\\" in name or posix.is_absolute() or windows.drive
                    or any(part in {".", ".."} or ":" in part or part.endswith((" ", ".")) for part in posix.parts)
                    or not posix.parts or posix.parts[0] != expected_root
                    or stat.S_ISLNK(mode) or item.flag_bits & 1):
                raise ValueError(f"Unsafe or unexpected ZIP member: {name}")
            if any(re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part) for part in posix.parts):
                raise ValueError(f"Reserved ZIP filename: {name}")
            normalized = str(posix).casefold()
            if normalized in names:
                raise ValueError(f"Duplicate ZIP member: {name}")
            names.add(normalized)
            total_bytes += item.file_size
            if total_bytes > 4 * 1024 ** 3:
                raise ValueError("Archive exceeds the bounded data extraction size")
            members.append(item)
        if bundle.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        temporary.mkdir()
        for item in members:
            target = (temporary / item.filename).resolve()
            if not target.is_relative_to(temporary.resolve()):
                raise ValueError("ZIP member escaped staging directory")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("xb") as sink:
                    while block := source.read(1024 * 1024):
                        sink.write(block)
                inventory[item.filename] = sha256(target)
    atomic_json(temporary / ".verified_archive.json", {"archive_sha256": expected_sha256, "root": expected_root, "files": inventory})
    # Both paths have been resolved and checked within the intended work parent.
    if destination.exists():
        raise FileExistsError("Extraction destination appeared while preparing data")
    temporary.rename(destination)
    return destination / expected_root


def find_source_checkpoint(drive_root, expected_sha256, explicit_path=""):
    """Search only the known V2 experiment's best-model files, never all Drive."""
    require_digest(expected_sha256)
    root = Path(drive_root)
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_file() or sha256(candidate) != expected_sha256:
            raise ValueError("Explicit source checkpoint is missing or has the wrong checksum")
        return candidate
    known = root / "experiments/speedometer_arc_v2_augmented/runs"
    exact = known / "20261017_120108_959025_fee960/best_model.pt"
    candidates = [exact] + sorted(known.glob("*/best_model.pt"))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and sha256(candidate) == expected_sha256:
            return candidate
    raise FileNotFoundError("Approved epoch34 source not found in the known V2 experiment. Set SOURCE_CHECKPOINT explicitly or upload that approved file.")


def prepare_experiment(drive_root, experiment_name, mode, config, *, code_commit,
                       public_archive_sha256, private_archive_sha256, runtime):
    """Create once, restore saved settings on resume, or give fresh a new folder."""
    require_digest(code_commit, 40, "Pinned code commit")
    require_digest(public_archive_sha256)
    require_digest(private_archive_sha256)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,70}", experiment_name):
        raise ValueError("Invalid experiment name")
    if mode not in {"transfer", "fresh", "resume_latest", "resume_best"}:
        raise ValueError("Unknown restart mode")
    if type(config.get("epochs")) is not int or not 1 <= config["epochs"] <= 500:
        raise ValueError("Epoch target must be an integer from1 to500")
    if type(config.get("batch_size")) is not int or not 1 <= config["batch_size"] <= 64:
        raise ValueError("Batch size must be an integer from1 to64")
    root = Path(drive_root) / "experiments"
    if mode == "fresh":
        experiment_name += "_fresh_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    experiment = root / experiment_name
    manifest_path = experiment / "experiment_config.json"
    identity = {"code_commit": code_commit, "public_archive_sha256": public_archive_sha256,
                "private_archive_sha256": private_archive_sha256, "runtime": runtime}
    if mode.startswith("resume_"):
        if not manifest_path.is_file():
            raise FileNotFoundError("No saved experiment manifest; resume never silently starts fresh")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity:
            raise ValueError("Saved code/data/runtime differs. Use its original notebook/runtime or a new experiment.")
        restored = dict(saved["config"])
        for field in ("data_dir", "real_data_dir", "device"):
            if field in config:
                restored[field] = config[field]
        restored["output_dir"] = str(experiment / "runs")
        return experiment, restored, saved
    if experiment.exists():
        raise FileExistsError("Experiment already exists. Choose resume_latest/resume_best, or fresh for a new experiment.")
    experiment.mkdir(parents=True)
    config = dict(config, output_dir=str(experiment / "runs"))
    manifest = {"format_version": 1, "initialization_mode": mode, "config": config,
                "identity": identity, "evaluation_scope": "Synthetic held-out source plus same-recording real diagnostics; no independent real test"}
    atomic_json(manifest_path, manifest)
    return experiment, config, manifest
