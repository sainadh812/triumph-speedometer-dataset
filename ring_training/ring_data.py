"""Verified inputs for the ring geometry pilot; never invent dataset splits.

The current review20 manifest is deliberately marked ``review_only``. Loading it
is useful for an explicitly authorized smoke check, but is not permission to
train or report validation accuracy. Trainers must gate this using ``integrity``
and call ``verify_dataset(..., allow_review=False)`` for ordinary experiments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Reuse the actual legacy implementation, including bilinear interpolation and
# ImageNet normalization. This also works when ring_training is added to sys.path.
_LEGACY_DIR = Path(__file__).resolve().parents[1] / "training_pilot"
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))
from train_core import preprocess_image  # noqa: E402


INPUT_SIZE = 512
TICK_NAMES = ("tick_0", "tick_40", "tick_80", "tick_120", "tick_160", "tick_180")
MASK_NAMES = (
    "needle_shaft", "arc_amodal", "arc_visible", "arc_occluded_by_needle",
    "source60_uncertainty", "arc_ignore", "arc_loss_valid", "image_footprint",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "Empty dataset path")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    _require(not posix.is_absolute() and not windows.is_absolute()
             and not windows.drive and ".." not in posix.parts
             and "\\" not in relative and ":" not in relative,
             f"Unsafe dataset path: {relative}")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root) and path.is_file(),
             f"Missing file or path outside data_dir: {relative}")
    return path


def _verified_file(root, relative, expected, used_paths, context):
    path = _data_path(root, relative)
    _require(path not in used_paths, f"Reused image/mask path: {context}")
    used_paths.add(path)
    _require(isinstance(expected, str) and len(expected) == 64,
             f"Invalid SHA256: {context}")
    digest = _sha256(path)
    _require(digest == expected, f"Checksum mismatch: {context}")
    return path, digest


def _point(xy, normalized, context):
    point = np.asarray(xy, dtype=np.float64)
    norm = np.asarray(normalized, dtype=np.float64)
    _require(point.shape == norm.shape == (2,), f"Invalid point shape: {context}")
    _require(np.isfinite(point).all() and np.isfinite(norm).all(),
             f"Nonfinite point: {context}")
    _require(np.all((point >= 0) & (point <= INPUT_SIZE - 1)),
             f"Out-of-frame point: {context}")
    _require(np.all((norm >= 0) & (norm <= 1))
             and np.allclose(point / (INPUT_SIZE - 1), norm, atol=1e-6, rtol=0),
             f"Normalized point mismatch: {context}")
    return point


def _dilate_one_pixel(mask):
    """Boolean 3x3 dilation, equivalent to the generator's one-pixel dilation."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)
    for y in range(3):
        for x in range(3):
            dilated |= padded[y:y + height, x:x + width]
    return dilated


def _verify_masks(root, row, used_paths):
    sid = row["sample_id"]
    _require(set(row["masks"]) == set(MASK_NAMES)
             and set(row["mask_sha256"]) == set(MASK_NAMES),
             f"Expected all eight mask paths and hashes: {sid}")
    masks = {}
    for name in MASK_NAMES:
        path, _ = _verified_file(root, row["masks"][name], row["mask_sha256"][name],
                                 used_paths, f"{sid}/{name}")
        with Image.open(path) as image:
            _require(image.mode == "L" and image.size == (INPUT_SIZE, INPUT_SIZE),
                     f"Expected grayscale 512x512 mask: {sid}/{name}")
            pixels = np.asarray(image).copy()
        _require(np.isin(pixels, (0, 255)).all(), f"Nonbinary mask: {sid}/{name}")
        masks[name] = pixels > 0

    footprint = masks["image_footprint"]
    for name, mask in masks.items():
        _require(not np.any(mask & ~footprint), f"Mask outside image footprint: {sid}/{name}")
    for name in ("needle_shaft", "arc_amodal", "arc_visible", "source60_uncertainty", "arc_loss_valid"):
        _require(masks[name].any(), f"Empty required mask: {sid}/{name}")

    dilated = _dilate_one_pixel(masks["needle_shaft"])
    occluded = masks["arc_amodal"] & dilated
    ignore = masks["source60_uncertainty"] | occluded
    expected = {
        "arc_occluded_by_needle": occluded,
        "arc_ignore": ignore,
        "arc_visible": masks["arc_amodal"] & ~ignore,
        "arc_loss_valid": footprint & ~ignore,
    }
    for name, wanted in expected.items():
        _require(np.array_equal(masks[name], wanted), f"Invalid mask semantics: {sid}/{name}")
    _require(not np.any(masks["arc_visible"] & ~masks["arc_loss_valid"]),
             f"Visible arc has invalid supervision: {sid}")
    return masks, dilated


def verify_dataset(data_dir, allow_review=True):
    """Return unchanged manifest records and an integrity report.

    All images and all eight masks are hashed and inspected. Review data retain
    their real split and are rejected when ``allow_review=False``. No source
    artifact outside this dataset is opened; declared source hashes are provenance,
    not an assertion that the physical dial or reconstructed source was verified.
    """
    _require(isinstance(allow_review, bool), "allow_review must be a boolean")
    root = Path(data_dir).resolve()
    labels_path = root / "labels.json"
    document = json.loads(labels_path.read_text(encoding="utf-8"))
    _require(isinstance(document, dict), "Expected an object in labels.json")
    records = document.get("samples")
    _require(isinstance(records, list) and len(records) > 0, "Expected nonempty samples")
    _require(type(document.get("count")) is int and len(records) == document["count"],
             "labels.json count mismatch")
    _require(document.get("resolution") == [INPUT_SIZE, INPUT_SIZE],
             "Ring supervision requires 512x512 source images and masks")
    schema = document.get("schema_version")
    _require(isinstance(schema, str) and bool(schema), "Missing schema_version")
    _require(all(isinstance(row, dict) for row in records), "Invalid sample record")
    counts = Counter(row.get("dataset_split") for row in records)
    _require(set(counts) <= {"review_only", "train", "validation", "test"}, "Unknown dataset split")
    review_only = "review_only" in counts
    _require(not review_only or set(counts) == {"review_only"},
             "Review examples cannot be mixed with train/validation/test splits")
    _require(not schema.startswith("arc_needle_review") or review_only,
             "A review-only schema cannot be relabeled as a training dataset")
    _require(not review_only or allow_review, "Review-only data are disabled for this operation")
    if "split_counts" in document:
        _require(document["split_counts"] == dict(counts), "Split counts disagree with manifest")

    ids, used_paths, image_hashes = set(), set(), set()
    groups, group_speeds = defaultdict(set), defaultdict(set)
    tick_visibility = Counter()
    for row in records:
        sid = row.get("sample_id")
        _require(isinstance(sid, str) and bool(sid) and sid not in ids, "Duplicate or invalid sample ID")
        ids.add(sid)
        _require(row.get("width") == INPUT_SIZE and row.get("height") == INPUT_SIZE
                 and row.get("channels") == 1, f"Unexpected dimensions/channels: {sid}")
        image_path, digest = _verified_file(root, row["image"], row["image_sha256"], used_paths, sid)
        _require(digest not in image_hashes, f"Duplicate image content: {sid}")
        image_hashes.add(digest)
        with Image.open(image_path) as image:
            _require(image.mode == "L" and image.size == (INPUT_SIZE, INPUT_SIZE),
                     f"Expected grayscale 512x512 image: {sid}")
            image.load()
        speed = float(row["speed_kmh"])
        _require(math.isfinite(speed) and 0 <= speed <= 180, f"Invalid speed: {sid}")
        group = row.get("source_pose_group")
        _require(isinstance(group, str) and bool(group), f"Missing source pose group: {sid}")
        groups[group].add(row["dataset_split"])
        group_speeds[group].add(speed)
        _require(set(row["needle_position_px"]) == {"pivot", "tip", "tick_anchor"}
                 and set(row["needle_position_normalized"]) == {"pivot", "tip", "tick_anchor"},
                 f"Invalid needle keypoint inventory: {sid}")
        for name in ("pivot", "tip", "tick_anchor"):
            _point(row["needle_position_px"][name], row["needle_position_normalized"][name], f"{sid}/{name}")
        _require(np.linalg.norm(np.asarray(row["needle_position_px"]["tip"])
                                - np.asarray(row["needle_position_px"]["pivot"])) > 1,
                 f"Degenerate needle: {sid}")
        arc_points = np.asarray(row["arc_centerline_px"], dtype=np.float64)
        _require(arc_points.ndim == 2 and arc_points.shape[1] == 2 and len(arc_points) >= 3
                 and np.isfinite(arc_points).all()
                 and np.all((arc_points >= 0) & (arc_points <= INPUT_SIZE - 1)),
                 f"Invalid or out-of-frame arc centerline: {sid}")
        masks, dilated = _verify_masks(root, row, used_paths)
        landmarks = row["tick_landmarks"]
        _require(isinstance(landmarks, list) and len(landmarks) == len(TICK_NAMES)
                 and {tick["name"] for tick in landmarks} == set(TICK_NAMES),
                 f"Expected six distinct named tick landmarks: {sid}")
        visible_count = 0
        for tick in landmarks:
            name = tick["name"]
            point = _point(tick["xy"], tick["xy_normalized"], f"{sid}/{name}")
            _require(tick["speed_kmh"] == int(name.split("_")[1]), f"Tick speed/name mismatch: {sid}/{name}")
            _require(type(tick["visible"]) is bool and type(tick["heatmap_loss_valid"]) is bool,
                     f"Tick visibility must be boolean: {sid}/{name}")
            x, y = np.rint(point).astype(int)
            visible = bool(masks["image_footprint"][y, x]
                           and not dilated[y, x] and not masks["source60_uncertainty"][y, x])
            _require(tick["visible"] == tick["heatmap_loss_valid"] == visible,
                     f"Tick visibility disagrees with occlusion masks: {sid}/{name}")
            visible_count += int(visible)
            tick_visibility[name] += int(visible)
        _require(type(row["visible_tick_count"]) is int and row["visible_tick_count"] == visible_count,
                 f"Visible tick count mismatch: {sid}")

    leaking = [group for group, splits in groups.items() if len(splits) != 1]
    _require(not leaking, f"Source pose groups cross splits: {leaking[:5]}")
    _require(all(len(speeds) == 1 for speeds in group_speeds.values()),
             "A source pose group contains inconsistent speeds")
    report = {
        "schema_version": schema,
        "count": len(records),
        "split_counts": dict(counts),
        "review_only": review_only,
        "pose_groups": len(groups),
        "pose_group_leakage": False,
        "all_image_hashes_verified": True,
        "all_mask_hashes_verified": True,
        "mask_files_verified": len(records) * len(MASK_NAMES),
        "binary_mask_semantics_verified": True,
        "tick_visibility_verified": True,
        "visible_tick_counts": dict(tick_visibility),
        "tick_order": list(TICK_NAMES),
        "labels_sha256": _sha256(labels_path),
        "source_master_sha256_declared": document.get("source_master_sha256"),
        "source_annotations_sha256_declared": document.get("source_annotations_sha256"),
        "external_source_artifacts_verified": False,
        "scope": ("Correlated synthetic review variants; no training/validation/test allocation"
                  if review_only else "Synthetic geometry labels; no independent real-dashboard accuracy claim"),
    }
    return records, report


class RingDataset(Dataset):
    """512px legacy image input and aligned needle, named tick and visible arc targets.

    Supplied ``records`` must be unchanged entries from this dataset's manifest.
    Every construction verifies the full manifest before selecting those entries.
    Only visible arc and loss-valid masks enter training; amodal/ignore masks are
    used for integrity checks and never silently substituted as positive targets.
    """

    def __init__(self, data_dir, records=None, input_size=512):
        _require(type(input_size) is int and input_size == INPUT_SIZE,
                 "RingDataset currently requires input_size=512 for thin arc supervision")
        self.data_dir = Path(data_dir).resolve()
        self.root = self.data_dir
        self.input_size = input_size
        manifest_records, self.integrity = verify_dataset(self.data_dir, allow_review=True)
        if records is None:
            self.records = manifest_records
        else:
            manifest_by_id = {row["sample_id"]: row for row in manifest_records}
            selected, seen = [], set()
            for row in records:
                _require(isinstance(row, dict), "Supplied records must be manifest objects")
                sid = row.get("sample_id")
                _require(isinstance(sid, str) and sid in manifest_by_id and sid not in seen,
                         "Unknown or duplicate supplied record")
                _require(row == manifest_by_id[sid], f"Supplied record differs from verified manifest: {sid}")
                seen.add(sid)
                selected.append(manifest_by_id[sid])
            self.records = selected

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        ticks = {tick["name"]: tick for tick in row["tick_landmarks"]}
        with Image.open(_data_path(self.data_dir, row["masks"]["arc_visible"])) as mask:
            arc_mask = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0).unsqueeze(0)
        with Image.open(_data_path(self.data_dir, row["masks"]["arc_loss_valid"])) as mask:
            arc_valid = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy() / 255.0).unsqueeze(0)
        return {
            "image": preprocess_image(_data_path(self.data_dir, row["image"]), self.input_size),
            "speed": torch.tensor(float(row["speed_kmh"]), dtype=torch.float32),
            "points": torch.tensor([row["needle_position_normalized"][name] for name in ("pivot", "tip")],
                                   dtype=torch.float32),
            "tick_points": torch.tensor([ticks[name]["xy_normalized"] for name in TICK_NAMES], dtype=torch.float32),
            "tick_valid": torch.tensor([float(ticks[name]["heatmap_loss_valid"]) for name in TICK_NAMES],
                                       dtype=torch.float32),
            "arc_mask": arc_mask,
            "arc_valid": arc_valid,
            "index": int(index),
        }
