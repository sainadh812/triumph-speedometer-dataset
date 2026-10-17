"""Verified 512px inputs for the augmented synthetic ring dataset.

Three PNG masks preserve full supervision: visible arc, arc-valid pixels, and a
packed provenance mask. The latter is decoded for validation, never used as an
image feature. Train/validation/test variants of one pose family never mix.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training_pilot.train_core import preprocess_image

SIZE = 512
TICK_NAMES = ("tick_0", "tick_40", "tick_80", "tick_120", "tick_160", "tick_180")
MASK_NAMES = ("arc_visible", "arc_valid", "supervision_bits")
BITS = {"arc_amodal": 1, "source60_uncertainty": 2, "needle_dilated": 4,
        "image_footprint": 8, "patch_ignore": 16, "needle_shaft": 32}
SPLIT_COUNTS = {"train": 500, "validation": 100, "test": 100}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _path(root, relative):
    _require(isinstance(relative, str) and bool(relative), "Invalid dataset path")
    p, w = PurePosixPath(relative), PureWindowsPath(relative)
    _require(not p.is_absolute() and not w.drive and ".." not in p.parts
             and "\\" not in relative and ":" not in relative, "Unsafe dataset path: " + relative)
    path = (root / relative).resolve()
    _require(path.is_relative_to(root) and path.is_file(), "Missing/outside dataset file: " + relative)
    return path


def _gray(path):
    with Image.open(path) as im:
        _require(im.mode == "L" and im.size == (SIZE, SIZE), "Expected grayscale 512x512: " + str(path))
        return np.asarray(im).copy()


def _project(points, h):
    p = np.asarray(points, dtype=float)
    q = np.c_[p, np.ones(len(p))] @ h.T
    _require(np.isfinite(q).all() and np.all(np.abs(q[:, 2]) > 1e-8), "Invalid projective coordinates")
    return q[:, :2] / q[:, 2:]


def _warp(mask, h):
    shift = np.array([[1, 0, 16], [0, 1, 16], [0, 0, 1.]], float)
    return cv2.warpPerspective(mask[16:-16, 16:-16], h @ shift, (SIZE, SIZE),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0


def _point(xy, norm, context):
    xy, norm = np.asarray(xy, float), np.asarray(norm, float)
    _require(xy.shape == norm.shape == (2,) and np.isfinite(xy).all() and np.isfinite(norm).all(),
             "Invalid point: " + context)
    _require(np.all((xy >= 0) & (xy <= 511)) and np.allclose(xy / 511, norm, atol=1e-7, rtol=0),
             "Out-of-frame or normalized point mismatch: " + context)
    return xy


def verify_dataset(data_dir):
    root = Path(data_dir).resolve()
    doc = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    _require(doc.get("schema_version") == "arc_needle_augmented_v2", "Unsupported ring dataset schema")
    rows = doc["samples"]
    _require(doc["count"] == len(rows) == 700 and doc["resolution"] == [512, 512], "Wrong sample count/resolution")
    counts = dict(Counter(r["dataset_split"] for r in rows))
    _require(counts == doc["split_counts"] == SPLIT_COUNTS, "Expected 500/100/100 declared splits")
    _require(doc["mask_bits"] == BITS and doc["mirrors_enabled"] is False, "Mask/mirror conventions differ")
    _require(set(doc["snapshot_sha256"]) == {"source_annotations_snapshot.json", "calibration_snapshot.json",
                                             "source_arc_amodal.png", "source60_uncertainty.png"},
             "Expected all four source snapshot hashes")
    for rel, digest in doc["snapshot_sha256"].items():
        _require(sha256(_path(root, rel)) == digest, "Snapshot checksum mismatch: " + rel)
    annotation = json.loads((root / "source_annotations_snapshot.json").read_text(encoding="utf-8"))
    calibration = json.loads((root / "calibration_snapshot.json").read_text(encoding="utf-8"))
    _require(annotation["source"]["sha256"] == calibration["source_sha256"] == doc["source_master_sha256"],
             "Source snapshots disagree")
    source_ticks = {t["name"]: t["xy_source"] for t in annotation["named_tick_landmarks"]}
    source_pivot = np.asarray(calibration["pivot_xy"], float)
    calibrated = sorted(calibration["ticks"], key=lambda t: t["speed_kmh"])
    speeds = np.array([t["speed_kmh"] for t in calibrated], float)
    vectors = np.array([t["inner_tick_xy"] for t in calibrated]) - source_pivot
    angles, radii = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0])), np.linalg.norm(vectors, axis=1)
    source_arc = cv2.imread(str(root / "source_arc_amodal.png"), 0)
    source_static = cv2.imread(str(root / "source60_uncertainty.png"), 0)
    source_footprint = np.full(source_arc.shape, 255, np.uint8)
    ids, paths, image_hashes = set(), set(), set()
    groups, pose_splits = defaultdict(list), defaultdict(set)
    visibility = {split: Counter() for split in counts}
    occluded_counts, clean_counts = Counter(), Counter()
    max_projection_error = 0.0
    for row in rows:
        sid, split = row["sample_id"], row["dataset_split"]
        _require(sid not in ids, "Duplicate sample: " + sid)
        ids.add(sid)
        _require((row["width"], row["height"], row["channels"]) == (512, 512, 1), "Invalid image metadata")
        image = _path(root, row["image"])
        _require(image not in paths and sha256(image) == row["image_sha256"], "Image path/hash mismatch: " + sid)
        paths.add(image)
        _require(row["image_sha256"] not in image_hashes, "Duplicate rendered input: " + sid)
        image_hashes.add(row["image_sha256"])
        _gray(image)
        _require(set(row["masks"]) == set(row["mask_sha256"]) == set(MASK_NAMES), "Three-mask inventory mismatch")
        masks = {}
        for name in MASK_NAMES:
            path = _path(root, row["masks"][name])
            _require(path not in paths and sha256(path) == row["mask_sha256"][name], "Mask path/hash mismatch: " + sid)
            paths.add(path)
            masks[name] = _gray(path)
        _require(np.isin(masks["arc_visible"], (0, 255)).all() and np.isin(masks["arc_valid"], (0, 255)).all(),
                 "Arc masks must be binary: " + sid)
        packed = masks["supervision_bits"]
        _require(np.all(packed <= 63), "Unknown supervision bit: " + sid)
        b = {name: (packed & bit) != 0 for name, bit in BITS.items()}
        h = np.asarray(row["source_to_output_homography_3x3"], float)
        _require(h.shape == (3, 3) and np.isfinite(h).all() and np.linalg.det(h) > 0, "Invalid/mirrored homography")
        _require(row["camera"]["mirrored"] is False, "Mirrors are not part of v2")
        for name, source in (("arc_amodal", source_arc), ("source60_uncertainty", source_static),
                             ("image_footprint", source_footprint)):
            _require(np.array_equal(b[name], _warp(source, h)), "Warped source mask mismatch: " + sid + "/" + name)
        dilated = cv2.dilate(b["needle_shaft"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        _require(np.array_equal(b["needle_dilated"], dilated), "Needle dilation mismatch: " + sid)
        patch = np.zeros((SIZE, SIZE), np.uint8)
        for ellipse in row["occlusion"]["patches"]:
            cv2.ellipse(patch, tuple(ellipse["center_px"]), tuple(ellipse["axes_px"]), ellipse["angle_degrees"],
                        0, 360, 255, -1, cv2.LINE_8)
        patch_ignore = cv2.dilate(patch, np.ones((5, 5), np.uint8)) > 0
        _require(np.array_equal(b["patch_ignore"], patch_ignore), "Patch mask does not match ellipse metadata: " + sid)
        _require(not np.any(patch_ignore & b["needle_shaft"]), "Controlled patch covers needle: " + sid)
        ignored = b["source60_uncertainty"] | (b["arc_amodal"] & dilated) | patch_ignore
        _require(np.array_equal(masks["arc_visible"] > 0, b["arc_amodal"] & ~ignored), "Visible arc rule mismatch")
        _require(np.array_equal(masks["arc_valid"] > 0, b["image_footprint"] & ~ignored), "Arc-valid rule mismatch")
        speed = float(row["speed_kmh"])
        _require(np.isfinite(speed) and 0 <= speed <= 180, "Invalid speed label")
        fingerprint = hashlib.sha256(json.dumps({"speed": speed, "H": h.tolist()}, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        _require(row["pose_fingerprint"] == fingerprint, "Pose fingerprint differs from actual camera and speed")
        angle, radius = np.interp(speed, speeds, angles), np.interp(speed, speeds, radii)
        anchor = source_pivot + radius * np.array([np.cos(angle), np.sin(angle)])
        tip = source_pivot + (anchor - source_pivot) * doc["needle_to_inner_tick_radius_ratio"]
        expected_points = _project([source_pivot, tip, anchor], h)
        for name, expected in zip(("pivot", "tip", "tick_anchor"), expected_points):
            point = _point(row["needle_position_px"][name], row["needle_position_normalized"][name], sid + "/" + name)
            error = float(np.max(np.abs(point - expected)))
            max_projection_error = max(max_projection_error, error)
            _require(error < 1e-6, "Needle speed/calibration/transform mismatch: " + sid)
        ticks = row["tick_landmarks"]
        _require(len(ticks) == 6 and {t["name"] for t in ticks} == set(TICK_NAMES), "Invalid tick order/inventory")
        visible_count = 0
        for tick in ticks:
            point = _point(tick["xy"], tick["xy_normalized"], sid + "/" + tick["name"])
            _require(np.allclose(point, _project([source_ticks[tick["name"]]], h)[0], atol=1e-6, rtol=0), "Tick transform mismatch")
            x, y = np.rint(point).astype(int)
            visible = bool(b["image_footprint"][y, x] and not dilated[y, x]
                           and not b["source60_uncertainty"][y, x] and not patch_ignore[y, x])
            _require(type(tick["visible"]) is bool and tick["visible"] == tick["heatmap_loss_valid"] == visible,
                     "Tick visibility mismatch: " + sid)
            visible_count += int(visible)
            visibility[split][tick["name"] + ("_visible" if visible else "_hidden")] += 1
        _require(row["visible_tick_count"] == visible_count, "Visible count mismatch")
        groups[row["source_pose_group"]].append(row)
        pose_splits[row["pose_fingerprint"]].add(split)
        occluded_counts[split] += int(bool(row["occlusion"]["patches"]))
        clean_counts[split] += int(row["difficulty"] == "clean_light")
    for group, siblings in groups.items():
        _require(len(siblings) == 2 and len({r["dataset_split"] for r in siblings}) == 1,
                 "Pose siblings cross splits or family size differs: " + group)
        _require(len({r["pose_fingerprint"] for r in siblings}) == 1
                 and len({r["speed_kmh"] for r in siblings}) == 1, "Inconsistent pose family: " + group)
        _require(np.array_equal(siblings[0]["source_to_output_homography_3x3"],
                               siblings[1]["source_to_output_homography_3x3"]), "Sibling camera transforms differ")
    _require(len(groups) == len(pose_splits) == 350, "Expected 350 distinct camera/needle families")
    _require(all(len(splits) == 1 for splits in pose_splits.values()), "Identical camera/needle poses cross splits")
    _require(dict(occluded_counts) == {"train": 200, "validation": 40, "test": 40}, "Occlusion sampling differs from plan")
    _require(dict(clean_counts) == {"train": 125, "validation": 25, "test": 25}, "Clean/light sampling differs from plan")
    report = {"schema_version": doc["schema_version"], "count": len(rows), "split_counts": counts,
              "pose_groups": len(groups), "pose_group_leakage": False, "review_only": False,
              "labels_sha256": sha256(root / "labels.json"), "image_files_verified": len(rows),
              "mask_files_verified": 3 * len(rows), "all_image_hashes_verified": True,
              "all_mask_hashes_verified": True, "mask_semantics_verified": True,
              "tick_visibility_verified": True, "occluded_samples": dict(occluded_counts),
              "clean_light_samples": dict(clean_counts), "visibility_counts": {k: dict(v) for k, v in visibility.items()},
              "max_needle_projection_error_px": max_projection_error,
              "source_master_sha256": doc["source_master_sha256"], "mirrors_enabled": False,
              "scope": "Single-master synthetic holdouts by camera/needle/occlusion family; not independent real-camera accuracy"}
    return rows, report


class RingDataset(Dataset):
    def __init__(self, data_dir, records=None, input_size=512):
        _require(input_size == 512, "Thin arc supervision requires input_size=512")
        self.root = Path(data_dir).resolve()
        self.data_dir, self.input_size = self.root, input_size
        all_records, self.integrity = verify_dataset(self.root)
        by_id = {r["sample_id"]: r for r in all_records}
        self.records = all_records if records is None else list(records)
        seen = set()
        for row in self.records:
            sid = row.get("sample_id")
            _require(sid in by_id and sid not in seen and row == by_id[sid], "Records must be unchanged manifest entries")
            seen.add(sid)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        ticks = {tick["name"]: tick for tick in row["tick_landmarks"]}
        def mask(name):
            return torch.from_numpy(_gray(_path(self.root, row["masks"][name])).astype(np.float32) / 255).unsqueeze(0)
        return {"image": preprocess_image(_path(self.root, row["image"]), 512),
                "speed": torch.tensor(row["speed_kmh"], dtype=torch.float32),
                "points": torch.tensor([row["needle_position_normalized"][k] for k in ("pivot", "tip")], dtype=torch.float32),
                "tick_points": torch.tensor([ticks[k]["xy_normalized"] for k in TICK_NAMES], dtype=torch.float32),
                "tick_valid": torch.tensor([float(ticks[k]["heatmap_loss_valid"]) for k in TICK_NAMES], dtype=torch.float32),
                "arc_mask": mask("arc_visible"), "arc_valid": mask("arc_valid"), "index": int(index)}


class AugmentedRingDataset(RingDataset):
    """Convenience split selection; ``val`` is an alias for ``validation``."""
    def __init__(self, data_dir, split="train", records=None, input_size=512):
        split = "validation" if split == "val" else split
        _require(split in SPLIT_COUNTS, "Expected train, val/validation, or test")
        super().__init__(data_dir, records=records, input_size=input_size)
        if records is None:
            self.records = [row for row in self.records if row["dataset_split"] == split]
        else:
            _require(all(row["dataset_split"] == split for row in self.records), "Records disagree with requested split")
        self.split = split
