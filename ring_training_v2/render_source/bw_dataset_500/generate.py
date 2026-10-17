"""Build exactly 500 inputs: 20 approved originals plus 480 fresh synthetic images.

Run python make_plan.py, then python generate.py. All rendering is local.
The approved source, its preparation, and the 20-image pilot remain unchanged.
"""
import copy
import csv
import json
import math
from collections import Counter
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from render_core import camera_matrix, dump_json, photo_effects, save_png, sha256, transform_points, warp_source

ROOT = Path(__file__).resolve().parent
SIZE = 512
CSV_FIELDS = ["id", "sample_id", "image", "speed_kmh", "pivot_x", "pivot_y", "tip_x", "tip_y",
              "pivot_x_normalized", "pivot_y_normalized", "tip_x_normalized", "tip_y_normalized",
              "image_angle_degrees", "needle_mask", "profile", "group_id", "dataset_split", "origin",
              "anchor_method", "inferred_60_influence", "review_status", "rng_seed"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetRenderer:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.calibration = read_json(self.root / "calibration.json")
        self.preparation = read_json(self.root / "source/preparation.json")
        self.review = read_json(self.root / "review_seed/labels.json")
        self.review_by_id = {r["id"]: r for r in self.review["samples"]}
        master = self.root / self.calibration["source_file"]
        assert sha256(master) == self.calibration["source_sha256"] == self.preparation["source_sha256"]
        assert sha256(self.root / "source/clean_plate.png") == self.review["source"]["clean_plate_sha256"]
        assert sha256(self.root / "source/needle_layer.npz") == self.review["source"]["needle_layer_sha256"]
        self.plate = cv2.imread(str(self.root / "source/clean_plate.png"), cv2.IMREAD_UNCHANGED)
        self.cap = cv2.imread(str(self.root / "source/cap_mask.png"), cv2.IMREAD_UNCHANGED)
        assert self.plate.ndim == 2 and self.plate.dtype == np.uint8
        assert self.cap.shape == self.plate.shape == (1254, 1254)
        with np.load(self.root / "source/needle_layer.npz") as data:
            self.alpha, self.premult = data["alpha"].copy(), data["premult"].copy()
        assert self.alpha.shape == self.premult.shape == self.plate.shape
        self.pivot = np.array(self.preparation["pivot_xy"], float)
        self.source_tip = np.array(self.preparation["source_needle_tip_xy"], float)
        self.ratio = float(self.preparation["needle_to_inner_tick_radius_ratio"])
        self.ticks = sorted(self.calibration["ticks"], key=lambda t: t["speed_kmh"])
        self.tick_speeds = np.array([t["speed_kmh"] for t in self.ticks], float)
        vectors = np.array([t["inner_tick_xy"] for t in self.ticks], float) - self.pivot
        self.tick_angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        self.tick_radii = np.linalg.norm(vectors, axis=1)
        assert np.all(np.diff(self.tick_angles) > 0)

    def target_tick(self, speed):
        assert 0 <= speed <= 180
        exact = np.flatnonzero(np.isclose(self.tick_speeds, speed, atol=1e-9, rtol=0))
        if len(exact):
            i = int(exact[0])
            tick = self.ticks[i]
            inferred = tick["method"] != "manual_visible_tick"
            return np.array(tick["inner_tick_xy"], float), tick["method"], float(tick["estimated_annotation_uncertainty_px"]), {
                "lower_speed_kmh": speed, "upper_speed_kmh": speed, "fraction": 0.0,
                "inferred_60_influence": inferred,
            }
        right = int(np.searchsorted(self.tick_speeds, speed))
        left = right - 1
        lo, hi = self.ticks[left], self.ticks[right]
        fraction = float((speed - self.tick_speeds[left]) / (self.tick_speeds[right] - self.tick_speeds[left]))
        angle = (1 - fraction) * self.tick_angles[left] + fraction * self.tick_angles[right]
        radius = (1 - fraction) * self.tick_radii[left] + fraction * self.tick_radii[right]
        anchor = self.pivot + radius * np.array([np.cos(angle), np.sin(angle)])
        inferred = lo["method"] != "manual_visible_tick" or hi["method"] != "manual_visible_tick"
        method = "piecewise_interpolated_inferred_tick" if inferred else "piecewise_interpolated_visible_ticks"
        uncertainty = float(max(lo["estimated_annotation_uncertainty_px"], hi["estimated_annotation_uncertainty_px"]))
        return anchor, method, uncertainty, {
            "lower_speed_kmh": float(self.tick_speeds[left]), "upper_speed_kmh": float(self.tick_speeds[right]),
            "fraction": fraction, "inferred_60_influence": inferred,
            "interpolation": "linear in unwrapped source-view angle and radius, not a physical 3D dial angle",
        }

    def source_composite(self, speed):
        anchor, method, uncertainty, interpolation = self.target_tick(speed)
        tip = self.pivot + (anchor - self.pivot) * self.ratio
        u, v = self.source_tip - self.pivot, tip - self.pivot
        a = float(u @ v / (u @ u))
        b = float((u[0] * v[1] - u[1] * v[0]) / (u @ u))
        linear = np.array([[a, -b], [b, a]])
        transform = np.eye(3)
        transform[:2] = np.column_stack([linear, self.pivot - linear @ self.pivot])
        assert np.allclose(transform_points([self.pivot, self.source_tip], transform), [self.pivot, tip])
        h, w = self.plate.shape
        alpha = cv2.warpAffine(self.alpha, transform[:2], (w, h), flags=cv2.INTER_LINEAR)
        premult = cv2.warpAffine(self.premult, transform[:2], (w, h), flags=cv2.INTER_LINEAR)
        alpha[self.cap > 0] = 0
        premult[self.cap > 0] = 0
        composite = np.rint(self.plate * (1 - alpha) + premult).clip(0, 255).astype(np.uint8)
        assert np.array_equal(composite[alpha == 0], self.plate[alpha == 0])
        mask = (alpha > .35).astype(np.uint8) * 255
        return composite, mask, tip, anchor, transform, method, uncertainty, interpolation


def render_sample(plan_row, renderer=None):
    """Return image, mask and labels in memory, without changing any files."""
    renderer = renderer or DatasetRenderer()
    ident = plan_row["id"]
    sample_id = plan_row["sample_id"]
    image_path, mask_path = f"images/{sample_id}.png", f"masks/{sample_id}.png"
    if plan_row["origin"] == "approved_review":
        original = renderer.review_by_id[plan_row["original_review_id"]]
        image_file = renderer.root / "review_seed" / original["image"]
        assert sha256(image_file) == original["image_sha256"]
        assert original["speed_kmh"] == plan_row["speed_kmh"]
        image = cv2.imread(str(image_file), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(renderer.root / "review_seed" / original["needle_mask"]), cv2.IMREAD_UNCHANGED)
        label = copy.deepcopy(original)
        label.pop("clean_reference", None)
        label.pop("label_overlay", None)
        label.update({"review_status": "user_approved_pilot", "readability": "pilot_review_accepted",
                      "inferred_60_influence": original["speed_kmh"] in (60, 65),
                      "original_review_id": original["id"], "original_review_image_sha256": original["image_sha256"]})
    else:
        speed = float(plan_row["speed_kmh"])
        rng = np.random.default_rng(plan_row["rng_seed"])
        composite, source_mask, tip, anchor, needle_transform, method, uncertainty, interpolation = renderer.source_composite(speed)
        profile = plan_row["profile"]
        H, camera_params = camera_matrix(rng, renderer.plate.shape, "none" if profile == "clean" else "perspective")
        framed = warp_source(composite, H, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        image, photo_params = photo_effects(framed, rng, profile)
        mask = warp_source(source_mask, H, cv2.INTER_NEAREST)
        projected = transform_points([renderer.pivot, tip, anchor], H)
        assert np.all((projected >= 0) & (projected <= SIZE - 1)), f"Out-of-frame geometry: {sample_id}"
        positions = {name: point.tolist() for name, point in zip(["pivot", "tip", "tick_anchor"], projected)}
        offsets = uncertainty * np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        output_uncertainty = float(np.linalg.norm(transform_points(anchor + offsets, H) - projected[2], axis=1).max())
        direction = projected[1] - projected[0]
        label = {
            "speed_kmh": speed, "speed_label_type": "intended_synthetic_speed",
            "width": SIZE, "height": SIZE, "channels": 1,
            "needle_position_px": positions,
            "needle_position_normalized": {name: (np.array(point) / (SIZE - 1)).tolist() for name, point in positions.items()},
            "needle_angle_image_degrees_clockwise_from_right": float(math.degrees(math.atan2(direction[1], direction[0])) % 360),
            "needle_angle_physical_degrees": None,
            "mask_semantics": "rendered shaft support before blur; hub excluded",
            "source_pivot_xy": renderer.pivot.tolist(), "source_target_tip_xy": tip.tolist(), "source_tick_anchor_xy": anchor.tolist(),
            "needle_transform_source_3x3": needle_transform.tolist(), "source_to_output_homography_3x3": H.tolist(),
            "anchor_method": method, "interpolation": interpolation,
            "inferred_60_influence": interpolation["inferred_60_influence"],
            "estimated_anchor_uncertainty_source_px": uncertainty,
            "estimated_anchor_uncertainty_output_px": output_uncertainty,
            "keypoints_in_frame": True, "readability": "manual_review_pending", "review_status": "pending",
            "augmentation": {"camera": camera_params, "photometric": photo_params},
        }
    label.update({"id": ident, "sample_id": sample_id, "image": image_path, "needle_mask": mask_path,
                  "group_id": plan_row["group_id"], "variant_index": plan_row["variant_index"],
                  "origin": plan_row["origin"], "dataset_split": plan_row["split"], "rng_seed": plan_row["rng_seed"]})
    return {"image": image, "mask": mask, "label": label}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            p, n = row["needle_position_px"], row["needle_position_normalized"]
            values = [row["id"], row["sample_id"], row["image"], row["speed_kmh"], *p["pivot"], *p["tip"], *n["pivot"], *n["tip"],
                      row["needle_angle_image_degrees_clockwise_from_right"], row["needle_mask"],
                      row["augmentation"]["photometric"]["profile"], row["group_id"], row["dataset_split"], row["origin"],
                      row["anchor_method"], row["inferred_60_influence"], row["review_status"], row["rng_seed"]]
            writer.writerow(dict(zip(CSV_FIELDS, values)))


def main():
    started = time.perf_counter()
    plan = read_json(ROOT / "generation_plan.json")
    assert len(plan["samples"]) == 500
    assert Counter(r["split"] for r in plan["samples"]) == {"train": 400, "validation": 100}
    renderer = DatasetRenderer()
    rows = []
    for plan_row in plan["samples"]:
        result = render_sample(plan_row, renderer)
        row = result["label"]
        image_path, mask_path = ROOT / row["image"], ROOT / row["needle_mask"]
        image_path.parent.mkdir(exist_ok=True)
        mask_path.parent.mkdir(exist_ok=True)
        if row["origin"] == "approved_review":
            original = renderer.review_by_id[row["original_review_id"]]
            shutil.copyfile(ROOT / "review_seed" / original["image"], image_path)
            shutil.copyfile(ROOT / "review_seed" / original["needle_mask"], mask_path)
        else:
            save_png(image_path, result["image"])
            save_png(mask_path, result["mask"])
        row["image_sha256"] = sha256(image_path)
        row["needle_mask_sha256"] = sha256(mask_path)
        rows.append(row)
        if len(rows) % 50 == 0:
            print(f"Generated {len(rows)}/500 images ({time.perf_counter()-started:.1f}s)", flush=True)
    profile_counts = {split: dict(Counter(r["augmentation"]["photometric"]["profile"] for r in rows if r["dataset_split"] == split))
                      for split in ("train", "validation")}
    bins = {split: [sum(r["dataset_split"] == split and min(int(r["speed_kmh"] // 10), 17) == b for r in rows) for b in range(18)]
            for split in ("train", "validation")}
    source = {
        "file": renderer.calibration["source_file"], "sha256": sha256(ROOT / renderer.calibration["source_file"]),
        "clean_plate_sha256": sha256(ROOT / "source/clean_plate.png"), "needle_layer_sha256": sha256(ROOT / "source/needle_layer.npz"),
        "origin": "user-approved ImageGen grayscale master; same frozen source as the20-image pilot",
    }
    document = {
        "schema_version": "2.0", "purpose": "500-image synthetic training pilot; no real test set",
        "count": len(rows), "approved_originals": 20, "new_images": 480,
        "source": source, "generation_plan_sha256": sha256(ROOT / "generation_plan.json"),
        "coordinate_convention": "x right, y down, zero-based pixel centers; normalized by width-1 and height-1",
        "split_counts": {"train": 400, "validation": 100},
        "split_rule": "Every clean needle pose and all of its variants are kept in one split. All20 approved originals are training inputs.",
        "profile_counts": profile_counts, "speed_bin_counts_10_kmh": bins,
        "calibration_status": "approved synthetic appearance; physical real-dashboard calibration remains unverified",
        "samples": rows,
    }
    dump_json(ROOT / "labels.json", document)
    write_csv(ROOT / "labels.csv", rows)
    for split in ("train", "validation"):
        write_csv(ROOT / f"splits/{split}.csv", [r for r in rows if r["dataset_split"] == split])
    summary = {"count": len(rows), "approved_originals": 20, "new_images": 480,
               "source_sha256": source["sha256"], "split_counts": document["split_counts"],
               "clean_pose_groups": len({r["group_id"] for r in rows}),
               "speed_range_kmh": [min(r["speed_kmh"] for r in rows), max(r["speed_kmh"] for r in rows)],
               "profile_counts": profile_counts, "speed_bin_counts_10_kmh": bins,
               "inferred_60_influence_count": sum(r["inferred_60_influence"] for r in rows),
               "training_started": False}
    dump_json(ROOT / "dataset_summary.json", summary)
    print(json.dumps({"generated": len(rows), "new_images": 480, "retained_approved": 20, "train": 400, "validation": 100,
                      "seconds": round(time.perf_counter()-started, 2)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
