"""Build the deterministic 500/100/100 augmented ring dataset from frozen layers.

Uses only local OpenCV/Pillow operations. No AI generation, source modifications,
training, or network access. Run from the published checkout with render_source/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SEED = 202610180701
SIZE = 512
TICKS = ("tick_0", "tick_40", "tick_80", "tick_120", "tick_160", "tick_180")
SPLITS = {"train": 500, "validation": 100, "test": 100}
BITS = {"arc_amodal": 1, "source60_uncertainty": 2, "needle_dilated": 4,
        "image_footprint": 8, "patch_ignore": 16, "needle_shaft": 32}
SOURCE_FILES = tuple("bw_dataset_500/" + name for name in
                     ("generate.py", "render_core.py", "calibration.json", "source/preparation.json",
                      "source/approved_imagegen_master.png", "source/clean_plate.png", "source/cap_mask.png",
                      "source/needle_layer.npz", "review_seed/labels.json")) + (
                     "source_annotations.json", "masks/arc_amodal_inferred.png", "masks/arc_ignore.png")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project(points, h):
    p = np.asarray(points, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ h.T
    if not np.isfinite(q).all() or np.any(q[:, 2] <= 1e-8):
        raise ValueError("Invalid perspective transform")
    return q[:, :2] / q[:, 2:]


def warp(pixels, h, *, mask=False, border=0):
    shift = np.array([[1, 0, 16], [0, 1, 16], [0, 0, 1.]], float)
    return cv2.warpPerspective(pixels[16:-16, 16:-16], h @ shift, (SIZE, SIZE),
                              flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def camera(rng, labelled_points):
    b = np.array([[511 / 1221, 0, -16 * 511 / 1221], [0, 511 / 1221, -16 * 511 / 1221], [0, 0, 1.]])
    center = np.array([[1, 0, 255.5], [0, 1, 255.5], [0, 0, 1.]], float)
    corners = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], np.float32)
    for attempt in range(100):
        roll = float(rng.uniform(-180, 180))
        scale = float(rng.uniform(.85, 1.05))
        translation = rng.uniform(-24, 24, 2)
        shear = rng.uniform(-6, 6, 2)
        s = np.array([[1, math.tan(math.radians(shear[0])), 0],
                      [math.tan(math.radians(shear[1])), 1, 0], [0, 0, 1.]], float)
        s = center @ s @ np.linalg.inv(center)
        r = np.eye(3)
        r[:2] = cv2.getRotationMatrix2D((255.5, 255.5), -roll, scale)
        r[:2, 2] += translation
        jitter = float(rng.uniform(16, 28))
        offsets = rng.uniform(-jitter, jitter, (4, 2)).astype(np.float32)
        p = cv2.getPerspectiveTransform(corners, corners + offsets)
        h = p @ r @ s @ b
        transformed = project(labelled_points, h)
        if np.all((transformed >= 5) & (transformed <= 506)):
            return h, {"clockwise_rotation_degrees": roll, "scale": scale,
                       "translation_px": translation.tolist(), "shear_degrees_xy": shear.tolist(),
                       "perspective_corner_max_jitter_px": jitter, "perspective_corner_offsets_px": offsets.tolist(),
                       "source_crop_xywh": [16, 16, 1222, 1222], "border_mode": "constant",
                       "mirrored": False, "sampling_attempts": attempt + 1,
                       "transform_order": "source crop/resize, centered shear, clockwise roll/scale/translation, perspective"}
    raise RuntimeError("Could not sample an uncropped labelled dial")


def controlled_occlusion(image, ticks, needle, static, count, usage, rng):
    raw = np.zeros((512, 512), np.uint8)
    patches, chosen = [], []
    safety = cv2.dilate(needle, np.ones((13, 13), np.uint8)) > 0
    priorities = {name: (usage[name], float(rng.random())) for name in TICKS}
    candidates = sorted(TICKS, key=lambda name: priorities[name])
    for name in candidates:
        if len(chosen) == count:
            break
        point = np.rint(ticks[name]).astype(int)
        x, y = point
        if static[y, x] or safety[y, x]:
            continue
        for _ in range(12):
            axes = (int(rng.integers(9, 15)), int(rng.integers(9, 17)))
            angle = float(rng.uniform(0, 180))
            candidate = np.zeros_like(raw)
            cv2.ellipse(candidate, (int(x), int(y)), axes, angle, 0, 360, 255, -1, cv2.LINE_8)
            if not np.any((candidate > 0) & safety):
                break
        else:
            continue
        raw |= candidate
        chosen.append(name)
        usage[name] += 1
        patches.append({"tick_name": name, "center_px": [int(x), int(y)], "axes_px": list(axes),
                        "angle_degrees": angle, "kind": "opaque textured elliptical obstruction"})
    if len(chosen) != count:
        raise RuntimeError("Unable to place requested occlusions without covering needle")
    if count:
        shade = float(rng.uniform(18, 110))
        texture = np.clip(shade + rng.normal(0, 2, image.shape), 0, 255)
        alpha = cv2.GaussianBlur(raw.astype(np.float32) / 255, (0, 0), .6)
        image = np.rint(image * (1 - alpha) + texture * alpha).clip(0, 255).astype(np.uint8)
    else:
        shade = None
    ignored = cv2.dilate(raw, np.ones((5, 5), np.uint8))
    return image, ignored, {"patches": patches, "target_tick_names": chosen, "shade_gray": shade,
                            "edge_softening_sigma_px": .6 if count else 0,
                            "supervision_padding_px": 2, "needle_overlap_avoided": True}


def appearance(image, rng, profile, old_photo):
    if profile == "light":
        gain, gamma = float(rng.uniform(.95, 1.05)), float(rng.uniform(.98, 1.02))
        if abs(gain - 1) < .01:
            gain += .015
        result = np.rint(np.clip(gain * (image.astype(np.float32) / 255) ** gamma, 0, 1) * 255).astype(np.uint8)
        metadata = {"profile": profile, "gain": gain, "gamma": gamma, "noise_std_255": 0,
                    "gaussian_sigma_px": 0, "motion_length_px": 0}
    else:
        result, metadata = old_photo(image, rng, profile)
    metadata["storage_codec"] = "grayscale JPEG"
    metadata["storage_jpeg_quality"] = 95
    return result, metadata


def font(size, bold=False):
    for name in ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
                 "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def overlay(image, row, visible, valid):
    canvas = Image.fromarray(image).convert("RGBA")
    rgba = np.zeros((512, 512, 4), np.uint8)
    rgba[visible > 0] = [40, 217, 232, 100]
    canvas = Image.alpha_composite(canvas, Image.fromarray(rgba)).convert("RGB")
    d = ImageDraw.Draw(canvas)
    d.line([tuple(row["needle_position_px"]["pivot"]), tuple(row["needle_position_px"]["tip"])], fill="#ff66cc", width=2)
    for tick in row["tick_landmarks"]:
        x, y = tick["xy"]
        color = "#ffe066" if tick["visible"] else "#ff784f"
        d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
        d.text((x + 7, y - 7), str(tick["speed_kmh"]), font=font(12, True), fill=color, stroke_width=1, stroke_fill="#111111")
    return canvas


def contact_sheets(root, rows):
    rng, selected = np.random.default_rng(SEED + 777), []
    quotas = [("train", "clean", 4), ("train", "occluded", 6), ("train", "other", 2),
              ("validation", "clean", 1), ("validation", "occluded", 2), ("validation", "other", 1),
              ("test", "clean", 1), ("test", "occluded", 2), ("test", "other", 1)]
    for split, kind, count in quotas:
        candidates = [r for r in rows if r["dataset_split"] == split and r not in selected
                      and (r["difficulty"] == "clean_light" if kind == "clean" else
                           bool(r["occlusion"]["patches"]) if kind == "occluded" else
                           r["difficulty"] != "clean_light" and not r["occlusion"]["patches"])]
        chosen = rng.choice(len(candidates), count, replace=False)
        selected.extend(candidates[int(i)] for i in chosen)
    qa = root / "qa"
    (qa / "overlays").mkdir(parents=True, exist_ok=True)
    tile, gap, header, footer = 256, 12, 98, 78
    for labels in (False, True):
        sheet = Image.new("RGB", (4 * tile + 5 * gap, header + 5 * (tile + footer + gap)), "#111d2d")
        d = ImageDraw.Draw(sheet)
        d.text((gap, 14), "AUGMENTED V2 | " + ("LABEL OVERLAYS" if labels else "GRAYSCALE MODEL INPUTS"), font=font(24, True), fill="white")
        d.text((gap, 52), "700 synthetic images | camera-family holdouts | no mirrors | not real-camera accuracy", font=font(14), fill="#b9cbe1")
        d.text((gap, 73), "Cyan: visible arc | Pink: needle | Yellow: visible tick | Orange: occluded tick", font=font(13), fill="#b9cbe1")
        for i, row in enumerate(selected):
            image = cv2.imread(str(root / row["image"]), 0)
            im = Image.fromarray(image).convert("RGB")
            if labels:
                visible = cv2.imread(str(root / row["masks"]["arc_visible"]), 0)
                valid = cv2.imread(str(root / row["masks"]["arc_valid"]), 0)
                im = overlay(image, row, visible, valid)
                im.save(qa / "overlays" / (row["sample_id"] + ".png"))
            x, y = gap + (i % 4) * (tile + gap), header + (i // 4) * (tile + footer + gap)
            sheet.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
            d.text((x, y + tile + 5), f"{row['dataset_split']} | {row['speed_kmh']:g} km/h", font=font(15, True), fill="white")
            d.text((x, y + tile + 28), f"roll {row['camera']['clockwise_rotation_degrees']:+.1f} | {row['appearance']['profile']}", font=font(13), fill="#b9cbe1")
            names = ",".join(n.replace("tick_", "") for n in row["occlusion"]["target_tick_names"]) or "none"
            d.text((x, y + tile + 48), "Covered ticks: " + names, font=font(13), fill="#ffa47b")
        sheet.save(qa / ("contact_sheet_labels.png" if labels else "contact_sheet_inputs.png"))
    dump(qa / "selected_samples.json", [r["sample_id"] for r in selected])


def generate(output, source):
    started = time.time()
    output, source = Path(output).resolve(), Path(source).resolve()
    if output.exists():
        raise ValueError("Output must be a new directory; existing datasets are preserved")
    for relative in SOURCE_FILES:
        if not (source / relative).is_file():
            raise FileNotFoundError("Missing bundled frozen renderer asset: " + relative)
    source_hashes = {relative: sha(source / relative) for relative in SOURCE_FILES}
    renderer_root = source / "bw_dataset_500"
    sys.path.insert(0, str(renderer_root))
    spec = importlib.util.spec_from_file_location("ring_v2_frozen_renderer", renderer_root / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    renderer = module.DatasetRenderer(renderer_root)
    annotation = json.loads((source / "source_annotations.json").read_text(encoding="utf-8"))
    if annotation["source"]["sha256"] != renderer.calibration["source_sha256"]:
        raise ValueError("Frozen calibration and arc annotation disagree")
    named = {t["name"]: np.asarray(t["xy_source"], float) for t in annotation["named_tick_landmarks"]}
    centerline = np.asarray(annotation["scale_arc"]["centerline_xy"], float)
    source_arc = cv2.imread(str(source / "masks/arc_amodal_inferred.png"), 0)
    source_static = cv2.imread(str(source / "masks/arc_ignore.png"), 0)
    footprint = np.full(renderer.plate.shape, 255, np.uint8)
    output.mkdir(parents=True)
    snapshots = {"source_annotations_snapshot.json": source / "source_annotations.json",
                 "calibration_snapshot.json": renderer_root / "calibration.json",
                 "source_arc_amodal.png": source / "masks/arc_amodal_inferred.png",
                 "source60_uncertainty.png": source / "masks/arc_ignore.png"}
    for name, path in snapshots.items():
        shutil.copy2(path, output / name)
    rows, patch_usage = [], {}
    for split_index, (split, count) in enumerate(SPLITS.items()):
        split_rng = np.random.default_rng(SEED + 100000 * split_index)
        families = count // 2
        occluded_families = set(map(int, split_rng.choice(families, int(families * .4), replace=False)))
        clean_slots = set(map(int, split_rng.choice(count, count // 4, replace=False)))
        usage = Counter({name: 0 for name in TICKS})
        occluded_index = 0
        for family_index in range(families):
            family_seed = SEED + 1000000 * (split_index + 1) + 7919 * family_index
            rng = np.random.default_rng(family_seed)
            speed = (0.0 if family_index == 0 else 180.0 if family_index == 1
                     else float(round((family_index % 18) * 10 + rng.uniform(.1, 9.9), 2)))
            composite, source_needle, tip, anchor, _, method, uncertainty, interpolation = renderer.source_composite(speed)
            labelled = np.vstack(([renderer.pivot, tip, anchor], centerline, list(named.values())))
            h, camera_meta = camera(rng, labelled)
            edges = np.r_[composite[16, 16:-16], composite[-17, 16:-16], composite[16:-16, 16], composite[16:-16, -17]]
            border = int(np.median(edges))
            camera_meta["border_gray_value"] = border
            framed = warp(composite, h, border=border)
            needle, arc, static, frame = [warp(mask, h, mask=True) for mask in (source_needle, source_arc, source_static, footprint)]
            dilated = cv2.dilate(needle, np.ones((3, 3), np.uint8))
            positions = {k: v.tolist() for k, v in zip(("pivot", "tip", "tick_anchor"), project([renderer.pivot, tip, anchor], h))}
            ticks = {name: project([point], h)[0] for name, point in named.items()}
            n_patches = 0
            if family_index in occluded_families:
                n_patches = 1 + occluded_index % 2
                occluded_index += 1
            framed, patch, occlusion = controlled_occlusion(framed, ticks, needle, static, n_patches, usage, rng)
            bits = {"arc_amodal": arc > 0, "source60_uncertainty": static > 0,
                    "needle_dilated": dilated > 0, "image_footprint": frame > 0,
                    "patch_ignore": patch > 0, "needle_shaft": needle > 0}
            ignored = bits["source60_uncertainty"] | (bits["arc_amodal"] & bits["needle_dilated"]) | bits["patch_ignore"]
            visible = (bits["arc_amodal"] & ~ignored).astype(np.uint8) * 255
            valid = (bits["image_footprint"] & ~ignored).astype(np.uint8) * 255
            packed = sum(bits[name].astype(np.uint8) * value for name, value in BITS.items()).astype(np.uint8)
            fingerprint = hashlib.sha256(json.dumps({"speed": speed, "H": h.tolist()}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            family = f"v2_{split}_pose_{family_index:04d}"
            for variant in range(2):
                sample_index = 2 * family_index + variant
                sample_seed = family_seed + (variant + 1) * 104729
                photo_rng = np.random.default_rng(sample_seed)
                if sample_index in clean_slots:
                    profile = "clean" if variant == 0 else "light"
                else:
                    profile = ["dim_noise", "motion_noise", "soft_focus", "shadow_noise", "reflection_noise", "bright_soft", "mixed", "compression"][int(photo_rng.integers(8))]
                image, photo = appearance(framed, photo_rng, profile, module.photo_effects)
                sid = f"v2_{split}_{sample_index + 1:04d}"
                image_name = f"images/{split}/{sid}.jpg"
                path = output / image_name
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError("Image save failed")
                mask_names = {name: f"masks/{name}/{split}/{sid}.png" for name in ("arc_visible", "arc_valid", "supervision_bits")}
                for name, mask in (("arc_visible", visible), ("arc_valid", valid), ("supervision_bits", packed)):
                    path = output / mask_names[name]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                        raise OSError("Mask save failed")
                landmarks = []
                for name in TICKS:
                    point = ticks[name]
                    x, y = np.rint(point).astype(int)
                    reasons = [reason for reason, covered in (("patch_occlusion", patch[y, x]),
                               ("needle_occlusion", dilated[y, x]), ("source60_uncertainty", static[y, x])) if covered]
                    is_visible = not reasons and bool(frame[y, x])
                    landmarks.append({"name": name, "speed_kmh": int(name.split("_")[1]),
                                      "xy": point.tolist(), "xy_normalized": (point / 511).tolist(),
                                      "visible": bool(is_visible), "heatmap_loss_valid": bool(is_visible),
                                      "visibility_reason": "+".join(reasons) if reasons else "visible_geometrically"})
                rows.append({"sample_id": sid, "id": len(rows) + 1, "image": image_name,
                             "image_sha256": sha(output / image_name), "width": 512, "height": 512, "channels": 1,
                             "dataset_split": split, "source_pose_group": family, "group_id": family,
                             "pose_fingerprint": fingerprint, "variant_index": variant, "rng_seed": sample_seed,
                             "speed_kmh": speed, "speed_label_type": "intended_synthetic_speed",
                             "needle_position_px": positions,
                             "needle_position_normalized": {name: (np.asarray(p) / 511).tolist() for name, p in positions.items()},
                             "needle_image_angle_degrees": float(np.degrees(np.arctan2(*(np.asarray(positions["tip"]) - positions["pivot"])[::-1])) % 360),
                             "tick_landmarks": landmarks, "visible_tick_count": sum(t["visible"] for t in landmarks),
                             "arc_centerline_px": project(centerline[::10], h).round(6).tolist(),
                             "masks": mask_names, "mask_sha256": {name: sha(output / rel) for name, rel in mask_names.items()},
                             "source_to_output_homography_3x3": h.tolist(), "camera": camera_meta,
                             "appearance": photo, "photometric": photo,
                             "difficulty": "clean_light" if profile in ("clean", "light") else "augmented",
                             "occlusion": occlusion, "anchor_method": method, "estimated_anchor_uncertainty_source_px": uncertainty,
                             "interpolation": interpolation, "inferred_60_influence": interpolation["inferred_60_influence"]})
            if len(rows) % 50 == 0:
                print(f"Rendered {len(rows)}/700 images in {time.time() - started:.1f}s", flush=True)
        patch_usage[split] = dict(usage)
    document = {"schema_version": "arc_needle_augmented_v2", "count": 700, "resolution": [512, 512],
                "split_counts": SPLITS, "seed": SEED, "mirrors_enabled": False, "mask_bits": BITS,
                "source_master_sha256": renderer.calibration["source_sha256"],
                "needle_to_inner_tick_radius_ratio": renderer.ratio,
                "snapshot_sha256": {name: sha(output / name) for name in snapshots},
                "frozen_source_sha256": source_hashes, "generator_sha256": sha(Path(__file__)),
                "grouping": "350 camera/needle/occlusion families assigned to splits before rendering; two appearance siblings per family",
                "appearance_note": "25% clean/light photometric variants; camera geometry and optional controlled occlusion remain",
                "image_encoding": "512x512 single-channel JPEG quality95; all label masks are lossless PNG",
                "mask_semantics": {"arc_visible": "amodal & ~(static60 | (amodal & dilated_needle) | patch_ignore)",
                                   "arc_valid": "image_footprint & ~(static60 | (amodal & dilated_needle) | patch_ignore)",
                                   "supervision_bits": "Packed independent geometry masks, never an image feature; bit values in mask_bits",
                                   "patch_ignore": "rendered opaque ellipse dilated by2pixels to exclude uncertain softened edges"},
                "limitations": ["Every image uses one AI-reconstructed grayscale source; no real-camera test data.",
                                "60km/h calibration and source reconstruction remain physically unverified.",
                                "Validation/test hold out synthetic pose families, not independent source dashboards.",
                                "Controlled patches and constant border fill are synthetic; real occlusion generalization is unproven."],
                "samples": rows}
    dump(output / "labels.json", document)
    contact_sheets(output, rows)
    if source_hashes != {relative: sha(source / relative) for relative in SOURCE_FILES}:
        raise RuntimeError("Frozen source assets changed during generation")
    sys.path.insert(0, str(HERE))
    from data import verify_dataset
    _, report = verify_dataset(output)
    report.update(patch_target_counts_per_family=patch_usage, rendering_seconds=time.time() - started,
                  photometric_profiles={split: dict(Counter(r["appearance"]["profile"] for r in rows if r["dataset_split"] == split)) for split in SPLITS},
                  visible_tick_count_histogram=dict(Counter(str(r["visible_tick_count"]) for r in rows)))
    dump(output / "validation_report.json", report)
    (output / "README.txt").write_text(
        "Augmented synthetic ring/needle dataset v2\n\n"
        "500 training, 100 validation, 100 test images; 512 x 512 grayscale JPEG95.\n"
        "350 independent camera/needle/occlusion families, two appearance siblings each.\n"
        "Split assignment precedes rendering; no sibling family crosses splits.\n"
        "360-degree camera roll, scale 0.85-1.05, translation up to 24 px, shear up to 6 degrees,\n"
        "perspective corner jitter 16-28 px; all labelled dial geometry retained in frame.\n"
        "25% clean/light appearance; 40% of samples have 1-2 controlled tick occlusions.\n"
        "Opaque patches avoid the needle; labels use geometry before photo blur/noise.\n"
        "No mirrors. Speed does not change under camera or appearance augmentation.\n"
        "Three mask PNGs: visible arc, arc valid, and packed provenance bits. See labels.json.\n"
        "Only visible arc/valid mask and visible tick targets enter loss.\n"
        "Validation/test are synthetic family holdouts from one reconstructed master,\n"
        "not independent real-camera accuracy. Inferred 60 km/h calibration remains provisional.\n"
        "Reproduce from published ring_training_v2/generate_dataset.py and render_source/.\n"
        "The frozen source assets are separate from this dataset archive.\n", encoding="utf-8")
    archive = output.parent / "arc_needle_augmented700.zip"
    if archive.exists():
        raise ValueError("Archive already exists; refusing to replace it")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, Path(output.name) / path.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise OSError("Archive CRC verification failed")
    manifest = {"archive": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size,
                "root_directory": output.name, "labels_sha256": sha(output / "labels.json"),
                "split_counts": SPLITS, "source_master_sha256": renderer.calibration["source_sha256"],
                "frozen_source_sha256": source_hashes, "mirrors_enabled": False}
    dump(output.parent / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "dataset_v2")
    parser.add_argument("--source", type=Path, default=HERE / "render_source")
    arguments = parser.parse_args()
    cv2.setNumThreads(4)
    generate(arguments.output, arguments.source)
