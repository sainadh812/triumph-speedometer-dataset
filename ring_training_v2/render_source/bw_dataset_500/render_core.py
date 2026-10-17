"""Frozen rendering helpers from the user-approved 20-image pipeline.

Local OpenCV operations; no image-generation API or training.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SIZE = 512
SOURCE_MARGIN = 16


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_png(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), data), str(path)


def dump_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transform_points(points, matrix):
    p = np.c_[np.atleast_2d(points), np.ones(len(np.atleast_2d(points)))] @ matrix.T
    return p[:, :2] / p[:, 2:]


def camera_matrix(rng, source_shape, geometry):
    height, width = source_shape
    # Trim the ImageGen master's thin white border, then frame pixel centers.
    cw, ch = width - 2 * SOURCE_MARGIN, height - 2 * SOURCE_MARGIN
    sx, sy = (SIZE - 1) / (cw - 1), (SIZE - 1) / (ch - 1)
    base = np.array([[sx, 0, -SOURCE_MARGIN * sx], [0, sy, -SOURCE_MARGIN * sy], [0, 0, 1.]])
    crop = [SOURCE_MARGIN, SOURCE_MARGIN, cw, ch]
    if geometry == "none":
        return base, {"rotation_degrees": 0, "scale": 1, "translation_px": [0, 0], "corner_jitter_px": [[0, 0]] * 4,
                      "source_crop_xywh": crop}
    rotation = float(rng.uniform(-2.5, 2.5))
    scale = float(rng.uniform(0.985, 1.025))
    shift = rng.uniform(-4.0, 4.0, 2)
    affine = np.eye(3)
    affine[:2] = cv2.getRotationMatrix2D(((SIZE - 1) / 2, (SIZE - 1) / 2), rotation, scale)
    affine[:2, 2] += shift
    corners = np.array([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]], np.float32)
    jitter = rng.uniform(-3.5, 3.5, (4, 2)).astype(np.float32) if geometry == "perspective" else np.zeros((4, 2), np.float32)
    projective = cv2.getPerspectiveTransform(corners, corners + jitter)
    return projective @ affine @ base, {
        "rotation_degrees": rotation, "scale": scale, "translation_px": shift.tolist(),
        "corner_jitter_px": jitter.tolist(), "border_mode": "reflect101",
        "source_crop_xywh": crop,
    }


def warp_source(im, H, interpolation, border_mode=cv2.BORDER_CONSTANT):
    # Physically crop before warping so camera jitter cannot reintroduce the
    # discarded source border. Labels keep the full-source coordinate system.
    cropped = im[SOURCE_MARGIN:-SOURCE_MARGIN, SOURCE_MARGIN:-SOURCE_MARGIN]
    uncrop = np.array([[1, 0, SOURCE_MARGIN], [0, 1, SOURCE_MARGIN], [0, 0, 1.]])
    return cv2.warpPerspective(cropped, H @ uncrop, (SIZE, SIZE), flags=interpolation, borderMode=border_mode)


def photo_effects(im, rng, kind):
    """Order: exposure/gamma, shadow/reflection, blur, sensor noise, JPEG."""
    p = {"profile": kind, "gain": 1.0, "gamma": 1.0, "shadow": None, "reflection": None,
         "gaussian_sigma_px": 0.0, "motion_length_px": 0, "motion_angle_degrees": None,
         "noise_std_255": 0.0, "jpeg_quality": None}
    f = im.astype(np.float32) / 255
    if kind in ("dim", "dim_noise"):
        p["gain"] = float(rng.uniform(.76, .86))
        p["gamma"] = float(rng.uniform(1.08, 1.18))
    elif kind in ("bright", "bright_soft"):
        p["gain"] = float(rng.uniform(1.08, 1.16))
        p["gamma"] = float(rng.uniform(.88, .95))
    elif kind in ("mixed", "shadow_noise", "reflection_noise"):
        p["gain"] = float(rng.uniform(.9, 1.07))
        p["gamma"] = float(rng.uniform(.95, 1.08))
    f = p["gain"] * np.power(f, p["gamma"])
    yy, xx = np.mgrid[:SIZE, :SIZE].astype(np.float32)
    if kind in ("shadow", "shadow_noise", "mixed"):
        theta = float(rng.uniform(-math.pi, math.pi))
        offset = float(rng.uniform(-70, 70))
        width = float(rng.uniform(45, 100))
        strength = float(rng.uniform(.18, .32))
        axis = (xx - SIZE / 2) * math.cos(theta) + (yy - SIZE / 2) * math.sin(theta)
        shadow = (1 + np.tanh((axis - offset) / width)) / 2
        f *= 1 - strength * shadow
        p["shadow"] = {"angle_radians": theta, "offset_px": offset, "softness_px": width, "strength": strength}
    if kind in ("reflection", "reflection_noise", "mixed"):
        theta = float(rng.uniform(-math.pi, math.pi))
        offset = float(rng.uniform(-120, 120))
        width = float(rng.uniform(30, 65))
        strength = float(rng.uniform(.08, .16))
        axis = (xx - SIZE / 2) * math.cos(theta) + (yy - SIZE / 2) * math.sin(theta)
        band = np.exp(-.5 * ((axis - offset) / width) ** 2)
        envelope = np.exp(-.5 * (((xx - SIZE * .48) / (SIZE * .48)) ** 2 + ((yy - SIZE * .50) / (SIZE * .48)) ** 2))
        f += strength * band * envelope * (1 - np.clip(f, 0, 1))
        p["reflection"] = {"angle_radians": theta, "offset_px": offset, "width_px": width, "strength": strength}
    if kind in ("soft_focus", "bright_soft", "mixed"):
        p["gaussian_sigma_px"] = float(rng.uniform(.45, .85))
        f = cv2.GaussianBlur(f, (0, 0), p["gaussian_sigma_px"])
    if kind in ("motion", "motion_noise"):
        length = int(rng.choice([3, 5]))
        angle = float(rng.uniform(-85, 85))
        kernel = np.zeros((7, 7), np.float32)
        vector = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))]) * (length - 1) / 2
        cv2.line(kernel, tuple(np.rint(3 - vector).astype(int)), tuple(np.rint(3 + vector).astype(int)), 1.0, 1, cv2.LINE_8)
        kernel /= kernel.sum()
        f = cv2.filter2D(f, -1, kernel)
        p["motion_length_px"], p["motion_angle_degrees"] = length, angle
        p["motion_kernel"] = kernel.tolist()
    if kind in ("noise", "dim_noise", "shadow_noise", "reflection_noise", "motion_noise", "mixed"):
        p["noise_std_255"] = float(rng.uniform(2, 5.5))
        f += rng.normal(0, p["noise_std_255"] / 255, f.shape).astype(np.float32)
    result = np.rint(np.clip(f, 0, 1) * 255).astype(np.uint8)
    if kind in ("compression", "mixed"):
        p["jpeg_quality"] = int(rng.integers(70, 86))
        ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg_quality"]])
        assert ok
        result = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return result, p


