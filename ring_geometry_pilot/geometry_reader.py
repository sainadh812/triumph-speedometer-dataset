"""Landmark-based synthetic dial reader; no learned detector or oracle transform.

The caller supplies observed image coordinates for named ticks and the needle.
A homography estimated ONLY from those ticks maps the needle into the approved
source template. Speed is piecewise-linear in that template's calibrated angle.
The template is an AI reconstruction with an inferred 60-km/h tick: numerical
agreement here is not physical-dashboard accuracy or independent label truth.

Missing/occluded ticks must be omitted or set to None by the caller. At least
four well-spaced ticks are required. Four correspondences have no redundant
reprojection check; a small fitting residual is never a confidence score.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
TICK_NAMES = ("tick_0", "tick_40", "tick_80", "tick_120", "tick_160", "tick_180")
DEFAULT_LIMITS = {
    "endpoint_tolerance_degrees": 1e-6,
    "minimum_axis_ratio": 0.02,
    "maximum_dlt_condition": 1e8,
    "maximum_reprojection_rmse_source_px": 6.0,
    "maximum_reprojection_error_source_px": 12.0,
    "maximum_pivot_offset_source_px": 12.0,
}


def _point(value, label):
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain two finite coordinates") from exc
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{label} must contain two finite coordinates")
    return point


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_template(annotation_path=None, calibration_path=None):
    """Load a serializable, validated template without reading any image pixels.

    Explicit paths are useful for controlled calibration experiments. Defaults
    refer to the user's approved source annotation and its existing calibration.
    Invalid/inconsistent template files raise ValueError; they are not treated
    as a rejected observation. Review status and source uncertainty are retained.
    """
    annotation_path = Path(annotation_path) if annotation_path is not None else (
        PROJECT / "ring_annotation_review/source_annotations.json")
    calibration_path = Path(calibration_path) if calibration_path is not None else (
        PROJECT / "generated_bw_dataset/calibration.json")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if annotation["source"]["sha256"] != calibration["source_sha256"]:
        raise ValueError("Annotation and calibration refer to different source images")
    pivot = _point(calibration["pivot_xy"], "calibration pivot")
    annotation_pivot = _point(annotation["needle"]["pivot"]["xy_source"], "annotation pivot")
    if not np.allclose(pivot, annotation_pivot, atol=1e-8, rtol=0):
        raise ValueError("Annotation and calibration pivots differ")
    landmarks = annotation["named_tick_landmarks"]
    by_name = {row["name"]: row for row in landmarks}
    if len(by_name) != len(landmarks) or set(by_name) != set(TICK_NAMES):
        raise ValueError("Template must identify the six distinct named tick landmarks")
    ticks = sorted(calibration["ticks"], key=lambda row: float(row["speed_kmh"]))
    speeds = np.asarray([row["speed_kmh"] for row in ticks], dtype=np.float64)
    if speeds.shape != (19,) or not np.array_equal(speeds, np.arange(0, 181, 10)):
        raise ValueError("Expected the 19 calibrated ticks from 0 through 180 km/h")
    points = np.stack([_point(row["inner_tick_xy"], "calibration tick") for row in ticks])
    vectors = points - pivot
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-6):
        raise ValueError("A calibration tick coincides with the pivot")
    angles = np.rad2deg(np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0])))
    if np.any(np.diff(angles) <= 0) or not (0 < angles[-1] - angles[0] < 360):
        raise ValueError("Calibration angles must increase in speed order across less than one turn")
    anchors = {}
    for name in TICK_NAMES:
        row = by_name[name]
        speed = int(name.split("_")[1])
        point = _point(row["xy_source"], name)
        if row["speed_kmh"] != speed or not np.allclose(point, points[speed // 10], atol=1e-8, rtol=0):
            raise ValueError(f"Named landmark differs from calibration: {name}")
        anchors[name] = point.tolist()
    template = {
        "schema_version": "landmark_geometry_template_v1",
        "tick_points_source": anchors,
        "pivot_source_xy": pivot.tolist(),
        "calibration_tick_points_source": points.tolist(),
        "speed_knots_kmh": speeds.tolist(),
        "angle_knots_deg_unwrapped": angles.tolist(),
        "source_sha256": calibration["source_sha256"],
        "annotation_sha256": _digest(annotation_path),
        "calibration_sha256": _digest(calibration_path),
        "annotation_status": annotation.get("status", "unspecified"),
        "calibration_status": calibration.get("calibration_status", "unspecified"),
        "calibration_methods": [row.get("method", "unspecified") for row in ticks],
        "limits": dict(DEFAULT_LIMITS),
        "coordinate_convention": "x right, y down; angles clockwise from positive x in source coordinates",
        "limitation": "Synthetic-template geometry only; inferred 60-km/h anchor and physical calibration remain uncertain",
        "quality_limits_status": "Engineering sanity gates; not empirically calibrated detector confidence thresholds",
    }
    _validate_template(template)
    return template


def _validate_template(template):
    if not isinstance(template, Mapping):
        raise ValueError("template must be the mapping returned by load_template")
    required = {"tick_points_source", "pivot_source_xy", "calibration_tick_points_source",
                "speed_knots_kmh", "angle_knots_deg_unwrapped"}
    if not required.issubset(template):
        raise ValueError("Template is missing required geometry/calibration fields")
    anchors = template["tick_points_source"]
    if not isinstance(anchors, Mapping) or set(anchors) != set(TICK_NAMES):
        raise ValueError("Template requires all six named landmarks")
    for name in TICK_NAMES:
        _point(anchors[name], f"template {name}")
    _point(template["pivot_source_xy"], "template pivot")
    points = np.asarray(template["calibration_tick_points_source"], dtype=np.float64)
    speeds = np.asarray(template["speed_knots_kmh"], dtype=np.float64)
    angles = np.asarray(template["angle_knots_deg_unwrapped"], dtype=np.float64)
    if (points.shape != (19, 2) or speeds.shape != (19,) or angles.shape != (19,)
            or not np.isfinite(points).all() or not np.isfinite(speeds).all()
            or not np.isfinite(angles).all() or np.any(np.diff(speeds) <= 0)
            or np.any(np.diff(angles) <= 0) or not (0 < angles[-1] - angles[0] < 360)
            or speeds[0] != 0 or speeds[-1] != 180):
        raise ValueError("Malformed template calibration arrays")
    limits = {**DEFAULT_LIMITS, **template.get("limits", {})}
    if any(not np.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError("Template quality limits must be positive finite numbers")
    if limits["endpoint_tolerance_degrees"] > 1e-4:
        raise ValueError("Endpoint tolerance is numerical only, not an out-of-range clamp")
    return limits


def _normalize(points, minimum_axis_ratio):
    center = points.mean(axis=0)
    centered = points - center
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 1e-12 or singular[1] / singular[0] < minimum_axis_ratio:
        raise ValueError("degenerate_or_poorly_spread_tick_points")
    radius = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    separation = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(separation, np.inf)
    if np.min(separation) <= radius * 1e-6:
        raise ValueError("duplicate_or_nearly_duplicate_tick_points")
    scale = np.sqrt(2.0) / radius
    transform = np.array([[scale, 0, -scale * center[0]],
                          [0, scale, -scale * center[1]], [0, 0, 1]], dtype=np.float64)
    return centered * scale, transform


def _estimate_homography(image_points, source_points, limits):
    image, image_normalization = _normalize(image_points, limits["minimum_axis_ratio"])
    source, source_normalization = _normalize(source_points, limits["minimum_axis_ratio"])
    equations = []
    for (x, y), (u, v) in zip(image, source):
        equations.extend(([-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                          [0, 0, 0, -x, -y, -1, v*x, v*y, v]))
    _, singular, right = np.linalg.svd(np.asarray(equations), full_matrices=True)
    condition = float(singular[0] / max(singular[7], np.finfo(float).tiny))
    if not np.isfinite(condition) or condition > limits["maximum_dlt_condition"]:
        raise ValueError("degenerate_or_ill_conditioned_tick_correspondences")
    normalized = right[-1].reshape(3, 3)
    if abs(np.linalg.det(normalized)) < 1e-12:
        raise ValueError("singular_homography")
    matrix = np.linalg.inv(source_normalization) @ normalized @ image_normalization
    matrix /= np.linalg.norm(matrix)
    if abs(matrix[2, 2]) > 1e-12:
        matrix /= matrix[2, 2]
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite_homography")
    return matrix, condition


def _project(matrix, points):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = homogeneous @ matrix.T
    denominator = mapped[:, 2]
    relative_scale = np.linalg.norm(matrix[2]) * np.linalg.norm(homogeneous, axis=1)
    if np.any(np.abs(denominator) <= 1e-12 * relative_scale):
        raise ValueError("point_at_projective_horizon")
    if not (np.all(denominator > 0) or np.all(denominator < 0)):
        raise ValueError("projective_horizon_crosses_observed_dial")
    result = mapped[:, :2] / denominator[:, None]
    if not np.isfinite(result).all():
        raise ValueError("nonfinite_rectified_coordinates")
    return result


def read_speed(tick_points, pivot, tip, template=None):
    """Read from observed points, returning accepted=False for invalid geometry.

    tick_points maps names in TICK_NAMES to image [x,y]. Missing values and None
    are omitted. Nonfinite or malformed supplied points reject the reading.
    Caller-provided visibility filtering is supported by leaving occluded ticks
    out; this function never obtains a transform from synthetic label metadata.

    Reprojection errors are reported separately in source and input-image pixels.
    The source-angle endpoints tolerate only floating-point roundoff. Needles
    beyond the calibrated arc are rejected, never clamped to 0 or 180 km/h.
    """
    template = load_template() if template is None else template
    limits = _validate_template(template)
    result = {
        "accepted": False, "speed_kmh": None, "reason": None,
        "image_to_source_homography_3x3": None,
        "rectified_pivot_xy": None, "rectified_tip_xy": None,
        "reprojection_rmse_source_px": None, "reprojection_rmse_input_px": None,
        "maximum_reprojection_error_source_px": None,
        "rectified_pivot_offset_source_px": None,
        "needle_angle_source_deg_unwrapped": None, "calibration_interval_kmh": None,
        "used_tick_names": [], "redundancy_available": False,
        "inferred_60_region": None, "dlt_condition": None,
        "fit_method": "Normalized DLT using all supplied named tick anchors only",
        "scope": "Synthetic-template geometric consistency; not physical-dashboard accuracy",
    }
    try:
        if not isinstance(tick_points, Mapping):
            raise ValueError("tick_points_must_be_a_mapping")
        if set(tick_points) - set(TICK_NAMES):
            raise ValueError("unknown_tick_landmark_name")
        names = [name for name in TICK_NAMES if tick_points.get(name) is not None]
        result.update(used_tick_names=names, redundancy_available=len(names) > 4)
        if len(names) < 4:
            raise ValueError("insufficient_visible_tick_landmarks")
        image_points = np.stack([_point(tick_points[name], name) for name in names])
        source_points = np.asarray([template["tick_points_source"][name] for name in names])
        needle = np.stack([_point(pivot, "pivot"), _point(tip, "tip")])
        if np.linalg.norm(needle[1] - needle[0]) <= 1e-8:
            raise ValueError("zero_length_needle")
        matrix, condition = _estimate_homography(image_points, source_points, limits)
        result.update(image_to_source_homography_3x3=matrix.tolist(), dlt_condition=condition)
        mapped = _project(matrix, np.concatenate([image_points, needle]))
        rectified_pivot, rectified_tip = mapped[-2:]
        result.update(rectified_pivot_xy=rectified_pivot.tolist(), rectified_tip_xy=rectified_tip.tolist())
        inverse = np.linalg.inv(matrix)
        # Verify the entire calibrated source dial remains on one side of the
        # projective horizon, including landmarks currently hidden from view.
        _project(inverse, np.vstack([template["calibration_tick_points_source"],
                                     template["pivot_source_xy"]]))
        pivot_denominator = float(matrix[2] @ np.r_[needle[0], 1.0])
        if np.linalg.det(matrix) / pivot_denominator**3 <= 0:
            raise ValueError("mirrored_or_reversed_tick_correspondences")
        residual_source = np.linalg.norm(mapped[:len(names)] - source_points, axis=1)
        residual_input = np.linalg.norm(_project(inverse, source_points) - image_points, axis=1)
        rmse_source = float(np.sqrt(np.mean(residual_source**2)))
        rmse_input = float(np.sqrt(np.mean(residual_input**2)))
        maximum = float(residual_source.max())
        pivot_offset = float(np.linalg.norm(rectified_pivot - template["pivot_source_xy"]))
        if not np.isfinite([rmse_source, rmse_input, maximum, pivot_offset]).all():
            raise ValueError("nonfinite_geometric_residuals")
        result.update(reprojection_rmse_source_px=rmse_source,
                      reprojection_rmse_input_px=rmse_input,
                      maximum_reprojection_error_source_px=maximum,
                      rectified_pivot_offset_source_px=pivot_offset)
        if (rmse_source > limits["maximum_reprojection_rmse_source_px"]
                or maximum > limits["maximum_reprojection_error_source_px"]):
            raise ValueError("inconsistent_tick_reprojection")
        if pivot_offset > limits["maximum_pivot_offset_source_px"]:
            raise ValueError("needle_pivot_inconsistent_with_template")
        vector = rectified_tip - rectified_pivot
        if np.linalg.norm(vector) <= 1e-8:
            raise ValueError("zero_length_rectified_needle")
        knots = np.asarray(template["angle_knots_deg_unwrapped"])
        angle = float(np.rad2deg(np.arctan2(vector[1], vector[0])))
        angle += 360.0 * round((float((knots[0] + knots[-1]) / 2) - angle) / 360.0)
        result["needle_angle_source_deg_unwrapped"] = angle
        tolerance = limits["endpoint_tolerance_degrees"]
        if angle < knots[0] - tolerance or angle > knots[-1] + tolerance:
            raise ValueError("needle_outside_calibrated_arc")
        # Snapping is confined to the stated numerical epsilon around endpoints.
        if abs(angle - knots[0]) <= tolerance:
            angle = float(knots[0])
        elif abs(angle - knots[-1]) <= tolerance:
            angle = float(knots[-1])
        speeds = np.asarray(template["speed_knots_kmh"])
        speed = float(np.interp(angle, knots, speeds))
        interval = min(max(int(np.searchsorted(knots, angle, side="right")) - 1, 0), len(knots) - 2)
        result.update(accepted=True, speed_kmh=speed, reason="ok",
                      needle_angle_source_deg_unwrapped=angle,
                      calibration_interval_kmh=speeds[interval:interval+2].tolist(),
                      inferred_60_region=bool(50.0 + 1e-7 < speed < 70.0 - 1e-7))
    except (TypeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result["reason"] = str(exc)
    return result
