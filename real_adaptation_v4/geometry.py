"""Perspective-aware speed from predicted semantic ticks and needle endpoints.

Inputs are 12 predicted points in pixel coordinates: pivot, needle_tip, then
tick_0 through tick_180 in steps of 20. No image-specific labels or rendering
transform are accepted by this API. A robust homography is estimated solely
from the available predicted semantic ticks. Landmark availability is amodal:
a tick covered by the needle still has a supervised physical location. Quality
gates measure geometric
consistency, not correctness or calibrated model confidence. Coherently wrong
semantic correspondences can remain geometrically plausible.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import cv2
import numpy as np

TICK_NAMES = tuple(f'tick_{speed}' for speed in range(0, 181, 20))
LANDMARK_NAMES = ('pivot', 'needle_tip') + TICK_NAMES
KEYPOINT_NAMES = LANDMARK_NAMES
SPEEDS = tuple(range(0, 181, 20))
ENDPOINT_TOLERANCE_KMH = 2.0
DEFAULT_LIMITS = {
    'visibility_threshold': .5,
    'minimum_tick_inliers': 6,
    'minimum_inlier_fraction': .75,
    'maximum_tick_outliers': 2,
    'minimum_speed_support_span': 120.,
    'maximum_lowest_supported_speed': 40.,
    'minimum_highest_supported_speed': 140.,
    'minimum_axis_ratio': .06,
    'minimum_tick_separation_px': 3.,
    'maximum_dlt_condition': 1e5,
    'ransac_threshold_canonical_px': 4.5,
    'maximum_inlier_rmse_canonical_px': 3.5,
    'maximum_inlier_error_canonical_px': 5.5,
    'maximum_leave_one_out_error_canonical_px': 8.,
    'maximum_pivot_offset_canonical_px': 16.,
    'minimum_needle_length_canonical_px': 40.,
    'maximum_needle_length_canonical_px': 200.,
    'endpoint_tolerance_kmh': ENDPOINT_TOLERANCE_KMH,
    'allow_reflection': False,
    'require_endpoint_ticks': True,
}


def _calibration(value):
    """Validate raw V3/V4 calibration or the result of load_calibration."""
    if not isinstance(value, Mapping):
        raise ValueError('Calibration must be a mapping')
    if 'ticks_xy' in value and 'speeds' in value:
        pivot = np.asarray(value['pivot_xy'], dtype=float)
        ticks = np.asarray(value['ticks_xy'], dtype=float)
        speeds = np.asarray(value['speeds'], dtype=float)
        raw = value.get('raw', value)
    else:
        pivot = np.asarray(value['pivot_xy'], dtype=float)
        rows = sorted(value['ticks'], key=lambda row: row['speed_kmh'])
        ticks = np.asarray([row['xy'] for row in rows], dtype=float)
        speeds = np.asarray([row['speed_kmh'] for row in rows], dtype=float)
        raw = value
    if (pivot.shape != (2,) or ticks.shape != (10, 2)
            or not np.isfinite(pivot).all() or not np.isfinite(ticks).all()
            or speeds.tolist() != list(SPEEDS)):
        raise ValueError('Expected finite pivot and ten canonical ticks from 0 to 180')
    angles = np.unwrap(np.arctan2(ticks[:, 1] - pivot[1], ticks[:, 0] - pivot[0]))
    if not np.all(np.diff(angles) > 0) or not 0 < angles[-1]-angles[0] < 2*math.pi:
        raise ValueError('Canonical calibration angles must increase in speed order')
    return {'pivot_xy': pivot.tolist(), 'ticks_xy': ticks.tolist(),
            'speeds': speeds.tolist(), 'angles': angles.tolist(), 'raw': dict(raw)}


def load_calibration(path):
    return _calibration(json.loads(Path(path).read_text(encoding='utf-8')))


def _default_calibration():
    local = Path(__file__).resolve().parent / 'calibration.json'
    if not local.is_file():
        local = Path(__file__).resolve().parents[1] / 'real_adaptation_v3/annotations/reference_ticks.json'
    return load_calibration(local)


def _project(points, homography):
    points = np.asarray(points, dtype=float)
    q = np.c_[points, np.ones(len(points))] @ homography.T
    scale = np.linalg.norm(homography[2]) * np.linalg.norm(np.c_[points, np.ones(len(points))], axis=1)
    if np.any(np.abs(q[:, 2]) <= 1e-9 * np.maximum(scale, 1e-12)):
        raise ValueError('point_at_projective_horizon')
    if not (np.all(q[:, 2] > 0) or np.all(q[:, 2] < 0)):
        raise ValueError('projective_horizon_crosses_observed_dial')
    mapped = q[:, :2] / q[:, 2, None]
    if not np.isfinite(mapped).all():
        raise ValueError('nonfinite_rectified_coordinates')
    return mapped


def _spread(points, limits):
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] < 1e-9 or singular[1] / singular[0] < limits['minimum_axis_ratio']:
        raise ValueError('degenerate_or_poorly_spread_tick_points')
    separation = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(separation, np.inf)
    if separation.min() < limits['minimum_tick_separation_px']:
        raise ValueError('duplicate_or_unresolved_tick_points')
    radius = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
    return centered * (np.sqrt(2) / radius)


def _dlt_condition(source, target, limits):
    source = _spread(source, limits)
    target = _spread(target, limits)
    matrix = []
    for (x, y), (u, v) in zip(source, target):
        matrix.extend([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                       [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    condition = float(singular[0] / max(singular[7], np.finfo(float).tiny))
    if not np.isfinite(condition) or condition > limits['maximum_dlt_condition']:
        raise ValueError('ill_conditioned_tick_correspondences')
    return condition


def _support(indices, limits):
    speeds = np.asarray(SPEEDS)[indices]
    if (speeds[-1] - speeds[0] < limits['minimum_speed_support_span']
            or speeds[0] > limits['maximum_lowest_supported_speed']
            or speeds[-1] < limits['minimum_highest_supported_speed']):
        raise ValueError('insufficient_arc_support')


def _limits(overrides):
    limits = {**DEFAULT_LIMITS, **(overrides or {})}
    if set(limits) != set(DEFAULT_LIMITS):
        raise ValueError('Unknown geometry limit')
    for name, value in limits.items():
        if name in ('allow_reflection', 'require_endpoint_ticks'):
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean')
        elif not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid geometry limit: {name}')
    if not 0 < limits['visibility_threshold'] <= 1 or not 0 < limits['minimum_inlier_fraction'] <= 1:
        raise ValueError('Probability/fraction limits must be in (0,1]')
    if not 6 <= limits['minimum_tick_inliers'] <= 10:
        raise ValueError('At least six tick inliers are required for redundant checks')
    return limits


def read_speed(predicted_points_xy, visibility=None, calibration=None, *, limits=None):
    """Decode 12 predicted landmarks; return explicit rejection instead of speed.

    visibility is the legacy API name for optional length12 (all landmarks)
    or length10 (ticks only) AMODAL landmark-availability probability/bool array.
    It does not mean directly observable stroke pixels: a needle-covered tick
    is still available when its in-frame location can be inferred from the
    dial. None means all predictions are available; it does not assert model
    confidence. Coordinates of unavailable ticks may be nonfinite.
    The default excludes reflections because this experiment has no mirrored
    camera images. In-plane rotation, translation, scale and projectivity do
    not have a fixed camera-displacement constraint. Both semantic endpoints
    are required by default: an almost uniform dial permits a coherently
    shifted tick labelling to mimic a different rotation if endpoints are
    silently omitted. An available endpoint may never be discarded as an
    outlier. Direct stroke visibility is separate dataset-audit metadata and
    is deliberately not an input to this decoder.
    """
    cal = _default_calibration() if calibration is None else _calibration(calibration)
    limits = _limits(limits)
    result = {
        'accepted': False, 'speed_kmh': None, 'reason': None,
        'raw_indicated_speed_kmh': None, 'endpoint_clamped': False,
        'endpoint_tolerance_kmh': float(limits['endpoint_tolerance_kmh']),
        'image_to_canonical_h': None, 'rectified_pivot_xy': None,
        'rectified_tip_xy': None, 'needle_angle_degrees_unwrapped': None,
        'available_tick_names': [], 'inlier_tick_names': [], 'outlier_tick_names': [],
        'inlier_count': 0, 'inlier_fraction': None, 'support_speed_range_kmh': None,
        'inlier_rmse_canonical_px': None, 'maximum_inlier_error_canonical_px': None,
        'inlier_rmse_input_px': None, 'maximum_leave_one_out_error_canonical_px': None,
        'pivot_offset_canonical_px': None, 'needle_length_canonical_px': None,
        'dlt_condition': None, 'orientation_preserved': None,
        'method': 'predicted_semantic_ticks_RANSAC_homography_then_predicted_needle_angle',
        'uses_ground_truth_transform': False,
        'limitation': 'Geometric consistency is not calibrated confidence or proof of correct semantic tick labels.',
    }
    try:
        points = np.asarray(predicted_points_xy, dtype=float)
        if points.shape != (12, 2):
            raise ValueError('expected_twelve_landmarks')
        probabilities = np.ones(12, dtype=float)
        if visibility is not None:
            supplied = np.asarray(visibility, dtype=float)
            if supplied.shape == (10,):
                probabilities[2:] = supplied
            elif supplied.shape == (12,):
                probabilities = supplied
            else:
                raise ValueError('invalid_visibility_shape')
            if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError('invalid_visibility_probabilities')
        visible = probabilities >= limits['visibility_threshold']
        if not visible[:2].all():
            raise ValueError('needle_landmark_unavailable')
        if not np.isfinite(points[visible]).all():
            raise ValueError('nonfinite_available_landmark')
        ids = np.flatnonzero(visible[2:])
        result['available_tick_names'] = [TICK_NAMES[i] for i in ids]
        if len(ids) < limits['minimum_tick_inliers']:
            raise ValueError('insufficient_available_ticks')
        if limits['require_endpoint_ticks'] and not (visible[2] and visible[11]):
            raise ValueError('semantic_endpoint_ticks_unavailable')
        _support(ids, limits)
        source = points[2:][ids]
        canonical = np.asarray(cal['ticks_xy'])[ids]
        _spread(source, limits)
        cv2.setRNGSeed(0)
        h, ransac_mask = cv2.findHomography(source, canonical, cv2.RANSAC,
            limits['ransac_threshold_canonical_px'], maxIters=3000, confidence=.999)
        if h is None or ransac_mask is None or not np.isfinite(h).all():
            raise ValueError('tick_homography_estimation_failed')
        selected = ransac_mask.ravel().astype(bool)
        inliers = ids[selected]
        fraction = len(inliers) / len(ids)
        result.update(inlier_count=len(inliers), inlier_fraction=float(fraction),
            inlier_tick_names=[TICK_NAMES[i] for i in inliers],
            outlier_tick_names=[TICK_NAMES[i] for i in ids[~selected]])
        if len(inliers) < limits['minimum_tick_inliers']:
            raise ValueError('insufficient_tick_inliers')
        if fraction < limits['minimum_inlier_fraction'] or len(ids)-len(inliers) > limits['maximum_tick_outliers']:
            raise ValueError('too_many_inconsistent_tick_predictions')
        if any(endpoint in ids and endpoint not in inliers for endpoint in (0, 9)):
            raise ValueError('semantic_endpoint_tick_inconsistent')
        _support(inliers, limits)
        result['support_speed_range_kmh'] = [int(SPEEDS[inliers[0]]), int(SPEEDS[inliers[-1]])]
        a, b = source[selected], canonical[selected]
        result['dlt_condition'] = _dlt_condition(a, b, limits)
        h, _ = cv2.findHomography(a, b, 0)
        if h is None or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError('singular_tick_homography')
        h /= np.linalg.norm(h)
        # The projective denominator must retain one sign over the observed
        # dial and needle. The sign-normalized Jacobian determines handedness.
        all_observed = np.vstack([points[:2], a])
        rectified = _project(all_observed, h)
        denominator = float(np.r_[points[0], 1.] @ h[2])
        orientation = bool(np.linalg.det(h) / denominator**3 > 0)
        result['orientation_preserved'] = orientation
        if not orientation and not limits['allow_reflection']:
            raise ValueError('reflected_or_reversed_semantic_ticks')
        residual = np.linalg.norm(rectified[2:] - b, axis=1)
        rmse, maximum = float(np.sqrt(np.mean(residual**2))), float(residual.max())
        result.update(inlier_rmse_canonical_px=rmse, maximum_inlier_error_canonical_px=maximum)
        if rmse > limits['maximum_inlier_rmse_canonical_px'] or maximum > limits['maximum_inlier_error_canonical_px']:
            raise ValueError('tick_reprojection_error_too_large')
        inverse = np.linalg.inv(h)
        input_residual = np.linalg.norm(_project(b, inverse) - a, axis=1)
        result['inlier_rmse_input_px'] = float(np.sqrt(np.mean(input_residual**2)))
        # Six or more correspondences leave redundancy. Check each withheld
        # point against a refit, so an unstable exact fit cannot pass silently.
        loo = []
        for index in range(len(a)):
            keep = np.arange(len(a)) != index
            _dlt_condition(a[keep], b[keep], limits)
            held_h, _ = cv2.findHomography(a[keep], b[keep], 0)
            if held_h is None or not np.isfinite(held_h).all():
                raise ValueError('unstable_leave_one_tick_out_fit')
            loo.append(float(np.linalg.norm(_project(a[index:index+1], held_h)[0] - b[index])))
        result['maximum_leave_one_out_error_canonical_px'] = max(loo)
        if max(loo) > limits['maximum_leave_one_out_error_canonical_px']:
            raise ValueError('unstable_leave_one_tick_out_fit')
        p, t = rectified[:2]
        offset = float(np.linalg.norm(p - np.asarray(cal['pivot_xy'])))
        length = float(np.linalg.norm(t-p))
        result.update(image_to_canonical_h=h.tolist(), rectified_pivot_xy=p.tolist(),
            rectified_tip_xy=t.tolist(), pivot_offset_canonical_px=offset,
            needle_length_canonical_px=length)
        if offset > limits['maximum_pivot_offset_canonical_px']:
            raise ValueError('pivot_inconsistent_with_tick_calibration')
        if not limits['minimum_needle_length_canonical_px'] < length < limits['maximum_needle_length_canonical_px']:
            raise ValueError('implausible_needle_length')
        # Remaining semantic tick predictions must occur in the expected order
        # around the rectified *predicted* pivot, rather than a target pivot.
        observed_angles = np.unwrap(np.arctan2(rectified[2:, 1]-p[1], rectified[2:, 0]-p[0]))
        if not np.all(np.diff(observed_angles) > 0) or observed_angles[-1]-observed_angles[0] >= 2*math.pi:
            raise ValueError('inconsistent_semantic_tick_order')
        ticks = np.asarray(cal['ticks_xy'])
        angles = np.unwrap(np.arctan2(ticks[:, 1]-p[1], ticks[:, 0]-p[0]))
        if not np.all(np.diff(angles) > 0):
            raise ValueError('nonmonotonic_reference_tick_angles')
        theta = math.atan2(*(t-p)[::-1])
        theta += round(((angles[0]+angles[-1])/2-theta)/(2*math.pi))*2*math.pi
        speeds = np.asarray(cal['speeds'])
        if theta < angles[0]:
            raw = speeds[0] + (theta-angles[0])*(speeds[1]-speeds[0])/(angles[1]-angles[0])
        elif theta > angles[-1]:
            raw = speeds[-1] + (theta-angles[-1])*(speeds[-1]-speeds[-2])/(angles[-1]-angles[-2])
        else:
            raw = np.interp(theta, angles, speeds)
        result.update(raw_indicated_speed_kmh=float(raw), needle_angle_degrees_unwrapped=math.degrees(theta))
        tolerance = limits['endpoint_tolerance_kmh']
        if raw < speeds[0]-tolerance-1e-7 or raw > speeds[-1]+tolerance+1e-7:
            raise ValueError('needle_outside_calibrated_arc')
        clamped = bool(raw < speeds[0]-1e-7 or raw > speeds[-1]+1e-7)
        result.update(accepted=True, speed_kmh=float(np.clip(raw, speeds[0], speeds[-1])),
            endpoint_clamped=clamped, reason='accepted_with_endpoint_tolerance' if clamped else 'accepted')
    except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error) as error:
        result['reason'] = str(error) if isinstance(error, ValueError) else 'invalid_or_unstable_landmark_geometry'
    return result


decode_speed = read_speed
