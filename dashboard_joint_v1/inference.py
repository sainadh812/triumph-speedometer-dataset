"""New dashboard/LCD + speed + RPM pipeline; neural execution is Colab CUDA only.

The numerical helpers are CPU-safe and contain no model execution. All output
coordinates refer to original video pixels. No image matching, old model heads,
fixed camera crops, previous readings, or source-video-specific offsets are used.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

try:
    from .preprocess import (SIZES, POINT_NAMES, letterbox_matrix, dashboard_matrix,
                             rpm_from_roi_matrix, warp_input)
    from .dataset import PREPROCESSING_ID
except ImportError:
    from preprocess import (SIZES, POINT_NAMES, letterbox_matrix, dashboard_matrix,
                            rpm_from_roi_matrix, warp_input)
    from dataset import PREPROCESSING_ID


# Reviewed OUTER-rail intersections from TRAIN night_11, never a test frame.
# The earlier models used different tick conventions and are deliberately unused.
# This is a perspective reference plane, not a metrologically calibrated dial.
# Piecewise angle interpolation is exact at labeled ticks; between-tick physical
# speed accuracy remains unverified, including perspective of this reference.
SPEED_REFERENCE = {
    'source_id': 'night_11', 'split': 'train', 'pivot': [486., 312.],
    'annotation_sha256': 'd003fc9d14ae2973a0f6fe82e4c5512ff0e0b8eecb0b73a5ce0b0e2885939912',
    'ticks': [[331., 472.], [277., 395.], [256., 321.], [266., 239.],
              [306., 166.], [374., 111.], [455., 95.], [534., 108.],
              [608., 145.], [655., 203.]],
    'speeds': list(range(0, 181, 20)),
    'scope': 'Reviewed real training-frame geometry; physical speed accuracy unverified',
}

DEFAULT_THRESHOLDS = {
    'visibility': .5, 'dashboard_presence': .5, 'lcd_mask': .5,
    'minimum_dashboard_side_input_px': 24., 'minimum_lcd_mask_input_px': 24,
    'minimum_roi_area_input_px2': 32.,
    'speed_ransac_reference_px': 10., 'speed_min_tick_span_kmh': 60.,
    'speed_max_pivot_offset_fraction': .30, 'speed_endpoint_tolerance_deg': 3.,
    'rpm_max_curve_distance_input_px': 24., 'rpm_max_anchor_gap': 4000.,
    'rpm_endpoint_tolerance_input_px': 3.,
}


def project_points(points, matrix):
    """Project without silently mapping a projective horizon to [0,0]."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    h = np.asarray(matrix, np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or not np.isfinite(p).all():
        raise ValueError('nonfinite_projective_geometry')
    q = np.c_[p, np.ones(len(p))] @ h.T
    if np.any(np.abs(q[:, 2]) < 1e-8):
        raise ValueError('projective_horizon')
    if len(q) and not (np.all(q[:, 2] > 0) or np.all(q[:, 2] < 0)):
        raise ValueError('projective_horizon_crosses_geometry')
    return q[:, :2] / q[:, 2:3]


def decode_heatmaps(logits, input_hw):
    """Same argmax/pixel-center convention used by training metrics."""
    values = np.asarray(logits)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError('invalid_heatmaps')
    n, k, h, w = values.shape
    indices = values.reshape(n, k, -1).argmax(-1)
    x = (indices % w + .5) * input_hw[1] / w - .5
    y = (indices // w + .5) * input_hw[0] / h - .5
    return np.stack([np.clip(x, 0, input_hw[1]-1), np.clip(y, 0, input_hw[0]-1)], -1)


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(np.asarray(values, np.float64), -60, 60)))


def read_speed(points, probabilities, thresholds=None, reference=None):
    """Fit named ticks to a reviewed plane, then interpolate needle angle.

    RANSAC rejects isolated mistaken ticks; all raw predictions remain available
    to the caller. Four inliers have no redundant consistency evidence.
    Thresholds are declared engineering limits, not calibrated confidence.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    ref = reference or SPEED_REFERENCE
    result = {'value_kmh': None, 'reason': None, 'diagnostics': {
        'method': 'predicted_tick_homography_then_reference_angle_interpolation',
        'reference_source_id': ref.get('source_id'),
        'reference_annotation_sha256': ref.get('annotation_sha256'), 'physical_accuracy_verified': False}}
    d = result['diagnostics']
    try:
        p = np.asarray(points, np.float64)
        v = np.asarray(probabilities, np.float64)
        if p.shape != (12, 2) or v.shape != (12,) or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite_speed_landmarks')
        if np.any(v[:2] < limits['visibility']):
            raise ValueError('needle_pivot_or_tip_not_visible')
        selected = np.flatnonzero(v[2:] >= limits['visibility'])
        d['visible_tick_count'] = int(len(selected))
        if len(selected) < 4:
            raise ValueError('fewer_than_four_visible_speed_ticks')
        rp = np.asarray(ref['ticks'], np.float64)
        speeds = np.asarray(ref['speeds'], np.float64)
        center = np.asarray(ref['pivot'], np.float64)
        if np.ptp(speeds[selected]) < limits['speed_min_tick_span_kmh']:
            raise ValueError('speed_ticks_cover_too_short_an_arc')
        src, dst = p[2:][selected], rp[selected]
        singular = np.linalg.svd(src - src.mean(0), compute_uv=False)
        if singular[0] < 1 or singular[-1] / singular[0] < .03:
            raise ValueError('degenerate_speed_tick_geometry')
        h, flags = cv2.findHomography(src, dst, cv2.RANSAC,
                                      limits['speed_ransac_reference_px'], maxIters=1000, confidence=.995)
        if h is None or flags is None or not np.isfinite(h).all():
            raise ValueError('speed_homography_failed')
        inside = flags.ravel().astype(bool)
        used = selected[inside]
        d.update(inlier_tick_count=int(len(used)), used_tick_speeds=speeds[used].tolist(),
                 rejected_tick_speeds=speeds[selected[~inside]].tolist(), redundancy_available=len(used) > 4)
        if len(used) < max(4, math.ceil(len(selected)*.5)):
            raise ValueError('too_few_consistent_speed_ticks')
        if np.ptp(speeds[used]) < limits['speed_min_tick_span_kmh']:
            raise ValueError('consistent_ticks_cover_too_short_an_arc')
        needle = project_points(p[:2], h)
        mapped_ticks = project_points(src[inside], h)
        project_points(np.vstack([rp, center]), np.linalg.inv(h))
        radius = float(np.median(np.linalg.norm(rp-center, axis=1)))
        offset = float(np.linalg.norm(needle[0]-center))
        d.update(image_to_reference_homography=h.tolist(), reference_pivot_offset_px=offset,
                 reference_tick_rmse_px=float(np.sqrt(np.mean(np.sum((mapped_ticks-dst[inside])**2, axis=1)))))
        if offset > limits['speed_max_pivot_offset_fraction']*radius:
            raise ValueError('predicted_pivot_inconsistent_with_tick_plane')
        direction = needle[1]-needle[0]
        if np.linalg.norm(direction) < .05*radius:
            raise ValueError('needle_is_too_short')
        knots = np.unwrap(np.arctan2((rp-center)[:, 1], (rp-center)[:, 0]))
        if np.any(np.diff(knots) <= 0) or np.ptp(knots) >= 2*np.pi:
            raise ValueError('invalid_speed_reference_order')
        angle = math.atan2(direction[1], direction[0])
        angle += 2*np.pi*round(((knots[0]+knots[-1])/2-angle)/(2*np.pi))
        tolerance = math.radians(limits['speed_endpoint_tolerance_deg'])
        d['reference_needle_angle_deg'] = float(np.rad2deg(angle))
        if angle < knots[0]-tolerance or angle > knots[-1]+tolerance:
            raise ValueError('needle_outside_calibrated_speed_arc')
        bounded = float(np.clip(angle, knots[0], knots[-1]))
        d['endpoint_snapped_within_tolerance'] = bounded != angle
        result.update(value_kmh=float(np.interp(bounded, knots, speeds)), reason=None)
    except (ValueError, cv2.error, np.linalg.LinAlgError, OverflowError) as exc:
        result['reason'] = str(exc)
    return result


def read_rpm(points, probabilities, thresholds=None):
    """Interpolate the observed endpoint along the predicted numeric scale.

    Input is the rectified RPM model plane, so thresholds are resolution-stable.
    Values are indicated RPM estimates, not independently measured engine RPM.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    result = {'value': None, 'reason': None, 'diagnostics': {'method': 'predicted_scale_polyline_interpolation',
                                                          'physical_accuracy_verified': False}}
    try:
        p, v = np.asarray(points, np.float64), np.asarray(probabilities, np.float64)
        if p.shape != (8, 2) or v.shape != (8,) or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite_rpm_landmarks')
        if v[7] < limits['visibility']:
            raise ValueError('rpm_fill_endpoint_not_visible')
        selected = np.flatnonzero(v[:7] >= limits['visibility'])
        if len(selected) < 2:
            raise ValueError('fewer_than_two_visible_rpm_anchors')
        candidates = []
        for a, b in zip(selected[:-1], selected[1:]):
            vector = p[b]-p[a]
            length = float(np.linalg.norm(vector))
            if length < 4 or (b-a)*2000 > limits['rpm_max_anchor_gap']:
                continue
            raw_t = float(np.dot(p[7]-p[a], vector)/(length*length))
            t = float(np.clip(raw_t, 0., 1.))
            distance = float(np.linalg.norm(p[7]-(p[a]+t*vector)))
            candidates.append((distance, int(a), int(b), raw_t, t, length))
        if not candidates:
            raise ValueError('rpm_scale_missing_usable_anchor_intervals')
        distance, a, b, raw_t, t, length = min(candidates)
        result['diagnostics'].update(curve_distance_input_px=distance, interval_rpm=[a*2000, b*2000],
                                     visible_anchor_count=int(len(selected)), endpoint_snapped=t != raw_t)
        if distance > limits['rpm_max_curve_distance_input_px']:
            raise ValueError('rpm_endpoint_too_far_from_scale')
        overshoot = abs(raw_t-t)*length
        if overshoot > limits['rpm_endpoint_tolerance_input_px']:
            raise ValueError('rpm_endpoint_outside_visible_anchor_interval')
        result['value'] = float((a+(b-a)*t)*2000)
    except (ValueError, OverflowError) as exc:
        result['reason'] = str(exc)
    return result


def validated_roi(corners, probabilities, thresholds=None):
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    q = np.asarray(corners, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError('invalid_rpm_roi_points')
    if np.any(np.asarray(probabilities) < limits['visibility']):
        raise ValueError('rpm_roi_corner_not_visible')
    if not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < limits['minimum_roi_area_input_px2']:
        raise ValueError('degenerate_rpm_roi')
    return q


def select_lcd_mask(probability, dashboard_box, thresholds=None):
    """Keep one learned LCD component inside the detected dashboard."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    value = np.asarray(probability)
    h, w = value.shape
    x0, y0, x1, y1 = np.asarray(dashboard_box, float)
    region = np.zeros((h, w), np.uint8)
    lo = np.maximum(np.floor([x0, y0]).astype(int), [0, 0])
    hi = np.minimum(np.ceil([x1, y1]).astype(int)+1, [w, h])
    if np.any(hi <= lo):
        return region, []
    region[lo[1]:hi[1], lo[0]:hi[0]] = (value[lo[1]:hi[1], lo[0]:hi[0]] >= limits['lcd_mask']).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region, 8)
    if count < 2:
        return region*0, []
    best = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == best).astype(np.uint8)*255
    if stats[best, cv2.CC_STAT_AREA] < limits['minimum_lcd_mask_input_px']:
        return mask*0, []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, .75, True).reshape(-1, 2).astype(float)
    return mask, polygon.tolist()


def _colab_guard():
    try:
        available = importlib.util.find_spec('google.colab') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not (available or os.environ.get('COLAB_RELEASE_TAG')):
        raise RuntimeError('Neural inference is restricted to Google Colab CUDA; no local execution')


class DashboardPredictor:
    """Load three NEW selected checkpoints and process every supplied frame.

    Large batches are split after CUDA OOM, never sent to a CPU model fallback.
    Checkpoint identity must agree across tasks and with this architecture.
    """
    def __init__(self, checkpoints, batch_size=32, thresholds=None):
        _colab_guard()  # Before importing torch or inspecting any weight file.
        import torch
        try:
            from .model import InstrumentModel, require_colab_cuda, ARCHITECTURE_ID
        except ImportError:
            from model import InstrumentModel, require_colab_cuda, ARCHITECTURE_ID
        self.torch, self.device = torch, require_colab_cuda()
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        if set(self.thresholds) != set(DEFAULT_THRESHOLDS) or any(not math.isfinite(float(v)) or v < 0 for v in self.thresholds.values()):
            raise ValueError('Unknown or invalid inference threshold')
        self.models, self.metadata = {}, {}
        identities = set()
        for task in ('localizer', 'speed', 'rpm'):
            payload = torch.load(Path(checkpoints[task]), map_location='cpu', weights_only=True)
            identity = payload['identity']
            expected_hw = list(SIZES[task][::-1])
            if (payload.get('checkpoint_kind') != 'dashboard_joint_v1_inference' or payload.get('format_version') != 1
                    or identity.get('task') != task or identity.get('architecture_id') != ARCHITECTURE_ID
                    or identity.get('preprocessing_id') != PREPROCESSING_ID
                    or identity.get('point_names') != POINT_NAMES[task] or identity.get('input_hw') != expected_hw):
                raise ValueError('New-model checkpoint identity mismatch: '+task)
            identities.add(identity['dataset_manifest_sha256'])
            model = InstrumentModel(task, expected_hw, pretrained=False).to(self.device)
            model.load_state_dict(payload['model_state_dict'], strict=True)
            model.eval()
            self.models[task] = model
            self.metadata[task] = {'best_epoch': payload['best_epoch'], 'identity': identity}
            del payload
        if len(identities) != 1:
            raise ValueError('The three checkpoints must belong to the same new dataset')
        self.last_batch_timing = {}
        self.effective_batch_size = {task: self.batch_size for task in self.models}
        self._mean = torch.tensor([.485, .456, .406], device=self.device).reshape(1, 3, 1, 1)
        self._std = torch.tensor([.229, .224, .225], device=self.device).reshape(1, 3, 1, 1)
        torch.set_num_threads(min(torch.get_num_threads(), 2))

    def _run(self, task, images):
        if not images:
            return []
        torch, results = self.torch, []
        cursor = 0
        while cursor < len(images):
            size = min(self.effective_batch_size[task], len(images)-cursor)
            tensor = output = logits = indices = x = y = None
            try:
                # Upload one uint8 gray channel, then do identical training
                # normalization on CUDA. Avoid 12x larger CPU float uploads.
                array = np.stack(images[cursor:cursor+size])
                tensor = torch.from_numpy(array).to(self.device).float().div_(255.)[:, None]
                tensor = (tensor.expand(-1, 3, -1, -1)-self._mean)/self._std
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = self.models[task](tensor)
                # Argmax on CUDA avoids copying dense landmark heatmaps to CPU.
                logits = output['heatmap_logits']
                h, w = logits.shape[-2:]
                indices = logits.flatten(2).argmax(-1)
                input_w, input_h = SIZES[task]
                x = ((indices % w+.5)*input_w/w-.5).clamp(0, input_w-1)
                y = ((indices // w+.5)*input_h/h-.5).clamp(0, input_h-1)
                decoded = torch.stack([x, y], -1).float().cpu().numpy()
                names = ['visibility_logits']
                if task == 'localizer':
                    names += ['lcd_logits', 'dashboard_box', 'presence_logits']
                values = {k: output[k].float().cpu().numpy() for k in names}
                vis = sigmoid(values['visibility_logits'])
                for index in range(size):
                    row = {'points': decoded[index], 'visibility': vis[index]}
                    if task == 'localizer':
                        row.update(mask_probability=sigmoid(values['lcd_logits'][index, 0]),
                                   box=values['dashboard_box'][index]*[512, 512, 512, 512],
                                   presence=float(sigmoid(values['presence_logits'][index])))
                    results.append(row)
                cursor += size
                del tensor, output, logits, indices, x, y
            except torch.cuda.OutOfMemoryError:
                del tensor, output, logits, indices, x, y
                torch.cuda.empty_cache()
                if size == 1:
                    raise RuntimeError('CUDA OOM at batch 1; preserve outputs and release runtime')
                self.effective_batch_size[task] = max(1, size//2)
        return results

    def predict_frames(self, frames):
        started = time.perf_counter()
        if not frames:
            self.last_batch_timing = {'frames': 0, 'total_seconds': 0.}
            return []
        local_inputs, matrices = [], []
        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError('Each frame must be a uint8 BGR image')
            h, w = frame.shape[:2]
            matrix = letterbox_matrix([0, 0, w, h], SIZES['localizer'])
            local_inputs.append(warp_input(frame, matrix, 'localizer')[0])
            matrices.append(matrix)
        local_start = time.perf_counter()
        local = self._run('localizer', local_inputs)
        local_seconds = time.perf_counter()-local_start
        rows, speed_inputs, rpm_inputs, speed_pending, rpm_pending = [], [], [], [], []
        for index, (frame, prediction, matrix) in enumerate(zip(frames, local, matrices)):
            inv = np.linalg.inv(matrix)
            original_points = project_points(prediction['points'], inv)
            box = prediction['box']
            box_quad = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
            source_box = project_points(box_quad, inv)
            mask, polygon = select_lcd_mask(prediction['mask_probability'], box, self.thresholds)
            original_polygon = project_points(polygon, inv).tolist() if polygon else []
            row = {'coordinate_space': 'original_frame_pixels', 'frame_size_wh': [frame.shape[1], frame.shape[0]],
                   'dashboard_box': [*source_box.min(0).tolist(), *source_box.max(0).tolist()],
                   'lcd_polygon': original_polygon, 'lcd_corners': original_points[:4].tolist(),
                   'rpm_roi_corners': original_points[4:].tolist(),
                   'localizer': {'presence_probability': prediction['presence'],
                                 'lcd_status': 'detected' if polygon else 'unavailable',
                                 'point_visibility': dict(zip(POINT_NAMES['localizer'], prediction['visibility'].tolist())),
                                 'lcd_mask_pixels_input': int(np.count_nonzero(mask))},
                   'speed': {'value_kmh': None, 'reason': 'not_processed', 'pivot': None, 'tip': None, 'ticks': {}, 'point_visibility': {}, 'diagnostics': {}},
                   'rpm': {'value': None, 'reason': 'not_processed', 'anchors': {}, 'bar_end': None, 'point_visibility': {}, 'diagnostics': {}}}
            rows.append(row)
            if (prediction['presence'] < self.thresholds['dashboard_presence'] or
                    np.min(np.array(box[2:])-box[:2]) < self.thresholds['minimum_dashboard_side_input_px']):
                row['speed']['reason'] = row['rpm']['reason'] = 'dashboard_not_confidently_localized'
                row['localizer']['lcd_status'] = 'unavailable_dashboard'
                continue
            # Decode original pixels once, mask in SPEED coordinates with the exact
            # learned mask, retaining the same gray128 convention as training.
            speed_matrix = dashboard_matrix(source_box, frame.shape)
            speed_image = warp_input(frame, speed_matrix, 'speed')[0]
            speed_mask = cv2.warpPerspective(mask, speed_matrix@inv, SIZES['speed'], flags=cv2.INTER_NEAREST)
            speed_image[speed_mask > 0] = 128
            speed_inputs.append(speed_image)
            speed_pending.append((index, speed_matrix))
            row['speed']['diagnostics']['lcd_mask_available'] = bool(polygon)
            try:
                validated_roi(prediction['points'][4:], prediction['visibility'][4:], self.thresholds)
                rpm_matrix = rpm_from_roi_matrix(original_points[4:])
                project_points(original_points[4:], rpm_matrix)
                rpm_inputs.append(warp_input(frame, rpm_matrix, 'rpm')[0])
                rpm_pending.append((index, rpm_matrix))
            except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
                row['rpm']['reason'] = str(exc)
        speed_start = time.perf_counter()
        speed_predictions = self._run('speed', speed_inputs)
        speed_seconds = time.perf_counter()-speed_start
        for (index, matrix), pred in zip(speed_pending, speed_predictions):
            points = project_points(pred['points'], np.linalg.inv(matrix))
            decoded = read_speed(points, pred['visibility'], self.thresholds)
            old_diagnostics = rows[index]['speed']['diagnostics']
            decoded['diagnostics'].update(old_diagnostics)
            decoded.update(pivot=points[0].tolist(), tip=points[1].tolist(),
                           ticks={str(n): points[2+i].tolist() for i, n in enumerate(range(0, 181, 20))},
                           point_visibility=dict(zip(POINT_NAMES['speed'], pred['visibility'].tolist())))
            rows[index]['speed'] = decoded
        rpm_start = time.perf_counter()
        rpm_predictions = self._run('rpm', rpm_inputs)
        rpm_seconds = time.perf_counter()-rpm_start
        for (index, matrix), pred in zip(rpm_pending, rpm_predictions):
            points = project_points(pred['points'], np.linalg.inv(matrix))
            decoded = read_rpm(pred['points'], pred['visibility'], self.thresholds)
            decoded.update(anchors={str(n): points[i].tolist() for i, n in enumerate(range(0, 13, 2))},
                           bar_end=points[7].tolist(),
                           point_visibility=dict(zip(POINT_NAMES['rpm'], pred['visibility'].tolist())))
            rows[index]['rpm'] = decoded
        self.torch.cuda.synchronize()
        total = time.perf_counter()-started
        self.last_batch_timing = {'frames': len(frames), 'total_seconds': total,
                                 'localizer_prepare_predict_seconds': local_seconds,
                                 'speed_prepare_predict_seconds': speed_seconds,
                                 'rpm_prepare_predict_seconds': rpm_seconds,
                                 'effective_batch_sizes': dict(self.effective_batch_size),
                                 'note': 'Stage timings include normalization/transfers; total includes all geometry and crop preparation. Excludes video decode/encode.'}
        return rows
