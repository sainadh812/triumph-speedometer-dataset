"""Learned current-frame LCD mask followed by the unchanged V5 speed model.

No SIFT, reference pose, image warp, source annotation, or previous-frame
fallback participates in prediction. A segmentation score and downstream
geometry checks are not guarantees that an incorrect mask preserved a needle.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIZE = 512
PIPELINE_ID = 'learned_lcd_mask128_to_frozen_v5_speed_v1'
LCD_PREPROCESSING_ID = 'gray512_fullframe_letterbox_unmasked_lcd_seg_v1_imagenet_normalized'
SPEED_CHECKPOINT_SHA256 = 'de08c9772c4eacc2b78f9037ff6c45c980d63ee6223739199b4ad28fc4777e1a'
SPEED_CALIBRATION_SHA256 = '2581a049a418d0f337a9997dcd6d196aef7d49d4d9194acc2bc176f14441d2f5'
DEFAULT_GATE_CONFIG = {
    'pixel_threshold': .70,
    'min_visible_pixels': 350,
    'max_sensor_area_fraction': .45,
    'min_largest_component_fraction': .95,
    'min_component_mean_probability': .80,
    'max_boundary_uncertain_fraction': .65,
    'uncertainty_low': .25,
    'uncertainty_high': .75,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def project(points, matrix):
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError('Invalid coordinate transform shape')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite coordinate transform')
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-8):
        raise ValueError('Invalid projective denominator')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def sensor_mask(footprint):
    """Accept a binary valid-pixel map or the runtime letterbox polygon."""
    footprint = np.asarray(footprint)
    if footprint.shape == (SIZE, SIZE):
        if not np.isfinite(footprint).all() or not np.isin(footprint, [0, 1, 255]).all():
            raise ValueError('Sensor map must be binary')
        return footprint > 0
    if footprint.ndim != 2 or footprint.shape[1] != 2 or len(footprint) < 3 or not np.isfinite(footprint).all():
        raise ValueError('Invalid sensor footprint')
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.fillPoly(mask, [np.rint(footprint).astype(np.int32)], 1)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    mask[(xx < footprint[:, 0].min()) | (xx > footprint[:, 0].max()) |
         (yy < footprint[:, 1].min()) | (yy > footprint[:, 1].max())] = 0
    return mask.astype(bool)


def letterbox(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim not in (2, 3):
        raise ValueError('Expected uint8 gray/BGR image')
    if frame.ndim == 3 and frame.shape[2] != 3:
        raise ValueError('Expected three BGR channels')
    height, width = frame.shape[:2]
    if min(height, width) < 2:
        raise ValueError('Image is too small')
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    square = cv2.copyMakeBorder(gray, top, side-height-top, left, side-width-left,
                                cv2.BORDER_CONSTANT, value=0)
    gray512 = cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    scale = SIZE / side
    matrix = np.array([[scale, 0, (left+.5)*scale-.5],
                       [0, scale, (top+.5)*scale-.5], [0, 0, 1]], float)
    footprint = project([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], matrix)
    return gray512, matrix, sensor_mask(footprint)


def validate_gate_config(config):
    if not isinstance(config, dict) or set(config) != set(DEFAULT_GATE_CONFIG):
        raise ValueError('Checkpoint must contain the complete supported LCD gate configuration')
    cfg = {key: float(value) for key, value in config.items()}
    if not all(np.isfinite(value) for value in cfg.values()):
        raise ValueError('Nonfinite LCD gate configuration')
    if not .5 <= cfg['pixel_threshold'] < 1 or not 0 < cfg['min_visible_pixels'] < SIZE*SIZE:
        raise ValueError('Invalid mask threshold/area')
    for key in ('max_sensor_area_fraction', 'min_largest_component_fraction',
                'min_component_mean_probability', 'max_boundary_uncertain_fraction'):
        if not 0 < cfg[key] <= 1:
            raise ValueError('Invalid probability/fraction gate: '+key)
    if not 0 <= cfg['uncertainty_low'] < cfg['uncertainty_high'] <= 1:
        raise ValueError('Invalid boundary uncertainty band')
    return cfg


def gate_lcd_probability(probability512, footprint512, gate_config):
    """Pure mask gate shared with validation; never changes the image pose.

    Keep the largest component only when its share passes the configured gate.
    Boundary erosion is used to MEASURE uncertainty, never to change the mask.
    No hull, dilation, hole filling, reference, or landmark is used here.
    """
    cfg = validate_gate_config(gate_config)
    report = {'accepted': False, 'reason': None, 'method': 'learned_lcd_segmentation',
              'gate_config': cfg, 'scores_are_calibrated_confidence': False,
              'source_annotations_used': False, 'input_image_warped': False,
              'previous_frame_fallback': False, 'mask_applied': False}
    probability = np.asarray(probability512)
    if probability.shape != (SIZE, SIZE) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        return None, dict(report, reason='invalid_lcd_probabilities')
    valid = sensor_mask(footprint512)
    sensor_pixels = int(np.count_nonzero(valid))
    if not sensor_pixels:
        return None, dict(report, reason='empty_sensor_footprint')
    foreground = probability >= cfg['pixel_threshold']
    clipped_pixels = int(np.count_nonzero(foreground & ~valid))
    foreground &= valid
    total_pixels = int(np.count_nonzero(foreground))
    report.update(sensor_pixels=sensor_pixels, threshold_foreground_pixels=total_pixels,
                  foreground_pixels_clipped_to_sensor=clipped_pixels)
    if total_pixels < cfg['min_visible_pixels']:
        return None, dict(report, reason='lcd_not_resolved_or_too_small')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    component = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    mask = labels == component
    fraction, coverage = area / total_pixels, area / sensor_pixels
    mean_probability = float(np.mean(probability[mask]))
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    sensor_inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    boundary = mask & ~eroded & sensor_inner
    boundary_pixels = int(np.count_nonzero(boundary))
    boundary_probability = probability[boundary]
    uncertain = ((boundary_probability >= cfg['uncertainty_low']) &
                 (boundary_probability <= cfg['uncertainty_high']))
    boundary_uncertain = float(np.mean(uncertain)) if boundary_pixels else 1.
    touches_sensor = bool(np.any(mask & ~sensor_inner))
    report.update(component_count=count-1, mask_pixels=area,
                  largest_component_fraction=fraction, sensor_coverage_fraction=coverage,
                  component_mean_probability=mean_probability,
                  boundary_pixels=boundary_pixels,
                  boundary_mean_probability=float(np.mean(boundary_probability)) if boundary_pixels else None,
                  boundary_uncertain_fraction=boundary_uncertain,
                  touches_sensor_boundary=touches_sensor,
                  partial_lcd_visibility_possible=touches_sensor,
                  boundary_excludes_sensor_cut=True)
    if area < cfg['min_visible_pixels']:
        return None, dict(report, reason='lcd_component_too_small')
    if coverage > cfg['max_sensor_area_fraction']:
        return None, dict(report, reason='implausibly_large_lcd_mask')
    if fraction < cfg['min_largest_component_fraction']:
        return None, dict(report, reason='fragmented_or_multiple_lcd_candidates')
    if mean_probability < cfg['min_component_mean_probability']:
        return None, dict(report, reason='low_lcd_component_score')
    if not boundary_pixels or boundary_uncertain > cfg['max_boundary_uncertain_fraction']:
        return None, dict(report, reason='uncertain_lcd_boundary')
    return mask.astype(np.uint8)*255, dict(report, accepted=True, reason='lcd_gate_passed')


def mask_contours(mask):
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2).astype(float) for contour in contours if len(contour) >= 3]


def verify_implementation_hashes(identity):
    for relative, expected in identity.get('implementation_hashes', {}).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Checkpoint implementation changed: '+relative)


class LearnedLCDSpeedReader:
    def __init__(self, lcd_checkpoint, speed_checkpoint, speed_calibration, device='cuda'):
        # Lazy imports keep the shared gate usable by detector training without
        # a train -> evaluation -> inference -> train import cycle.
        import torch
        from learned_lcd_v1.train import load_model as load_lcd
        from mixed_view_v5.train_landmarks import load_model as load_speed
        from real_adaptation_v4.geometry import load_calibration, LANDMARK_NAMES
        if sha256(speed_checkpoint) != SPEED_CHECKPOINT_SHA256:
            raise ValueError('Expected the frozen V5 best epoch11 speed weights')
        if sha256(speed_calibration) != SPEED_CALIBRATION_SHA256:
            raise ValueError('Unexpected frozen speed calibration')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else str(device)
        self.lcd_model, lcd_payload = load_lcd(lcd_checkpoint, self.device)
        if lcd_payload.get('preprocessing') != LCD_PREPROCESSING_ID:
            raise ValueError('Unexpected LCD model preprocessing')
        if type(lcd_payload.get('ready_for_integration')) is not bool:
            raise ValueError('LCD checkpoint is missing its validation integration decision')
        self.integration_ready = lcd_payload['ready_for_integration']
        self.gates = validate_gate_config(lcd_payload.get('gate_config'))
        self.speed_model, speed_payload = load_speed(speed_checkpoint, self.device)
        if speed_payload.get('best_epoch') != 11 or speed_payload.get('keypoint_names') != list(LANDMARK_NAMES):
            raise ValueError('Unexpected frozen speed epoch or landmark order')
        for payload in (lcd_payload, speed_payload):
            identity = payload.get('identity', {})
            if identity.get('opencv_version') != cv2.__version__:
                raise ValueError('OpenCV differs from the recorded checkpoint runtime')
            verify_implementation_hashes(identity)
        speed_identity = speed_payload['identity']
        if speed_identity['dataset']['calibration_sha256'] != sha256(speed_calibration):
            raise ValueError('Speed checkpoint calibration identity mismatch')
        self.calibration = load_calibration(speed_calibration)
        self.lcd_model.eval(); self.speed_model.eval()
        for model in (self.lcd_model, self.speed_model):
            for parameter in model.parameters(): parameter.requires_grad_(False)
        self.metadata = {
            'pipeline': PIPELINE_ID, 'device': self.device, 'opencv': cv2.__version__,
            'numpy': np.__version__, 'lcd_checkpoint_sha256': sha256(lcd_checkpoint),
            'lcd_best_epoch': lcd_payload.get('best_epoch'),
            'lcd_preprocessing': LCD_PREPROCESSING_ID, 'gate_config': self.gates,
            'ready_for_integration': self.integration_ready,
            'lcd_dataset_manifest_sha256': lcd_payload['identity']['dataset']['manifest_sha256'],
            'speed_checkpoint_sha256': SPEED_CHECKPOINT_SHA256, 'speed_best_epoch': 11,
            'speed_training_preprocessing': speed_identity['preprocessing'],
            'speed_training_dataset_manifest_sha256': speed_identity['dataset']['manifest_sha256'],
            'speed_calibration_sha256': SPEED_CALIBRATION_SHA256,
            'inference_code_sha256': sha256(__file__),
            'speed_geometry_code_sha256': sha256(ROOT/'real_adaptation_v4/geometry.py'),
            'speed_weights_changed': False, 'sift_used': False, 'source_pose_used': False,
            'input_image_warped': False, 'input_roi_cropped': False,
            'previous_frame_fallback': False, 'smoothing': False,
            'limitation': 'A learned mask can be wrong despite high scores. Post-mask landmark checks cannot guarantee preservation of an already hidden needle. Indicated needle speed is not road speed during startup sweeps.'}

    def predict(self, frame_bgr, include_model_input=False):
        import torch
        from PIL import Image
        from learned_lcd_v1.model import normalize_gray
        from training_pilot.train_core import preprocess_image
        from real_adaptation_v4.geometry import read_speed
        result = {'accepted': False, 'speed_kmh': None, 'reason': None, 'failure_stage': None,
                  'lcd_model_evaluated': False, 'model_evaluated': False,
                  'lcd_detection': None, 'geometry': None, 'points_input_xy': None,
                  'points_source_xy': None, 'availability': None, 'lcd_contours_source_xy': [],
                  'annotation_origin': 'model_predictions_not_ground_truth'}
        if include_model_input:
            result.update(model_input=None, candidate_masked_input=None, lcd_probability=None, lcd_mask=None)
        gray, matrix, footprint = letterbox(frame_bgr)
        raw_tensor = normalize_gray(gray)[None].to(self.device)
        with torch.inference_mode(): logits = self.lcd_model(raw_tensor)['lcd_logits']
        result['lcd_model_evaluated'] = True
        if tuple(logits.shape) != (1, 1, SIZE, SIZE):
            return dict(result, reason='invalid_lcd_output_shape', failure_stage='lcd_detector')
        if not torch.isfinite(logits).all():
            return dict(result, reason='nonfinite_lcd_logits', failure_stage='lcd_detector')
        probability = logits[0, 0].detach().float().sigmoid().cpu().numpy()
        mask, detection = gate_lcd_probability(probability, footprint, self.gates)
        result['lcd_detection'] = detection
        if include_model_input: result['lcd_probability'] = probability
        inverse = np.linalg.inv(matrix)
        if np.isfinite(probability).all():
            candidate = ((probability >= self.gates['pixel_threshold']) & footprint).astype(np.uint8)*255
            result['lcd_contours_source_xy'] = [project(contour, inverse).tolist() for contour in mask_contours(candidate)]
        result['lcd_outline_status'] = 'candidate_rejected' if mask is None else 'gate_passed'
        if mask is None:
            return dict(result, reason=detection['reason'], failure_stage='lcd_mask_gate')
        result['lcd_contours_source_xy'] = [project(contour, inverse).tolist() for contour in mask_contours(mask)]
        masked = gray.copy(); masked[mask > 0] = 128
        if not self.integration_ready:
            if include_model_input: result.update(candidate_masked_input=masked, lcd_mask=mask)
            return dict(result, reason='lcd_checkpoint_not_ready_for_integration', failure_stage='lcd_validation_readiness')
        detection['mask_applied'] = True
        if include_model_input: result.update(model_input=masked, lcd_mask=mask)
        tensor = preprocess_image(Image.fromarray(masked), SIZE)[None].to(self.device)
        with torch.inference_mode(): output = self.speed_model(tensor)
        result['model_evaluated'] = True
        points = output['points'][0].detach().float().cpu().numpy()*511
        availability = output['visibility'][0].detach().float().cpu().numpy()
        if points.shape != (12, 2) or availability.shape != (12,) or not np.isfinite(points).all() or not np.isfinite(availability).all():
            return dict(result, reason='invalid_speed_landmarks', failure_stage='speed_model')
        source = project(points, inverse)
        result.update(points_input_xy=points.tolist(), points_source_xy=source.tolist(), availability=availability.tolist())
        active = availability >= .5; active[:2] = True
        height, width = frame_bgr.shape[:2]
        inside = (source[:, 0] >= 0) & (source[:, 0] <= width-1) & (source[:, 1] >= 0) & (source[:, 1] <= height-1)
        if not inside[active].all():
            return dict(result, reason='available_landmark_outside_sensor', failure_stage='speed_geometry')
        rounded = np.clip(np.rint(points).astype(int), 0, 511)
        if np.any(mask[rounded[active, 1], rounded[active, 0]] > 0):
            return dict(result, reason='predicted_landmark_intersects_lcd_mask', failure_stage='post_mask_safety')
        needle = np.zeros_like(mask)
        cv2.line(needle, tuple(rounded[0]), tuple(rounded[1]), 255, 2)
        if np.any((needle > 0) & (mask > 0)):
            return dict(result, reason='predicted_needle_crosses_lcd_mask', failure_stage='post_mask_safety')
        reading = read_speed(points, visibility=availability, calibration=self.calibration)
        result.update(accepted=bool(reading['accepted']), speed_kmh=reading['speed_kmh'],
                      reason=reading['reason'], failure_stage=None if reading['accepted'] else 'speed_geometry', geometry=reading)
        return result
