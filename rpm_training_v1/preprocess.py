"""Shared deterministic preprocessing for training and registered-frame inference.

Input must already be registered to the reviewed 478x850 LCD reference. This
module does not locate the LCD or recognize startup; callers must run those
full-frame/full-LCD gates separately. N/trip pixels are outside the curved mask.
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
INPUT_HEIGHT, INPUT_WIDTH = 256, 192
CROP_XYXY = (120, 316, 254, 485)
BACKGROUND_VALUE = 128
PREPROCESS_VERSION = 'rpm_curved_mask_letterbox_v1'


def load_calibration(path=None):
    return json.loads(Path(path or HERE/'calibration/calibration_draft.json').read_text(encoding='utf-8'))


def canonical_mask(calibration=None):
    """Return native-reference 850x478 mask; preserve strip and printed scale."""
    cal = calibration or load_calibration()
    mask = np.zeros((int(cal['image_height']), int(cal['image_width'])), np.uint8)
    left = np.asarray(cal['left_edge_xy'], np.float32).copy()
    left[:, 0] -= 4
    # Right boundary follows the scale so even-number labels remain visible.
    right = np.asarray([a['xy'] for a in cal['scale_anchors']], np.float32)
    right[:, 0] += 23
    right[0, 1] = 480
    polygon = np.concatenate([left, right], axis=0)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    for polygon in cal['static_exclusion_polygons_xy']:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 0)
    x0, y0, x1, y1 = CROP_XYXY
    bounds = np.zeros_like(mask)
    bounds[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(mask, bounds)


def reference_to_input_matrix():
    x0, y0, x1, y1 = CROP_XYXY
    w, h = x1-x0, y1-y0
    scale = min(INPUT_WIDTH/w, INPUT_HEIGHT/h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    ox, oy = (INPUT_WIDTH-nw)//2, (INPUT_HEIGHT-nh)//2
    # OpenCV pixel-center resize mapping, after the integer crop.
    sx, sy = nw/w, nh/h
    return np.array([[sx, 0, ox+(0.5-x0)*sx-0.5],
                     [0, sy, oy+(0.5-y0)*sy-0.5], [0, 0, 1]], np.float64)


def transform_points(points_xy, matrix):
    pts = np.asarray(points_xy, np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(matrix, np.float64).T
    return homogeneous[:, :2]/homogeneous[:, 2:3]


def preprocess_aligned(aligned, calibration=None, return_mask=False):
    """Convert aligned BGR/gray frame to uint8 [256,192], same at inference."""
    if aligned is None or aligned.shape[:2] != (850, 478):
        raise ValueError('Expected an aligned native reference frame with shape (850,478[,3])')
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned
    if gray.dtype != np.uint8:
        raise ValueError('Expected uint8 image')
    mask = canonical_mask(calibration)
    x0, y0, x1, y1 = CROP_XYXY
    crop = gray[y0:y1, x0:x1].copy()
    cm = mask[y0:y1, x0:x1]
    crop[cm == 0] = BACKGROUND_VALUE
    scale = min(INPUT_WIDTH/crop.shape[1], INPUT_HEIGHT/crop.shape[0])
    nw, nh = int(round(crop.shape[1]*scale)), int(round(crop.shape[0]*scale))
    ox, oy = (INPUT_WIDTH-nw)//2, (INPUT_HEIGHT-nh)//2
    out = np.full((INPUT_HEIGHT, INPUT_WIDTH), BACKGROUND_VALUE, np.uint8)
    out_mask = np.zeros_like(out)
    out[oy:oy+nh, ox:ox+nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out_mask[oy:oy+nh, ox:ox+nw] = cv2.resize(cm, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out[out_mask == 0] = BACKGROUND_VALUE
    return (out, out_mask) if return_mask else out


def preprocessing_metadata():
    return {'version': PREPROCESS_VERSION, 'input_size_hw': [INPUT_HEIGHT, INPUT_WIDTH],
            'channels': 1, 'dtype': 'uint8', 'network_normalization': 'float32 / 255',
            'canonical_reference_size_hw': [850, 478], 'crop_xyxy': list(CROP_XYXY),
            'aspect_ratio': 'preserved by letterboxing', 'background_value': BACKGROUND_VALUE,
            'reference_to_input_matrix': reference_to_input_matrix().tolist(),
            'mask': 'curved tachometer and printed scale only; static top/bottom decorations removed',
            'excluded': ['gear N/1..6', 'trip', 'clock', 'fuel'],
            'external_requirements': ['LCD/reference registration', 'full-LCD startup/readability gate'],
            'calibration': 'draft printed-scale coordinates; not verified engine-RPM calibration'}
