"""Pure NumPy metrics; no model execution or guessed speed/RPM ground truth."""
from __future__ import annotations

import numpy as np


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {'count': int(values.size), 'mean': float(values.mean()) if values.size else None,
            'median': float(np.median(values)) if values.size else None,
            'p95': float(np.percentile(values, 95)) if values.size else None}


def heatmap_to_points(logits, input_hw):
    """Argmax at heatmap pixel centers, mapped to input image pixel centers."""
    values = np.asarray(logits)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError('Expected finite N,K,H,W heatmaps')
    height, width = values.shape[-2:]
    index = values.reshape(*values.shape[:2], -1).argmax(axis=-1)
    x, y = index % width, index // width
    out = np.stack([(x + .5) * input_hw[1] / width - .5,
                    (y + .5) * input_hw[0] / height - .5], axis=-1)
    return np.clip(out, [0., 0.], [input_hw[1] - 1., input_hw[0] - 1.])


def binary_scores(prediction, target):
    pred, truth = np.asarray(prediction, bool), np.asarray(target, bool)
    if pred.shape != truth.shape:
        raise ValueError('Binary metric shapes differ')
    tp, fp, fn, tn = (int(np.sum(pred & truth)), int(np.sum(pred & ~truth)),
                      int(np.sum(~pred & truth)), int(np.sum(~pred & ~truth)))
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'accuracy': (tp + tn) / pred.size if pred.size else None}


def box_iou(prediction, target):
    a, b = np.asarray(prediction, float), np.asarray(target, float)
    intersection = np.maximum(np.minimum(a[..., 2:], b[..., 2:]) -
                              np.maximum(a[..., :2], b[..., :2]), 0).prod(axis=-1)
    area_a = np.maximum(a[..., 2:] - a[..., :2], 0).prod(axis=-1)
    area_b = np.maximum(b[..., 2:] - b[..., :2], 0).prod(axis=-1)
    return intersection / np.maximum(area_a + area_b - intersection, np.finfo(float).eps)


class MetricAccumulator:
    def __init__(self, point_names):
        self.names = list(point_names)
        self.errors = [[] for _ in self.names]
        self.visibility_prediction, self.visibility_target = [], []
        self.visibility_unknown_count = 0
        self.presence_prediction, self.presence_target = [], []
        self.positive_mask_iou, self.absent_mask_fraction, self.box_ious = [], [], []
        self.count = 0

    def add(self, points, targets, visible, visibility_probability, *, visibility_known=None, presence=None,
            present=None, masks=None, mask_targets=None, boxes=None, box_targets=None):
        points, targets, visible = np.asarray(points), np.asarray(targets), np.asarray(visible, bool)
        if points.shape != targets.shape or points.shape[:-1] != visible.shape:
            raise ValueError('Landmark metric shapes differ')
        known = np.ones_like(visible, bool) if visibility_known is None else np.asarray(visibility_known)
        probability = np.asarray(visibility_probability)
        if known.shape != visible.shape or probability.shape != visible.shape or not np.isin(known, [0, 1]).all():
            raise ValueError('Invalid visibility metric supervision')
        known = known.astype(bool)
        if np.any(visible & ~known):
            raise ValueError('A visible landmark must have known visibility')
        error = np.linalg.norm(points - targets, axis=-1)
        for index in range(len(self.names)):
            self.errors[index].extend(error[:, index][visible[:, index]].tolist())
        self.visibility_prediction.extend((probability[known] >= .5).tolist())
        self.visibility_target.extend(visible[known].tolist())
        self.visibility_unknown_count += int(np.count_nonzero(~known))
        self.count += len(points)
        if presence is not None:
            truth = np.asarray(present, bool)
            self.presence_prediction.extend((np.asarray(presence) >= .5).tolist())
            self.presence_target.extend(truth.tolist())
            self.box_ious.extend(box_iou(boxes, box_targets)[truth].tolist())
            for pred, target in zip(np.asarray(masks, bool), np.asarray(mask_targets, bool)):
                if target.any():
                    self.positive_mask_iou.append(float(np.sum(pred & target) / np.sum(pred | target)))
                else:
                    self.absent_mask_fraction.append(float(pred.mean()))

    def result(self):
        result = {'samples': self.count,
                  'visible_point_error_px': _summary([x for group in self.errors for x in group]),
                  'per_landmark_error_px': {name: _summary(values) for name, values in zip(self.names, self.errors)},
                  'visibility_at_0_5': binary_scores(self.visibility_prediction, self.visibility_target),
                  'visibility_known_point_count': len(self.visibility_target),
                  'visibility_unknown_point_count_excluded': self.visibility_unknown_count,
                  'metric_scope': 'Input-pixel landmark error on visible labels; no speed or physical RPM accuracy claim.'}
        if self.presence_target:
            result.update(dashboard_presence_at_0_5=binary_scores(self.presence_prediction, self.presence_target),
                          dashboard_box_iou=_summary(self.box_ious),
                          lcd_iou_on_nonempty_targets=_summary(self.positive_mask_iou),
                          lcd_false_positive_fraction_on_empty_targets=_summary(self.absent_mask_fraction))
        return result
