"""Software smoke checks for V4 video plumbing; no trained-model accuracy claim."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from real_adaptation_v4.infer_video import build_timeline, predict_items, row_from_item, render, BOARD_SIZE
from real_adaptation_v4.test_geometry import CAL, landmarks


class StubProcessor:
    """Deterministic preprocessing fixture, not a substitute for real registration."""
    fixed_mask = np.zeros((512, 512), np.uint8)
    fixed_mask[10:20, 400:450] = 255
    def process(self, image, apply_mask=True):
        valid = bool(image[0, 0])
        metadata = {'accepted': valid, 'reason': 'accepted' if valid else 'missing_features',
                    'lcd_mask_applied': valid, 'image_to_reference_h': np.eye(3).tolist() if valid else None}
        if not valid:
            return None, metadata
        image = image.copy()
        image[self.fixed_mask > 0] = 128
        return image, metadata


class StubModel:
    """Known-point fixture tests IO/decoder wiring only; it is not a neural result."""
    def __init__(self):
        self.calls = []
    def __call__(self, batch):
        self.calls.append(tuple(batch.shape))
        n = len(batch)
        return {'points': torch.tensor(landmarks(73)/511, dtype=torch.float32).repeat(n, 1, 1),
                'visibility': torch.ones(n, 12)}


def item(valid=True, index=0):
    return {'gray': np.full((512, 512), 80 if valid else 0, np.uint8),
            'color': np.full((512, 512, 3), 80, np.uint8), 'output_frame_index': index,
            'source_frame_index': index, 'source_timestamp_seconds': index/30,
            'output_timestamp_seconds': index/30}


class VideoSmokeTests(unittest.TestCase):
    def test_timestamp_sampling_preserves_duration_with_explicit_duplicates(self):
        timestamps = np.array([0., .04, .08, .12, .18])
        indices, times = build_timeline(timestamps, .22)
        self.assertEqual(len(times), 7)
        self.assertEqual(set(indices), set(range(5)))
        self.assertEqual(len(indices)-len(set(indices)), 2)
        self.assertLessEqual(abs(len(times)/30-.22), .5/30)
        self.assertTrue(np.all(np.diff(indices) >= 0))

    def test_invalid_timestamps_rejected(self):
        for timestamps, duration in [([0, 0], 1), ([0, np.nan], 1), ([0, 1], .9), ([1, 2], 3)]:
            with self.assertRaises(ValueError):
                build_timeline(timestamps, duration)

    def test_alignment_failure_never_feeds_model_or_reuses_previous_speed(self):
        model = StubModel()
        items = [item(True, 0), item(False, 1), item(True, 2)]
        count = predict_items(model, StubProcessor(), items, CAL)
        self.assertEqual(count, 2)
        self.assertEqual(model.calls, [(2, 3, 512, 512)])
        self.assertTrue(items[0]['reading']['accepted'])
        self.assertAlmostEqual(items[0]['reading']['speed_kmh'], 73, delta=.002)
        self.assertFalse(items[1]['reading']['accepted'])
        self.assertIsNone(items[1]['reading']['speed_kmh'])
        self.assertIsNone(items[1]['points'])
        self.assertIsNone(items[1]['masked'])
        self.assertEqual(items[1]['reading']['reason'], 'preprocessing_rejected:missing_features')
        self.assertTrue(items[2]['reading']['accepted'])

    def test_all_alignment_failures_skip_model_batch(self):
        model = StubModel()
        items = [item(False)]
        self.assertEqual(predict_items(model, StubProcessor(), items, CAL), 0)
        self.assertEqual(model.calls, [])

    def test_render_and_logging_preserve_rejected_status(self):
        items = [item(True), item(False, 1)]
        predict_items(StubModel(), StubProcessor(), items, CAL)
        for source in items:
            board = render(source, epoch='fixture', total=2)
            self.assertEqual(board.shape, (BOARD_SIZE[1], BOARD_SIZE[0], 3))
            self.assertEqual(board.dtype, np.uint8)
            row = row_from_item(source)
            self.assertEqual(row['geometry_accepted'], source['reading']['accepted'])
            if not row['geometry_accepted']:
                self.assertIsNone(row['primary_speed_kmh'])
                self.assertIsNone(row['predicted_points_aligned_xy'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
