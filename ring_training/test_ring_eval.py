"""Ensure the evaluator cannot obtain a correct reading from label shortcuts."""
from pathlib import Path
import copy
import unittest
import torch
from torch.utils.data import DataLoader

from ring_data import RingDataset
from ring_eval import evaluate, load_template

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'ring_geometry_pilot/review20'


class FixtureModel(torch.nn.Module):
    def __init__(self, item, hide=False, bad=False):
        super().__init__()
        self.item, self.hide, self.bad = item, hide, bad

    def forward(self, images):
        b = len(images)
        tick = self.item['tick_points'].clone()
        if self.bad:
            tick[:] = .5
        return {
            'speed': self.item['speed'].repeat(b),
            'points': self.item['points'][None].repeat(b, 1, 1),
            'tick_points': tick[None].repeat(b, 1, 1),
            'tick_visibility_logits': ((self.item['tick_valid'] * 2 - 1) * 20 if not self.hide
                                       else torch.full((6,), -20.))[None].repeat(b, 1),
            'arc_logits': ((self.item['arc_mask'] * 2 - 1) * 20)[None].repeat(b, 1, 1, 1),
        }


class EvaluationIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        all_data = RingDataset(DATA)
        # Pick a non-endpoint sample so float32 coordinate error is not out-of-arc.
        cls.dataset = RingDataset(DATA, [all_data.records[3]])
        cls.loader = DataLoader(cls.dataset, batch_size=1)
        cls.item = cls.dataset[0]
        cls.template = load_template(DATA / 'source_annotations_snapshot.json', DATA / 'calibration_snapshot.json')

    def test_supplied_perfect_predictions_recover_speed(self):
        metrics, rows = evaluate(FixtureModel(self.item), self.loader, 'cpu', self.template)
        self.assertTrue(rows[0]['geometry_accepted'])
        self.assertLess(rows[0]['geometry_error_kmh'], .001)
        self.assertEqual(metrics['arc']['valid_pixel_iou'], 1)

    def test_hidden_predictions_reject_despite_visible_ground_truth(self):
        metrics, rows = evaluate(FixtureModel(self.item, hide=True), self.loader, 'cpu', self.template)
        self.assertFalse(rows[0]['geometry_accepted'])
        self.assertEqual(metrics['geometric_speed']['within_2_kmh_all_inputs_percent'], 0)
        self.assertIsNone(metrics['geometric_speed']['accepted_only']['mae_kmh'])

    def test_degenerate_predictions_cannot_use_oracle_transform(self):
        metrics, rows = evaluate(FixtureModel(self.item, bad=True), self.loader, 'cpu', self.template)
        self.assertFalse(rows[0]['geometry_accepted'])
        self.assertEqual(metrics['direct_speed']['within_2_kmh_percent'], 100)
        self.assertEqual(metrics['geometric_speed']['coverage_percent'], 0)


if __name__ == '__main__':
    unittest.main()
