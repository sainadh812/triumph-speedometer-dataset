"""Pure synthetic array/coordinate tests; no model or video execution."""
import unittest
import numpy as np
from learned_lcd_v1.inference import (DEFAULT_GATE_CONFIG, gate_lcd_probability,
                                    letterbox, project, sensor_mask, validate_gate_config)


class LearnedMaskTests(unittest.TestCase):
    def setUp(self):
        self.valid = np.ones((512, 512), np.uint8)*255
        self.probability = np.zeros((512, 512), np.float32)
        self.probability[120:220, 350:460] = .99

    def gate(self, probability=None, valid=None, config=None):
        return gate_lcd_probability(self.probability if probability is None else probability,
                                    self.valid if valid is None else valid,
                                    dict(DEFAULT_GATE_CONFIG) if config is None else config)

    def test_confident_mask_has_no_expansion(self):
        mask, report = self.gate()
        self.assertTrue(report['accepted'])
        self.assertTrue(np.array_equal(mask > 0, self.probability >= .7))
        self.assertEqual(report['mask_pixels'], 11000)
        self.assertFalse(report['scores_are_calibrated_confidence'])

    def test_nonrectangular_sensor_map_clips_mask(self):
        yy, xx = np.ogrid[:512, :512]
        valid = ((xx >= 90) & (xx < 410) & (yy >= 20) & (xx+yy < 600)).astype(np.uint8)
        mask, report = self.gate(valid=valid)
        self.assertTrue(report['accepted'])
        self.assertFalse(np.any((mask > 0) & (valid == 0)))
        self.assertTrue(report['touches_sensor_boundary'])
        self.assertGreater(report['foreground_pixels_clipped_to_sensor'], 0)

    def test_partial_screen_boundary_is_not_uncertain_edge(self):
        valid = self.valid.copy(); valid[:, 400:] = 0
        probability = self.probability.copy(); probability[125:215, 399] = .72
        mask, report = self.gate(probability, valid)
        self.assertTrue(report['accepted'])
        self.assertEqual(report['boundary_uncertain_fraction'], 0.)
        self.assertFalse(np.any(mask[:, 400:]))

    def test_multiple_large_components_are_rejected(self):
        probability = self.probability.copy(); probability[300:400, 100:210] = .99
        mask, report = self.gate(probability)
        self.assertIsNone(mask)
        self.assertEqual(report['reason'], 'fragmented_or_multiple_lcd_candidates')

    def test_unknown_or_bad_mask_is_rejected(self):
        for probability, reason in [(np.zeros((512, 512)), 'lcd_not_resolved_or_too_small'),
                                    (np.ones((512, 512)), 'implausibly_large_lcd_mask'),
                                    (np.full((512, 512), np.nan), 'invalid_lcd_probabilities')]:
            mask, report = self.gate(probability)
            self.assertIsNone(mask); self.assertEqual(report['reason'], reason)

    def test_low_component_score_and_uncertain_boundary_are_separate(self):
        probability = np.where(self.probability > 0, .76, 0).astype(np.float32)
        mask, report = self.gate(probability)
        self.assertIsNone(mask); self.assertEqual(report['reason'], 'low_lcd_component_score')
        probability = self.probability.copy()
        probability[120, 350:460] = .72; probability[219, 350:460] = .72
        probability[120:220, 350] = .72; probability[120:220, 459] = .72
        mask, report = self.gate(probability)
        self.assertIsNone(mask); self.assertEqual(report['reason'], 'uncertain_lcd_boundary')

    def test_tiny_disconnected_fragments_do_not_expand_primary_mask(self):
        probability = self.probability.copy(); probability[40:42, 40:42] = .99
        mask, report = self.gate(probability)
        self.assertTrue(report['accepted']); self.assertEqual(report['component_count'], 2)
        self.assertTrue(np.array_equal(mask > 0, self.probability > 0))

    def test_full_frame_letterbox_matches_frozen_v5(self):
        from mixed_view_v5.inference import letterbox as frozen_letterbox
        frame = np.arange(850*534, dtype=np.uint8).reshape(850, 534)
        actual, matrix, valid = letterbox(frame)
        expected, old_matrix, footprint = frozen_letterbox(frame)
        self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(np.array_equal(matrix, old_matrix))
        self.assertTrue(np.array_equal(valid, sensor_mask(footprint)))
        points = np.array([[0, 0], [533, 849], [270, 400]], float)
        self.assertTrue(np.allclose(project(project(points, matrix), np.linalg.inv(matrix)), points))

    def test_checkpoint_gate_must_be_complete_and_conservative(self):
        for patch in [{'pixel_threshold': .4}, {'min_visible_pixels': -1}, {'uncertainty_high': .1}]:
            with self.assertRaises(ValueError): validate_gate_config({**DEFAULT_GATE_CONFIG, **patch})
        with self.assertRaises(ValueError): validate_gate_config({'pixel_threshold': .7})
        with self.assertRaises(ValueError): sensor_mask(np.full((512, 512), .5))


if __name__ == '__main__': unittest.main()
