"""Independent geometry, provenance, and split checks for camera-roll data.

Run after generation: python test_rotation_dataset.py
No source images, labels, or model weights are modified. Analytic clockwise
geometry and exact quarter-turn pixel comparisons are independent of OpenCV's
matrix-building helper used by the generator.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PARENTS = PROJECT / "bw_dataset_500"
OUTPUT = PROJECT / "bw_dataset_rotation_v2"
SPEC = importlib.util.spec_from_file_location(
    "rotation_generator_under_test", PROJECT / "rotation_dataset" / "generate_rotations.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clockwise(points, angle, size=512):
    points = np.asarray(points, dtype=float)
    center = (size - 1) / 2
    theta = math.radians(angle)
    dx, dy = points[:, 0] - center, points[:, 1] - center
    return np.column_stack((center + math.cos(theta) * dx - math.sin(theta) * dy,
                            center + math.sin(theta) * dx + math.cos(theta) * dy))


def analytic_matrix(angle):
    theta = math.radians(angle)
    c, s, center = math.cos(theta), math.sin(theta), 255.5
    return np.array([[c, -s, center * (1 - c + s)],
                     [s, c, center * (1 - c - s)], [0, 0, 1]], dtype=float)


class RotationGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        document = json.loads((PARENTS / "labels.json").read_text(encoding="utf-8"))
        cls.parent = next(row for row in document["samples"] if row["sample_id"] == "sample_0205")
        cls.image = cv2.imread(str(PARENTS / cls.parent["image"]), cv2.IMREAD_UNCHANGED)
        cls.mask = cv2.imread(str(PARENTS / cls.parent["needle_mask"]), cv2.IMREAD_UNCHANGED)

    def test_clockwise_sign_center_endpoints_and_inverse(self):
        points = np.array([[255.5, 255.5], [355.5, 255.5], [231.2, 379.8], [0, 0], [511, 511]])
        expected90 = np.array([[255.5, 255.5], [255.5, 355.5]])
        np.testing.assert_allclose(GENERATOR.transform_points(points[:2], GENERATOR.rotation_matrix(90)),
                                   expected90, atol=1e-10, rtol=0)
        for angle in (-180, -150, -90, -30, 0, 15, 30, 90, 180, 360):
            with self.subTest(angle=angle):
                matrix = GENERATOR.rotation_matrix(angle)
                np.testing.assert_allclose(matrix, analytic_matrix(angle), atol=1e-10, rtol=0)
                actual = GENERATOR.transform_points(points, matrix)
                np.testing.assert_allclose(actual, clockwise(points, angle), atol=1e-10, rtol=0)
                restored = GENERATOR.transform_points(actual, GENERATOR.rotation_matrix(-angle))
                np.testing.assert_allclose(restored, points, atol=1e-10, rtol=0)

    def test_quarter_turn_pixels_and_mask_match_exact_array_rotation(self):
        # This verifies direction and pixel-center convention against the pixels,
        # not merely agreement between two label-transform functions.
        for angle, turns in ((0, 0), (90, -1), (180, 2), (-90, 1)):
            with self.subTest(angle=angle):
                result = GENERATOR.rotate_sample(self.parent, self.image, self.mask, angle)
                np.testing.assert_array_equal(result["image"], np.rot90(self.image, turns))
                np.testing.assert_array_equal(result["mask"], np.rot90(self.mask, turns))
                self.assertEqual(result["image"].dtype, np.uint8)

    def test_all_keypoints_homography_speed_and_parent_are_preserved(self):
        before = copy.deepcopy(self.parent)
        names = list(before["needle_position_px"])
        original = [before["needle_position_px"][name] for name in names]
        for angle in (15, 30, 75, 150, 180, -135, -30):
            row = GENERATOR.rotate_sample(self.parent, self.image, self.mask, angle)["label"]
            np.testing.assert_allclose([row["needle_position_px"][name] for name in names],
                                       clockwise(original, angle), atol=1e-9, rtol=0)
            np.testing.assert_allclose([row["needle_position_normalized"][name] for name in names],
                                       clockwise(original, angle) / 511, atol=1e-12, rtol=0)
            np.testing.assert_allclose(row["source_to_output_homography_3x3"],
                                       analytic_matrix(angle) @ np.asarray(before["source_to_output_homography_3x3"]),
                                       atol=1e-9, rtol=0)
            for key in ("speed_kmh", "dataset_split", "group_id", "needle_angle_physical_degrees",
                        "source_pivot_xy", "source_target_tip_xy", "source_tick_anchor_xy",
                        "needle_transform_source_3x3"):
                self.assertEqual(row[key], before[key], key)
            self.assertEqual(row["parent_sample_id"], before["sample_id"])
            self.assertEqual(row["parent_group_id"], before["group_id"])
            self.assertEqual(row["parent_image_sha256"], before["image_sha256"])
            expected_angle = (before["needle_angle_image_degrees_clockwise_from_right"] + angle) % 360
            error = (row["needle_angle_image_degrees_clockwise_from_right"] - expected_angle + 180) % 360 - 180
            self.assertAlmostEqual(error, 0, places=9)
        self.assertEqual(self.parent, before)

    def test_missing_corners_are_constant_fill_and_never_reflected_mask(self):
        result = GENERATOR.rotate_sample(self.parent, self.image, self.mask, 30)
        border = np.concatenate((self.image[0], self.image[-1], self.image[:, 0], self.image[:, -1]))
        fill = int(np.median(border))
        for y, x in ((0, 0), (0, 511), (511, 0), (511, 511)):
            self.assertEqual(int(result["image"][y, x]), fill)
            self.assertEqual(int(result["mask"][y, x]), 0)
        self.assertEqual(set(np.unique(result["mask"])), {0, 255})


class RotationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent_document = json.loads((PARENTS / "labels.json").read_text(encoding="utf-8"))
        cls.parents = {r["sample_id"]: r for r in cls.parent_document["samples"]}
        cls.main = json.loads((OUTPUT / "labels.json").read_text(encoding="utf-8"))
        cls.robust = json.loads((OUTPUT / "robustness" / "labels.json").read_text(encoding="utf-8"))

    def test_counts_parent_coverage_and_group_split_isolation(self):
        self.assertEqual(self.main["count"], 6000)
        self.assertEqual(self.main["split_counts"], {"train": 4800, "validation": 1200})
        self.assertEqual(self.robust["count"], 1200)
        self.assertEqual(self.robust["split_counts"], {"validation": 1200})
        for document in (self.main, self.robust):
            self.assertEqual(document["parent_labels_sha256"], digest(PARENTS / "labels.json"))
            self.assertEqual(len(document["samples"]), document["count"])
            self.assertEqual(len({r["sample_id"] for r in document["samples"]}), document["count"])
            actual_counts = Counter(r["dataset_split"] for r in document["samples"])
            self.assertEqual(dict(actual_counts), document["split_counts"])
        descendants = defaultdict(list)
        splits = defaultdict(set)
        for row in self.main["samples"]:
            descendants[row["parent_sample_id"]].append(row)
        self.assertEqual(set(descendants), set(self.parents))
        for parent, rows in descendants.items():
            self.assertEqual(len(rows), 12, parent)
            self.assertEqual({r["rotation_degrees"] for r in rows}, set(GENERATOR.ANGLES), parent)
        for row in self.main["samples"] + self.robust["samples"]:
            parent = self.parents[row["parent_sample_id"]]
            self.assertEqual(row["group_id"], parent["group_id"])
            self.assertEqual(row["parent_group_id"], parent["group_id"])
            self.assertEqual(row["dataset_split"], parent["dataset_split"])
            self.assertEqual(row["speed_kmh"], parent["speed_kmh"])
            self.assertEqual(row["parent_image_sha256"], parent["image_sha256"])
            splits[row["group_id"]].add(row["dataset_split"])
        self.assertTrue(all(len(value) == 1 for value in splits.values()))
        training = {r["group_id"] for r in self.main["samples"] if r["dataset_split"] == "train"}
        validation = {r["group_id"] for r in self.main["samples"] if r["dataset_split"] == "validation"}
        robustness = {r["group_id"] for r in self.robust["samples"]}
        self.assertEqual((len(training), len(validation), len(robustness)), (210, 50, 50))
        self.assertTrue(training.isdisjoint(validation | robustness))
        self.assertEqual(validation, robustness)  # Correlated diagnostic, not a new independent test set.

    def test_unseen_angles_are_held_out_from_main_and_use_validation_parents_only(self):
        main_angles = {angle % 360 for angle in self.main["angles_degrees"]}
        unseen_angles = {angle % 360 for angle in self.robust["angles_degrees"]}
        self.assertTrue(main_angles.isdisjoint(unseen_angles))
        self.assertEqual(unseen_angles, {(angle + 15) % 360 for angle in main_angles})
        parent_counts = Counter(r["parent_sample_id"] for r in self.robust["samples"])
        expected = {sid for sid, row in self.parents.items() if row["dataset_split"] == "validation"}
        self.assertEqual(set(parent_counts), expected)
        self.assertEqual(set(parent_counts.values()), {12})
        self.assertTrue(all(r["dataset_split"] == "validation" and r["evaluation_set"] == "unseen_angles_robustness"
                            for r in self.robust["samples"]))

    def test_every_generated_coordinate_uses_the_same_analytic_affine(self):
        for row in self.main["samples"] + self.robust["samples"]:
            parent = self.parents[row["parent_sample_id"]]
            angle = row["rotation_degrees"]
            names = list(parent["needle_position_px"])
            expected = clockwise([parent["needle_position_px"][name] for name in names], angle)
            actual = np.asarray([row["needle_position_px"][name] for name in names])
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0, err_msg=row["sample_id"])
            np.testing.assert_allclose([row["needle_position_normalized"][name] for name in names],
                                       expected / 511, atol=1e-12, rtol=0)
            np.testing.assert_allclose(row["source_to_output_homography_3x3"],
                                       analytic_matrix(angle) @ np.asarray(parent["source_to_output_homography_3x3"]),
                                       atol=1e-9, rtol=0)
            self.assertTrue(np.all((actual >= 0) & (actual <= 511)), row["sample_id"])
            self.assertIsNone(row["needle_angle_physical_degrees"])

    def test_all_zero_angle_original_bytes_are_preserved(self):
        originals = [r for r in self.main["samples"] if r["rotation_degrees"] == 0]
        self.assertEqual(len(originals), 500)
        for row in originals:
            parent = self.parents[row["parent_sample_id"]]
            for key in ("image", "needle_mask"):
                expected = digest(PARENTS / parent[key])
                self.assertEqual(digest(OUTPUT / row[key]), expected, row["sample_id"])
                self.assertEqual(row[key + "_sha256"], expected)

    def test_sampled_images_hashes_binary_masks_and_tip_alignment(self):
        # Sample every main and unseen angle, with multiple original appearances.
        main_parents = {"sample_0001", "sample_0205", "sample_0500"}
        val_parent = next(r["parent_sample_id"] for r in self.robust["samples"])
        selections = [(OUTPUT, r) for r in self.main["samples"] if r["parent_sample_id"] in main_parents]
        selections += [(OUTPUT / "robustness", r) for r in self.robust["samples"]
                       if r["parent_sample_id"] in {val_parent, "sample_0205"}]
        self.assertGreaterEqual(len(selections), 48)
        for root, row in selections:
            image = cv2.imread(str(root / row["image"]), cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(str(root / row["needle_mask"]), cv2.IMREAD_UNCHANGED)
            self.assertEqual(image.shape, (512, 512))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertEqual(digest(root / row["image"]), row["image_sha256"])
            self.assertEqual(digest(root / row["needle_mask"]), row["needle_mask_sha256"])
            ys, xs = np.nonzero(mask)
            tip_x, tip_y = row["needle_position_px"]["tip"]
            distance = np.sqrt((xs - tip_x) ** 2 + (ys - tip_y) ** 2).min()
            self.assertLessEqual(float(distance), 2.5, row["sample_id"])


if __name__ == "__main__":
    cv2.setNumThreads(1)
    unittest.main(verbosity=2)
