"""Align this camera's frame to its reviewed reference, then use one fixed LCD mask.

This module never hides input pixels until current-frame image registration passes.
The fixed polygon is in reference coordinates, not the coordinates of another video.
"""
import cv2
import numpy as np


def prepare_frame(frame):
    height, width = frame.shape[:2]
    side = max(height, width)
    top, left = (side-height)//2, (side-width)//2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    square = cv2.copyMakeBorder(gray, top, side-height-top, left, side-width-left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(square, (512, 512), interpolation=cv2.INTER_LINEAR)


def project(points, h):
    p = np.c_[np.asarray(points, float), np.ones(len(points))] @ h.T
    if np.any(np.abs(p[:, 2]) < 1e-8):
        raise ValueError('Point crosses projective horizon')
    return p[:, :2] / p[:, 2:]


class FixedViewPreprocessor:
    def __init__(self, reference, lcd_polygon, ticks_xy, pivot_xy, mask_value=128):
        self.reference = np.asarray(reference, np.uint8)
        if self.reference.shape != (512, 512):
            raise ValueError('Reference must be 512x512 grayscale')
        self.lcd_polygon = np.asarray(lcd_polygon, float)
        self.mask_value = int(mask_value)
        # Stabilization may expose sensor pixels beyond the original letterbox.
        # Keep the reviewed reference's content footprint (including black padding).
        nonzero_y, nonzero_x = np.nonzero(self.reference)
        self.reference_content_mask = np.zeros((512, 512), np.uint8)
        if len(nonzero_x) == 0:
            raise ValueError('Reference is empty')
        self.reference_content_xyxy = [int(nonzero_x.min()), int(nonzero_y.min()), int(nonzero_x.max()+1), int(nonzero_y.max()+1)]
        x0, y0, x1, y1 = self.reference_content_xyxy
        self.reference_content_mask[y0:y1, x0:x1] = 255
        self.fixed_mask = np.zeros((512, 512), np.uint8)
        cv2.fillPoly(self.fixed_mask, [np.rint(self.lcd_polygon).astype(np.int32)], 255)
        self.clahe = cv2.createCLAHE(2., (8, 8))
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.012)
        feature_mask = np.zeros_like(self.reference)
        cv2.polylines(feature_mask, [np.rint(ticks_xy).astype(np.int32)], False, 255, 48)
        feature_mask[self.fixed_mask > 0] = 0
        cv2.circle(feature_mask, tuple(np.rint(pivot_xy).astype(int)), 30, 0, -1)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.clahe.apply(self.reference), feature_mask)

    def process(self, gray, apply_mask=True):
        gray = np.asarray(gray, np.uint8)
        if gray.shape != (512, 512):
            raise ValueError('Input must be 512x512 grayscale')
        meta = {'accepted': False, 'reason': None, 'method': 'day_reference_SIFT_static_scale_RANSAC',
                'inliers': 0, 'matches': 0, 'rmse_px': None, 'lcd_mask_applied': False,
                'image_to_reference_h': None}
        cv2.setRNGSeed(0)
        keypoints, descriptors = self.sift.detectAndCompute(self.clahe.apply(gray), None)
        if descriptors is None or self.descriptors is None:
            meta['reason'] = 'missing_features'
            return None, meta
        pairs = cv2.BFMatcher().knnMatch(descriptors, self.descriptors, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .72*p[1].distance]
        # Prevent multiple source descriptors from inflating one reference match.
        unique = {}
        for match in sorted(matches, key=lambda x: x.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        meta['matches'] = len(matches)
        if len(matches) < 12:
            meta['reason'] = 'insufficient_scale_matches'
            return None, meta
        source = np.float32([keypoints[m.queryIdx].pt for m in matches])
        target = np.float32([self.keypoints[m.trainIdx].pt for m in matches])
        h, inside = cv2.findHomography(source, target, cv2.RANSAC, 2.2, maxIters=5000)
        if h is None or inside is None or not np.isfinite(h).all() or abs(h[2, 2]) < 1e-8:
            meta['reason'] = 'invalid_alignment'
            return None, meta
        h /= h[2, 2]
        inside = inside.ravel().astype(bool)
        meta['inliers'] = int(inside.sum())
        meta['rmse_px'] = float(np.sqrt(np.mean(np.sum((project(source[inside], h)-target[inside])**2, axis=1))))
        meta['inlier_fraction'] = float(inside.mean())
        meta['reference_support_area_px2'] = float(cv2.contourArea(cv2.convexHull(target[inside])))
        corners = np.array([[0., 0.], [511., 0.], [511., 511.], [0., 511.]])
        denominators = np.c_[corners, np.ones(4)] @ h[2]
        if np.any(denominators <= 1e-6) or np.linalg.det(h) <= 0:
            meta['reason'] = 'invalid_projective_orientation'
            return None, meta
        warped_corners = project(corners, h)
        area = abs(float(cv2.contourArea(warped_corners.astype(np.float32)))) / (511*511)
        meta['whole_frame_area_ratio'] = area
        valid = (meta['inliers'] >= 12 and meta['inlier_fraction'] >= .4 and meta['rmse_px'] < 1.6
                 and meta['reference_support_area_px2'] >= 1500 and .4 < area < 2.5
                 and cv2.isContourConvex(warped_corners.astype(np.float32)))
        if not valid:
            meta['reason'] = 'scale_alignment_quality_failed'
            return None, meta
        aligned = cv2.warpPerspective(gray, h, (512, 512), flags=cv2.INTER_LINEAR)
        aligned[self.reference_content_mask == 0] = 0
        result = aligned.copy()
        if apply_mask:
            result[self.fixed_mask > 0] = self.mask_value
        meta.update(accepted=True, reason='accepted', lcd_mask_applied=bool(apply_mask),
                    reference_content_xyxy=self.reference_content_xyxy, image_to_reference_h=h.tolist())
        return result, meta
