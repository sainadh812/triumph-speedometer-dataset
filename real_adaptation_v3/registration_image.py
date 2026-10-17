"""Image-only fixed-camera registration, shared by synthetic and real inference.

No label coordinates or known source/rendering homographies enter estimation.
Initialize ImageOnlyRegistrar once with masked_reference.png, then estimate each
final masked grayscale512 frame. Rejected estimates never use an oracle fallback.
"""
import cv2
import numpy as np


def _project(xy, h):
    points = np.c_[np.atleast_2d(xy), np.ones(len(np.atleast_2d(xy)))] @ h.T
    return points[:, :2] / points[:, 2:]


class ImageOnlyRegistrar:
    def __init__(self, reference):
        reference = np.asarray(reference)
        if reference.shape != (512, 512) or reference.dtype != np.uint8:
            raise ValueError('Reference must be a grayscale512x512 uint8 image')
        self.clahe = cv2.createCLAHE(2., (8, 8))
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.012)
        yy, xx = np.mgrid[:512, :512]
        radius = np.hypot(xx - 211, yy - 225)
        self.mask = ((radius > 93) & (radius < 188) & (xx < 270)
                     & (yy > 40) & (yy < 390)).astype(np.uint8) * 255
        self.keypoints, self.descriptors = self.sift.detectAndCompute(
            self.clahe.apply(reference), self.mask)

    def estimate(self, image, seed=0):
        """Return (image_to_reference_H_list_or_None, diagnostic_dict).

        The H may be present but rejected. Only use it for a trustworthy reading
        when diagnostics['accepted'] is True. RMSE describes feature fit, not
        speed accuracy. The gates are engineering checks for this fixed camera.
        """
        image = np.asarray(image)
        if image.shape != (512, 512) or image.dtype != np.uint8:
            raise ValueError('Image must be the final masked grayscale512x512 uint8 input')
        cv2.setRNGSeed(int(seed))
        meta = {
            'status': 'failed', 'accepted': False,
            'method': 'image_only_SIFT_CLAHE_static_reference_annulus_RANSAC',
            'reference_image': 'masked_reference.png', 'matched_features': 0,
            'inliers': 0, 'inlier_rmse_px': None, 'max_corner_displacement_px': None,
            'positive_projective_horizon': False, 'orientation_preserved': False,
            'max_corner_displacement_gate_px': 100.,
            'uses_render_transform_or_landmark_labels': False,
        }
        keypoints, descriptors = self.sift.detectAndCompute(self.clahe.apply(image), None)
        if descriptors is None or self.descriptors is None:
            meta['reason'] = 'missing_image_features'
            return None, meta
        pairs = cv2.BFMatcher().knnMatch(descriptors, self.descriptors, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .72 * p[1].distance]
        meta['matched_features'] = len(matches)
        if len(matches) < 8:
            meta['reason'] = 'insufficient_matches'
            return None, meta
        source = np.float32([keypoints[m.queryIdx].pt for m in matches])
        target = np.float32([self.keypoints[m.trainIdx].pt for m in matches])
        h, mask = cv2.findHomography(source, target, cv2.RANSAC, 2.2, maxIters=5000)
        if h is None or mask is None or not np.all(np.isfinite(h)):
            meta['reason'] = 'homography_estimation_failed'
            return None, meta
        if abs(h[2, 2]) < 1e-12:
            meta['reason'] = 'invalid_homography_normalization'
            return None, meta
        h = h / h[2, 2]
        inside = mask.ravel().astype(bool)
        meta['inliers'] = int(inside.sum())
        residual = _project(source[inside], h) - target[inside]
        meta['inlier_rmse_px'] = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        corners = np.array([[0., 0.], [511., 0.], [511., 511.], [0., 511.]])
        denominators = np.c_[corners, np.ones(4)] @ h[2]
        horizon_ok = bool(np.all(np.isfinite(denominators)) and np.all(denominators > 1e-6))
        meta['positive_projective_horizon'] = horizon_ok
        orientation_ok = bool(horizon_ok and np.linalg.det(h) > 0)
        meta['orientation_preserved'] = orientation_ok
        if not horizon_ok or not orientation_ok:
            meta['status'] = 'rejected'
            meta['reason'] = 'projective_horizon_or_orientation_invalid'
            return h.tolist(), meta
        transformed = _project(corners, h)
        finite = np.all(np.isfinite(transformed))
        maximum = float(np.max(np.linalg.norm(transformed - corners, axis=1))) if finite else None
        meta['max_corner_displacement_px'] = maximum
        valid = (meta['inliers'] >= 12 and meta['inlier_rmse_px'] < 1.6
                 and maximum is not None and maximum < 100.
                 and cv2.isContourConvex(transformed.astype(np.float32)))
        meta['accepted'] = bool(valid)
        meta['status'] = 'accepted' if valid else 'rejected'
        meta['reason'] = None if valid else 'inliers_residual_or_fixed_camera_plausibility_gate'
        return h.tolist(), meta


def register_image(image, reference, seed=0):
    """One-shot convenience API; reuse ImageOnlyRegistrar for video/datasets."""
    return ImageOnlyRegistrar(reference).estimate(image, seed=seed)
