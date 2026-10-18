"""Shared CPU geometry for training preparation and later Colab inference.

All transforms map ORIGINAL frame pixels to model pixels. Semantic corner order
is TL,TR,BR,BL in the instrument's own orientation, never image-axis sorting.
"""
import cv2
import numpy as np

SIZES = {'localizer': (512, 512), 'speed': (512, 512), 'rpm': (256, 384)}
SPEED_NAMES = ['pivot', 'tip'] + [f'tick_{n}' for n in range(0, 181, 20)]
RPM_NAMES = [f'rpm_{n}' for n in range(0, 13, 2)] + ['bar_end']
POINT_NAMES = {'localizer': ['lcd_tl', 'lcd_tr', 'lcd_br', 'lcd_bl', 'rpm_roi_tl', 'rpm_roi_tr', 'rpm_roi_br', 'rpm_roi_bl'],
               'speed': SPEED_NAMES, 'rpm': RPM_NAMES}

def project(points, matrix):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if not len(points):
        return points
    return cv2.perspectiveTransform(points[None], np.asarray(matrix, np.float64))[0]

def letterbox_matrix(box, size):
    x0, y0, x1, y1 = map(float, box)
    w, h = size
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Empty crop')
    scale = min(w / (x1-x0), h / (y1-y0))
    return np.array([[scale, 0, (w-(x1-x0)*scale)/2-x0*scale],
                     [0, scale, (h-(y1-y0)*scale)/2-y0*scale], [0,0,1]], np.float64)

def dashboard_matrix(polygon, image_shape, margin=.07, jitter=(0,0,1)):
    p = np.asarray(polygon, np.float32)
    if len(p) < 3:
        raise ValueError('No dashboard polygon')
    lo, hi = p.min(0), p.max(0)
    center, extent = (lo+hi)/2, (hi-lo)*(1+2*margin)
    dx,dy,scale = jitter
    center += extent * [dx,dy]
    extent *= scale
    return letterbox_matrix([*(center-extent/2), *(center+extent/2)], SIZES['speed'])

def rpm_matrix(corners, jitter=None):
    """Preserve the complete RPM scale, its fill bar, and adjacent numbers.

    Canonical LCD is 640x320. A generous fixed band [64,360] x [-16,336]
    retains the curved RPM strip. It is NOT derived from ground-truth anchors.
    Training can perturb estimated corners to simulate detector error.
    """
    q = np.asarray(corners, np.float32)
    if q.shape != (4,2) or not np.isfinite(q).all():
        raise ValueError('Four finite semantic corners required for rectification')
    if jitter is not None:
        q = q + np.asarray(jitter, np.float32)
    if abs(cv2.contourArea(q)) < 20 or not cv2.isContourConvex(q.astype(np.float32)):
        raise ValueError('Degenerate LCD quadrilateral')
    rect = cv2.getPerspectiveTransform(q, np.array([[0,0],[640,0],[640,320],[0,320]], np.float32))
    return letterbox_matrix([64,-16,360,336], SIZES['rpm']) @ rect

def rpm_roi_corners(lcd_corners):
    """Virtual crop corners covering the visible scale, not physical LCD corners.

    These let the shared localizer find the RPM context even when the far-right
    LCD corner is outside the video. Only in-frame, justified ROI targets train.
    """
    quad=np.asarray(lcd_corners,np.float32)
    rect=cv2.getPerspectiveTransform(quad,np.array([[0,0],[640,0],[640,320],[0,320]],np.float32))
    return project([[64,-16],[360,-16],[360,336],[64,336]],np.linalg.inv(rect))

def rpm_from_roi_matrix(roi_corners):
    """Inference equivalent of rpm_matrix using four learned context corners."""
    destination=project([[64,-16],[360,-16],[360,336],[64,336]],
                        letterbox_matrix([64,-16,360,336],SIZES['rpm']))
    return cv2.getPerspectiveTransform(np.asarray(roi_corners,np.float32),destination.astype(np.float32))

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

def warp_input(image, matrix, task, lcd_polygon=None):
    result = cv2.warpPerspective(gray(image), matrix, SIZES[task], flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=128)
    mask = np.zeros(result.shape, np.uint8)
    if lcd_polygon is not None and len(lcd_polygon) >= 3:
        cv2.fillPoly(mask, [np.round(project(lcd_polygon,matrix)).astype(np.int32)], 255)
    if task == 'speed':
        result[mask > 0] = 128
    return result, mask

def map_labels(points, visibility, matrix, size):
    out = project(points, matrix)
    w,h = size
    vis = np.asarray(visibility, bool) & np.isfinite(out).all(1)
    vis &= (out[:,0]>=0)&(out[:,0]<=w-1)&(out[:,1]>=0)&(out[:,1]<=h-1)
    out[~vis] = 0
    return out, vis
