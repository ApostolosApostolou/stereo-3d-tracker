import numpy as np
import cv2

from .config import MAX_Z


def get_3d_points(u_left, v_left, u_right, v_right, P_l, P_r):
    pts_left = np.array([[u_left], [v_left]], dtype=np.float32)
    pts_right = np.array([[u_right], [v_right]], dtype=np.float32)
    X_hom = cv2.triangulatePoints(P_l, P_r, pts_left, pts_right)
    X = X_hom[:3] / X_hom[3]
    return X.ravel()


def project_3d_to_2d(P, X, Y, Z):
    pt_3d = np.array([[X], [Y], [Z], [1.0]], dtype=np.float32)
    uvw = P @ pt_3d
    u = uvw[0, 0] / uvw[2, 0]
    v = uvw[1, 0] / uvw[2, 0]
    return float(u), float(v)


def triangulate_box_sgbm(box_left, disp, P_l, P_r, min_disp: float = 1.0):
    """
    Disparity-based triangulation from a YOLO box using a bottom band
    of the box in the disparity map.
    """
    h_img, w_img = disp.shape[:2]
    x1, y1, x2, y2 = map(int, box_left.xyxy[0])

    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    if x2 <= x1 or y2 <= y1:
        return np.nan, np.nan, np.nan

    # Use only center region in width
    drop_percent = 0.25
    w_box = x2 - x1
    side_margin = int(drop_percent * w_box)
    x1_c = x1 + side_margin
    x2_c = x2 - side_margin
    if x2_c <= x1_c:
        return np.nan, np.nan, np.nan

    # Use bottom band in height
    band_frac = 0.3
    h_box = y2 - y1
    band_h = max(3, int(band_frac * h_box))
    y_band1 = max(0, y2 - band_h)
    y_band2 = y2

    disp_band = disp[y_band1:y_band2, x1_c:x2_c]
    if disp_band.size == 0:
        return np.nan, np.nan, np.nan

    valid = disp_band[disp_band > min_disp]
    if valid.size == 0:
        return np.nan, np.nan, np.nan

    d_med = float(np.median(valid))

    uL = 0.5 * (x1 + x2)
    vL = 0.5 * (y1 + y2)
    uR = uL - d_med
    vR = vL

    return get_3d_points(uL, vL, uR, vR, P_l, P_r)


def basic_3d_checks(X, Y, Z) -> bool:
    if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z):
        return False
    if Z <= 0 or Z > MAX_Z:
        return False
    return True
