from __future__ import annotations

import cv2
import numpy as np

from .config import BEV_X_MAX, BEV_Z_MAX, CLASS_NAMES, EDGE_MARGIN, P_LEFT
from .stereo import project_3d_to_2d
from .kalman import Track3D


# BEV helpers

def project_to_bev(X: float, Z: float, bev_w: int, bev_h: int,
                   x_max: float = BEV_X_MAX,
                   z_max: float = BEV_Z_MAX) -> tuple[int, int]:
    """Project (X, Z) into BEV image coordinates."""
    X_clamped = max(-x_max, min(X, x_max))
    Z_clamped = max(0.0, min(Z, z_max))

    u = (X_clamped + x_max) / (2.0 * x_max)  # 0..1
    v = 1.0 - (Z_clamped / z_max)            # 1..0

    px = int(u * (bev_w - 1))
    py = int(v * (bev_h - 1))
    return px, py


def create_bev_canvas(w_img: int, h_img: int) -> np.ndarray:
    """Create BEV background with axes and legend."""
    bev_h = h_img // 2
    bev_vis = np.zeros((bev_h, w_img, 3), dtype=np.uint8)
    bev_vis[:] = (30, 30, 30)

    # Center vertical line (X=0)
    cv2.line(bev_vis, (w_img // 2, 0), (w_img // 2, bev_h - 1), (80, 80, 80), 1)

    origin_x = w_img // 2
    origin_y = bev_h - 1
    axis_len = 40

    # Z axis (forward)
    cv2.arrowedLine(
        bev_vis,
        (origin_x, origin_y),
        (origin_x, origin_y - axis_len),
        (220, 220, 220),
        2,
        tipLength=0.3,
    )
    cv2.putText(
        bev_vis, "Z",
        (origin_x - 15, origin_y - axis_len - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
    )

    # X axis (right)
    cv2.arrowedLine(
        bev_vis,
        (origin_x, origin_y),
        (origin_x + axis_len, origin_y),
        (220, 220, 220),
        2,
        tipLength=0.3,
    )
    cv2.putText(
        bev_vis, "X",
        (origin_x + axis_len + 5, origin_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1,
    )

    # Legend
    legend_items = [
        ("Person",  (0, 0, 255)),
        ("Bicycle", (0, 255, 255)),
        ("Car",     (0, 255, 0)),
        ("Motorcycle", (255, 0, 0)),
        ("Bus",     (255, 255, 0)),
    ]
    legend_x = 10
    legend_y = 20
    for i, (name, color) in enumerate(legend_items):
        cy = legend_y + i * 20
        cv2.circle(bev_vis, (legend_x + 5, cy - 5), 4, color, -1)
        cv2.putText(
            bev_vis,
            name,
            (legend_x + 15, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

    return bev_vis


def draw_bev_marker(bev_img: np.ndarray, X: float, Z: float, cls_id: int) -> None:
    """Draw a class-colored dot in BEV for (X, Z)."""
    if cls_id == 0:
        color = (0, 0, 255)      # Person
    elif cls_id == 1:
        color = (0, 255, 255)    # Bicycle
    elif cls_id == 2:
        color = (0, 255, 0)      # Car
    elif cls_id == 3:
        color = (255, 0, 0)      # Motorcycle
    elif cls_id == 5:
        color = (255, 255, 0)    # Bus
    else:
        color = (255, 255, 255)  # Others

    h_bev, w_bev = bev_img.shape[:2]
    px, py = project_to_bev(X, Z, w_bev, h_bev)
    cv2.circle(bev_img, (px, py), 4, color, -1)


# 2D bounding box drawing

def draw_measurement_bbox(
    vis_img: np.ndarray,
    box_left,
    X: float,
    Y: float,
    Z: float,
    cls_id: int,
    w_img: int,
    h_img: int,
) -> None:
    """
    Draw a green bounding box and labels for a measurement update.

    Uses:
      - projected 3D point as box center
      - original YOLO box size
    """
    # Project 3D point back to image
    u_proj, v_proj = project_3d_to_2d(P_LEFT, X, Y, Z)

    x1_raw, y1_raw, x2_raw, y2_raw = map(int, box_left.xyxy[0])
    w_box = x2_raw - x1_raw
    h_box = y2_raw - y1_raw

    x1 = int(u_proj - w_box / 2)
    x2 = int(u_proj + w_box / 2)
    y1 = int(v_proj - h_box / 2)
    y2 = int(v_proj + h_box / 2)

    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    class_name = CLASS_NAMES.get(cls_id, "Unknown")
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label_lines = [
        f"{class_name}",
        f"X: {X:.2f} m",
        f"Y: {Y:.2f} m",
        f"Z: {Z:.2f} m",
    ]
    text_x = x1
    text_y = max(15, y1 - 60)
    line_height = 18
    for i, line in enumerate(label_lines):
        cv2.putText(
            vis_img,
            line,
            (text_x, text_y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def draw_prediction_bbox(
    vis_img: np.ndarray,
    track: Track3D,
    Xp: float,
    Yp: float,
    Zp: float,
    w_img: int,
    h_img: int,
) -> bool:
    """
    Draw a blue bounding box and labels for a prediction-only track.

    Returns:
        keep_track (bool): False if box is too close to edges and track
                           should be dropped.
    """
    if track.last_bbox is None:
        return False

    u_pred, v_pred = project_3d_to_2d(P_LEFT, Xp, Yp, Zp)
    x1_last, y1_last, x2_last, y2_last = track.last_bbox

    w_box = x2_last - x1_last
    h_box = y2_last - y1_last

    x1 = int(u_pred - w_box / 2)
    x2 = int(u_pred + w_box / 2)
    y1 = int(v_pred - h_box / 2)
    y2 = int(v_pred + h_box / 2)

    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    too_close_to_edge = (
        x1 <= EDGE_MARGIN
        or y1 <= EDGE_MARGIN
        or x2 >= (w_img - 1 - EDGE_MARGIN)
        or y2 >= (h_img - 1 - EDGE_MARGIN)
    )
    if too_close_to_edge:
        return False

    class_id = getattr(track, "class_id", None)
    class_name = CLASS_NAMES.get(class_id, "Unknown")

    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label_lines = [
        f"{class_name}",
        f"X: {Xp:.2f} m",
        f"Y: {Yp:.2f} m",
        f"Z: {Zp:.2f} m",
    ]
    text_x = x1
    text_y = max(15, y1 - 60)
    line_height = 18
    for i, line in enumerate(label_lines):
        cv2.putText(
            vis_img,
            line,
            (text_x, text_y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return True
