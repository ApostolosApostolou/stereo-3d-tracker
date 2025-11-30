from collections import defaultdict

import cv2
import numpy as np

from .config import (
    DT,
    P_LEFT,
    P_RIGHT,
    CLASSES_TO_DETECT,
    MIN_DETECT_FRAMES_FOR_PRED,
)
from .config import get_max_missed_frames_for_class
from .kalman import Track3D
from .stereo import triangulate_box_sgbm, basic_3d_checks
from .detection import object_detect_and_track, get_image_coordinates
from .visualization import (
    create_bev_canvas,
    draw_bev_marker,
    draw_measurement_bbox,
    draw_prediction_bbox,
)


class MultiObject3DTracker:
    """
    Wraps:
        - 3D Kalman filters (Track3D)
        - YOLO detections + BoT-SORT
        - SGBM disparity
        - Visualization (2D + BEV via visualization.py)
    """

    def __init__(self, model, stereo_matcher):
        self.model = model
        self.stereo = stereo_matcher

        # track_id -> Track3D
        self.kf_tracks: dict[int, Track3D] = {}
        # track_id -> list of (X, Y, Z, frame_idx)
        self.tracks_3d_history = defaultdict(list)

    def process_frame(self, left_image, right_image, frame_idx: int, dt_frame: float = DT):
        """
        Process one stereo frame pair. Returns a stacked image:
        [RGB view; BEV view].
        """
        final_vis = left_image.copy()
        h_img, w_img = final_vis.shape[:2]

        bev_vis = create_bev_canvas(w_img, h_img)

        # Disparity
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        disp_raw = self.stereo.compute(left_gray, right_gray).astype(np.float32)
        disp = disp_raw / 16.0

        # Detections + tracking (2D)
        tracked_boxes_left = object_detect_and_track(
            final_vis, self.model, CLASSES_TO_DETECT, conf=0.55
        )

        # Kalman prediction step for all existing tracks
        for track in self.kf_tracks.values():
            track.predict(dt_frame)

        updated_ids = set()

        # Measurement update loop for detected boxes
        for box_left in tracked_boxes_left:
            _, _, track_id = get_image_coordinates(box_left)
            if track_id is None:
                continue

            cls_id = int(box_left.cls[0])

            X_meas, Y_meas, Z_meas = triangulate_box_sgbm(
                box_left, disp, P_LEFT, P_RIGHT, min_disp=0.2
            )
            if not basic_3d_checks(X_meas, Y_meas, Z_meas):
                continue

            # Create new track if needed
            if track_id not in self.kf_tracks:
                self.kf_tracks[track_id] = Track3D(
                    X_meas, Y_meas, Z_meas, dt=dt_frame, class_id=cls_id
                )

            track = self.kf_tracks[track_id]
            track.class_id = cls_id
            track.update(X_meas, Y_meas, Z_meas)
            track.last_bbox = tuple(map(int, box_left.xyxy[0]))

            Xf, Yf, Zf = track.get_xyz()
            track.last_xyz = (Xf, Yf, Zf)
            self.tracks_3d_history[track_id].append((Xf, Yf, Zf, frame_idx))
            updated_ids.add(track_id)

            # Visualization: 2D bbox + BEV marker
            draw_measurement_bbox(final_vis, box_left, Xf, Yf, Zf, cls_id, w_img, h_img)
            draw_bev_marker(bev_vis, Xf, Zf, cls_id)

        # Prediction-only tracks (ghosts)
        for track_id, track in list(self.kf_tracks.items()):
            if track_id in updated_ids:
                continue

            cid = getattr(track, "class_id", None)
            max_missed = get_max_missed_frames_for_class(cid)

            if track.frames_since_update > max_missed:
                # Let it silently fade
                continue

            if track.detect_count < MIN_DETECT_FRAMES_FOR_PRED:
                continue

            Xp, Yp, Zp = self._predict_clamped_xyz(track)
            if not basic_3d_checks(Xp, Yp, Zp):
                continue

            # Visualization: prediction bbox + BEV marker.
            keep_track = draw_prediction_bbox(final_vis, track, Xp, Yp, Zp, w_img, h_img)
            if not keep_track:
                # Too close to edges -> delete
                del self.kf_tracks[track_id]
                continue

            if cid is not None:
                draw_bev_marker(bev_vis, Xp, Zp, cid)

        stacked = cv2.vconcat([final_vis, bev_vis])
        return stacked

    # Internal helpers 

    def _predict_clamped_xyz(self, track: Track3D) -> tuple[float, float, float]:
        """
        Clamp 3D motion during ghost prediction, enforcing:
          - per-class max dX, dZ
          - flat ground assumption on Y
        """
        Xp_raw, Yp_raw, Zp_raw = track.get_xyz()

        if track.last_xyz is None:
            track.last_xyz = (Xp_raw, Yp_raw, Zp_raw)
            return Xp_raw, Yp_raw, Zp_raw

        X_prev, Y_prev, Z_prev = track.last_xyz
        dX = Xp_raw - X_prev
        dZ = Zp_raw - Z_prev

        if track.MAX_PRED_DX is not None:
            dX = np.clip(dX, -track.MAX_PRED_DX, track.MAX_PRED_DX)
        if track.MAX_PRED_DZ is not None:
            dZ = np.clip(dZ, -track.MAX_PRED_DZ, track.MAX_PRED_DZ)

        Xp = X_prev + dX
        Yp = Y_prev
        Zp = Z_prev + dZ

        # Write back to state (enforce flat-ground & clamped motion)
        track.x[0, 0] = Xp
        track.x[1, 0] = Yp
        track.x[2, 0] = Zp
        track.x[4, 0] = 0.0  # Vy
        track.x[7, 0] = 0.0  # Ay

        track.last_xyz = (Xp, Yp, Zp)
        return Xp, Yp, Zp
