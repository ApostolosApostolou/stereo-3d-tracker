import numpy as np
import cv2

# Frame rate and time step
FPS = 10.0
DT = 1.0 / FPS

# Max missed frames per class
MAX_MISSED_FRAMES_PERSON = 30
MAX_MISSED_FRAMES_BICYCLE = 25
MAX_MISSED_FRAMES_CAR = 20
MAX_MISSED_FRAMES_MOTORCYCLE = 20
MAX_MISSED_FRAMES_BUS = 20

# Min detections before enabling prediction-only tracking
MIN_DETECT_FRAMES_FOR_PRED = 10

# If predicted box gets closer than this to an edge, the track is dropped
EDGE_MARGIN = 10  # pixels

# 3D sanity
MAX_Z = 100.0

# Per-class Kalman filter parameters
CLASS_KF_PARAMS = {
    0: {"MAX_PRED_DX": 1.0, "MAX_PRED_DZ": 1.0},   # Person
    1: {"MAX_PRED_DX": 0.8, "MAX_PRED_DZ": 0.5},   # Bicycle
    2: {"MAX_PRED_DX": 5.0, "MAX_PRED_DZ": 5.0},   # Car
    3: {"MAX_PRED_DX": 5.0, "MAX_PRED_DZ": 5.0},   # Motorcycle
    5: {"MAX_PRED_DX": 5.0, "MAX_PRED_DZ": 5.0},   # Bus
}

# Stereo projection matrices (rectified)
P_LEFT = np.array([
    [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02,  4.485728000000e+01],
    [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02,  2.163791000000e-01],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00,  2.745884000000e-03]
], dtype=np.float32)

P_RIGHT = np.array([
    [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.395242000000e+02],
    [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02,  2.199936000000e+00],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00,  2.729905000000e-03]
], dtype=np.float32)

# Stereo SGBM parameters
SGBM_BLOCK_SIZE = 5
SGBM_NUM_DISPARITIES = 16 * 10

# BEV params
BEV_X_MAX = 50.0
BEV_Z_MAX = 100.0

# Classes
CLASSES_TO_DETECT = [0, 1, 2, 3, 5]
CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
}


def create_stereo_matcher():
    """Create and return a configured StereoSGBM matcher."""
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=SGBM_NUM_DISPARITIES,
        blockSize=SGBM_BLOCK_SIZE,
        P1=8 * 3 * SGBM_BLOCK_SIZE ** 2,
        P2=32 * 3 * SGBM_BLOCK_SIZE ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def get_max_missed_frames_for_class(class_id: int) -> int:
    """Return per-class maximum allowed missed frames."""
    if class_id == 0:
        return MAX_MISSED_FRAMES_PERSON
    if class_id == 1:
        return MAX_MISSED_FRAMES_BICYCLE
    if class_id == 2:
        return MAX_MISSED_FRAMES_CAR
    if class_id == 3:
        return MAX_MISSED_FRAMES_MOTORCYCLE
    if class_id == 5:
        return MAX_MISSED_FRAMES_BUS
    # Fallback
    return MAX_MISSED_FRAMES_CAR
