import os

# YOLO & BoT-SORT params file path
file_path = os.path.dirname(__file__) # current directory
params_path = os.path.join(file_path, "botsort_params.yaml") # path to params file

def object_detect_and_track(image, model, classes_to_detect, conf: float = 0.3):
    """
    Detect + track objects with YOLO and BoT-SORT.
    Returns list of YOLO boxes with track IDs.
    """
    results = model.track(
        image,
        classes=classes_to_detect,
        conf=conf,
        imgsz=960,
        persist=True,
        tracker=params_path,
        verbose=False,
    )

    tracked_boxes = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) in classes_to_detect:
                tracked_boxes.append(box)
    return tracked_boxes


def get_image_coordinates(box):
    """
    Return (u, v, track_id) for a YOLO box.
    u: center in X, v: bottom edge in Y.
    """
    x1, y1, x2, y2 = box.xyxy[0]
    u = (x1 + x2) / 2
    v = y2

    track_id = None
    if hasattr(box, "id") and box.id is not None:
        track_id = int(box.id[0])

    return float(u), float(v), track_id
