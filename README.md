# Stereo 3D Multi-Object Tracking System  
This project repreents a modular stereo-vision system that detects multiple objects using a **stereo camera setup** and estimates their **3D position in real time**, even during **temporary occlusions**.  
The system demonstrates the core components used in **autonomous driving** and **robot perception**, including:

- **Stereo vision**
- **Dense disparity estimation (SGBM)**
- **YOLO + BoT-SORT multi-object tracking with Re-Identification**
- **3D motion estimation with a Kalman filter**

It processes synchronized stereo frames, triangulates accurate 3D positions for each detected object, predicts their motion through occlusions, and visualizes the results both in the camera view and in a bird’s-eye view (BEV).  

---

## Features

### 3D Tracking From Stereo  
Each detected object is triangulated using disparity information from OpenCV’s StereoSGBM.  
A custom 3D Kalman filter smooths trajectories and predicts motion during short-term occlusions.

### Multi-Object 2D Tracking (BoT-SORT)  
YOLO provides:
- Detection  
- Re-identification  
- Track IDs  

These IDs are matched with 3D tracks for stable identity tracking.

### Prediction Mode (Ghost Tracking)  
If an object is temporarily occluded:
- The Kalman filter predicts its 3D motion
- Motion is clamped to per-class physical limits
- Prediction-only tracks are visualized with blue bounding boxes, whereas detected/measured tracks use green bounding boxes

### Bird’s-Eye View (BEV)  
A clean BEV shows:
- X-Z ground plane positions of all objects  
- Class-colored markers  
- Camera coordinate axes  
- Legend for quick interpretation

### Modular, Production-Friendly Codebase  
All components are split into logical modules:
- Tracking logic  
- 3D geometry  
- Visualization  
- Detection wrapper  
- Configuration  
- Standalone main file  

This makes the system easy to extend, maintain, and reuse in other projects.

---

## Project Structure

```text
├── main.py
└── src/
    └── stereo_3d_tracker/
        ├── __init__.py
        ├── config.py
        ├── kalman.py
        ├── botsort_params.yaml
        ├── stereo.py
        ├── detection.py
        ├── visualization.py
        └── tracker.py
```

### `config.py`  
Centralized configuration:
- FPS, time-step, class IDs
- Per-class tracking parameters
- Stereo projection matrices (P_left, P_right)
- SGBM stereo matcher factory
- BEV map limits  
Keeps all system parameters in one place.

### `stereo.py`  
3D geometry utilities:
- Triangulation from stereo
- Projection of 3D → 2D
- Disparity-based depth estimation  
This is the mathematical backbone of the 3D system.

### `kalman.py`  
Defines the **Track3D** class:
- 9-state constant-acceleration Kalman filter  
- Prediction and update steps  
- Class-specific motion limits  
Each object in the scene has its own Track3D instance.

### `detection.py`  
YOLO + BoT-SORT interface:
- Runs detection + 2D tracking + Re-identification
- Extracts bounding boxes, classes, IDs  
- Parameters defined in the `botsort_params.yaml` 

Keeps the tracker independent from YOLO internals.

### `tracker.py`  
The orchestrator:
- Handles YOLO detections
- Predicts Kalman states
- Updates tracks with new 3D measurements
- Handles prediction-only tracks
- Calls visualization components  
This class glues the entire system together.

### `visualization.py`  
All drawing logic:
- 2D bounding boxes (measurement + prediction)
- BEV canvas creation
- BEV marker rendering  
Only this file controls visuals → easy customization.

### `__init__.py`
Package entry point:
- Exposes the main classes and functions (e.g., `MultiObject3DTracker`, `create_stereo_matcher`, constants)
- Allows clean imports from the package without referencing internal file paths  
This file defines the **public API**, making the library easier to use and integrate.


### `main.py`  
Demo runner:
- Loads YOLO
- Creates the stereo matcher
- Instantiates the multi-object tracker
- Processes every stereo frame and displays the output  
Users interact with this file to run the system.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ApostolosApostolou/stereo-3d-tracker.git
cd stereo-3d-tracker
```

### 2. Instal dependencies
```bash
pip install -r requirements.txt
```

### 3. Istall PyTorch with CUDA (if needed)
For torch and torchvision, ensure to install with the appropriate CUDA version if needed.

Otherwise, the CPU-only versions will be installed from the `requirements.txt` file.

You can find the correct installation command for torch and torchvision at:

https://pytorch.org/get-started/locally/

---

## Usage

### Run the main script
```bash
python main.py
```

#### The system will:
- Load YOLO
- Load stereo matcing functionality
- Track objects in 3D
- Show a combined output window:
    - Top: camera view with 2D boxes (green=measurement, blue=prediction)
    - Bottom: BEV representation

Press Q to exit.

## Key Parameters (high-level)

| Parameter | Description |
|----------|-------------|
| **DT** | Time step (derived from FPS) |
| **MAX_Z** | Maximum accepted triangulation depth |
| **CLASS_KF_PARAMS** | Per-class motion constraints for prediction |
| **MIN_DETECT_FRAMES_FOR_PRED** | Minimum measurements before enabling prediction mode |
| **P_LEFT / P_RIGHT** | Stereo projection matrices |
| **BEV_X_MAX / BEV_Z_MAX** | Size of BEV map in meters |

These live in **`config.py`** for easy tuning.

---

## Extending the System

Because everything is modular:

- Want a **new detector?** Replace `detection.py`.  
- Want a **new Kalman model?** Modify `kalman.py`.  
- Want a **new BEV visualization?** Edit only `visualization.py`.  
- And so on ...

Each piece is isolated and can be replaced without touching the rest.

---

## Dataset & Testing

This system was tested on the **KITTI Vision Benchmark Suite**.  
The dataset provides:

- **Rectified stereo image pairs**  
- **Precomputed projection matrices** (`P_left`, `P_right`)  
- **Consistent calibration parameters across sequences (provided in `calibration_results.txt`)**

Because KITTI’s images are already rectified and include their own stereo projection matrices, the system can directly perform:

- Disparity estimation  
- 3D triangulation  

without requiring additional calibration.

If you use this system with your own cameras, you must first:

1. **Calibrate the stereo rig** (intrinsics + extrinsics)  
2. **Rectify both image streams**  
3. **Extract or compute projection matrices** for the left and right cameras  

Only then can stereo triangulation and 3D tracking work correctly.

---

## Results

The video below demonstrates the full system in action:

https://www.youtube.com/watch?v=3aPpbea5wNk 

It shows real-time multi-object 3D tracking from a stereo camera setup, with both the camera view and the bird’s-eye view (BEV) rendered simultaneously.

Overall, the system:

- Accurately detects and tracks road users in 3D  
- Maintains stable identities for most objects  
- Continues tracking through short-term **occlusions** using Kalman prediction  
- Provides smooth 3D trajectories and intuitive BEV visualization  

### Limitations

While the system performs well, a few limitations appear in challenging scenes:

- **Re-identification mismatches**:  
  When an object disappears briefly and reappears, the BoT-SORT re-ID may occasionally fail.  
  In these casses, the system treats the re-appearing object as a *new* track instead of continuing the old one.

- **Ghost predictions**:  
  If an object is misidentified after occlusion, the old track may continue predicting motion even though no object is present.  
  This creates “ghost boxes”, predicted motions drifting in the frame for a few frames until the system times them out.

These issues can be improved with stronger appearance embeddings and temporal consistency checks. Better embeddings help the tracker recognize when an object that reappears after an occlusion is the same one seen earlier, reducing ID switches. Temporal checks ensure that object motion remains physically reasonable, for example, preventing tracks from suddenly jumping in depth, moving sideways unrealistically fast, or drifting when no detection matches the predicted location. Together, these improvements make multi-object tracking more stable and resilient in challenging scenarios.

---

## License

MIT License.  
Feel free to use this system in your own robotics/computer vision projects.




