import numpy as np

from .config import CLASS_KF_PARAMS, DT


class Track3D:
    """
    Kalman filter to track 3D position of an object.

    State:
        x = [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]^T
    Measurement:
        z = [X, Y, Z]^T
    """

    def __init__(self, X, Y, Z, dt: float = DT, class_id: int | None = None):
        self.dt = dt
        self.class_id = class_id

        # Per-class clamping parameters
        params = CLASS_KF_PARAMS.get(class_id, {})
        self.MAX_PRED_DX = params.get("MAX_PRED_DX")
        self.MAX_PRED_DZ = params.get("MAX_PRED_DZ")

        # State vector
        self.x = np.zeros((9, 1), dtype=np.float32)
        self.x[0, 0] = X
        self.x[1, 0] = Y
        self.x[2, 0] = Z

        # Covariance
        self.P = np.eye(9, dtype=np.float32) * 50.0

        # Transition matrix
        self.F = np.eye(9, dtype=np.float32)
        self._update_F(dt)

        # Measurement model
        self.H = np.zeros((3, 9), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise
        self.Q = np.diag([
            0.3, 0.3, 0.3,
            0.5, 0.5, 0.5,
            1.0, 1.0, 1.0
        ]).astype(np.float32)

        # Measurement noise
        self.R = np.diag([1.5, 0.5, 2.0]).astype(np.float32)

        # Tracking meta
        self.frames_since_update = 0
        self.last_bbox = None
        self.detect_count = 1
        self.last_xyz = (X, Y, Z)

    def _update_F(self, dt: float) -> None:
        self.F = np.eye(9, dtype=np.float32)
        dt2 = 0.5 * dt * dt

        # X
        self.F[0, 3] = dt
        self.F[0, 6] = dt2
        self.F[3, 6] = dt

        # Y
        self.F[1, 4] = dt
        self.F[1, 7] = dt2
        self.F[4, 7] = dt

        # Z
        self.F[2, 5] = dt
        self.F[2, 8] = dt2
        self.F[5, 8] = dt

    def predict(self, dt: float):
        self.dt = dt
        self._update_F(dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.frames_since_update += 1
        return self.x

    def update(self, X: float, Y: float, Z: float) -> None:
        z = np.array([[X], [Y], [Z]], dtype=np.float32)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(9, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        self.frames_since_update = 0
        self.detect_count += 1

    def get_xyz(self) -> tuple[float, float, float]:
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
        )
