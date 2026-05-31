from __future__ import annotations
import numpy as np
from config import load_config


class KalmanTracker:
    def __init__(self, initial_x: float, initial_y: float, dt: float = 1/30):
        cfg = load_config()
        q = cfg["behavioral"]["kalman_process_noise"]
        r = cfg["behavioral"]["kalman_measurement_noise"]

        self.dt = dt
        self.n_frames = 0

        self.x = np.array([initial_x, initial_y, 0.0, 0.0], dtype=float)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.Q = np.eye(4) * q
        self.R = np.eye(2) * r
        self.P = np.eye(4) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: np.ndarray):
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.n_frames += 1
        return self.x[:2].copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def heading(self) -> float:
        vx, vy = self.velocity
        return float(np.arctan2(vy, vx))

    @property
    def is_mature(self) -> bool:
        cfg = load_config()
        return self.n_frames >= cfg["behavioral"]["min_track_frames"]