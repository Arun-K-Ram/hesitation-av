from __future__ import annotations
import numpy as np
from collections import deque
from config import load_config


def ttc_risk(distance: float, relative_velocity: float) -> float:
    cfg     = load_config()
    eps     = cfg["risk"]["ttc_velocity_floor"]
    tau_ref = cfg["risk"]["tau_ref"]
    v_safe  = max(abs(relative_velocity), eps)
    ttc     = distance / v_safe
    return float(np.exp(-ttc / tau_ref))


def trajectory_conflict(
    ego_path:       np.ndarray,
    agent_position: np.ndarray,
    agent_velocity: np.ndarray,
) -> float:
    cfg    = load_config()
    K      = cfg["risk"]["trajectory_samples"]
    H      = cfg["risk"]["trajectory_horizon"]
    dt     = 1.0 / 10.0
    steps  = int(H / dt)
    radius = 0.5

    conflicts = 0
    for _ in range(K):
        noise = np.random.randn(2) * 0.3
        v     = agent_velocity + noise
        pos   = agent_position.copy()
        hit   = False
        for _ in range(steps):
            pos = pos + v * dt
            if np.any(np.linalg.norm(ego_path - pos, axis=1) < radius):
                hit = True
                break
        if hit:
            conflicts += 1
    return float(conflicts / K)


class CorrectionMonitor:
    def __init__(self):
        cfg = load_config()
        self._window  = cfg["risk"]["correction_window"]
        self._jerk_th = cfg["risk"]["jerk_threshold"]
        self._path_th = cfg["risk"]["path_deviation_threshold"]

        self._active   = False
        self._t_commit = None
        self._last_v   = None
        self._jerk_history = deque()
        self._path_history = deque()

    def on_commit(self, t: float):
        self._active   = True
        self._t_commit = t
        self._jerk_history.clear()
        self._path_history.clear()
        self._last_v = None

    def update(self, t: float, velocity: float, path_deviation: float) -> float:
        if not self._active:
            return 0.0
        if t - self._t_commit > self._window:
            self._active = False
            return 0.0
        if self._last_v is not None:
            jerk = abs(velocity - self._last_v) / (1.0 / 30.0)
            self._jerk_history.append(min(jerk / self._jerk_th, 1.0))
        self._last_v = velocity
        self._path_history.append(min(path_deviation / self._path_th, 1.0))
        j = float(np.mean(self._jerk_history)) if self._jerk_history else 0.0
        p = float(np.mean(self._path_history)) if self._path_history else 0.0
        return float(np.clip(0.5 * j + 0.5 * p, 0.0, 1.0))


class RiskComposite:
    def __init__(self):
        cfg = load_config()
        self._gamma = cfg["risk"]["gamma"]
        self._eta   = cfg["risk"]["eta"]
        self._xi    = cfg["risk"]["xi"]
        self.correction_monitor = CorrectionMonitor()
        self._history = deque(maxlen=30)

    def compute(self, ttc_r: float, traj_c: float,
                correction_s: float) -> dict:
        R = float(np.clip(
            self._gamma * ttc_r + self._eta * traj_c + self._xi * correction_s,
            0.0, 1.0
        ))
        self._history.append(R)
        dR_dt = 0.0
        if len(self._history) >= 5:
            y     = np.array(self._history)
            x     = np.arange(len(y), dtype=float)
            dR_dt = float(np.polyfit(x, y, 1)[0])
        return {
            "risk":       R,
            "ttc_risk":   ttc_r,
            "traj_conf":  traj_c,
            "correction": correction_s,
            "dR_dt":      dR_dt,
        }