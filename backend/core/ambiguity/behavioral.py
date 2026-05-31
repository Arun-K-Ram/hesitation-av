from __future__ import annotations
import numpy as np
from collections import deque
from config import load_config


class BehavioralAmbiguity:
    def __init__(self):
        cfg = load_config()
        self._N             = cfg["ambiguity"]["window_frames"]
        self._speed_bins    = cfg["behavioral"]["velocity_bins_speed"]
        self._heading_bins  = cfg["behavioral"]["velocity_bins_heading"]
        self._velocity_window: deque[np.ndarray] = deque(maxlen=self._N)
        self._max_speed_seen: float = 1.0

    def update(self, tracker) -> float:
        if tracker is None or not tracker.is_mature:
            return 0.0
        v     = tracker.velocity.copy()
        speed = tracker.speed
        if speed > self._max_speed_seen:
            self._max_speed_seen = speed
        self._velocity_window.append(v)
        return self.value

    @property
    def value(self) -> float:
        if len(self._velocity_window) < 5:
            return 0.0
        vels     = np.array(self._velocity_window)
        speeds   = np.linalg.norm(vels, axis=1)
        headings = np.arctan2(vels[:, 1], vels[:, 0])

        speed_idx = np.floor(
            np.clip(speeds / (self._max_speed_seen + 1e-6), 0.0, 1.0 - 1e-9)
            * self._speed_bins
        ).astype(int)

        heading_idx = np.floor(
            (headings + np.pi) / (2 * np.pi + 1e-9) * self._heading_bins
        ).astype(int)

        combined   = speed_idx * self._heading_bins + heading_idx
        total_bins = self._speed_bins * self._heading_bins
        counts     = np.bincount(combined, minlength=total_bins).astype(float)
        probs      = counts / counts.sum()
        probs      = probs[probs > 0]
        entropy    = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(total_bins)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def reset(self):
        self._velocity_window.clear()