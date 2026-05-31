from __future__ import annotations
import numpy as np
from collections import deque
from config import load_config


class PerceptualAmbiguity:
    def __init__(self):
        cfg = load_config()
        self._N        = cfg["ambiguity"]["window_frames"]
        self._lambda   = cfg["perceptual"]["lambda_aleatoric"]
        self._min_conf = cfg["perceptual"]["min_detection_confidence"]
        self._confidence_window: deque[float] = deque(maxlen=self._N)

    def update(self, confidence: float | None) -> float:
        c = confidence if confidence is not None else self._min_conf
        c = float(np.clip(c, 0.0, 1.0))
        self._confidence_window.append(c)
        return self.value

    @property
    def value(self) -> float:
        if len(self._confidence_window) < 2:
            return 0.5
        arr        = np.array(self._confidence_window)
        aleatoric  = float(np.mean(1.0 - arr))
        epistemic  = float(np.clip(np.std(arr) * 4.0, 0.0, 1.0))
        return float(np.clip(
            self._lambda * aleatoric + (1.0 - self._lambda) * epistemic,
            0.0, 1.0
        ))

    @property
    def aleatoric(self) -> float:
        if not self._confidence_window:
            return 0.5
        return float(np.mean(1.0 - np.array(self._confidence_window)))

    @property
    def epistemic(self) -> float:
        if len(self._confidence_window) < 2:
            return 0.0
        return float(np.clip(np.std(np.array(self._confidence_window)) * 4.0, 0.0, 1.0))

    def reset(self):
        self._confidence_window.clear()