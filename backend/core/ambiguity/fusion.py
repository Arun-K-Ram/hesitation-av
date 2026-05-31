from __future__ import annotations
import numpy as np
from collections import deque
from config import load_config
from core.ambiguity.perceptual import PerceptualAmbiguity
from core.ambiguity.behavioral import BehavioralAmbiguity


class AmbiguityFusion:
    def __init__(self):
        cfg         = load_config()
        self._alpha = cfg["ambiguity"]["alpha"]
        self._beta  = cfg["ambiguity"]["beta"]
        self._N     = cfg["ambiguity"]["window_frames"]

        self.perceptual = PerceptualAmbiguity()
        self.behavioral = BehavioralAmbiguity()
        self._history:   deque[float] = deque(maxlen=self._N)
        self._t_history: deque[float] = deque(maxlen=self._N)

    def update(self, Ap: float, Ab: float, t: float) -> dict:
        A = float(np.clip(self._alpha * Ap + self._beta * Ab, 0.0, 1.0))
        self._history.append(A)
        self._t_history.append(t)
        return {
            "A":     A,
            "Ap":    Ap,
            "Ab":    Ab,
            "dA_dt": self._derivative(),
            "osc":   self._oscillation(),
        }

    def _derivative(self) -> float:
        if len(self._history) < 5:
            return 0.0
        y = np.array(self._history)
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    def _oscillation(self) -> float:
        if len(self._history) < 5:
            return 0.0
        return float(np.var(np.array(self._history)))

    def reset(self):
        self._history.clear()
        self._t_history.clear()
        self.perceptual.reset()
        self.behavioral.reset()