from __future__ import annotations
from collections import deque
from core.state_machine.states import State
from config import load_config


class StateMachineMemory:
    def __init__(self):
        cfg = load_config()["state_machine"]
        self._t_cooldown       = cfg["t_cooldown"]
        self._n_abort_max      = cfg["n_abort_max"]
        self._rho_unresolvable = cfg["rho_unresolvable"]
        self._N                = 30 * 30

        self._last_probe_exit_t: float = -999.0
        self._consecutive_aborts: int  = 0
        self._ambiguity_high_flags: deque[bool] = deque(maxlen=self._N)

    #  Cooldown 
    def record_probe_exit(self, t: float):
        self._last_probe_exit_t = t

    def time_since_probe_exit(self, t: float) -> float:
        return t - self._last_probe_exit_t

    #  Abort counter
    def record_abort(self):
        self._consecutive_aborts += 1

    def record_commit(self):
        self._consecutive_aborts = 0

    @property
    def should_yield(self) -> bool:
        return self._consecutive_aborts >= self._n_abort_max

    #  Unresolvable scene detection 
    def record_ambiguity_level(self, A: float, tau_high: float):
        self._ambiguity_high_flags.append(A > tau_high)

    @property
    def is_scene_unresolvable(self) -> bool:
        if len(self._ambiguity_high_flags) < 30:
            return False
        flags = list(self._ambiguity_high_flags)
        return sum(flags) / len(flags) >= self._rho_unresolvable

    #  Reset
    def reset_abort_counter(self):
        self._consecutive_aborts = 0

    def full_reset(self):
        self._last_probe_exit_t = -999.0
        self._consecutive_aborts = 0
        self._ambiguity_high_flags.clear()