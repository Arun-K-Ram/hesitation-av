from enum import Enum, auto


class State(Enum):
    CRUISE = auto()
    PROBE  = auto()
    HOLD   = auto()
    COMMIT = auto()
    ABORT  = auto()
    YIELD  = auto()


STATE_LABELS = {
    State.CRUISE: "CRUISE",
    State.PROBE:  "PROBE",
    State.HOLD:   "HOLD",
    State.COMMIT: "COMMIT",
    State.ABORT:  "ABORT",
    State.YIELD:  "YIELD",
}

STATE_COLORS = {
    State.CRUISE: "#22c55e",
    State.PROBE:  "#eab308",
    State.HOLD:   "#f97316",
    State.COMMIT: "#3b82f6",
    State.ABORT:  "#ef4444",
    State.YIELD:  "#a855f7",
}