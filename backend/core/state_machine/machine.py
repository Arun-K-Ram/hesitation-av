"""
HesitationStateMachine

The behavioral engine. Consumes ambiguity signals and risk scores,
applies G1–G8 transition guards, manages memory, emits state + actions.

Core design principle:
  No transition depends on A(t) alone.
  Every transition depends on the trajectory of A(t) over time.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from core.state_machine.states import State, STATE_LABELS, STATE_COLORS
from core.state_machine.guards import (
    G1, G1_spike, G2, G3, G4, G4_early, G5, G6, G7, G8_emergency,
    G_yield_exit,
)
from core.state_machine.memory import StateMachineMemory
from config import load_config


@dataclass
class MachineInput:
    """Everything the state machine needs to make a decision."""
    t:              float    # current timestamp (seconds)
    A:              float    # composite ambiguity ∈ [0,1]
    dA_dt:          float    # ambiguity velocity
    osc:            float    # ambiguity oscillation (variance)
    risk:           float    # composite risk ∈ [0,1]
    dR_dt:          float    # risk velocity
    risk_projected: float    # risk projected t_horizon seconds forward


@dataclass
class MachineOutput:
    """Everything the state machine emits per tick."""
    state:           State
    state_label:     str
    state_color:     str
    t_in_state:      float
    action:          str        # "cruise" | "probe" | "hold" | "commit" | "abort" | "yield"
    transition_fired: Optional[str]   # e.g. "G3" or None
    flags: dict = field(default_factory=dict)


class HesitationStateMachine:
    def __init__(self):
        self._state       = State.CRUISE
        self._state_entry_t: float = 0.0
        self._memory      = StateMachineMemory()
        self._cfg         = load_config()["state_machine"]
        self._transition_log: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    def tick(self, inp: MachineInput) -> MachineOutput:
        """
        Advance the state machine by one frame.
        Returns a complete output snapshot.
        """
        t_in_state   = inp.t - self._state_entry_t
        fired        = None

        # Record ambiguity level for unresolvability tracking
        self._memory.record_ambiguity_level(inp.A, self._cfg["tau_high"])

        # ── Emergency override (any state) ──────────────────────────────────
        if self._state not in (State.ABORT, State.YIELD):
            if G8_emergency(inp.risk):
                fired = "G8_emergency"
                self._transition(State.ABORT, inp.t)

        # ── Per-state transitions ────────────────────────────────────────────
        if fired is None:
            fired = self._evaluate_transitions(inp, t_in_state)

        t_in_state_final = inp.t - self._state_entry_t

        output = MachineOutput(
            state            = self._state,
            state_label      = STATE_LABELS[self._state],
            state_color      = STATE_COLORS[self._state],
            t_in_state       = t_in_state_final,
            action           = self._state.name.lower(),
            transition_fired = fired,
            flags = {
                "unresolvable":    self._memory.is_scene_unresolvable,
                "yield_pending":   self._memory.should_yield,
                "consecutive_aborts": self._memory._consecutive_aborts,
            }
        )
        return output

    # ── Private: transition evaluator ────────────────────────────────────────

    def _evaluate_transitions(self, inp: MachineInput, t_in_state: float) -> Optional[str]:
        s = self._state

        if s == State.CRUISE:
            # Spike bypass (no confirmation window)
            if G1_spike(inp.A, inp.dA_dt):
                self._transition(State.PROBE, inp.t)
                return "G1_spike"
            # Normal entry
            since_exit = self._memory.time_since_probe_exit(inp.t)
            if G1(inp.A, t_in_state, since_exit):
                if self._memory.should_yield:
                    self._transition(State.YIELD, inp.t)
                    return "G_yield_on_entry"
                self._transition(State.PROBE, inp.t)
                return "G1"

        elif s == State.PROBE:
            # G3 checked BEFORE G2 - commit opportunity takes priority
            if G3(inp.A, inp.dA_dt, inp.osc, inp.risk, t_in_state):
                self._memory.record_commit()
                self._transition(State.COMMIT, inp.t)
                return "G3"
            if G2(inp.A, inp.dA_dt, inp.osc, t_in_state):
                self._memory.record_probe_exit(inp.t)
                self._transition(State.HOLD, inp.t)
                return "G2"

        elif s == State.HOLD:
            if G6(t_in_state, inp.risk):
                self._memory.record_abort()
                self._memory.record_probe_exit(inp.t)
                self._transition(State.ABORT, inp.t)
                return "G6"
            if G5(inp.A, inp.dA_dt, t_in_state):
                self._transition(State.PROBE, inp.t)
                return "G5"

        elif s == State.COMMIT:
            # Anticipatory abort (HOLD, not ABORT - less drastic)
            if G4_early(inp.risk, inp.dR_dt, inp.risk_projected):
                self._transition(State.HOLD, inp.t)
                return "G4_early"
            # Hard abort
            if G4(inp.risk, inp.dA_dt):
                self._memory.record_abort()
                self._transition(State.ABORT, inp.t)
                return "G4"
            # Maneuver complete - back to cruise when scene clears
            if G7(inp.A, inp.risk, inp.dA_dt, t_in_state):
                self._transition(State.CRUISE, inp.t)
                return "G7_from_commit"

        elif s == State.ABORT:
            if G7(inp.A, inp.risk, inp.dA_dt, t_in_state):
                self._transition(State.CRUISE, inp.t)
                return "G7"

        elif s == State.YIELD:
            if G_yield_exit(inp.A, inp.risk, t_in_state):
                self._memory.reset_abort_counter()
                self._transition(State.CRUISE, inp.t)
                return "G_yield_exit"

        return None  # no transition this tick

    def _transition(self, new_state: State, t: float):
        self._transition_log.append({
            "from": self._state.name,
            "to":   new_state.name,
            "t":    t,
        })
        self._state        = new_state
        self._state_entry_t = t

    @property
    def transition_log(self) -> list[dict]:
        return list(self._transition_log)

    def reset(self):
        self._state        = State.CRUISE
        self._state_entry_t = 0.0
        self._memory.full_reset()
        self._transition_log.clear()
