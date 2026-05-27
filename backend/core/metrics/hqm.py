"""
HesitationQualityMetric (HQM)

Episode-level metric decomposing hesitation quality into:
  S  - Safety Gain          ∈ [-1, 1]
  E  - Temporal Efficiency  ∈ [0, 1]
  B  - Behavioral Stability ∈ [0, 1]
  R  - Resolution Quality   ∈ {0, 0.5, 1}

HQM = α·S + β·E + γ·B + δ·R     ∈ [-α, 1]

Negative HQM = hesitation hurt more than it helped.
Greedy baseline HQM ≈ 0.60 (by construction - S=0, E=1, B=1, R=1).
Your system must beat 0.60 to justify its complexity.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from core.state_machine.states import State
from config import load_config


class ResolutionType(Enum):
    EARNED_COMMIT   = 1.0   # G3 - earned commit
    ANTICIPATORY_ABORT = 1.0  # G4_early - predicted danger
    FORCED_ABORT    = 0.5   # G6-timeout - ambiguity never resolved
    DEADLOCK_YIELD  = 0.0   # N_abort_max reached


@dataclass
class Episode:
    """A single hesitation episode from PROBE-entry to resolution."""
    t_start:          float
    risk_at_greedy:   float   # Risk(t_greedy) - counterfactual

    # Filled during episode
    state_sequence:   list = field(default_factory=list)
    A_history:        list = field(default_factory=list)
    n_transitions:    int  = 0
    t_g3_eligible:    Optional[float] = None   # first moment G3 was possible

    # Filled at resolution
    t_end:            Optional[float] = None
    risk_at_commit:   Optional[float] = None
    resolution_type:  Optional[ResolutionType] = None

    @property
    def is_complete(self) -> bool:
        return self.t_end is not None

    @property
    def duration(self) -> float:
        if self.t_end is None:
            return 0.0
        return self.t_end - self.t_start


class HQMComputer:
    def __init__(self):
        cfg = load_config()["hqm"]
        self._alpha = cfg["alpha_s"]
        self._beta  = cfg["beta_e"]
        self._gamma = cfg["gamma_b"]
        self._delta = cfg["delta_r"]
        self._T_budget = cfg["t_budget"]
        self._lambda   = cfg["lambda_transitions"]
        self._w1       = cfg["w1_stability"]
        self._w2       = cfg["w2_stability"]

        self._current_episode: Optional[Episode] = None
        self._completed_episodes: list[dict] = []

    # ── Episode lifecycle ─────────────────────────────────────────────────────

    def on_probe_enter(self, t: float, risk_at_greedy: float):
        """Call when state machine enters PROBE."""
        self._current_episode = Episode(
            t_start        = t,
            risk_at_greedy = risk_at_greedy,
        )

    def on_tick(
        self,
        state: State,
        A: float,
        risk: float,
        dA_dt: float,
        osc: float,
        t: float,
        g3_eligible: bool,
    ):
        """Call every frame while episode is active."""
        ep = self._current_episode
        if ep is None or ep.is_complete:
            return

        ep.state_sequence.append(state)
        ep.A_history.append(A)

        # Track first moment G3 conditions were satisfiable
        if g3_eligible and ep.t_g3_eligible is None:
            ep.t_g3_eligible = t

    def on_transition(self, transition: str, t: float, risk: float):
        """Call on every state machine transition."""
        ep = self._current_episode
        if ep is None or ep.is_complete:
            return

        # Count state transitions within episode
        ep.n_transitions += 1

        # Detect resolution events
        if transition == "G3":
            ep.t_end          = t
            ep.risk_at_commit = risk
            ep.resolution_type = ResolutionType.EARNED_COMMIT
            self._finalise()

        elif transition == "G4_early":
            ep.t_end          = t
            ep.risk_at_commit = risk
            ep.resolution_type = ResolutionType.ANTICIPATORY_ABORT
            self._finalise()

        elif transition in ("G6", "G8_emergency"):
            ep.t_end          = t
            ep.risk_at_commit = risk
            ep.resolution_type = ResolutionType.FORCED_ABORT
            self._finalise()

        elif transition == "G_yield_on_entry":
            ep.t_end          = t
            ep.risk_at_commit = risk
            ep.resolution_type = ResolutionType.DEADLOCK_YIELD
            self._finalise()

    # ── HQM computation ───────────────────────────────────────────────────────

    def _finalise(self):
        ep = self._current_episode
        if ep is None:
            return

        hqm_dict = self.compute(ep)
        self._completed_episodes.append(hqm_dict)
        self._current_episode = None

    def compute(self, ep: Episode) -> dict:
        S = self._safety_gain(ep)
        E = self._efficiency(ep)
        B = self._stability(ep)
        R = ep.resolution_type.value if ep.resolution_type else 0.0

        hqm = self._alpha*S + self._beta*E + self._gamma*B + self._delta*R

        return {
            "hqm":        round(float(hqm), 4),
            "S":          round(float(S), 4),
            "E":          round(float(E), 4),
            "B":          round(float(B), 4),
            "R":          R,
            "duration":   round(ep.duration, 3),
            "resolution": ep.resolution_type.name if ep.resolution_type else "UNKNOWN",
            "n_transitions": ep.n_transitions,
        }

    def _safety_gain(self, ep: Episode) -> float:
        """S = Risk(t_greedy) - Risk(t_commit)  ∈ [-1, 1]"""
        if ep.risk_at_commit is None:
            return 0.0
        S = ep.risk_at_greedy - ep.risk_at_commit
        return float(np.clip(S, -1.0, 1.0))

    def _efficiency(self, ep: Episode) -> float:
        """E = exp(-max(0, T_actual - T_necessary) / T_budget)"""
        if ep.resolution_type == ResolutionType.FORCED_ABORT:
            return 0.0
        if ep.resolution_type == ResolutionType.ANTICIPATORY_ABORT:
            return 1.0  # aborting early when danger predicted is efficient

        T_actual    = ep.duration
        T_necessary = (ep.t_g3_eligible - ep.t_start) if ep.t_g3_eligible else T_actual
        T_necessary = max(0.0, T_necessary)

        excess = max(0.0, T_actual - T_necessary)
        return float(np.exp(-excess / self._T_budget))

    def _stability(self, ep: Episode) -> float:
        """B = w1·exp(-λ·N_transitions) + w2·(1 - norm_oscillation)"""
        B1 = float(np.exp(-self._lambda * ep.n_transitions))

        osc_mean = float(np.var(ep.A_history)) if ep.A_history else 0.0
        B2 = float(np.clip(1.0 - osc_mean * 10.0, 0.0, 1.0))

        return float(self._w1 * B1 + self._w2 * B2)

    # ── Analytics ─────────────────────────────────────────────────────────────

    @property
    def completed_episodes(self) -> list[dict]:
        return list(self._completed_episodes)

    @property
    def macro_hqm(self) -> Optional[float]:
        if not self._completed_episodes:
            return None
        return float(np.mean([e["hqm"] for e in self._completed_episodes]))

    @property
    def greedy_baseline_hqm(self) -> float:
        """Greedy: S=0, E=1, B=1, R=1 → 0 + β + γ + δ"""
        return self._beta + self._gamma + self._delta
