"""
Transition guards G1–G8.

Each guard is a pure function:
  inputs  →  bool

No side effects. No state mutation. Testable in isolation.

Guard index:
  G1  CRUISE  → PROBE     (ambiguity entered lower band)
  G2  PROBE   → HOLD      (ambiguity escalating or probe timeout)
  G3  PROBE   → COMMIT    (earned commit: all four conditions met)
  G4  COMMIT  → ABORT     (risk spiked post-commit)
  G5  HOLD    → PROBE     (ambiguity recovering)
  G6  HOLD    → ABORT     (timeout or critical risk in hold)
  G7  ABORT   → CRUISE    (scene cleared after abort)
  G8  (any)   → ABORT     (emergency override - critical risk anywhere)
  G_cooldown  helper      (re-entry cooldown check)
"""

from config import load_config


def _p() -> dict:
    return load_config()["state_machine"]


# G1: CRUISE → PROBE 

def G1(A: float, t_in_state: float, t_since_last_probe_exit: float) -> bool:
    """Normal ambiguity entry (confirmation window + cooldown)."""
    p = _p()
    normal_path = (
        A > p["tau_low"]
        and t_in_state >= p["t_entry"]
        and t_since_last_probe_exit >= p["t_cooldown"]
    )
    return normal_path


def G1_spike(A: float, dA_dt: float) -> bool:
    """Spike bypass - skips t_entry for fast-onset events."""
    p = _p()
    return A > p["tau_spike"] and dA_dt > p["delta_spike"]


# G2: PROBE → HOLD ──

def G2(A: float, dA_dt: float, osc: float, t_in_state: float) -> bool:
    p = _p()
    high_and_stable   = A > p["tau_high"] and t_in_state >= p["t_escalate"]
    rising_oscillating = (
        dA_dt > 0
        and osc > p["sigma_osc"]
        and t_in_state >= p["t_escalate"]
    )
    probe_timeout     = t_in_state >= p["t_probe_max"]
    return high_and_stable or rising_oscillating or probe_timeout


# G3: PROBE → COMMIT ─

def G3(A: float, dA_dt: float, osc: float, risk: float, t_in_state: float) -> bool:
    """
    Earned commit: all four conditions must hold simultaneously
    for the minimum sustained duration.
    """
    p = _p()
    exit_threshold = p["tau_low"] * (1.0 - p["hysteresis"])
    return (
        A        < exit_threshold
        and dA_dt  < 0.0
        and osc    < p["sigma_stable"]
        and risk   < p["rho_commit"]
        and t_in_state >= p["t_commit_min"]
    )


# G4: COMMIT → ABORT ─

def G4(risk: float, dA_dt: float) -> bool:
    """Post-commit risk spike or sudden ambiguity jump."""
    p = _p()
    return risk > p["rho_critical"] or dA_dt > p["delta_spike"]


def G4_early(risk: float, dR_dt: float, risk_projected: float) -> bool:
    """
    Anticipatory abort: risk is rising toward critical.
    Transition to HOLD (not ABORT) to preserve options.
    """
    p = _p()
    return (
        risk         > p["rho_warn"]
        and dR_dt    > p["delta_risk_rise"]
        and risk_projected > p["rho_critical"]
    )


# G5: HOLD → PROBE ───

def G5(A: float, dA_dt: float, t_in_state: float) -> bool:
    p = _p()
    exit_threshold = p["tau_high"] * (1.0 - p["hysteresis"])
    return (
        A       < exit_threshold
        and dA_dt < 0.0
        and t_in_state >= p["t_recovery"]
    )


# G6: HOLD → ABORT ───

def G6(t_in_state: float, risk: float) -> bool:
    p = _p()
    return t_in_state >= p["t_timeout"] or risk > p["rho_critical"]


# G7: ABORT → CRUISE ─

def G7(A: float, risk: float, dA_dt: float, t_in_state: float) -> bool:
    p = _p()
    return (
        A      < p["tau_low"]
        and risk  < p["rho_safe"]
        and dA_dt <= 0.0
        and t_in_state >= p["t_reset_min"]
    )


# G8: Emergency override (any → ABORT) ─────────────────────────────────────

def G8_emergency(risk: float) -> bool:
    p = _p()
    return risk > p["rho_critical"]


# Yield re-entry check 

def G_yield_exit(A: float, risk: float, t_in_state: float) -> bool:
    """Exit YIELD after mandatory pause and scene clearance."""
    p = _p()
    return (
        t_in_state >= p["t_yield"]
        and A    < p["tau_low"]
        and risk < p["rho_safe"]
    )
