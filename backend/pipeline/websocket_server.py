"""
WebSocket server - streams live telemetry to React dashboard.

Message format (JSON, sent every frame):
{
  "t":        1234567890.123,
  "state":    "PROBE",
  "color":    "#eab308",
  "t_in_state": 1.2,
  "ambiguity": { "A": 0.61, "Ap": 0.40, "Ab": 0.78, "dA_dt": 0.04, "osc": 0.03 },
  "risk":      { "risk": 0.45, "ttc_risk": 0.30, "traj_conf": 0.60, "correction": 0.20 },
  "hqm":       { "macro": 0.68, "last": { ... }, "greedy_baseline": 0.60 },
  "flags":     { "unresolvable": false, "yield_pending": false },
  "transition": "G3"   or null
}
"""

import asyncio
import json
import websockets
import time
import numpy as np
from typing import Optional

from config import load_config
from core.ambiguity.perceptual import PerceptualAmbiguity
from core.ambiguity.behavioral import BehavioralAmbiguity
from core.ambiguity.fusion import AmbiguityFusion
from core.risk.composite import RiskComposite, ttc_risk, trajectory_conflict
from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.metrics.hqm import HQMComputer
from core.state_machine.states import State


class HesitationPipeline:
    """
    Full pipeline: ingests per-frame sensor data,
    runs all layers, returns telemetry dict for broadcast.
    """

    def __init__(self):
        self._fusion   = AmbiguityFusion()
        self._risk     = RiskComposite()
        self._machine  = HesitationStateMachine()
        self._hqm      = HQMComputer()
        self._t0       = time.time()

        # Greedy risk reference (for HQM safety gain)
        self._greedy_risk_at_probe: float = 0.0

    def process_frame(
        self,
        confidence: Optional[float],
        agent_velocity: Optional[np.ndarray],
        distance: float,
        relative_velocity: float,
        ego_path: Optional[np.ndarray] = None,
        ego_speed: float = 0.0,
        path_deviation: float = 0.0,
        tracker=None,
    ) -> dict:
        t = time.time() - self._t0

        # ── Ambiguity ───────────────────────────────────────────────────────
        Ap = self._fusion.perceptual.update(confidence)
        Ab = self._fusion.behavioral.update(tracker)
        amb = self._fusion.update(Ap, Ab, t)

        # ── Risk ────────────────────────────────────────────────────────────
        ttc_r = ttc_risk(distance, relative_velocity)

        agent_pos = np.array([0.0, distance]) if agent_velocity is None else None
        if agent_pos is None and tracker is not None:
            agent_pos = tracker.position

        if ego_path is None:
            ego_path = np.array([[0, i] for i in np.linspace(0, 5, 20)])

        if agent_pos is not None and agent_velocity is not None:
            tc = trajectory_conflict(ego_path, agent_pos, agent_velocity)
        else:
            tc = 0.0

        cs = self._risk.correction_monitor.update(t, ego_speed, path_deviation)
        risk_dict = self._risk.compute(ttc_r, tc, cs)

        # Project risk forward (simple linear extrapolation)
        risk_projected = float(np.clip(
            risk_dict["risk"] + risk_dict["dR_dt"] * 1.0, 0.0, 1.0
        ))

        # ── State machine ───────────────────────────────────────────────────
        machine_in = MachineInput(
            t=t,
            A=amb["A"], dA_dt=amb["dA_dt"], osc=amb["osc"],
            risk=risk_dict["risk"], dR_dt=risk_dict["dR_dt"],
            risk_projected=risk_projected,
        )
        out = self._machine.tick(machine_in)

        # ── HQM tracking ────────────────────────────────────────────────────
        prev_state = getattr(self, "_prev_state", State.CRUISE)
        if prev_state != State.PROBE and out.state == State.PROBE:
            self._greedy_risk_at_probe = risk_dict["risk"]
            self._hqm.on_probe_enter(t, self._greedy_risk_at_probe)

        from core.state_machine.guards import G3 as g3_check
        g3_eligible = g3_check(
            amb["A"], amb["dA_dt"], amb["osc"],
            risk_dict["risk"], out.t_in_state
        )
        self._hqm.on_tick(out.state, amb["A"], risk_dict["risk"],
                          amb["dA_dt"], amb["osc"], t, g3_eligible)

        if out.transition_fired:
            self._hqm.on_transition(out.transition_fired, t, risk_dict["risk"])
            if out.transition_fired == "G3":
                self._risk.correction_monitor.on_commit(t)

        self._prev_state = out.state

        # ── Telemetry payload ───────────────────────────────────────────────
        last_ep = (self._hqm.completed_episodes[-1]
                   if self._hqm.completed_episodes else None)

        return {
            "t":          round(t, 3),
            "state":      out.state_label,
            "color":      out.state_color,
            "t_in_state": round(out.t_in_state, 2),
            "transition": out.transition_fired,
            "ambiguity":  {k: round(float(v), 4) for k, v in amb.items()},
            "risk": {
                "risk":       round(risk_dict["risk"], 4),
                "ttc_risk":   round(ttc_r, 4),
                "traj_conf":  round(tc, 4),
                "correction": round(cs, 4),
                "dR_dt":      round(risk_dict["dR_dt"], 4),
            },
            "hqm": {
                "macro":           round(self._hqm.macro_hqm or 0.0, 4),
                "greedy_baseline": round(self._hqm.greedy_baseline_hqm, 4),
                "last":            last_ep,
                "n_episodes":      len(self._hqm.completed_episodes),
            },
            "flags": out.flags,
        }


# ── WebSocket server ─────────────────────────────────────────────────────────

_pipeline = HesitationPipeline()
_clients  = set()


async def broadcast(message: str):
    if _clients:
        await asyncio.gather(*[c.send(message) for c in _clients],
                             return_exceptions=True)


async def handler(websocket):
    _clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _clients.discard(websocket)


async def feed_loop():
    """Simulate or read real data and broadcast at ~30fps."""
    dt = 1.0 / 30.0
    t = 0.0
    while True:
        # ── Replace this block with real sensor data in live mode ──
        confidence      = float(np.clip(0.7 + 0.15 * np.sin(t * 1.3) + 0.05 * np.random.randn(), 0, 1))
        distance        = max(1.0, 4.0 - 0.5 * np.sin(t * 0.8))
        relative_v      = 0.3 * np.cos(t * 0.5) + 0.05 * np.random.randn()
        agent_velocity  = np.array([relative_v, 0.1 * np.random.randn()])
        # ─────────────────────────────────────────────────────────────

        payload = _pipeline.process_frame(
            confidence=confidence,
            agent_velocity=agent_velocity,
            distance=distance,
            relative_velocity=relative_v,
        )
        await broadcast(json.dumps(payload))
        await asyncio.sleep(dt)
        t += dt


async def main():
    cfg = load_config()["pipeline"]
    host = cfg["websocket_host"]
    port = cfg["websocket_port"]
    print(f"[WebSocket] Starting server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await feed_loop()


if __name__ == "__main__":
    asyncio.run(main())
