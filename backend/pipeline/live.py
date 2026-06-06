"""
backend/pipeline/live.py

Live pipeline - reads from video file or webcam,
runs YOLO detection, feeds ambiguity + state machine,
streams telemetry to React dashboard via WebSocket.

Usage:
    # Webcam:
    poetry run python backend/pipeline/live.py --source 0

    # Video file:
    poetry run python backend/pipeline/live.py --source C:/Users/Arun/Downloads/test.mp4

    # Record to disk:
    poetry run python backend/pipeline/live.py --source 0 --record

    # Scenario 1:
    poetry run python backend/pipeline/live.py --source 0 --record --label pedestrian_curb

    # Scenario 2:
    poetry run python backend/pipeline/live.py --source 0 --record --label merge_hesitation

    # Scenario 3:
    poetry run python backend/pipeline/live.py --source 0 --record --label occluded_intersection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from ml.scene_predictor import SceneAmbiguityPredictor
import asyncio
import websockets
import json
import argparse
import time
import csv
from collections import deque
from pathlib import Path

from config import load_config, reset_config
reset_config()

from core.perception.detector import Detector, Detection
from core.perception.tracker import KalmanTracker
from core.ambiguity.perceptual import PerceptualAmbiguity
from core.ambiguity.behavioral import BehavioralAmbiguity
from core.ambiguity.fusion import AmbiguityFusion
from core.risk.composite import RiskComposite, ttc_risk, trajectory_conflict
from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.state_machine.states import State
from core.metrics.hqm import HQMComputer
from core.state_machine.guards import G3 as g3_check


#  Globals 

_clients = set()


#  WebSocket server 

async def ws_handler(websocket):
    _clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _clients.discard(websocket)


async def broadcast(message: str):
    if _clients:
        await asyncio.gather(
            *[c.send(message) for c in _clients],
            return_exceptions=True
        )


#  Frame processor 

class LivePipeline:
    def __init__(self):
        self._scene_predictor = SceneAmbiguityPredictor()
        self._current_alpha   = 0.45
        self._current_beta    = 0.55
        self._detector   = Detector()
        self._fusion     = AmbiguityFusion()
        self._risk       = RiskComposite()
        self._machine    = HesitationStateMachine()
        self._hqm        = HQMComputer()
        self._trackers:  dict[int, KalmanTracker] = {}
        self._t0         = time.time()
        self._prev_state = State.CRUISE
        self._frame_count = 0
        self._risk_history: deque[float] = deque(maxlen=90)

    def process(self, frame: np.ndarray) -> dict:
        t  = time.time() - self._t0
        dt = 1.0 / 30.0
        h, w = frame.shape[:2]

        #  Detection 
        detections = self._detector.detect(frame)

        primary = None
        if detections:
            centre_x = w / 2
            bottom_y = h
            def proximity(d):
                cx, cy = d.center
                return abs(cx - centre_x) + abs(cy - bottom_y)
            primary = min(detections, key=proximity)

        # Scene classification + adaptive weights
        scene_pred = self._scene_predictor.predict(frame)
        self._current_alpha = scene_pred["alpha"]
        self._current_beta  = scene_pred["beta"]

        # Update fusion weights dynamically
        self._fusion._alpha = self._current_alpha
        self._fusion._beta  = self._current_beta

        #  Tracking 
        tracker = None
        if primary is not None:
            tid    = primary.track_id
            cx, cy = primary.center
            if tid not in self._trackers:
                self._trackers[tid] = KalmanTracker(cx, cy, dt)
            else:
                self._trackers[tid].predict()
                self._trackers[tid].update(np.array([cx, cy]))
            tracker = self._trackers[tid]

        #  Ambiguity 
        conf = primary.confidence if primary else None
        Ap   = self._fusion.perceptual.update(conf)
        Ab   = self._fusion.behavioral.update(tracker)
        amb  = self._fusion.update(Ap, Ab, t)

        #  Risk 
        if primary is not None:
            px_distance  = float(h - primary.center[1])
            distance_m   = max(0.5, px_distance * 0.01)
            rel_velocity = tracker.speed if tracker else 0.1
            agent_vel    = tracker.velocity if tracker else np.array([0.0, 0.0])
            agent_pos    = tracker.position if tracker else np.array([w/2, h/2])
        else:
            distance_m   = 5.0
            rel_velocity = 0.1
            agent_vel    = np.array([0.0, 0.0])
            agent_pos    = np.array([w/2.0, 0.0])

        ttc_r    = ttc_risk(distance_m, rel_velocity)
        ego_path = np.array([[w/2, h - i*10] for i in range(20)])
        tc       = trajectory_conflict(ego_path, agent_pos, agent_vel)
        cs       = self._risk.correction_monitor.update(t, rel_velocity, 0.0)
        risk_d   = self._risk.compute(ttc_r, tc, cs)

        risk_projected = float(np.clip(
            risk_d["risk"] + risk_d["dR_dt"] * 1.0, 0.0, 1.0
        ))
        self._risk_history.append(risk_d["risk"])

        #  State machine 
        inp = MachineInput(
            t=t,
            A=amb["A"], dA_dt=amb["dA_dt"], osc=amb["osc"],
            risk=risk_d["risk"], dR_dt=risk_d["dR_dt"],
            risk_projected=risk_projected,
        )
        out = self._machine.tick(inp)

        #  HQM tracking 
        if self._prev_state != State.PROBE and out.state == State.PROBE:
            greedy_risk = float(np.mean(self._risk_history)) if self._risk_history else 0.3
            self._hqm.on_probe_enter(t, greedy_risk)

        g3_ok = g3_check(amb["A"], amb["dA_dt"], amb["osc"],
                          risk_d["risk"], out.t_in_state)
        self._hqm.on_tick(out.state, amb["A"], risk_d["risk"],
                           amb["dA_dt"], amb["osc"], t, g3_ok)

        if out.transition_fired:
            self._hqm.on_transition(out.transition_fired, t, risk_d["risk"])
            if out.transition_fired == "G3":
                self._risk.correction_monitor.on_commit(t)

        self._prev_state  = out.state
        self._frame_count += 1

        #  Frame overlays 
        cv2.putText(frame, f"Scene: {scene_pred['scene_name']}",
            (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"a={scene_pred['alpha']:.2f} b={scene_pred['beta']:.2f}",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            color = (0, 255, 0) if det == primary else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        state_color_bgr = {
            "CRUISE": (34, 197, 94),
            "PROBE":  (234, 197, 34),
            "HOLD":   (249, 130, 34),
            "COMMIT": (59, 131, 235),
            "ABORT":  (239, 68, 68),
            "YIELD":  (247, 85, 168),
        }
        bgr = state_color_bgr.get(out.state_label, (255, 255, 255))
        cv2.putText(frame, f"STATE: {out.state_label}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr, 2)
        cv2.putText(frame, f"A(t):  {amb['A']:.3f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"Risk:  {risk_d['risk']:.3f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, f"dA/dt: {amb['dA_dt']:.3f}",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        last_ep = self._hqm.completed_episodes[-1] if self._hqm.completed_episodes else None

        return {
            "t":          round(t, 3),
            "state":      out.state_label,
            "color":      out.state_color,
            "t_in_state": round(out.t_in_state, 2),
            "transition": out.transition_fired,
            "ambiguity":  {k: round(float(v), 4) for k, v in amb.items()},
            "risk": {
                "risk":       round(risk_d["risk"], 4),
                "ttc_risk":   round(ttc_r, 4),
                "traj_conf":  round(tc, 4),
                "correction": round(cs, 4),
                "dR_dt":      round(risk_d["dR_dt"], 4),
            },
            "scene": {
                "type":       scene_pred["scene_type"],
                "name":       scene_pred["scene_name"],
                "confidence": round(scene_pred["scene_confidence"], 4),
                "alpha":      scene_pred["alpha"],
                "beta":       scene_pred["beta"],
                "source":     scene_pred["weight_source"],
            },
            "hqm": {
                "macro":           round(self._hqm.macro_hqm or 0.0, 4),
                "greedy_baseline": round(self._hqm.greedy_baseline_hqm, 4),
                "last":            last_ep,
                "n_episodes":      len(self._hqm.completed_episodes),
            },
            "flags":       out.flags,
            "detections": [
                {"class":      d.class_name,
                 "confidence": round(d.confidence, 3),
                 "bbox":       d.bbox.astype(int).tolist()}
                for d in detections
            ],
            "frame_count": self._frame_count,
        }


#  Main loop 

async def video_loop(source, show_window=True, record=False, label="unlabeled"):
    pipeline = LivePipeline()
    cfg      = load_config()

    # Recording setup
    csv_file   = None
    csv_writer = None
    frames_dir = None

    if record:
        record_dir = Path("experiments/recordings") / f"{label}_{int(time.time())}"
        frames_dir = record_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        csv_path   = record_dir / "telemetry.csv"
        csv_file = open(csv_path, "w", newline="", buffering=1)
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            "frame", "t", "state", "A", "Ap", "Ab",
            "dA_dt", "osc", "risk", "ttc_risk",
            "traj_conf", "transition"
        ])
        csv_writer.writeheader()
        print(f"[Record] Saving to {record_dir}")

    # Open source
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) \
          if isinstance(source, int) \
          else cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps_target = cfg["pipeline"]["fps_target"]
    dt         = 1.0 / fps_target

    print(f"[Live] Source opened: {source}")
    print(f"[Live] Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame   = cv2.resize(frame, (640, 480))
        payload = pipeline.process(frame)

        # Broadcast to dashboard
        await broadcast(json.dumps(payload))

        # Save if recording
        if record and csv_writer and frames_dir:
            frame_path = frames_dir / f"frame_{pipeline._frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            csv_writer.writerow({
                "frame":      pipeline._frame_count,
                "t":          payload["t"],
                "state":      payload["state"],
                "A":          payload["ambiguity"]["A"],
                "Ap":         payload["ambiguity"]["Ap"],
                "Ab":         payload["ambiguity"]["Ab"],
                "dA_dt":      payload["ambiguity"]["dA_dt"],
                "osc":        payload["ambiguity"]["osc"],
                "risk":       payload["risk"]["risk"],
                "ttc_risk":   payload["risk"]["ttc_risk"],
                "traj_conf":  payload["risk"]["traj_conf"],
                "transition": payload["transition"] or "",
            })

        # Show window
        if show_window:
            cv2.imshow("Hesitation-AV Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        await asyncio.sleep(dt)

    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
        print(f"[Record] Session saved.")


async def main(source, show_window=True, record=False, label="unlabeled"):
    cfg  = load_config()
    host = cfg["pipeline"]["websocket_host"]
    port = cfg["pipeline"]["websocket_port"]

    print(f"[WebSocket] ws://{host}:{port}")
    print(f"[Live] Starting pipeline...\n")

    async with websockets.serve(ws_handler, host, port):
        await video_loop(source, show_window, record, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (0,1,2) or video file path"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable OpenCV preview window"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save frames and telemetry to disk"
    )
    parser.add_argument(
    "--label",
    default="unlabeled",
    help="Scenario label for this recording session"
)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    asyncio.run(main(
    source,
    show_window=not args.no_window,
    record=args.record,
    label=args.label
))