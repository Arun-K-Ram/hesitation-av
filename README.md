# Hesitation-AV

**Ambiguity-Driven Hesitation in Autonomous Driving Systems**

A formal temporal decision framework for interpretable autonomous hesitation.

---

## Architecture

```
Video Feed → Perception → Ambiguity Layer → State Machine → HQM Evaluation
                          A(t) = α·Aₚ + β·Ab    G1–G8 guards    S·E·B·R
```

---

## Repo Structure

```
hesitation-av/
├── config/
│   └── params.yaml          ← every parameter lives here
├── core/
│   ├── ambiguity/
│   │   ├── perceptual.py    ← DetConf(t) - aleatoric/epistemic
│   │   ├── behavioral.py    ← MotionEntropy(t) - Kalman-denoised
│   │   └── fusion.py        ← A(t) composite + dA/dt + oscillation
│   ├── risk/
│   │   └── composite.py     ← TTC_risk + TrajectoryConflict + CorrectionSeverity
│   ├── state_machine/
│   │   ├── states.py        ← State enum
│   │   ├── guards.py        ← G1–G8 transition conditions
│   │   ├── machine.py       ← HesitationStateMachine
│   │   └── memory.py        ← cooldown / abort counter / resolution horizon
│   ├── metrics/
│   │   └── hqm.py           ← HQM: S, E, B, R components
│   └── perception/
│       ├── detector.py      ← YOLOv8 wrapper
│       └── tracker.py       ← Kalman filter tracker
├── pipeline/
│   └── websocket_server.py  ← streams telemetry to React at 30fps
├── dashboard/
│   └── src/App.jsx          ← React live dashboard
└── tests/
    └── test_state_machine.py ← 5 pathological case tests
```

---

## Quick Start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Run the pipeline (demo mode with synthetic data)
cd hesitation-av
python pipeline/websocket_server.py

# 3. In another terminal, run the dashboard
cd dashboard
npm install
npm run dev
# Open http://localhost:5173
```

---

## Hardware Shopping List - Live Demo

**Total estimated cost: $60–85**

### Required

| Item | Purpose | Where | ~Cost |
|------|---------|-------|-------|
| RC car (WLtoys A959 or similar) | The demo vehicle | Amazon | $30–40 |
| USB webcam (Logitech C270 or similar) | Overhead scene camera | Amazon | $20–25 |
| Small toy figures / traffic cones | "Pedestrian" props | Amazon / Dollar store | $5–10 |
| Zip ties or rubber bands | Mount phone/webcam on car | Hardware store | $2 |

**Total: ~$57–77**

### Optional (better demo)
| Item | Purpose | ~Cost |
|------|---------|-------|
| Raspberry Pi Zero 2W | Onboard compute for car | $15 |
| Pi Camera Module 3 | Better onboard video | $25 |
| Small portable monitor | Second screen for dashboard | (use laptop) |

---

## Demo Setup (No RC Autonomy Needed)

The RC car does **not** need to drive itself.

```
Setup A - Fixed overhead camera (recommended for LinkedIn video):
  1. Mount webcam above a small tabletop course
  2. Drive RC car manually through scenarios
  3. Pipeline watches the overhead feed
  4. Dashboard shows live analysis

Setup B - Onboard camera:
  1. Mount phone on RC car (zip tie)
  2. Stream via IP Webcam app (Android) or EpocCam (iOS)
  3. Set camera_index in params.yaml to the stream URL
  4. Same pipeline, first-person perspective
```

### Scenarios to film

| Scenario | Setup | What it shows |
|----------|-------|---------------|
| Pedestrian near curb | Toy figure 5cm from RC path | Behavioral ambiguity spike |
| Hesitant merge | Two RC cars approaching | Trajectory conflict |
| Occluded crossing | Cardboard box blocking view | Perceptual ambiguity |

### LinkedIn video structure (90 seconds)
```
0:00–0:10   Title card: "AVs fail at ambiguity, not just detection"
0:10–0:30   Greedy policy - RC car passes close to "pedestrian"
            Dashboard: Risk spikes AFTER commit
0:30–1:00   Hesitation-aware policy - same scenario
            Dashboard: PROBE state, oscillation detected, G3 earned commit
1:00–1:20   HQM comparison: 0.68 vs 0.60 baseline
1:20–1:30   GitHub link
```

---

## Key Research Claim

> Temporal confidence oscillation in the 2-second window preceding commitment
> is a statistically significant predictor of unsafe correction events,
> outperforming static ambiguity thresholds as a commit-gating signal.

---

## Parameters

All 14 state-machine parameters live in `config/params.yaml`.
Tune by editing the YAML - no code changes needed.

---

## Running Tests

```bash
pytest tests/ -v
```
