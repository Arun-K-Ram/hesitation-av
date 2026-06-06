# Hesitation-AV

**Hesitate or Commit: A Formal Framework for Ambiguity-Aware Decision-Making in Autonomous Driving**

A formal temporal decision framework for interpretable autonomous hesitation.

> *Hesitation is not a failure mode - it is a measurable, optimizable decision primitive.*

---

## Overview

Autonomous driving systems fail systematically in ambiguous scenarios where agent intent is unclear. This framework introduces:

- **Decomposed ambiguity** `A(t) = α·Aₚ(t) + β·Ab(t)` grounded in aleatoric/epistemic uncertainty theory
- **Six-state hesitation state machine** with eight formally defined transition guards (G1–G8)
- **Hesitation Quality Metric (HQM)** decomposing decision quality into Safety gain, Efficiency, Behavioral stability, and Resolution quality
- **Context-adaptive MLP** replacing fixed α,β weights with data-driven predictions (R²=0.814)
- **Two-phase validation**: RC car physical testbed + CARLA simulation (240 trials, zero collisions)

---

## Key Results

| Metric | Value |
|--------|-------|
| Hesitation HQM (pedestrian\_curb) | **0.747** vs greedy baseline 0.600 |
| Only policy beating greedy | All 3 scenario classes |
| Large-scale robustness (1,000 configs) | **100% beat greedy** |
| Counterfactual asymmetry at ±1.5s | **ΔHQM = +0.066** (p<0.001) |
| CARLA validation (240 trials) | **Zero collisions** |
| Scene classification - same-session | **99.6%** (EfficientNetB2) |
| Scene classification - cross-session | **87.5%** (EfficientNetB2) |

---

## Architecture

```
Video Feed → YOLOv11n Detection → Ambiguity Layer → State Machine → HQM Evaluation
                                  A(t) = α·Aₚ + β·Ab   G1–G8 guards   S·E·B·R
                                         ↑
                                    MLP adaptive weights
                                    (scene-conditioned)
```

---

## Repo Structure

```
hesitation-av/
├── backend/
│   ├── config/
│   │   └── params.yaml                  ← all parameters live here
│   ├── core/
│   │   ├── ambiguity/
│   │   │   ├── perceptual.py            ← Aₚ(t): aleatoric + epistemic
│   │   │   ├── behavioral.py            ← Ab(t): Kalman-smoothed entropy
│   │   │   └── fusion.py                ← A(t) composite, dA/dt, oscillation
│   │   ├── risk/
│   │   │   └── composite.py             ← TTC_risk + TrajectoryConflict + CorrectionSeverity
│   │   ├── state_machine/
│   │   │   ├── states.py                ← State enum (CRUISE/PROBE/HOLD/COMMIT/ABORT/YIELD)
│   │   │   ├── guards.py                ← G1–G8 transition conditions
│   │   │   ├── machine.py               ← HesitationStateMachine
│   │   │   └── memory.py                ← cooldown / abort counter / resolution horizon
│   │   ├── metrics/
│   │   │   └── hqm.py                   ← HQM: S, E, B, R components
│   │   └── perception/
│   │       ├── detector.py              ← YOLOv11n wrapper
│   │       └── tracker.py               ← Kalman filter tracker
│   ├── ml/
│   │   ├── train_mlp.py                 ← MLP ambiguity predictor (R²=0.814)
│   │   ├── train_cnn_v2.py              ← CNN architecture sweep (36 configs)
│   │   ├── train_cnn_crosssession.py    ← Cross-session CNN validation
│   │   ├── scene_predictor.py           ← EfficientNetB2 scene classifier
│   │   └── cnn_best_v2.pth              ← Best CNN weights (same-session)
│   └── pipeline/
│       └── live.py                      ← Live webcam pipeline + recording
├── experiments/
│   ├── run_ablation.py                  ← 6-policy comparison (540 trials)
│   ├── counterfactual.py                ← Commitment timing analysis
│   ├── sensitivity_analysis.py          ← ±20% perturbation + 1000-config sweep
│   ├── carla_validation.py              ← CARLA Phase 2 (240 trials)
│   ├── generate_visual_summary.py       ← Policy comparison figure
│   ├── generate_cnn_sweep_plot.py       ← CNN sweep figure
│   └── results/                         ← All experiment outputs
├── paper_figures/                       ← Publication-ready figures (white theme)
│   ├── hesitav_samples.png
│   ├── cnn_sweep_plot.png
│   ├── ablation.png
│   ├── counterfactual.png
│   ├── sensitivity.png
│   ├── carla_validation.png
│   ├── carla_scenarios.png
│   ├── visual_summary.png
│   └── camera_calibration.png
├── recordings/                          ← HesitAV-1564 dataset
│   ├── pedestrian_curb_*/               ← Sessions 1–3
│   ├── merge_hesitation_*/
│   └── occluded_intersection_*/
└── dashboard/
    └── src/App.jsx                      ← React live dashboard
```

---

## HesitAV-1564 Dataset

Custom RC car dataset recorded across three sessions and three scenario classes.

| Session | Conditions | Role |
|---------|-----------|------|
| Session 1 | Room lighting | Train |
| Session 2 | No lights / evening | Train |
| Session 3 | Lamp on (different illumination) | **Test (held out)** |

| Scenario | Frames | Description |
|----------|--------|-------------|
| pedestrian\_curb | 2,033 | Agent near ego path with ambiguous crossing intent |
| merge\_hesitation | 1,275 | Vehicle at uncertain merge point |
| occluded\_intersection | 255 | Agent partially occluded |

**Total: 3,563 frames** across 3 sessions, 3 scenario classes.

---

## CNN Architecture Sweep

Six architectures evaluated across 36 hyperparameter configurations.

| Architecture | Params | Val (%) | Cross-session (%) |
|-------------|--------|---------|-------------------|
| EfficientNetB2 | 9.1M | **99.8** | **87.5** |
| EfficientNetB0 | 5.3M | 99.6 | 75.2 |
| MobileNetV3 | 5.4M | 99.6 | 70.9 |
| MobileNetV2 | 3.4M | 99.4 | 71.1 |
| ResNet18 | 11M | 99.6 | 67.7 |
| ConvNeXt-Tiny | 28M | 99.6 | 52.6 |

EfficientNetB2 selected: best on both same-session and cross-session.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Arun-K-Ram/hesitation-av
cd hesitation-av
poetry install

# 2. Run live pipeline (webcam)
poetry run python backend/pipeline/live.py --source 0

# 3. Record a scenario
poetry run python backend/pipeline/live.py --source 0 --record --label pedestrian_curb

# 4. Run experiments
python experiments/run_ablation.py
python experiments/counterfactual.py
python experiments/sensitivity_analysis.py

# 5. Run CARLA validation (requires CARLA 0.9.15)
C:\carla\WindowsNoEditor\CarlaUE4.exe -quality-level=Low
python experiments/carla_validation.py
```

---

## Hesitation State Machine

Six states, eight guards. Core principle: no transition depends on A(t) alone - every transition depends on its temporal trajectory.

```
CRUISE → PROBE → COMMIT
           ↓        ↓
          HOLD    ABORT
           ↓        ↓
          PROBE   CRUISE
           ↓
          YIELD
```

**G3 (Earned Commit):** A < τₗ(1−h) ∧ dA/dt < 0 ∧ Ã < σₛ ∧ R < ρc ∧ Tₛ ≥ t_min

Compare to naive threshold policy A(t) < τₗ which commits on any momentary dip. G3 requires ambiguity to be decreasing, stable, and sustained simultaneously.

---

## HQM - Hesitation Quality Metric

```
HQM = αS·S + βE·E + γB·B + δR·R
      (0.40)  (0.25)  (0.20)  (0.15)
```

| Component | Description | Range |
|-----------|-------------|-------|
| S (Safety Gain) | Risk reduction vs greedy commit | [−1, 1] |
| E (Efficiency) | Penalty for unnecessary waiting | [0, 1] |
| B (Stability) | Penalizes oscillation and excess transitions | [0, 1] |
| R (Resolution) | Deadlock=0, timeout=0.5, earned=1.0 | {0, 0.5, 1.0} |

Greedy baseline: **HQM = 0.60** by construction. All 1,000 random parameter configurations beat this (100% win rate).

---

## Six-Policy Comparison

| Scenario | Greedy | Fixed | Random | Risk | TTC | **Hesitation** |
|----------|--------|-------|--------|------|-----|----------------|
| pedestrian\_curb | 0.600 | 0.436 | 0.484 | 0.576 | 0.576 | **0.747** |
| merge\_hesitation | 0.600 | 0.377 | 0.401 | 0.568 | 0.568 | **0.606** |
| occluded\_intersection | 0.600 | 0.549 | 0.579 | 0.616 | 0.712 | **0.655** |

Hesitation is the **only policy that consistently outperforms greedy** across all three scenario classes. Fixed and random delay score *below* greedy - confirming that delay is not hesitation.

---

## CARLA Phase 2 Validation

```
Town10HD_Opt | Tesla Model3 blueprint | 4 weather conditions
3 scenarios × 4 weather × 20 trials = 240 total trials
Zero collisions across all conditions
```

| Weather | HQM Mean | Collisions/Trial |
|---------|----------|-----------------|
| Clear | 0.362 | 0.000 |
| Night | 0.382 | 0.000 |
| Fog | 0.342 | 0.000 |
| Rain | 0.329 | 0.000 |

---

## Hardware

- **Laptop**: Windows, NVIDIA RTX 2060 6GB, CUDA 12.1
- **Camera**: Logitech C270
- **RC car**: Adventure Force 1:24
- **CARLA**: v0.9.15

---

## Paper

> Arunkumar Ramachandran. *Hesitate or Commit: A Formal Framework for Ambiguity-Aware Decision-Making in Autonomous Driving.* Independent Researcher, 2026.

Target: arXiv cs.RO (primary), cs.LG + cs.AI (secondary).

---

## Acknowledgments

Thanks to the open-source communities behind Ultralytics YOLO, PyTorch, OpenCV, and CARLA, and the authors of OnSiteVRU for releasing trajectory data under academic use terms.
