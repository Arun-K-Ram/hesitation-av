"""
experiments/run_ablation.py

Ablation study: Greedy vs Hesitation-Aware policy
across three scenario categories, N trials each.

Outputs:
  experiments/results/results.csv     - per-trial metrics
  experiments/results/summary.csv     - aggregated stats
  experiments/results/ablation.png    - comparison figure

Run from repo root:
  poetry run python experiments/run_ablation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

from config import load_config
from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.state_machine.states import State
from core.metrics.hqm import HQMComputer
from core.risk.composite import ttc_risk

from config import load_config, reset_config
reset_config()
cfg = load_config()

#  Output directory 

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


#  Scenario profiles 
#
# Each scenario generates a synthetic A(t) and Risk(t) trajectory.
# These are characteristic ambiguity profiles for each scenario class.
# CARLA integration replaces this with real simulation in future work.

def scenario_pedestrian_curb(n_frames: int, dt: float,
                              rng: np.random.Generator) -> dict:
    """
    Pedestrian near curb with ambiguous intent.

    Greedy commits at t≈1.0s — exactly when risk peaks
    (pedestrian closest to path, mid-step ambiguity).
    Hesitation waits for behavioral ambiguity to resolve at t≈3.5s
    when pedestrian clearly stops. Risk is low by then → positive S.
    """
    t = np.arange(n_frames) * dt

    # Ambiguity: rises at t=0.8s, oscillates (behavioral uncertainty),
    # resolves cleanly at t=3.2s
    rise       = np.clip((t - 0.8) / 1.0, 0, 1)
    oscillation = 0.10 * np.sin(2 * np.pi * t * 2.1) * np.clip((t - 0.8) / 0.5, 0, 1)
    decay = np.clip(1.0 - (t - 2.8) / 0.8, 0, 1)
    A = np.clip(0.58 * rise * decay + oscillation
                + rng.normal(0, 0.018, n_frames), 0, 1)

    # Risk: sharp peak at t=1.5s (pedestrian mid-step toward road),
    # drops to near-zero by t=3.5s (pedestrian backs away)
    risk = np.clip(
        0.62 * np.exp(-((t - 1.5) ** 2) / 0.35)   # peak at greedy commit zone
        + 0.08 * np.clip(1.0 - (t - 3.0) / 1.5, 0, 1)  # residual clears
        + rng.normal(0, 0.012, n_frames),
        0, 1
    )

    cfg = load_config()["state_machine"]
    # Greedy commits at first t>0.8s where A dips below tau_low
    greedy_mask = (t > 1.2) & (A < cfg["tau_low"])
    t_greedy = float(t[greedy_mask][0]) if greedy_mask.any() else float(t[-1])

    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy, "name": "pedestrian_curb"}


def scenario_merge_hesitation(n_frames: int, dt: float,
                               rng: np.random.Generator) -> dict:
    """
    Hesitant driver at merge. A(t) rises sharply above tau_high,
    stays high during conflict, drops after t=5s when driver yields.
    Greedy commits at t=2.0s into peak risk zone.
    Hesitation holds until ambiguity resolves at t=5s, low risk.
    """
    t = np.arange(n_frames) * dt

    # Ambiguity: sharp rise at t=1.5s, stays HIGH above tau_high=0.65,
    # drops sharply at t=5.0s when other driver finally yields
    A = np.clip(
        0.72 * np.clip((t - 1.5) / 0.5, 0, 1)
        * np.clip(1.0 - (t - 5.0) / 0.8, 0, 1)
        + rng.normal(0, 0.018, n_frames),
        0, 1
    )

    # Risk: peaks at t=2.5s (active conflict zone),
    # drops sharply after t=5.0s
    risk = np.clip(
        0.68 * np.clip((t - 1.5) / 0.8, 0, 1)
        * np.clip(1.0 - (t - 5.0) / 0.6, 0, 1)
        + rng.normal(0, 0.015, n_frames),
        0, 1
    )

    cfg = load_config()["state_machine"]
    # Greedy commits at first t > 2.0s where A dips below tau_low
    # During the high-ambiguity window this means committing into peak risk
    greedy_mask = (t > 2.0) & (A < cfg["tau_low"])
    t_greedy = float(t[greedy_mask][0]) if greedy_mask.any() else float(t[int(2.5*fps)])

    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy, "name": "merge_hesitation"}


def scenario_occluded_intersection(n_frames: int, dt: float,
                                    rng: np.random.Generator) -> dict:
    """
    Partially occluded intersection. Hidden agent reveals at t≈3.5s.

    Greedy commits at a misleading early dip in A(t) around t≈1.5s —
    before the hidden agent is visible. Risk is actually HIGH then
    (unknown agent may be approaching). After occlusion clears at t≈3.5s
    the agent is confirmed stationary — risk drops to near zero.
    Hesitation waits for full visibility → large positive S.
    """
    t = np.arange(n_frames) * dt

    # Ambiguity: starts high (can't see), has a MISLEADING dip at t=1.5s
    # (partial clearing), then rises again, then fully clears at t=3.5s
    A_base      = 0.70 * np.ones(n_frames)
    false_dip   = 0.25 * np.exp(-((t - 1.5) ** 2) / 0.15)  # misleading dip
    clear_t     = 3.5 + rng.normal(0, 0.25)
    full_clear  = np.clip((t - clear_t) / 0.6, 0, 1)
    A = np.clip(A_base - false_dip - 0.68 * full_clear
                + rng.normal(0, 0.022, n_frames), 0, 1)

    # Risk: HIGH during occlusion (unknown agent),
    # drops sharply once agent confirmed stationary at t=clear_t
    risk = np.clip(
        0.65 * np.clip(1.0 - (t - clear_t) / 0.7, 0, 1)
        + 0.10 * np.exp(-((t - 1.5) ** 2) / 0.3)   # spike at false dip
        + rng.normal(0, 0.015, n_frames),
        0, 1
    )

    cfg = load_config()["state_machine"]
    # Greedy falls for the false dip
    greedy_mask = (t > 1.0) & (A < cfg["tau_low"])
    t_greedy = float(t[greedy_mask][0]) if greedy_mask.any() else float(t[-1])

    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy, "name": "occluded_intersection"}


SCENARIOS = [
    scenario_pedestrian_curb,
    scenario_merge_hesitation,
    scenario_occluded_intersection,
]


#  Policy runners 

@dataclass
class TrialResult:
    scenario:        str
    policy:          str
    trial:           int
    hqm:             float
    S:               float
    E:               float
    B:               float
    R:               float
    unsafe_commits:  int
    decision_latency_ms: float
    resolution:      str
    n_transitions:   int


def run_hesitation_policy(scenario: dict, trial: int) -> TrialResult:
    """Run the full hesitation-aware state machine on a scenario."""
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    dt       = float(t_arr[1] - t_arr[0])

    machine = HesitationStateMachine()
    hqm_computer = HQMComputer()

    prev_state   = State.CRUISE
    unsafe_count = 0
    commit_t     = None
    cfg          = load_config()["state_machine"]

    from core.state_machine.guards import G3 as g3_check

    for i, (t, A, risk) in enumerate(zip(t_arr, A_arr, risk_arr)):
        # Finite difference derivative (central where possible)
        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1] - A_arr[i-1]) / (2 * dt)

        # Oscillation: rolling variance over 30 frames
        window_start = max(0, i - 30)
        osc = float(np.var(A_arr[window_start:i+1]))

        dR_dt = 0.0
        if i > 0:
            dR_dt = (risk_arr[i] - risk_arr[i-1]) / dt

        risk_proj = float(np.clip(risk + dR_dt * 1.0, 0, 1))

        inp = MachineInput(
            t=t, A=A, dA_dt=dA_dt, osc=osc,
            risk=risk, dR_dt=dR_dt, risk_projected=risk_proj
        )
        out = machine.tick(inp)

        # HQM tracking
        if prev_state != State.PROBE and out.state == State.PROBE:
            # Risk at greedy commit time = counterfactual
            greedy_idx = np.searchsorted(t_arr, scenario["t_greedy"])
            greedy_idx = min(greedy_idx, len(risk_arr)-1)
            risk_at_greedy = float(risk_arr[greedy_idx])
            hqm_computer.on_probe_enter(t, risk_at_greedy)

        g3_eligible = g3_check(A, dA_dt, osc, risk, out.t_in_state)
        hqm_computer.on_tick(out.state, A, risk, dA_dt, osc, t, g3_eligible)

        if out.transition_fired:
            hqm_computer.on_transition(out.transition_fired, t, risk)

        # Unsafe commit: committed with risk > rho_commit
        if out.transition_fired == "G3":
            commit_t = t
            if risk > cfg["rho_commit"]:
                unsafe_count += 1

        prev_state = out.state

    # Extract last completed episode
    episodes = hqm_computer.completed_episodes
    if episodes:
        last = episodes[-1]
        return TrialResult(
            scenario=scenario["name"],
            policy="hesitation",
            trial=trial,
            hqm=last["hqm"],
            S=last["S"],
            E=last["E"],
            B=last["B"],
            R=last["R"],
            unsafe_commits=unsafe_count,
            decision_latency_ms=round((commit_t - scenario["t_greedy"]) * 1000
                                      if commit_t else 0, 1),
            resolution=last["resolution"],
            n_transitions=last["n_transitions"],
        )

    # No episode completed (e.g. no PROBE entered) - treat as baseline-like
    return TrialResult(
        scenario=scenario["name"], policy="hesitation", trial=trial,
        hqm=hqm_computer.greedy_baseline_hqm,
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=unsafe_count,
        decision_latency_ms=0.0,
        resolution="NO_EPISODE",
        n_transitions=0,
    )


def run_greedy_policy(scenario: dict, trial: int) -> TrialResult:
    """
    Greedy baseline: commit at first frame A(t) < τ_low.
    S=0 by definition (it IS the reference).
    """
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    cfg      = load_config()["state_machine"]

    # Find first commit moment
    commit_mask = A_arr < cfg["tau_low"]
    if commit_mask.any():
        commit_idx  = int(np.argmax(commit_mask))
        commit_risk = float(risk_arr[commit_idx])
    else:
        commit_idx  = len(t_arr) - 1
        commit_risk = float(risk_arr[-1])

    unsafe_count = 1 if commit_risk > cfg["rho_commit"] else 0

    # Greedy HQM: S=0, E=1, B=1, R=1
    from core.metrics.hqm import HQMComputer
    hqm_computer = HQMComputer()
    greedy_hqm = hqm_computer.greedy_baseline_hqm

    return TrialResult(
        scenario=scenario["name"],
        policy="greedy",
        trial=trial,
        hqm=round(greedy_hqm, 4),
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=unsafe_count,
        decision_latency_ms=0.0,
        resolution="GREEDY_COMMIT",
        n_transitions=0,
    )


#  Main ablation loop 

def run_ablation(n_trials: int = 30, seed: int = 42, duration: float = 8.0,
                 fps: float = 30.0) -> pd.DataFrame:
    rng      = np.random.default_rng(seed)
    dt       = 1.0 / fps
    n_frames = int(duration * fps)
    results  = []

    total = len(SCENARIOS) * 2 * n_trials
    done  = 0

    print(f"\n{'='*60}")
    print(f"  Hesitation-AV Ablation Study")
    print(f"  {len(SCENARIOS)} scenarios × 2 policies × {n_trials} trials = {total} runs")
    print(f"{'='*60}\n")

    for scenario_fn in SCENARIOS:
        for trial in range(n_trials):
            scenario = scenario_fn(n_frames, dt, rng)

            # Greedy
            results.append(run_greedy_policy(scenario, trial))
            done += 1

            # Hesitation-aware
            results.append(run_hesitation_policy(scenario, trial))
            done += 1

            if (done % 20) == 0:
                pct = 100 * done / total
                print(f"  [{pct:5.1f}%]  {scenario['name']}  trial {trial+1}/{n_trials}")

    print(f"\n  [100.0%]  Complete - {total} trials finished\n")

    df = pd.DataFrame([vars(r) for r in results])
    return df


#  Summary statistics 

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["scenario", "policy"])
        .agg(
            hqm_mean=("hqm",            "mean"),
            hqm_std=("hqm",             "std"),
            hqm_median=("hqm",          "median"),
            S_mean=("S",                "mean"),
            E_mean=("E",                "mean"),
            B_mean=("B",                "mean"),
            R_mean=("R",                "mean"),
            ucr=("unsafe_commits",      "mean"),
            dl_mean_ms=("decision_latency_ms", "mean"),
            n_trials=("trial",          "count"),
        )
        .reset_index()
        .round(4)
    )
    return summary


#  Figures 

def plot_results(df: pd.DataFrame, summary: pd.DataFrame):
    scenarios   = df["scenario"].unique()
    n_scenarios = len(scenarios)
    colors      = {"greedy": "#ef4444", "hesitation": "#3b82f6"}

    fig = plt.figure(figsize=(16, 10), facecolor="#0f172a")
    fig.suptitle("Hesitation-AV Ablation Study", fontsize=14,
                 color="#e2e8f0", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, n_scenarios + 1, figure=fig,
                           hspace=0.45, wspace=0.35)

    #  Top row: HQM distributions per scenario 
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#020617")
        ax.set_title(scenario.replace("_", "\n"), fontsize=8,
                     color="#94a3b8", pad=6)

        for policy in ["greedy", "hesitation"]:
            vals = df[(df.scenario == scenario) & (df.policy == policy)]["hqm"]
            ax.hist(vals, bins=12, alpha=0.7, color=colors[policy],
                    label=policy, edgecolor="#020617", linewidth=0.5)

        ax.axvline(x=0.60, color="#475569", linestyle="--",
                   linewidth=0.8, label="baseline")
        ax.set_xlabel("HQM", fontsize=8, color="#64748b")
        ax.set_ylabel("Count", fontsize=8, color="#64748b")
        ax.tick_params(colors="#475569", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        if col == 0:
            ax.legend(fontsize=7, facecolor="#0f172a",
                      labelcolor="#94a3b8", edgecolor="#1e293b")

    #  Top right: Component breakdown bar chart 
    ax_comp = fig.add_subplot(gs[0, -1])
    ax_comp.set_facecolor("#020617")
    ax_comp.set_title("HQM Components\n(hesitation, mean)", fontsize=8,
                       color="#94a3b8", pad=6)

    comp_colors = {"S_mean":"#818cf8","E_mean":"#22c55e",
                   "B_mean":"#eab308","R_mean":"#f97316"}
    labels      = ["S (safety)", "E (efficiency)", "B (stability)", "R (resolution)"]
    hes_summary = summary[summary.policy == "hesitation"]
    comp_vals   = hes_summary[["S_mean","E_mean","B_mean","R_mean"]].mean()

    bars = ax_comp.bar(labels, comp_vals.values,
                       color=list(comp_colors.values()),
                       edgecolor="#020617", linewidth=0.5)
    ax_comp.set_ylim(0, 1.1)
    ax_comp.tick_params(colors="#475569", labelsize=6)
    ax_comp.set_xticklabels(labels, rotation=20, ha="right")
    for spine in ax_comp.spines.values():
        spine.set_edgecolor("#1e293b")

    for bar_obj, val in zip(bars, comp_vals.values):
        ax_comp.text(bar_obj.get_x() + bar_obj.get_width()/2,
                     val + 0.02, f"{val:.2f}",
                     ha="center", va="bottom",
                     color="#e2e8f0", fontsize=7)

    #  Bottom row: UCR and macro HQM comparison 
    ax_hqm = fig.add_subplot(gs[1, :2])
    ax_hqm.set_facecolor("#020617")
    ax_hqm.set_title("Macro HQM by Scenario & Policy",
                      fontsize=9, color="#94a3b8", pad=6)

    x = np.arange(n_scenarios)
    w = 0.35
    for i, policy in enumerate(["greedy", "hesitation"]):
        means = [summary[(summary.scenario == s) &
                          (summary.policy == policy)]["hqm_mean"].values[0]
                 for s in scenarios]
        stds  = [summary[(summary.scenario == s) &
                          (summary.policy == policy)]["hqm_std"].values[0]
                 for s in scenarios]
        ax_hqm.bar(x + i*w, means, w, yerr=stds, label=policy,
                   color=colors[policy], alpha=0.8,
                   capsize=4, ecolor="#475569", edgecolor="#020617")

    ax_hqm.axhline(y=0.60, color="#475569", linestyle="--",
                    linewidth=0.8, label="greedy baseline")
    ax_hqm.set_xticks(x + w/2)
    ax_hqm.set_xticklabels([s.replace("_", "\n") for s in scenarios],
                             fontsize=7, color="#64748b")
    ax_hqm.set_ylabel("HQM", color="#64748b", fontsize=8)
    ax_hqm.set_ylim(0, 1.0)
    ax_hqm.tick_params(colors="#475569", labelsize=7)
    for spine in ax_hqm.spines.values():
        spine.set_edgecolor("#1e293b")
    ax_hqm.legend(fontsize=7, facecolor="#0f172a",
                   labelcolor="#94a3b8", edgecolor="#1e293b")

    ax_ucr = fig.add_subplot(gs[1, 2:])
    ax_ucr.set_facecolor("#020617")
    ax_ucr.set_title("Unsafe Commit Rate by Scenario & Policy",
                      fontsize=9, color="#94a3b8", pad=6)

    for i, policy in enumerate(["greedy", "hesitation"]):
        ucrs = [summary[(summary.scenario == s) &
                         (summary.policy == policy)]["ucr"].values[0]
                for s in scenarios]
        ax_ucr.bar(x + i*w, ucrs, w, label=policy,
                   color=colors[policy], alpha=0.8, edgecolor="#020617")

    ax_ucr.set_xticks(x + w/2)
    ax_ucr.set_xticklabels([s.replace("_", "\n") for s in scenarios],
                             fontsize=7, color="#64748b")
    ax_ucr.set_ylabel("UCR (mean unsafe commits/trial)", color="#64748b", fontsize=8)
    ax_ucr.tick_params(colors="#475569", labelsize=7)
    for spine in ax_ucr.spines.values():
        spine.set_edgecolor("#1e293b")
    ax_ucr.legend(fontsize=7, facecolor="#0f172a",
                   labelcolor="#94a3b8", edgecolor="#1e293b")

    plt.savefig(RESULTS_DIR / "ablation.png", dpi=150,
                bbox_inches="tight", facecolor="#0f172a")
    print(f"  Figure saved → experiments/results/ablation.png")


#  Entry point 

if __name__ == "__main__":
    df      = run_ablation(n_trials=30)
    summary = compute_summary(df)

    # Save
    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"  Results saved → experiments/results/results.csv")
    print(f"  Summary saved → experiments/results/summary.csv\n")

    # Print summary table
    print("=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(summary[["scenario","policy","hqm_mean","hqm_std",
                   "ucr","dl_mean_ms"]].to_string(index=False))
    print()
    print("COMPONENT BREAKDOWN:")
    print(summary[["scenario","policy","S_mean","E_mean","B_mean","R_mean"]].to_string(index=False))
    print()

    # Headline finding
    for scenario in df["scenario"].unique():
        g = summary[(summary.scenario==scenario) & (summary.policy=="greedy")]["hqm_mean"].values[0]
        h = summary[(summary.scenario==scenario) & (summary.policy=="hesitation")]["hqm_mean"].values[0]
        delta = h - g
        print(f"  {scenario:<28}  Δ HQM = {delta:+.4f}  "
              f"({'✓ hesitation wins' if delta > 0 else '✗ no improvement'})")

    print()
    plot_results(df, summary)
    print("\n  Done.\n")