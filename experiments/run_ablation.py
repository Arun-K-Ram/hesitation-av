"""
experiments/run_ablation.py

Ablation study: Greedy vs Hesitation-Aware vs Fixed Delay vs Random Delay
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
from dataclasses import dataclass
from pathlib import Path

from config import load_config, reset_config
from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.state_machine.states import State
from core.metrics.hqm import HQMComputer
from core.state_machine.guards import G3 as g3_check

reset_config()
cfg = load_config()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

fps = 30.0

#  Scenario profiles

def scenario_pedestrian_curb(n_frames, dt, rng):
    t = np.arange(n_frames) * dt
    rise        = np.clip((t - 0.8) / 1.0, 0, 1)
    oscillation = 0.10 * np.sin(2 * np.pi * t * 2.1) * \
                  np.clip((t - 0.8) / 0.5, 0, 1)
    decay = np.clip(1.0 - (t - 2.8) / 0.8, 0, 1)
    A = np.clip(0.58 * rise * decay + oscillation
                + rng.normal(0, 0.018, n_frames), 0, 1)
    risk = np.clip(
        0.62 * np.exp(-((t - 1.5)**2) / 0.35)
        + 0.08 * np.clip(1.0 - (t - 3.0) / 1.5, 0, 1)
        + rng.normal(0, 0.012, n_frames), 0, 1)
    cfg_sm = load_config()["state_machine"]
    greedy_mask = (t > 1.2) & (A < cfg_sm["tau_low"])
    t_greedy = float(t[greedy_mask][0]) if greedy_mask.any() else float(t[-1])
    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy, "name": "pedestrian_curb"}


def scenario_merge_hesitation(n_frames, dt, rng):
    t = np.arange(n_frames) * dt
    A = np.clip(
        0.72 * np.clip((t - 1.5) / 0.5, 0, 1)
        * np.clip(1.0 - (t - 5.0) / 0.8, 0, 1)
        + rng.normal(0, 0.018, n_frames), 0, 1)
    risk = np.clip(
        0.68 * np.clip((t - 1.5) / 0.8, 0, 1)
        * np.clip(1.0 - (t - 5.0) / 0.6, 0, 1)
        + rng.normal(0, 0.015, n_frames), 0, 1)
    cfg_sm = load_config()["state_machine"]
    greedy_mask = (t > 2.0) & (A < cfg_sm["tau_low"])
    t_greedy = float(t[greedy_mask][0]) if greedy_mask.any() \
               else float(t[int(2.5 * fps)])
    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy, "name": "merge_hesitation"}


def scenario_occluded_intersection(n_frames, dt, rng):
    t          = np.arange(n_frames) * dt
    A_base     = 0.70 * np.ones(n_frames)
    false_dip  = 0.25 * np.exp(-((t - 1.5)**2) / 0.15)
    clear_t    = 3.5 + rng.normal(0, 0.25)
    full_clear = np.clip((t - clear_t) / 0.6, 0, 1)
    A = np.clip(A_base - false_dip - 0.68 * full_clear
                + rng.normal(0, 0.022, n_frames), 0, 1)
    risk = np.clip(
        0.65 * np.clip(1.0 - (t - clear_t) / 0.7, 0, 1)
        + 0.10 * np.exp(-((t - 1.5)**2) / 0.3)
        + rng.normal(0, 0.015, n_frames), 0, 1)
    cfg_sm = load_config()["state_machine"]
    greedy_mask = (t > 1.0) & (A < cfg_sm["tau_low"])
    t_greedy = float(t[greedy_mask][0]) if greedy_mask.any() \
               else float(t[-1])
    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy, "name": "occluded_intersection"}


SCENARIOS = [
    scenario_pedestrian_curb,
    scenario_merge_hesitation,
    scenario_occluded_intersection,
]


#  Trial result 

@dataclass
class TrialResult:
    scenario:            str
    policy:              str
    trial:               int
    hqm:                 float
    S:                   float
    E:                   float
    B:                   float
    R:                   float
    unsafe_commits:      int
    decision_latency_ms: float
    resolution:          str
    n_transitions:       int


#  Policy runners 

def run_hesitation_policy(scenario: dict, trial: int) -> TrialResult:
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    dt       = float(t_arr[1] - t_arr[0])

    machine      = HesitationStateMachine()
    hqm_computer = HQMComputer()
    prev_state   = State.CRUISE
    unsafe_count = 0
    commit_t     = None
    cfg_sm       = load_config()["state_machine"]

    for i, (t, A, risk) in enumerate(zip(t_arr, A_arr, risk_arr)):
        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1] - A_arr[i-1]) / (2 * dt)

        window_start = max(0, i - 30)
        osc      = float(np.var(A_arr[window_start:i+1]))
        dR_dt    = 0.0 if i == 0 else \
                   (risk_arr[i] - risk_arr[i-1]) / dt
        risk_proj = float(np.clip(risk + dR_dt * 1.0, 0, 1))

        inp = MachineInput(t=t, A=A, dA_dt=dA_dt, osc=osc,
                           risk=risk, dR_dt=dR_dt,
                           risk_projected=risk_proj)
        out = machine.tick(inp)

        if prev_state != State.PROBE and out.state == State.PROBE:
            greedy_idx = np.searchsorted(t_arr, scenario["t_greedy"])
            greedy_idx = min(greedy_idx, len(risk_arr)-1)
            hqm_computer.on_probe_enter(
                t, float(risk_arr[greedy_idx]))

        g3_ok = g3_check(A, dA_dt, osc, risk, out.t_in_state)
        hqm_computer.on_tick(out.state, A, risk, dA_dt, osc, t, g3_ok)

        if out.transition_fired:
            hqm_computer.on_transition(out.transition_fired, t, risk)

        if out.transition_fired == "G3":
            commit_t = t
            if risk > cfg_sm["rho_commit"]:
                unsafe_count += 1

        prev_state = out.state

    episodes = hqm_computer.completed_episodes
    if episodes:
        last = episodes[-1]
        return TrialResult(
            scenario=scenario["name"], policy="hesitation",
            trial=trial,
            hqm=last["hqm"], S=last["S"], E=last["E"],
            B=last["B"], R=last["R"],
            unsafe_commits=unsafe_count,
            decision_latency_ms=round(
                (commit_t - scenario["t_greedy"]) * 1000
                if commit_t else 0, 1),
            resolution=last["resolution"],
            n_transitions=last["n_transitions"],
        )

    return TrialResult(
        scenario=scenario["name"], policy="hesitation",
        trial=trial,
        hqm=hqm_computer.greedy_baseline_hqm,
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=unsafe_count,
        decision_latency_ms=0.0,
        resolution="NO_EPISODE", n_transitions=0,
    )


def run_greedy_policy(scenario: dict, trial: int) -> TrialResult:
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    cfg_sm   = load_config()["state_machine"]

    commit_mask = A_arr < cfg_sm["tau_low"]
    if commit_mask.any():
        commit_idx  = int(np.argmax(commit_mask))
        commit_risk = float(risk_arr[commit_idx])
    else:
        commit_idx  = len(t_arr) - 1
        commit_risk = float(risk_arr[-1])

    unsafe_count = 1 if commit_risk > cfg_sm["rho_commit"] else 0
    greedy_hqm   = HQMComputer().greedy_baseline_hqm

    return TrialResult(
        scenario=scenario["name"], policy="greedy",
        trial=trial,
        hqm=round(greedy_hqm, 4),
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=unsafe_count,
        decision_latency_ms=0.0,
        resolution="GREEDY_COMMIT", n_transitions=0,
    )


def run_fixed_delay_policy(scenario: dict, trial: int,
                            delay: float = 2.0) -> TrialResult:
    """Commit after fixed delay regardless of ambiguity."""
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    dt       = float(t_arr[1] - t_arr[0])

    hqm_comp  = HQMComputer()
    committed = False

    greedy_mask = t_arr > 0.5
    greedy_risk = float(np.mean(risk_arr[greedy_mask][:5])) \
                  if greedy_mask.any() else 0.3
    hqm_comp.on_probe_enter(t_arr[0], greedy_risk)

    for i in range(len(t_arr)):
        t    = t_arr[i]
        A    = A_arr[i]
        risk = risk_arr[i]

        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1] - A_arr[i-1]) / (2 * dt)

        window_start = max(0, i - 30)
        osc = float(np.var(A_arr[window_start:i+1]))

        if not committed and t >= delay:
            hqm_comp.on_transition("G3", t, risk)
            committed = True
            break

        hqm_comp.on_tick(State.PROBE, A, risk,
                         dA_dt, osc, t, False)

    episodes = hqm_comp.completed_episodes
    if episodes:
        ep = episodes[-1]
        return TrialResult(
            scenario=scenario["name"], policy="fixed_delay",
            trial=trial,
            hqm=ep["hqm"], S=ep["S"], E=ep["E"],
            B=ep["B"], R=ep["R"],
            unsafe_commits=0,
            decision_latency_ms=round(delay * 1000, 1),
            resolution="FIXED_DELAY", n_transitions=1,
        )

    return TrialResult(
        scenario=scenario["name"], policy="fixed_delay",
        trial=trial,
        hqm=hqm_comp.greedy_baseline_hqm,
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=0,
        decision_latency_ms=round(delay * 1000, 1),
        resolution="FIXED_DELAY", n_transitions=0,
    )


def run_random_delay_policy(scenario: dict, trial: int,
                             rng: np.random.Generator) -> TrialResult:
    """Commit at random time between 0 and 4 seconds."""
    delay  = float(rng.uniform(0.0, 4.0))
    result = run_fixed_delay_policy(scenario, trial, delay=delay)
    result.policy = "random_delay"
    result.resolution = "RANDOM_DELAY"
    return result
def run_risk_threshold_policy(scenario: dict,
                               trial: int) -> TrialResult:
    """
    Pure risk-threshold policy: commit when
    R(t) drops below rho_commit regardless of A(t).
    Represents a common reactive AV controller.
    """
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    cfg_sm   = load_config()["state_machine"]

    hqm_comp = HQMComputer()
    greedy_mask = t_arr > 0.5
    greedy_risk = float(np.mean(risk_arr[greedy_mask][:5])) \
                  if greedy_mask.any() else 0.3
    hqm_comp.on_probe_enter(t_arr[0], greedy_risk)

    dt = float(t_arr[1] - t_arr[0])
    committed = False

    for i in range(len(t_arr)):
        t    = t_arr[i]
        A    = A_arr[i]
        risk = risk_arr[i]

        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1] - A_arr[i-1]) / (2 * dt)

        window_start = max(0, i - 30)
        osc = float(np.var(A_arr[window_start:i+1]))

        # Commit when risk drops below threshold
        if not committed and \
           risk < cfg_sm["rho_commit"] and t > 0.5:
            hqm_comp.on_transition("G3", t, risk)
            committed = True
            break

        hqm_comp.on_tick(State.PROBE, A, risk,
                         dA_dt, osc, t, False)

    episodes = hqm_comp.completed_episodes
    if episodes:
        ep = episodes[-1]
        return TrialResult(
            scenario=scenario["name"],
            policy="risk_threshold",
            trial=trial,
            hqm=ep["hqm"], S=ep["S"], E=ep["E"],
            B=ep["B"], R=ep["R"],
            unsafe_commits=0,
            decision_latency_ms=0.0,
            resolution="RISK_THRESHOLD",
            n_transitions=1,
        )
    return TrialResult(
        scenario=scenario["name"],
        policy="risk_threshold", trial=trial,
        hqm=hqm_comp.greedy_baseline_hqm,
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=0,
        decision_latency_ms=0.0,
        resolution="NO_COMMIT", n_transitions=0,
    )


def run_ttc_only_policy(scenario: dict,
                         trial: int) -> TrialResult:
    """
    TTC-only policy: commit when TTC_risk drops
    below 0.3 (standard AV safety threshold).
    Ignores behavioral ambiguity entirely.
    """
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    TTC_THRESHOLD = 0.3

    hqm_comp = HQMComputer()
    greedy_mask = t_arr > 0.5
    greedy_risk = float(np.mean(risk_arr[greedy_mask][:5])) \
                  if greedy_mask.any() else 0.3
    hqm_comp.on_probe_enter(t_arr[0], greedy_risk)

    dt = float(t_arr[1] - t_arr[0])
    committed = False

    for i in range(len(t_arr)):
        t    = t_arr[i]
        A    = A_arr[i]
        risk = risk_arr[i]

        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1] - A_arr[i-1]) / (2 * dt)

        window_start = max(0, i - 30)
        osc = float(np.var(A_arr[window_start:i+1]))

        # TTC-only: commit when risk proxy below threshold
        if not committed and \
           risk < TTC_THRESHOLD and t > 0.5:
            hqm_comp.on_transition("G3", t, risk)
            committed = True
            break

        hqm_comp.on_tick(State.PROBE, A, risk,
                         dA_dt, osc, t, False)

    episodes = hqm_comp.completed_episodes
    if episodes:
        ep = episodes[-1]
        return TrialResult(
            scenario=scenario["name"],
            policy="ttc_only",
            trial=trial,
            hqm=ep["hqm"], S=ep["S"], E=ep["E"],
            B=ep["B"], R=ep["R"],
            unsafe_commits=0,
            decision_latency_ms=0.0,
            resolution="TTC_COMMIT",
            n_transitions=1,
        )
    return TrialResult(
        scenario=scenario["name"],
        policy="ttc_only", trial=trial,
        hqm=hqm_comp.greedy_baseline_hqm,
        S=0.0, E=1.0, B=1.0, R=1.0,
        unsafe_commits=0,
        decision_latency_ms=0.0,
        resolution="NO_COMMIT", n_transitions=0,
    )

#  Main ablation loop 

def run_ablation(n_trials=30, seed=42,
                 duration=8.0, fps=30.0):
    rng      = np.random.default_rng(seed)
    dt       = 1.0 / fps
    n_frames = int(duration * fps)
    results  = []

    n_policies = 6
    total = len(SCENARIOS) * n_policies * n_trials
    done  = 0

    print(f"\n{'='*60}")
    print(f"  Hesitation-AV Ablation Study")
    print(f"  {len(SCENARIOS)} scenarios × {n_policies} policies "
          f"× {n_trials} trials = {total} runs")
    print(f"{'='*60}\n")

    for scenario_fn in SCENARIOS:
        for trial in range(n_trials):
            scenario = scenario_fn(n_frames, dt, rng)

            results.append(run_greedy_policy(scenario, trial))
            done += 1

            results.append(run_hesitation_policy(scenario, trial))
            done += 1

            results.append(run_fixed_delay_policy(
                scenario, trial, delay=2.0))
            done += 1

            results.append(run_random_delay_policy(
                scenario, trial, rng))
            done += 1

            results.append(run_risk_threshold_policy(scenario, trial))
            done += 1

            results.append(run_ttc_only_policy(scenario, trial))
            done += 1

            if (done % 40) == 0:
                pct = 100 * done / total
                print(f"  [{pct:5.1f}%]  {scenario['name']}  "
                      f"trial {trial+1}/{n_trials}")

    print(f"\n  [100.0%]  Complete - {total} trials finished\n")
    return pd.DataFrame([vars(r) for r in results])


#  Summary 

def compute_summary(df):
    return (
        df.groupby(["scenario", "policy"])
        .agg(
            hqm_mean=("hqm",             "mean"),
            hqm_std=("hqm",              "std"),
            S_mean=("S",                 "mean"),
            E_mean=("E",                 "mean"),
            B_mean=("B",                 "mean"),
            R_mean=("R",                 "mean"),
            ucr=("unsafe_commits",       "mean"),
            dl_mean_ms=("decision_latency_ms", "mean"),
            n_trials=("trial",           "count"),
        )
        .reset_index()
        .round(4)
    )


#  Plot 

def plot_results(df, summary):
    scenarios   = df["scenario"].unique()
    n_scenarios = len(scenarios)

    policy_colors = {
        "greedy":          "#ef4444",
        "hesitation":      "#3b82f6",
        "fixed_delay":     "#22c55e",
        "random_delay":    "#eab308",
        "risk_threshold":  "#a855f7",
        "ttc_only":        "#f97316",
    }
    policy_labels = {
        "greedy":          "Greedy",
        "hesitation":      "Hesitation",
        "fixed_delay":     "Fixed Delay (2s)",
        "random_delay":    "Random Delay",
        "risk_threshold":  "Risk Threshold",
        "ttc_only":        "TTC-Only",
    }

    fig = plt.figure(figsize=(18, 10), facecolor="#0f172a")
    fig.suptitle("Hesitation-AV Ablation Study — 4 Policies",
                 fontsize=14, color="#e2e8f0",
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, n_scenarios + 1, figure=fig,
                           hspace=0.45, wspace=0.35)

    #  Top row: HQM distributions per scenario 
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#020617")
        ax.set_title(scenario.replace("_", "\n"),
                     fontsize=8, color="#94a3b8", pad=6)

        for policy in ["greedy", "hesitation",
                        "fixed_delay", "random_delay"]:
            vals = df[(df.scenario == scenario) &
                      (df.policy == policy)]["hqm"]
            ax.hist(vals, bins=12, alpha=0.6,
                    color=policy_colors[policy],
                    label=policy_labels[policy],
                    edgecolor="#020617", linewidth=0.5)

        ax.axvline(x=0.60, color="#475569",
                   linestyle="--", linewidth=0.8)
        ax.set_xlabel("HQM", fontsize=8, color="#64748b")
        ax.set_ylabel("Count", fontsize=8, color="#64748b")
        ax.tick_params(colors="#475569", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        if col == 0:
            ax.legend(fontsize=6, facecolor="#0f172a",
                      labelcolor="#94a3b8",
                      edgecolor="#1e293b")

    #  Top right: Component breakdown 
    ax_comp = fig.add_subplot(gs[0, -1])
    ax_comp.set_facecolor("#020617")
    ax_comp.set_title("HQM Components\n(hesitation, mean)",
                      fontsize=8, color="#94a3b8", pad=6)

    comp_colors = {
        "S_mean": "#818cf8", "E_mean": "#22c55e",
        "B_mean": "#eab308", "R_mean": "#f97316"
    }
    labels    = ["S", "E", "B", "R"]
    hes_summ  = summary[summary.policy == "hesitation"]
    comp_vals = hes_summ[["S_mean","E_mean",
                            "B_mean","R_mean"]].mean()

    bars = ax_comp.bar(labels, comp_vals.values,
                       color=list(comp_colors.values()),
                       edgecolor="#020617")
    ax_comp.set_ylim(0, 1.1)
    ax_comp.tick_params(colors="#475569", labelsize=8)
    for spine in ax_comp.spines.values():
        spine.set_edgecolor("#1e293b")
    for bar_obj, val in zip(bars, comp_vals.values):
        ax_comp.text(
            bar_obj.get_x() + bar_obj.get_width()/2,
            val + 0.02, f"{val:.2f}",
            ha="center", va="bottom",
            color="#e2e8f0", fontsize=8)

    #  Bottom: Macro HQM grouped bar 
    ax_hqm = fig.add_subplot(gs[1, :])
    ax_hqm.set_facecolor("#020617")
    ax_hqm.set_title(
        "Mean HQM by Scenario and Policy",
        fontsize=9, color="#94a3b8", pad=6)

    policies = ["greedy", "hesitation",
                "fixed_delay", "random_delay", "risk_threshold", "ttc_only"]
    x = np.arange(n_scenarios)
    w = 0.13

    for i, policy in enumerate(policies):
        means, stds = [], []
        for scenario in scenarios:
            row = summary[(summary.scenario == scenario) &
                          (summary.policy == policy)]
            means.append(
                row["hqm_mean"].values[0]
                if len(row) > 0 else 0.0)
            stds.append(
                row["hqm_std"].values[0]
                if len(row) > 0 else 0.0)

        ax_hqm.bar(x + i*w, means, w,
                   yerr=stds, capsize=3,
                   label=policy_labels[policy],
                   color=policy_colors[policy],
                   alpha=0.85, ecolor="#475569",
                   edgecolor="#020617")

    ax_hqm.axhline(y=0.60, color="#475569",
                    linestyle="--", linewidth=0.8,
                    label="Greedy baseline")
    ax_hqm.set_xticks(x + w * 2.5)
    ax_hqm.set_xticklabels(
        [s.replace("_", "\n") for s in scenarios],
        fontsize=8, color="#64748b")
    ax_hqm.set_ylabel("HQM", color="#64748b", fontsize=8)
    ax_hqm.set_ylim(0, 1.0)
    ax_hqm.tick_params(colors="#475569", labelsize=7)
    for spine in ax_hqm.spines.values():
        spine.set_edgecolor("#1e293b")
    ax_hqm.legend(fontsize=7, facecolor="#0f172a",
                   labelcolor="#94a3b8",
                   edgecolor="#1e293b")

    plt.savefig(RESULTS_DIR / "ablation.png", dpi=150,
                bbox_inches="tight", facecolor="#0f172a")
    print(f"  Figure saved → experiments/results/ablation.png")


#  Entry point 

if __name__ == "__main__":
    df      = run_ablation(n_trials=30)
    summary = compute_summary(df)

    df.to_csv(RESULTS_DIR / "results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"  Results saved → experiments/results/results.csv")
    print(f"  Summary saved → experiments/results/summary.csv\n")

    print("=" * 60)
    print("  SUMMARY TABLE")
    print("=" * 60)
    print(summary[["scenario","policy","hqm_mean",
                   "hqm_std","ucr",
                   "dl_mean_ms"]].to_string(index=False))
    print()
    print("COMPONENT BREAKDOWN:")
    print(summary[["scenario","policy","S_mean",
                   "E_mean","B_mean",
                   "R_mean"]].to_string(index=False))
    print()

    policies = ["greedy", "hesitation",
                "fixed_delay", "random_delay", "risk_threshold", "ttc_only"]
    for scenario in df["scenario"].unique():
        print(f"\n  {scenario}:")
        baseline = summary[
            (summary.scenario == scenario) &
            (summary.policy == "greedy")
        ]["hqm_mean"].values[0]
        for policy in policies[1:]:
            row = summary[
                (summary.scenario == scenario) &
                (summary.policy == policy)
            ]
            if len(row) > 0:
                hqm   = row["hqm_mean"].values[0]
                delta = hqm - baseline
                win   = "✓ wins" if delta > 0 else "✗ loses"
                print(f"    {policy:<15} "
                      f"HQM={hqm:.4f}  "
                      f"Δ={delta:+.4f}  {win}")

    print()
    plot_results(df, summary)
    print("\n  Done.\n")