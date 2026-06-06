"""
experiments/counterfactual.py

Counterfactual commitment timing analysis.

Question: What is the cost of committing too early or too late?

Method:
  For each scenario, run 30 trials with hesitation-aware policy.
  For each trial, simulate counterfactual commit times
  at offsets [-2.0, -1.5, -1.0, -0.5, 0, +0.5, +1.0, +1.5, +2.0] seconds.
  Recompute HQM at each counterfactual commit time.
  Test asymmetry: premature vs delayed commitment cost.

Outputs:
  experiments/results/counterfactual.csv
  experiments/results/counterfactual.png
  experiments/results/counterfactual_stats.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

from config import load_config, reset_config
reset_config()

from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.state_machine.states import State
from core.state_machine.guards import G3 as g3_check
from core.metrics.hqm import HQMComputer

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OFFSETS = [-2.0, -1.5, -1.0, -0.5, 0.0,
           0.5, 1.0, 1.5, 2.0]


#  Scenario profiles 

def scenario_pedestrian_curb(n_frames, dt, rng):
    t = np.arange(n_frames) * dt
    rise        = np.clip((t - 0.8) / 1.0, 0, 1)
    oscillation = 0.10 * np.sin(2 * np.pi * t * 2.1) * \
                  np.clip((t - 0.8) / 0.5, 0, 1)
    decay       = np.clip(1.0 - (t - 2.8) / 0.8, 0, 1)
    A    = np.clip(0.58 * rise * decay + oscillation +
                   rng.normal(0, 0.018, n_frames), 0, 1)
    risk = np.clip(
        0.62 * np.exp(-((t - 1.5)**2) / 0.35)
        + 0.08 * np.clip(1.0 - (t - 3.0) / 1.5, 0, 1)
        + rng.normal(0, 0.012, n_frames), 0, 1)
    cfg         = load_config()["state_machine"]
    greedy_mask = (t > 1.2) & (A < cfg["tau_low"])
    t_greedy    = float(t[greedy_mask][0]) \
                  if greedy_mask.any() else float(t[-1])
    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy,
            "name": "pedestrian_curb"}


def scenario_merge_hesitation(n_frames, dt, rng):
    t = np.arange(n_frames) * dt
    A = np.clip(
        0.72 * np.clip((t - 1.5) / 0.5, 0, 1)
        * np.clip(1.0 - (t - 5.0) / 0.8, 0, 1)
        + rng.normal(0, 0.018, n_frames), 0, 1)
    risk_base = 0.70 * np.clip((t - 0.8) / 1.0, 0, 1)
    risk_drop = np.clip(1.0 - (t - 3.0) / 0.4, 0, 1)
    risk = np.clip(risk_base * risk_drop +
                   rng.normal(0, 0.015, n_frames), 0, 1)
    cfg         = load_config()["state_machine"]
    greedy_mask = (t > 2.0) & (A < cfg["tau_low"])
    t_greedy    = float(t[greedy_mask][0]) \
                  if greedy_mask.any() \
                  else float(t[int(2.5 * 30)])
    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy,
            "name": "merge_hesitation"}


def scenario_occluded_intersection(n_frames, dt, rng):
    t          = np.arange(n_frames) * dt
    A_base     = 0.70 * np.ones(n_frames)
    false_dip  = 0.25 * np.exp(-((t - 1.5)**2) / 0.15)
    clear_t    = 3.5 + rng.normal(0, 0.25)
    full_clear = np.clip((t - clear_t) / 0.6, 0, 1)
    A    = np.clip(A_base - false_dip -
                   0.68 * full_clear +
                   rng.normal(0, 0.022, n_frames), 0, 1)
    risk = np.clip(
        0.65 * np.clip(1.0 - (t - clear_t) / 0.7, 0, 1)
        + 0.10 * np.exp(-((t - 1.5)**2) / 0.3)
        + rng.normal(0, 0.015, n_frames), 0, 1)
    cfg         = load_config()["state_machine"]
    greedy_mask = (t > 1.0) & (A < cfg["tau_low"])
    t_greedy    = float(t[greedy_mask][0]) \
                  if greedy_mask.any() else float(t[-1])
    return {"t": t, "A": A, "risk": risk,
            "t_greedy": t_greedy,
            "name": "occluded_intersection"}


SCENARIOS = [
    scenario_pedestrian_curb,
    scenario_merge_hesitation,
    scenario_occluded_intersection,
]


#  Get actual commit time 

def get_actual_commit_time(scenario, dt):
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    machine  = HesitationStateMachine()

    for i, (t, A, risk) in enumerate(
            zip(t_arr, A_arr, risk_arr)):
        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1] - A_arr[i-1]) / (2*dt)

        window_start = max(0, i - 30)
        osc       = float(np.var(A_arr[window_start:i+1]))
        dR_dt     = 0.0 if i == 0 else \
                    (risk_arr[i] - risk_arr[i-1]) / dt
        risk_proj = float(np.clip(risk + dR_dt, 0, 1))

        inp = MachineInput(t=t, A=A, dA_dt=dA_dt,
                           osc=osc, risk=risk,
                           dR_dt=dR_dt,
                           risk_projected=risk_proj)
        out = machine.tick(inp)

        if out.transition_fired == "G3":
            return float(t)

    return float(t_arr[-1])


#  Compute HQM at counterfactual commit time 

def compute_hqm_at_commit(scenario, t_commit,
                            fps=30.0):
    t_arr    = scenario["t"]
    A_arr    = scenario["A"]
    risk_arr = scenario["risk"]
    t_greedy = scenario["t_greedy"]

    t_commit   = float(np.clip(t_commit,
                               t_arr[0], t_arr[-1]))
    commit_idx = int(np.searchsorted(t_arr, t_commit))
    commit_idx = min(commit_idx, len(t_arr) - 1)

    risk_at_commit = float(risk_arr[commit_idx])
    greedy_idx     = int(np.searchsorted(
        t_arr, t_greedy))
    greedy_idx     = min(greedy_idx,
                         len(risk_arr) - 1)
    risk_at_greedy = float(risk_arr[greedy_idx])

    S = float(np.clip(
        risk_at_greedy - risk_at_commit, -1.0, 1.0))

    cfg      = load_config()
    T_budget = cfg["hqm"]["t_budget"]
    dt       = float(t_arr[1] - t_arr[0])

    t_g3_eligible = None
    for i in range(len(t_arr)):
        t   = t_arr[i]
        A   = A_arr[i]
        risk = risk_arr[i]
        if i == 0:
            dA_dt = 0.0
        elif i == len(A_arr) - 1:
            dA_dt = (A_arr[i] - A_arr[i-1]) / dt
        else:
            dA_dt = (A_arr[i+1]-A_arr[i-1])/(2*dt)
        window_start = max(0, i - 30)
        osc = float(np.var(A_arr[window_start:i+1]))
        if g3_check(A, dA_dt, osc, risk, t):
            t_g3_eligible = t
            break

    T_start     = t_arr[0]
    T_actual    = t_commit - T_start
    T_necessary = (t_g3_eligible - T_start) \
                  if t_g3_eligible else T_actual
    T_necessary = max(0.0, T_necessary)
    excess      = max(0.0, T_actual - T_necessary)
    E           = float(np.exp(-excess / T_budget))

    A_up_to_commit = A_arr[:commit_idx+1]
    osc_mean       = float(np.var(A_up_to_commit)) \
                     if len(A_up_to_commit) > 1 else 0.0
    B = float(np.clip(1.0 - osc_mean * 10.0, 0.0, 1.0))

    if t_g3_eligible and t_commit >= t_g3_eligible:
        R = 1.0
    else:
        R = 0.5

    alpha_s = cfg["hqm"]["alpha_s"]
    beta_e  = cfg["hqm"]["beta_e"]
    gamma_b = cfg["hqm"]["gamma_b"]
    delta_r = cfg["hqm"]["delta_r"]

    HQM = alpha_s*S + beta_e*E + gamma_b*B + delta_r*R

    return {
        "hqm":            round(float(HQM), 4),
        "S":              round(float(S), 4),
        "E":              round(float(E), 4),
        "B":              round(float(B), 4),
        "R":              float(R),
        "risk_at_commit": round(risk_at_commit, 4),
        "risk_at_greedy": round(risk_at_greedy, 4),
        "t_commit":       round(t_commit, 3),
    }


#  Main counterfactual loop 

def run_counterfactual(n_trials=30, seed=42,
                        duration=8.0, fps=30.0):
    rng      = np.random.default_rng(seed)
    dt       = 1.0 / fps
    n_frames = int(duration * fps)
    results  = []

    print(f"\n{'='*60}")
    print(f"  Counterfactual Commitment Timing Analysis")
    print(f"  {len(SCENARIOS)} scenarios × "
          f"{n_trials} trials × "
          f"{len(OFFSETS)} offsets")
    print(f"{'='*60}\n")

    for scenario_fn in SCENARIOS:
        scenario_name = scenario_fn.__name__\
                        .replace("scenario_", "")
        print(f"  Scenario: {scenario_name}")

        for trial in range(n_trials):
            scenario        = scenario_fn(
                n_frames, dt, rng)
            t_actual_commit = get_actual_commit_time(
                scenario, dt)

            if t_actual_commit >= scenario["t"][-1]:
                continue

            for offset in OFFSETS:
                t_cf     = t_actual_commit + offset
                hqm_dict = compute_hqm_at_commit(
                    scenario, t_cf, fps)
                results.append({
                    "scenario": scenario_name,
                    "trial":    trial,
                    "offset_s": offset,
                    "t_actual": round(t_actual_commit, 3),
                    "t_cf":     round(t_cf, 3),
                    **hqm_dict,
                })

        print(f"    {n_trials} trials complete")

    return pd.DataFrame(results)


#  Statistical test 

def test_asymmetry(df):
    lines = []
    lines.append(
        "ASYMMETRY TEST: Premature vs Delayed Commitment")
    lines.append("=" * 55)

    symmetric_pairs = [
        (-0.5, +0.5),
        (-1.0, +1.0),
        (-1.5, +1.5),
    ]

    for early_off, late_off in symmetric_pairs:
        early_hqm = df[
            df.offset_s == early_off]["hqm"].values
        late_hqm  = df[
            df.offset_s == late_off]["hqm"].values

        if len(early_hqm) == 0 or len(late_hqm) == 0:
            continue

        min_len   = min(len(early_hqm), len(late_hqm))
        early_hqm = early_hqm[:min_len]
        late_hqm  = late_hqm[:min_len]

        stat, p    = stats.wilcoxon(early_hqm, late_hqm)
        early_mean = np.mean(early_hqm)
        late_mean  = np.mean(late_hqm)
        direction  = "PREMATURE COSTS MORE" \
                     if early_mean < late_mean \
                     else "DELAYED COSTS MORE"

        lines.append(
            f"\n  Offset pair: "
            f"{early_off:+.1f}s vs {late_off:+.1f}s")
        lines.append(
            f"    Early mean HQM:  {early_mean:.4f}")
        lines.append(
            f"    Late mean HQM:   {late_mean:.4f}")
        lines.append(
            f"    Difference:      "
            f"{late_mean - early_mean:+.4f}")
        lines.append(
            f"    Wilcoxon p:      {p:.4f} "
            f"{'(significant)' if p < 0.05 else '(not significant)'}")
        lines.append(
            f"    Finding:         {direction}")

    return "\n".join(lines)


#  Plot - LIGHT THEME 

def plot_counterfactual(df):
    scenarios = df["scenario"].unique()
    colors    = {
        "pedestrian_curb":       "#3182ce",
        "merge_hesitation":      "#38a169",
        "occluded_intersection": "#dd6b20",
    }

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#cccccc",
        "axes.labelcolor":  "#222222",
        "xtick.color":      "#444444",
        "ytick.color":      "#444444",
        "text.color":       "#222222",
        "grid.color":       "#eeeeee",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.5,
    })

    fig = plt.figure(figsize=(18, 10),
                      facecolor="white")
    gs  = gridspec.GridSpec(
        2, len(scenarios) + 1,
        figure=fig, hspace=0.45, wspace=0.35)

    #  Top row: HQM vs offset per scenario 
    for col, scenario in enumerate(scenarios):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("white")

        sdf     = df[df.scenario == scenario]
        color   = colors.get(scenario, "#5a67d8")
        grouped = sdf.groupby("offset_s")["hqm"]\
                     .agg(["mean","std"])

        ax.plot(grouped.index, grouped["mean"],
                 color=color, linewidth=2.5,
                 marker="o", markersize=5,
                 label="Mean HQM")
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15, color=color)

        ax.axvline(x=0, color="#718096",
                    linestyle="--", linewidth=1,
                    label="Actual commit")
        ax.axhline(y=0.60, color="#e53e3e",
                    linestyle=":", linewidth=0.8,
                    label="Greedy baseline")

        ax.set_title(
            scenario.replace("_", "\n"),
            color="#333333", fontsize=8, pad=6)
        ax.set_xlabel("Commit offset (s)",
                       color="#444444", fontsize=8)
        ax.set_ylabel("HQM",
                       color="#444444", fontsize=8)
        ax.tick_params(colors="#444444",
                        labelsize=7)
        ax.grid(True, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        if col == 0:
            ax.legend(fontsize=7,
                       facecolor="white",
                       edgecolor="#cccccc")

    #  Top right: All scenarios overlaid 
    ax_all = fig.add_subplot(gs[0, -1])
    ax_all.set_facecolor("white")
    for scenario in scenarios:
        sdf     = df[df.scenario == scenario]
        color   = colors.get(scenario, "#5a67d8")
        grouped = sdf.groupby("offset_s")["hqm"].mean()
        ax_all.plot(grouped.index, grouped.values,
                     color=color, linewidth=2,
                     marker="o", markersize=4,
                     label=scenario.replace(
                         "_", "\n"))

    ax_all.axvline(x=0, color="#718096",
                    linestyle="--", linewidth=1)
    ax_all.axhline(y=0.60, color="#e53e3e",
                    linestyle=":", linewidth=0.8)
    ax_all.set_title("All Scenarios",
                      color="#333333",
                      fontsize=8, pad=6)
    ax_all.set_xlabel("Commit offset (s)",
                       color="#444444", fontsize=8)
    ax_all.set_ylabel("HQM",
                       color="#444444", fontsize=8)
    ax_all.tick_params(colors="#444444",
                        labelsize=7)
    ax_all.grid(True, alpha=0.5)
    for spine in ax_all.spines.values():
        spine.set_edgecolor("#cccccc")
    ax_all.legend(fontsize=6,
                   facecolor="white",
                   edgecolor="#cccccc")

    #  Bottom row: S and E components 
    ax_s = fig.add_subplot(gs[1, :2])
    ax_s.set_facecolor("white")
    ax_e = fig.add_subplot(gs[1, 2:])
    ax_e.set_facecolor("white")

    for scenario in scenarios:
        sdf   = df[df.scenario == scenario]
        color = colors.get(scenario, "#5a67d8")

        s_grouped = sdf.groupby(
            "offset_s")["S"].mean()
        e_grouped = sdf.groupby(
            "offset_s")["E"].mean()

        ax_s.plot(s_grouped.index,
                   s_grouped.values,
                   color=color, linewidth=2,
                   marker="o", markersize=4,
                   label=scenario.replace(
                       "_", "\n"))
        ax_e.plot(e_grouped.index,
                   e_grouped.values,
                   color=color, linewidth=2,
                   marker="o", markersize=4,
                   label=scenario.replace(
                       "_", "\n"))

    for ax, title in [
        (ax_s, "Safety Gain (S) vs Commit Offset"),
        (ax_e, "Efficiency (E) vs Commit Offset"),
    ]:
        ax.axvline(x=0, color="#718096",
                    linestyle="--", linewidth=1)
        ax.set_title(title, color="#333333",
                      fontsize=9, pad=6)
        ax.set_xlabel("Commit offset (s)",
                       color="#444444", fontsize=8)
        ax.tick_params(colors="#444444",
                        labelsize=7)
        ax.grid(True, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.legend(fontsize=7,
                   facecolor="white",
                   edgecolor="#cccccc")

    ax_s.set_ylabel("S (Safety Gain)",
                     color="#444444", fontsize=8)
    ax_e.set_ylabel("E (Efficiency)",
                     color="#444444", fontsize=8)

    plt.savefig(RESULTS_DIR / "counterfactual.png",
                dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"  Plot saved → "
          f"experiments/results/counterfactual.png")


#  Entry point 

if __name__ == "__main__":
    df = run_counterfactual(n_trials=30)

    df.to_csv(RESULTS_DIR / "counterfactual.csv",
               index=False)
    print(f"\n  Data saved → "
          f"experiments/results/counterfactual.csv")

    stats_text = test_asymmetry(df)
    print(f"\n{stats_text}")

    with open(RESULTS_DIR /
              "counterfactual_stats.txt", "w") as f:
        f.write(stats_text)

    print(f"\n  MEAN HQM BY OFFSET "
          f"(all scenarios combined):")
    print(f"  {'Offset':>8}  "
          f"{'Mean HQM':>10}  {'Std':>8}")
    print(f"  {'-'*32}")
    overall = df.groupby("offset_s")["hqm"]\
                .agg(["mean","std"])
    for offset, row in overall.iterrows():
        marker = "<- actual" if offset == 0.0 else ""
        print(f"  {offset:>+8.1f}s  "
              f"{row['mean']:>10.4f}  "
              f"{row['std']:>8.4f}  {marker}")

    plot_counterfactual(df)
    print(f"\n  Done.")