"""
experiments/sensitivity_analysis.py

Sensitivity analysis: perturb key parameters ±20%
and measure HQM impact across all three scenarios.

Parameters tested:
  tau_l   - lower ambiguity threshold
  rho_c   - risk commit threshold  
  sigma_s - oscillation stability threshold

Run:
  python experiments/sensitivity_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from copy import deepcopy

from config import load_config, reset_config
reset_config()

from core.ambiguity.fusion import AmbiguityFusion
from core.risk.composite import RiskComposite, ttc_risk
from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.state_machine.states import State
from core.metrics.hqm import HQMComputer
from core.state_machine.guards import G3 as g3_check

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Scenario profiles (same as run_ablation.py)

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
    return {"t": t, "A": A, "risk": risk, "name": "pedestrian_curb"}


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
    return {"t": t, "A": A, "risk": risk, "name": "merge_hesitation"}


def scenario_occluded_intersection(n_frames, dt, rng):
    t         = np.arange(n_frames) * dt
    A_base    = 0.70 * np.ones(n_frames)
    false_dip = 0.25 * np.exp(-((t - 1.5)**2) / 0.15)
    clear_t   = 3.5 + rng.normal(0, 0.25)
    full_clear = np.clip((t - clear_t) / 0.6, 0, 1)
    A    = np.clip(A_base - false_dip - 0.68 * full_clear +
                   rng.normal(0, 0.022, n_frames), 0, 1)
    risk = np.clip(
        0.65 * np.clip(1.0 - (t - clear_t) / 0.7, 0, 1)
        + 0.10 * np.exp(-((t - 1.5)**2) / 0.3)
        + rng.normal(0, 0.015, n_frames), 0, 1)
    return {"t": t, "A": A, "risk": risk,
            "name": "occluded_intersection"}


SCENARIOS = [
    scenario_pedestrian_curb,
    scenario_merge_hesitation,
    scenario_occluded_intersection,
]


# Run one trial with given config
def run_trial(scenario_fn, rng, dt, n_frames,
              param_overrides: dict) -> float:
    """Run one trial with parameter overrides and return HQM."""
    import config as config_module

    # Reset and load fresh config
    config_module.reset_config()
    cfg = config_module.load_config()

    # Apply overrides directly to the loaded dict
    for key, val in param_overrides.items():
        cfg["state_machine"][key] = val

    # _CONFIG is already mutated since it's the same dict object
    scenario   = scenario_fn(n_frames, dt, rng)
    t_arr      = scenario["t"]
    A_arr      = scenario["A"]
    risk_arr   = scenario["risk"]

    machine    = HesitationStateMachine()
    hqm_comp   = HQMComputer()
    prev_state = State.CRUISE
    risk_hist  = []

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
        osc      = float(np.var(A_arr[window_start:i+1]))
        dR_dt    = 0.0 if i == 0 else \
                   (risk_arr[i] - risk_arr[i-1]) / dt
        risk_proj = float(np.clip(risk + dR_dt, 0, 1))

        inp = MachineInput(t=t, A=A, dA_dt=dA_dt, osc=osc,
                           risk=risk, dR_dt=dR_dt,
                           risk_projected=risk_proj)
        out = machine.tick(inp)
        risk_hist.append(risk)

        if prev_state != State.PROBE and \
           out.state == State.PROBE:
            hqm_comp.on_probe_enter(
                t, float(np.mean(risk_hist)))

        g3_ok = g3_check(A, dA_dt, osc, risk, out.t_in_state)
        hqm_comp.on_tick(out.state, A, risk, dA_dt,
                         osc, t, g3_ok)

        if out.transition_fired:
            hqm_comp.on_transition(out.transition_fired,
                                   t, risk)
        prev_state = out.state

    # Reset config after trial so next trial starts clean
    config_module.reset_config()

    episodes = hqm_comp.completed_episodes
    if episodes:
        return episodes[-1]["hqm"]
    return hqm_comp.greedy_baseline_hqm


# Sensitivity sweep

def run_sensitivity(n_trials=30, seed=42,
                    duration=8.0, fps=30.0):

    rng      = np.random.default_rng(seed)
    dt       = 1.0 / fps
    n_frames = int(duration * fps)

    cfg = load_config()

    # Parameters to perturb
    params = {
        "tau_l":   cfg["state_machine"]["tau_low"],
        "rho_c":   cfg["state_machine"]["rho_commit"],
        "sigma_s": cfg["state_machine"]["sigma_stable"],
    }

    # Perturbation levels
    perturbations = [-0.20, -0.10, 0.00, +0.10, +0.20]

    results = []

    print(f"\n{'='*60}")
    print(f"  Sensitivity Analysis")
    print(f"  3 parameters × 5 perturbations × "
          f"3 scenarios × {n_trials} trials")
    print(f"{'='*60}\n")

    # Baseline first
    print("  Computing baseline...")
    for scenario_fn in SCENARIOS:
        hqm_vals = []
        for _ in range(n_trials):
            hqm = run_trial(scenario_fn, rng, dt,
                            n_frames, {})
            hqm_vals.append(hqm)
        results.append({
            "param":       "baseline",
            "perturbation": 0.0,
            "scenario":    scenario_fn.__name__.replace(
                           "scenario_", ""),
            "hqm_mean":    np.mean(hqm_vals),
            "hqm_std":     np.std(hqm_vals),
        })

    # Perturb each parameter
    param_keys = {
        "tau_l":   "tau_low",
        "rho_c":   "rho_commit",
        "sigma_s": "sigma_stable",
    }

    for param_name, cfg_key in param_keys.items():
        base_val = params[param_name]
        print(f"\n  Parameter: {param_name} "
              f"(baseline={base_val:.4f})")

        for pct in perturbations:
            if pct == 0.0:
                continue  # already have baseline
            perturbed_val = base_val * (1.0 + pct)
            override = {cfg_key: perturbed_val}

            for scenario_fn in SCENARIOS:
                hqm_vals = []
                for _ in range(n_trials):
                    hqm = run_trial(scenario_fn, rng,
                                    dt, n_frames, override)
                    hqm_vals.append(hqm)

                results.append({
                    "param":        param_name,
                    "perturbation": pct * 100,
                    "scenario":     scenario_fn.__name__\
                                    .replace("scenario_", ""),
                    "hqm_mean":     np.mean(hqm_vals),
                    "hqm_std":      np.std(hqm_vals),
                })

                print(f"    {pct:+.0%}  "
                      f"{scenario_fn.__name__.replace('scenario_',''):<25}"
                      f"  HQM={np.mean(hqm_vals):.4f} "
                      f"±{np.std(hqm_vals):.4f}")

    return pd.DataFrame(results)


# Print summary table

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  SENSITIVITY SUMMARY (mean HQM across scenarios)")
    print(f"{'='*60}")
    print(f"\n  {'Param':<12} {'Perturbation':>14} "
          f"{'Mean HQM':>10} {'Std':>8} {'Delta':>8}")
    print(f"  {'-'*55}")

    baseline = df[df.param == "baseline"]["hqm_mean"].mean()

    for param in ["tau_l", "rho_c", "sigma_s"]:
        param_df = df[df.param == param]
        for pct in sorted(param_df["perturbation"].unique()):
            row = param_df[param_df.perturbation == pct]
            mean_hqm = row["hqm_mean"].mean()
            std_hqm  = row["hqm_std"].mean()
            delta    = mean_hqm - baseline
            print(f"  {param:<12} {pct:>+13.0f}%  "
                  f"{mean_hqm:>10.4f} {std_hqm:>8.4f} "
                  f"{delta:>+8.4f}")
        print()

    print(f"  Baseline HQM: {baseline:.4f}")


# Plot

def plot_sensitivity(df: pd.DataFrame):
    params   = ["tau_l", "rho_c", "sigma_s"]
    scenarios = df["scenario"].unique()

    colors = {
        "pedestrian_curb":       "#3b82f6",
        "merge_hesitation":      "#22c55e",
        "occluded_intersection": "#f97316",
    }

    fig = plt.figure(figsize=(16, 5), facecolor="#0f172a")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            hspace=0.3, wspace=0.3)

    baseline_hqm = df[df.param == "baseline"] \
                   .groupby("scenario")["hqm_mean"].mean()

    for col, param in enumerate(params):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#020617")

        param_df = df[df.param == param]

        for scenario in scenarios:
            sdf   = param_df[param_df.scenario == scenario]
            pcts  = sorted(sdf["perturbation"].unique())
            means = []
            stds  = []

            for pct in pcts:
                row = sdf[sdf.perturbation == pct]
                means.append(row["hqm_mean"].values[0])
                stds.append(row["hqm_std"].values[0])

            # Add baseline at 0%
            base = baseline_hqm.get(scenario, 0.5)
            pcts_full  = [-20, -10, 0, 10, 20]

            # Insert baseline at 0%
            means_full = means[:2] + [base] + means[2:]
            stds_full  = stds[:2]  + [0.0]  + stds[2:]

            ax.plot(pcts_full, means_full,
                    color=colors.get(scenario, "#818cf8"),
                    linewidth=2, marker="o", markersize=4,
                    label=scenario.replace("_", "\n"))
            ax.fill_between(
                pcts_full,
                [m - s for m, s in zip(means_full, stds_full)],
                [m + s for m, s in zip(means_full, stds_full)],
                alpha=0.15,
                color=colors.get(scenario, "#818cf8")
            )

        ax.axhline(y=0.60, color="#ef4444",
                   linestyle="--", linewidth=1,
                   label="Greedy baseline")
        ax.axvline(x=0, color="#475569",
                   linestyle=":", linewidth=1)

        param_labels = {
            "tau_l":   r"$\tau_l$ perturbation (%)",
            "rho_c":   r"$\rho_c$ perturbation (%)",
            "sigma_s": r"$\sigma_s$ perturbation (%)",
        }
        ax.set_xlabel(param_labels[param],
                      color="#64748b", fontsize=8)
        ax.set_ylabel("Mean HQM", color="#64748b", fontsize=8)
        ax.set_title(f"Sensitivity: {param}",
                     color="#e2e8f0", fontsize=9)
        ax.tick_params(colors="#475569", labelsize=7)
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_xticklabels(["-20%", "-10%", "0%",
                             "+10%", "+20%"],
                           fontsize=7, color="#475569")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")

        if col == 0:
            ax.legend(fontsize=6, facecolor="#0f172a",
                      labelcolor="#94a3b8",
                      edgecolor="#1e293b",
                      loc="lower left")

    plt.savefig(RESULTS_DIR / "sensitivity.png",
                dpi=150, bbox_inches="tight",
                facecolor="#0f172a")
    print(f"\n  Plot saved → "
          f"experiments/results/sensitivity.png")

def run_random_param_sweep(n_configs=1000, n_trials=10,
                            seed=42, duration=8.0, fps=30.0):
    import config as config_module

    config_module.reset_config()
    base_cfg = config_module.load_config()
    base_sm  = base_cfg["state_machine"]

    rng      = np.random.default_rng(seed + 999)
    dt       = 1.0 / fps
    n_frames = int(duration * fps)

    param_ranges = {
        "tau_low":      (base_sm["tau_low"]      * 0.6,
                         base_sm["tau_low"]      * 1.4),
        "tau_high":     (base_sm["tau_high"]     * 0.6,
                         base_sm["tau_high"]     * 1.4),
        "rho_commit":   (base_sm["rho_commit"]   * 0.6,
                         base_sm["rho_commit"]   * 1.4),
        "sigma_stable": (base_sm["sigma_stable"] * 0.6,
                         base_sm["sigma_stable"] * 1.4),
    }

    results      = []
    beats_greedy = 0

    print(f"\n{'='*60}")
    print(f"  Random Parameter Sweep: {n_configs} configurations")
    print(f"  {n_trials} trials × 3 scenarios per config")
    print(f"{'='*60}\n")

    for config_idx in range(n_configs):
        sampled = {}
        for key, (lo, hi) in param_ranges.items():
            sampled[key] = float(rng.uniform(lo, hi))

        hqm_vals = []
        for scenario_fn in SCENARIOS:
            for _ in range(n_trials):
                hqm = run_trial(
                    scenario_fn, rng, dt,
                    n_frames, sampled)
                hqm_vals.append(hqm)

        mean_hqm = float(np.mean(hqm_vals))
        beats     = mean_hqm > 0.60
        if beats:
            beats_greedy += 1

        results.append({
            "config_idx":   config_idx,
            "mean_hqm":     round(mean_hqm, 4),
            "beats_greedy": beats,
            **{k: round(v, 6) for k, v in sampled.items()}
        })

        if config_idx % 100 == 0:
            pct      = 100 * config_idx / n_configs
            win_rate = 100 * beats_greedy / max(config_idx+1, 1)
            print(f"  [{pct:5.1f}%] config {config_idx:4d}  "
                  f"win rate so far: {win_rate:.1f}%")

    df       = pd.DataFrame(results)
    win_rate = 100 * beats_greedy / n_configs

    print(f"\n  {'='*40}")
    print(f"  Configurations tested:  {n_configs}")
    print(f"  Beat greedy (>0.60):    {beats_greedy} "
          f"({win_rate:.1f}%)")
    print(f"  Mean HQM across all:    "
          f"{df['mean_hqm'].mean():.4f}")
    print(f"  Min HQM observed:       "
          f"{df['mean_hqm'].min():.4f}")
    print(f"  Max HQM observed:       "
          f"{df['mean_hqm'].max():.4f}")

    return df, win_rate

# Entry point 
if __name__ == "__main__":
    df = run_sensitivity(n_trials=30)

    df.to_csv(RESULTS_DIR / "sensitivity.csv", index=False)
    print(f"\n  Data saved → experiments/results/sensitivity.csv")
    print_summary(df)
    plot_sensitivity(df)

    print("\n\nRunning 1000-configuration parameter sweep...")
    sweep_df, win_rate = run_random_param_sweep(
        n_configs=1000, n_trials=10
    )
    sweep_df.to_csv(
        RESULTS_DIR / "param_sweep_1000.csv", index=False)
    print(f"  Saved → experiments/results/param_sweep_1000.csv")
    print(f"\n  Done.")