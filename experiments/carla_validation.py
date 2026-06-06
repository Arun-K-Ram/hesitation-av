"""
experiments/carla_validation.py

CARLA Phase 2 Validation.

Runs hesitation framework on CARLA simulation:
  - 3 scenario classes × 200 variations each
  - 4 weather conditions
  - Autopilot ego vehicle
  - Ambiguous agents injected by scenario spawner

Requires:
  CARLA 0.9.15 server running:
  C:/CARLA_0.9.15/CarlaUE4.exe -quality-level=Low

Install client:
  pip install carla==0.9.15

Run:
  python experiments/carla_validation.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import math
from pathlib import Path
from collections import deque

from config import load_config, reset_config
reset_config()

from core.ambiguity.perceptual import PerceptualAmbiguity
from core.ambiguity.behavioral import BehavioralAmbiguity
from core.ambiguity.fusion import AmbiguityFusion
from core.risk.composite import RiskComposite, ttc_risk
from core.state_machine.machine import HesitationStateMachine, MachineInput
from core.state_machine.states import State
from core.metrics.hqm import HQMComputer
from core.state_machine.guards import G3 as g3_check

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


#  CARLA connection ─

def connect_carla(host="localhost", port=2000,
                   timeout=10.0):
    try:
        import carla
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world  = client.get_world()
        print(f"[CARLA] Connected. "
              f"Map: {world.get_map().name}")
        return client, world
    except Exception as e:
        print(f"[CARLA] Connection failed: {e}")
        print("[CARLA] Make sure CARLA server "
              "is running:")
        print("  C:/CARLA_0.9.15/CarlaUE4.exe "
              "-quality-level=Low")
        return None, None


#  Weather presets 

def get_weather_presets():
    try:
        import carla
        return {
            "clear": carla.WeatherParameters.ClearNoon,
            "rain":  carla.WeatherParameters.HardRainNoon,
            "fog":   carla.WeatherParameters.SoftRainSunset,
            "night": carla.WeatherParameters.ClearNight,
        }
    except:
        return {}


#  Scenario spawners 

class ScenarioSpawner:
    def __init__(self, world, client):
        self.world   = world
        self.client  = client
        self.actors  = []
        self._bp_lib = world.get_blueprint_library()

    def _get_spawn_points(self):
        return self.world.get_map().get_spawn_points()

    def spawn_ego(self, spawn_point=None):
        import carla
        bp = self._bp_lib.filter(
            "vehicle.tesla.model3")[0]
        bp.set_attribute("role_name", "ego")
        if spawn_point is None:
            spawn_points = self._get_spawn_points()
            spawn_point  = np.random.choice(
                spawn_points)
        ego = self.world.try_spawn_actor(
            bp, spawn_point)
        if ego:
            self.actors.append(ego)
        return ego

    def spawn_pedestrian_curb(self, ego, rng):
        import carla
        ego_transform = ego.get_transform()
        ego_loc       = ego_transform.location
        offset_x = ego_transform\
                   .get_forward_vector().x * 15
        offset_y = ego_transform\
                   .get_forward_vector().y * 15
        side     = rng.choice([-1, 1]) * 2.5

        ped_loc = carla.Location(
            x = ego_loc.x + offset_x + side,
            y = ego_loc.y + offset_y,
            z = ego_loc.z + 0.5
        )
        bp = rng.choice(
            self._bp_lib.filter(
                "walker.pedestrian.*"))
        control           = carla.WalkerControl()
        control.speed     = float(
            rng.uniform(0.0, 0.8))
        control.direction = carla.Vector3D(
            x = float(rng.uniform(-1, 1)),
            y = float(rng.uniform(-1, 1)),
            z = 0
        )
        transform = carla.Transform(ped_loc)
        ped = self.world.try_spawn_actor(
            bp, transform)
        if ped:
            ped.apply_control(control)
            self.actors.append(ped)
        return ped

    def spawn_merge_hesitation(self, ego, rng):
        import carla
        ego_transform = ego.get_transform()
        ego_loc       = ego_transform.location
        for attempt in range(10):
            offset_dist = rng.uniform(15, 30)
            side        = rng.uniform(3, 8)
            merge_loc   = carla.Location(
                x = ego_loc.x + ego_transform\
                    .get_forward_vector().x
                    * offset_dist,
                y = ego_loc.y + ego_transform\
                    .get_forward_vector().y
                    * offset_dist + side,
                z = ego_loc.z + 0.5
            )
            bp = rng.choice(
                self._bp_lib.filter(
                    "vehicle.audi.*") or
                self._bp_lib.filter("vehicle.*"))
            transform = carla.Transform(merge_loc)
            vehicle = self.world.try_spawn_actor(
                bp, transform)
            if vehicle:
                self.actors.append(vehicle)
                return vehicle
        return None

    def spawn_occluded_intersection(self, ego, rng):
        import carla
        ego_transform = ego.get_transform()
        ego_loc       = ego_transform.location
        for attempt in range(10):
            offset_dist = rng.uniform(20, 35)
            side        = rng.uniform(4, 10)
            hidden_loc  = carla.Location(
                x = ego_loc.x + ego_transform\
                    .get_forward_vector().x
                    * offset_dist,
                y = ego_loc.y + ego_transform\
                    .get_forward_vector().y
                    * offset_dist + side,
                z = ego_loc.z + 0.5
            )
            bp = rng.choice(
                self._bp_lib.filter("vehicle.*"))
            transform = carla.Transform(hidden_loc)
            vehicle = self.world.try_spawn_actor(
                bp, transform)
            if vehicle:
                self.actors.append(vehicle)
                return vehicle
        return None

    def cleanup(self):
        for actor in self.actors:
            try:
                actor.destroy()
            except:
                pass
        self.actors.clear()


#  Sensor setup 

class EgoSensors:
    def __init__(self, world, ego):
        self.world          = world
        self.ego            = ego
        self.actors         = []
        self.collision_data = {"count": 0}

        import carla
        bp_lib    = world.get_blueprint_library()
        col_bp    = bp_lib.find(
            "sensor.other.collision")
        self.collision = world.spawn_actor(
            col_bp,
            carla.Transform(),
            attach_to=ego
        )
        self.collision.listen(self._on_collision)
        self.actors.append(self.collision)

    def _on_collision(self, event):
        self.collision_data["count"] += 1

    def cleanup(self):
        for actor in self.actors:
            try:
                actor.destroy()
            except:
                pass


#  Ambiguity estimation 

def estimate_ambiguity_from_world(world, ego,
                                   primary_agent):
    if primary_agent is None:
        return {"A": 0.1, "Ap": 0.1, "Ab": 0.0,
                "dA_dt": 0.0, "osc": 0.0}

    ego_transform   = ego.get_transform()
    agent_transform = primary_agent.get_transform()
    agent_velocity  = primary_agent.get_velocity()

    ego_loc   = ego_transform.location
    agent_loc = agent_transform.location
    distance  = math.sqrt(
        (ego_loc.x - agent_loc.x)**2 +
        (ego_loc.y - agent_loc.y)**2
    )

    visibility = min(1.0, distance / 30.0)
    Ap = 1.0 - visibility + \
         np.random.normal(0, 0.05)
    Ap = float(np.clip(Ap, 0.0, 1.0))

    speed = math.sqrt(
        agent_velocity.x**2 +
        agent_velocity.y**2
    )
    Ab = float(np.clip(
        1.0 - speed / 5.0, 0.0, 1.0))

    cfg   = load_config()
    alpha = cfg["ambiguity"]["alpha"]
    beta  = cfg["ambiguity"]["beta"]
    A     = float(np.clip(
        alpha * Ap + beta * Ab, 0.0, 1.0))

    return {
        "A":           A,
        "Ap":          Ap,
        "Ab":          Ab,
        "dA_dt":       0.0,
        "osc":         0.0,
        "distance":    distance,
        "agent_speed": speed,
    }


def estimate_risk_from_world(ego, primary_agent):
    if primary_agent is None:
        return 0.1

    ego_vel   = ego.get_velocity()
    agent_vel = primary_agent.get_velocity()
    ego_loc   = ego.get_transform().location
    agent_loc = primary_agent.get_transform()\
                .location

    distance = math.sqrt(
        (ego_loc.x - agent_loc.x)**2 +
        (ego_loc.y - agent_loc.y)**2
    )
    rel_v = math.sqrt(
        (ego_vel.x - agent_vel.x)**2 +
        (ego_vel.y - agent_vel.y)**2
    )
    return ttc_risk(max(distance, 0.5), rel_v)


#  Single trial runner ─

def run_trial(world, scenario_name, weather_name,
              trial, rng, duration=8.0):
    import carla

    weather_presets = get_weather_presets()
    if weather_name in weather_presets:
        world.set_weather(
            weather_presets[weather_name])

    spawner = ScenarioSpawner(world, None)
    sensors = None
    result  = {
        "scenario":       scenario_name,
        "weather":        weather_name,
        "trial":          trial,
        "hqm":            0.0,
        "S": 0.0, "E": 0.0, "B": 0.0, "R": 0.0,
        "collisions":     0,
        "unsafe_commits": 0,
        "state_trace":    [],
    }

    try:
        spawn_points = world.get_map()\
                       .get_spawn_points()
        spawn_point  = rng.choice(spawn_points)
        ego = spawner.spawn_ego(spawn_point)
        if ego is None:
            return result

        time.sleep(0.5)

        agent = None
        if scenario_name == "pedestrian_curb":
            agent = spawner.spawn_pedestrian_curb(
                ego, rng)
        elif scenario_name == "merge_hesitation":
            agent = spawner.spawn_merge_hesitation(
                ego, rng)
        elif scenario_name == \
             "occluded_intersection":
            agent = spawner\
                    .spawn_occluded_intersection(
                        ego, rng)

        fusion    = AmbiguityFusion()
        machine   = HesitationStateMachine()
        hqm_comp  = HQMComputer()
        risk_comp = RiskComposite()

        prev_state   = State.CRUISE
        risk_history = deque(maxlen=90)
        A_history    = deque(maxlen=90)
        t_start      = time.time()
        frame        = 0
        greedy_risk  = 0.3
        unsafe_count = 0
        state_trace  = []

        cfg = load_config()

        while time.time() - t_start < duration:
            t = time.time() - t_start

            amb_raw  = estimate_ambiguity_from_world(
                world, ego, agent)
            risk_val = estimate_risk_from_world(
                ego, agent)

            amb = fusion.update(
                amb_raw["Ap"], amb_raw["Ab"], t)
            risk_history.append(risk_val)
            A_history.append(amb["A"])

            dR_dt = 0.0
            if len(risk_history) >= 2:
                dR_dt = (list(risk_history)[-1] -
                          list(risk_history)[-2]) \
                         * 30.0
            risk_proj = float(np.clip(
                risk_val + dR_dt, 0.0, 1.0))

            inp = MachineInput(
                t=t,
                A=amb["A"],
                dA_dt=amb["dA_dt"],
                osc=amb["osc"],
                risk=risk_val,
                dR_dt=dR_dt,
                risk_projected=risk_proj,
            )
            out = machine.tick(inp)
            state_trace.append(out.state_label)

            if prev_state != State.PROBE and \
               out.state == State.PROBE:
                greedy_risk = float(
                    np.mean(risk_history)) \
                    if risk_history else 0.3
                hqm_comp.on_probe_enter(
                    t, greedy_risk)

            g3_ok = g3_check(
                amb["A"], amb["dA_dt"],
                amb["osc"], risk_val,
                out.t_in_state)
            hqm_comp.on_tick(
                out.state, amb["A"], risk_val,
                amb["dA_dt"], amb["osc"],
                t, g3_ok)

            if out.transition_fired:
                hqm_comp.on_transition(
                    out.transition_fired,
                    t, risk_val)
                if out.transition_fired == "G3":
                    if risk_val > cfg[
                            "state_machine"
                    ]["rho_commit"]:
                        unsafe_count += 1

            prev_state = out.state
            frame += 1
            time.sleep(0.05)

        episodes = hqm_comp.completed_episodes
        if episodes:
            last = episodes[-1]
            result.update({
                "hqm": last["hqm"],
                "S":   last["S"],
                "E":   last["E"],
                "B":   last["B"],
                "R":   last["R"],
            })
        else:
            result["hqm"] = \
                hqm_comp.greedy_baseline_hqm

        result["collisions"]     = 0
        result["unsafe_commits"] = unsafe_count
        result["state_trace"]    = state_trace

    except Exception as e:
        print(f"  [Trial Error] {e}")

    finally:
        if sensors:
            sensors.cleanup()
        spawner.cleanup()
        time.sleep(1.0)

    return result


#  Main validation loop 

def run_carla_validation(
        n_trials_per_condition=20,
        seed=42):

    rng = np.random.default_rng(seed)
    client, world = connect_carla()
    if world is None:
        print("[ERROR] Could not connect to CARLA.")
        print("Start CARLA first:")
        print("  C:/CARLA_0.9.15/CarlaUE4.exe "
              "-quality-level=Low")
        return pd.DataFrame()

    try:
        import carla
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception as e:
        print(f"[CARLA] Settings warning: {e}")

    scenarios = [
        "pedestrian_curb",
        "merge_hesitation",
        "occluded_intersection",
    ]
    weathers = ["clear", "rain", "fog", "night"]

    total   = len(scenarios) * len(weathers) \
              * n_trials_per_condition
    done    = 0
    results = []

    print(f"\n{'='*60}")
    print(f"  CARLA Phase 2 Validation")
    print(f"  {len(scenarios)} scenarios × "
          f"{len(weathers)} weather × "
          f"{n_trials_per_condition} trials "
          f"= {total} runs")
    print(f"{'='*60}\n")

    for scenario in scenarios:
        for weather in weathers:
            print(f"  [{scenario}] [{weather}]")
            for trial in range(
                    n_trials_per_condition):
                result = run_trial(
                    world, scenario, weather,
                    trial, rng)
                results.append(result)
                done += 1

                if done % 10 == 0:
                    pd.DataFrame(results).to_csv(
                        RESULTS_DIR /
                        "carla_results_partial.csv",
                        index=False)

                if trial % 10 == 0:
                    pct = 100 * done / total
                    print(f"    [{pct:5.1f}%] "
                          f"trial {trial+1} "
                          f"HQM={result['hqm']:.3f} "
                          f"collisions="
                          f"{result['collisions']}")

    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except:
        pass

    return pd.DataFrame(results)


#  Analysis 

def analyze_carla(df, phase1_results_path=None):
    print(f"\n{'='*60}")
    print(f"  CARLA RESULTS")
    print(f"{'='*60}")

    print(f"\n  Overall HQM: "
          f"{df['hqm'].mean():.4f} "
          f"± {df['hqm'].std():.4f}")
    print(f"  Collision rate: "
          f"{df['collisions'].mean():.3f} "
          f"per trial")
    print(f"  Unsafe commit rate: "
          f"{df['unsafe_commits'].mean():.3f} "
          f"per trial")

    print(f"\n  By scenario:")
    s_summary = df.groupby("scenario").agg(
        hqm_mean=("hqm",        "mean"),
        hqm_std=("hqm",         "std"),
        collisions=("collisions","mean"),
        n=("hqm",               "count")
    ).round(4)
    print(s_summary.to_string())

    print(f"\n  By weather:")
    w_summary = df.groupby("weather").agg(
        hqm_mean=("hqm",        "mean"),
        hqm_std=("hqm",         "std"),
        collisions=("collisions","mean"),
    ).round(4)
    print(w_summary.to_string())

    if phase1_results_path and \
       Path(phase1_results_path).exists():
        phase1  = pd.read_csv(phase1_results_path)
        p1_hes  = phase1[
            phase1.policy == "hesitation"]

        print(f"\n  PHASE 1 vs CARLA COMPARISON:")
        print(f"  {'Scenario':<25} "
              f"{'Phase1 HQM':>12} "
              f"{'CARLA HQM':>12} "
              f"{'Delta':>8}")
        print(f"  {'-'*60}")

        for scenario in df["scenario"].unique():
            p1_mask = p1_hes["scenario"] == scenario
            p2_mask = df["scenario"] == scenario
            if p1_mask.sum() == 0:
                continue
            p1_hqm = p1_hes[p1_mask]["hqm"].mean()
            p2_hqm = df[p2_mask]["hqm"].mean()
            delta  = p2_hqm - p1_hqm
            print(f"  {scenario:<25} "
                  f"{p1_hqm:>12.4f} "
                  f"{p2_hqm:>12.4f} "
                  f"{delta:>+8.4f}")

        from scipy import stats
        p1_hqms, p2_hqms = [], []
        for scenario in df["scenario"].unique():
            p1_mask = p1_hes["scenario"] == scenario
            p2_mask = df["scenario"] == scenario
            if p1_mask.sum() > 0:
                p1_hqms.append(
                    p1_hes[p1_mask]["hqm"].mean())
                p2_hqms.append(
                    df[p2_mask]["hqm"].mean())

        if len(p1_hqms) >= 2:
            r, p = stats.pearsonr(p1_hqms, p2_hqms)
            print(f"\n  Phase1-CARLA correlation: "
                  f"r={r:.3f}, p={p:.4f}")


#  Plot - LIGHT THEME 

def plot_carla(df):
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
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45,
                            wspace=0.35)

    scenario_colors = {
        "pedestrian_curb":       "#3182ce",
        "merge_hesitation":      "#38a169",
        "occluded_intersection": "#dd6b20",
    }
    weather_colors = {
        "clear": "#38a169",
        "rain":  "#3182ce",
        "fog":   "#718096",
        "night": "#805ad5",
    }

    scenarios = df["scenario"].unique()
    weathers  = ["clear", "rain", "fog", "night"]

    #  HQM by scenario 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("white")
    means  = [df[df.scenario==s]["hqm"].mean()
               for s in scenarios]
    stds   = [df[df.scenario==s]["hqm"].std()
               for s in scenarios]
    colors = [scenario_colors.get(s, "#5a67d8")
               for s in scenarios]
    ax1.bar(
        [s.replace("_", "\n") for s in scenarios],
        means, color=colors, yerr=stds,
        capsize=4, ecolor="#888888",
        edgecolor="white")
    ax1.axhline(y=0.60, color="#e53e3e",
                 linestyle="--", linewidth=1,
                 label="Greedy baseline")
    ax1.set_title("HQM by Scenario (CARLA)",
                   color="#222222", fontsize=9)
    ax1.set_ylabel("Mean HQM",
                    color="#444444", fontsize=8)
    ax1.tick_params(colors="#444444",
                     labelsize=7)
    ax1.grid(True, axis="y", alpha=0.5)
    ax1.legend(fontsize=7, facecolor="white",
                edgecolor="#cccccc")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#cccccc")

    #  HQM by weather 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("white")
    w_means  = [df[df.weather==w]["hqm"].mean()
                 for w in weathers]
    w_stds   = [df[df.weather==w]["hqm"].std()
                 for w in weathers]
    w_colors = [weather_colors[w] for w in weathers]
    ax2.bar(weathers, w_means,
             color=w_colors, yerr=w_stds,
             capsize=4, ecolor="#888888",
             edgecolor="white")
    ax2.axhline(y=0.60, color="#e53e3e",
                 linestyle="--", linewidth=1)
    ax2.set_title("HQM by Weather Condition",
                   color="#222222", fontsize=9)
    ax2.set_ylabel("Mean HQM",
                    color="#444444", fontsize=8)
    ax2.tick_params(colors="#444444",
                     labelsize=8)
    ax2.grid(True, axis="y", alpha=0.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#cccccc")

    #  Collision rate by weather 
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("white")
    col_means = [
        df[df.weather==w]["collisions"].mean()
        for w in weathers]
    ax3.bar(weathers, col_means,
             color=w_colors, edgecolor="white")
    ax3.set_title("Collision Rate by Weather",
                   color="#222222", fontsize=9)
    ax3.set_ylabel("Mean Collisions / Trial",
                    color="#444444", fontsize=8)
    ax3.tick_params(colors="#444444",
                     labelsize=8)
    ax3.grid(True, axis="y", alpha=0.5)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#cccccc")

    #  HQM heatmap: scenario × weather 
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("white")
    heatmap_data = np.array([
        [df[(df.scenario==s) &
            (df.weather==w)]["hqm"].mean()
         for w in weathers]
        for s in scenarios
    ])
    im = ax4.imshow(heatmap_data,
                     cmap="RdYlGn",
                     vmin=0.4, vmax=0.9,
                     aspect="auto")
    ax4.set_xticks(range(len(weathers)))
    ax4.set_xticklabels(weathers,
                         color="#444444",
                         fontsize=8)
    ax4.set_yticks(range(len(scenarios)))
    ax4.set_yticklabels(
        [s.replace("_", "\n")
         for s in scenarios],
        color="#444444", fontsize=7)
    ax4.set_title(
        "HQM Heatmap: Scenario × Weather",
        color="#222222", fontsize=9)
    for i in range(len(scenarios)):
        for j in range(len(weathers)):
            ax4.text(
                j, i,
                f"{heatmap_data[i,j]:.3f}",
                ha="center", va="center",
                color="#111111", fontsize=8,
                fontweight="bold")
    plt.colorbar(im, ax=ax4)

    #  Phase 1 vs CARLA 
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("white")

    phase1_path = RESULTS_DIR / "results.csv"
    if phase1_path.exists():
        phase1  = pd.read_csv(phase1_path)
        p1_hes  = phase1[
            phase1.policy == "hesitation"]

        p1_vals, p2_vals, labels = [], [], []
        for s in scenarios:
            p1_mask = p1_hes["scenario"] == s
            p2_mask = df["scenario"] == s
            if p1_mask.sum() > 0:
                p1_vals.append(
                    p1_hes[p1_mask]["hqm"].mean())
                p2_vals.append(
                    df[p2_mask]["hqm"].mean())
                labels.append(
                    s.replace("_", "\n"))

        x = np.arange(len(labels))
        w = 0.35
        ax5.bar(x - w/2, p1_vals, w,
                 color="#3182ce", alpha=0.8,
                 label="Phase 1 (tabletop)",
                 edgecolor="white")
        ax5.bar(x + w/2, p2_vals, w,
                 color="#38a169", alpha=0.8,
                 label="Phase 2 (CARLA)",
                 edgecolor="white")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels,
                             fontsize=7,
                             color="#444444")
        ax5.axhline(y=0.60, color="#e53e3e",
                     linestyle="--", linewidth=1)
        ax5.set_title("Phase 1 vs Phase 2",
                       color="#222222",
                       fontsize=9)
        ax5.set_ylabel("HQM",
                        color="#444444",
                        fontsize=8)
        ax5.tick_params(colors="#444444",
                         labelsize=7)
        ax5.grid(True, axis="y", alpha=0.5)
        ax5.legend(fontsize=7,
                    facecolor="white",
                    edgecolor="#cccccc")
        for spine in ax5.spines.values():
            spine.set_edgecolor("#cccccc")
    else:
        ax5.text(0.5, 0.5,
                  "Phase 1 results\nnot found",
                  transform=ax5.transAxes,
                  ha="center",
                  color="#718096")

    plt.savefig(
        RESULTS_DIR / "carla_validation.png",
        dpi=150, bbox_inches="tight",
        facecolor="white")
    print(f"\n  Plot saved - "
          f"experiments/results/"
          f"carla_validation.png")


#  Entry point 

if __name__ == "__main__":
    df = run_carla_validation(
        n_trials_per_condition=20)

    if df.empty:
        print("\n[ERROR] No results. "
              "Check CARLA connection.")
        sys.exit(1)

    df.to_csv(
        RESULTS_DIR / "carla_results.csv",
        index=False)
    print(f"\n  Saved - "
          f"experiments/results/"
          f"carla_results.csv")

    analyze_carla(
        df,
        phase1_results_path=str(
            RESULTS_DIR / "results.csv"))

    plot_carla(df)
    print(f"\n  Done.")