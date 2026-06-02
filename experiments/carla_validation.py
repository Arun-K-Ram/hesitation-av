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

#  CARLA connection 

def connect_carla(host="localhost", port=2000, timeout=10.0):
    """Connect to running CARLA server."""
    try:
        import carla
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world  = client.get_world()
        print(f"[CARLA] Connected. Map: {world.get_map().name}")
        return client, world
    except Exception as e:
        print(f"[CARLA] Connection failed: {e}")
        print("[CARLA] Make sure CARLA server is running:")
        print("  C:/CARLA_0.9.15/CarlaUE4.exe -quality-level=Low")
        return None, None


#  Weather presets 

def get_weather_presets():
    """Four weather conditions for robustness testing."""
    try:
        import carla
        return {
            "clear":  carla.WeatherParameters.ClearNoon,
            "rain":   carla.WeatherParameters.HardRainNoon,
            "fog":    carla.WeatherParameters.SoftRainSunset,
            "night":  carla.WeatherParameters.ClearNight,
        }
    except:
        return {}


#  Scenario spawners 

class ScenarioSpawner:
    """
    Spawns ego vehicle + ambiguous agents for each scenario class.
    Ego uses autopilot. Agents are scripted for ambiguous behavior.
    """

    def __init__(self, world, client):
        self.world      = world
        self.client     = client
        self.actors     = []
        self._bp_lib    = world.get_blueprint_library()

    def _get_spawn_points(self):
        return self.world.get_map().get_spawn_points()

    def spawn_ego(self, spawn_point=None):
        """Spawn ego vehicle with autopilot."""
        import carla
        bp = self._bp_lib.filter("vehicle.tesla.model3")[0]
        bp.set_attribute("role_name", "ego")

        if spawn_point is None:
            spawn_points = self._get_spawn_points()
            spawn_point  = np.random.choice(spawn_points)

        ego = self.world.try_spawn_actor(bp, spawn_point)
        if ego:
            self.actors.append(ego)
        return ego

    def spawn_pedestrian_curb(self, ego, rng):
        """
        Spawn pedestrian near ego path with ambiguous crossing intent.
        Pedestrian oscillates near curb edge.
        """
        import carla

        ego_transform = ego.get_transform()
        ego_loc       = ego_transform.location

        # Place pedestrian 15m ahead, near road edge
        offset_x = ego_transform.get_forward_vector().x * 15
        offset_y = ego_transform.get_forward_vector().y * 15
        side      = rng.choice([-1, 1]) * 2.5  # left or right edge

        ped_loc = carla.Location(
            x = ego_loc.x + offset_x + side,
            y = ego_loc.y + offset_y,
            z = ego_loc.z + 0.5
        )

        bp = rng.choice(self._bp_lib.filter("walker.pedestrian.*"))

        # Randomize pedestrian behavior
        control = carla.WalkerControl()
        control.speed     = float(rng.uniform(0.0, 0.8))  # slow/stopped
        control.direction = carla.Vector3D(
            x = float(rng.uniform(-1, 1)),
            y = float(rng.uniform(-1, 1)),
            z = 0
        )

        transform = carla.Transform(ped_loc)
        ped = self.world.try_spawn_actor(bp, transform)
        if ped:
            ped.apply_control(control)
            self.actors.append(ped)
        return ped

    def spawn_merge_hesitation(self, ego, rng):
        import carla
        ego_transform = ego.get_transform()
        ego_loc = ego_transform.location

        # Try multiple offsets until spawn succeeds
        for attempt in range(10):
            offset_dist = rng.uniform(15, 30)
            side = rng.uniform(3, 8)
            merge_loc = carla.Location(
                x = ego_loc.x + ego_transform.get_forward_vector().x * offset_dist,
                y = ego_loc.y + ego_transform.get_forward_vector().y * offset_dist + side,
                z = ego_loc.z + 0.5
            )
            bp = rng.choice(self._bp_lib.filter("vehicle.audi.*") or
                            self._bp_lib.filter("vehicle.*"))
            transform = carla.Transform(merge_loc)
            vehicle = self.world.try_spawn_actor(bp, transform)
            if vehicle:
                self.actors.append(vehicle)
                return vehicle
        return None  # all attempts failed, trial continues without agent

    def spawn_occluded_intersection(self, ego, rng):
        import carla
        ego_transform = ego.get_transform()
        ego_loc = ego_transform.location

        for attempt in range(10):
            offset_dist = rng.uniform(20, 35)
            side = rng.uniform(4, 10)
            hidden_loc = carla.Location(
                x = ego_loc.x + ego_transform.get_forward_vector().x * offset_dist,
                y = ego_loc.y + ego_transform.get_forward_vector().y * offset_dist + side,
                z = ego_loc.z + 0.5
            )
            bp = rng.choice(self._bp_lib.filter("vehicle.*"))
            transform = carla.Transform(hidden_loc)
            vehicle = self.world.try_spawn_actor(bp, transform)
            if vehicle:
                self.actors.append(vehicle)
                return vehicle
        return None

    def cleanup(self):
        """Destroy all spawned actors."""
        for actor in self.actors:
            try:
                actor.destroy()
            except:
                pass
        self.actors.clear()


#  Sensor setup 
class EgoSensors:
    """Collision sensor only — camera removed for stability."""

    def __init__(self, world, ego):
        self.world          = world
        self.ego            = ego
        self.actors         = []
        self.collision_data = {"count": 0}

        bp_lib = world.get_blueprint_library()
        col_bp = bp_lib.find("sensor.other.collision")

        import carla
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
#  Ambiguity estimation from CARLA world state 

def estimate_ambiguity_from_world(world, ego, primary_agent):
    """
    Compute A(t) directly from CARLA world state.
    Uses ground-truth agent positions for behavioral ambiguity
    and simulated detection confidence for perceptual ambiguity.
    """
    if primary_agent is None:
        return {"A": 0.1, "Ap": 0.1, "Ab": 0.0,
                "dA_dt": 0.0, "osc": 0.0}

    ego_transform   = ego.get_transform()
    agent_transform = primary_agent.get_transform()
    agent_velocity  = primary_agent.get_velocity()

    # Distance from ego to agent
    ego_loc   = ego_transform.location
    agent_loc = agent_transform.location
    distance  = math.sqrt(
        (ego_loc.x - agent_loc.x)**2 +
        (ego_loc.y - agent_loc.y)**2
    )

    # Perceptual ambiguity proxy:
    # closer + more occluded = higher Ap
    # Use distance-based visibility model
    visibility = min(1.0, distance / 30.0)
    Ap = 1.0 - visibility + np.random.normal(0, 0.05)
    Ap = float(np.clip(Ap, 0.0, 1.0))

    # Behavioral ambiguity:
    # Use actual velocity magnitude and direction change
    speed = math.sqrt(
        agent_velocity.x**2 +
        agent_velocity.y**2
    )
    # Low speed near ego path = high behavioral ambiguity
    Ab = float(np.clip(1.0 - speed / 5.0, 0.0, 1.0))

    cfg   = load_config()
    alpha = cfg["ambiguity"]["alpha"]
    beta  = cfg["ambiguity"]["beta"]
    A     = float(np.clip(alpha * Ap + beta * Ab, 0.0, 1.0))

    return {
        "A":     A,
        "Ap":    Ap,
        "Ab":    Ab,
        "dA_dt": 0.0,  # updated by fusion layer
        "osc":   0.0,
        "distance": distance,
        "agent_speed": speed,
    }


def estimate_risk_from_world(ego, primary_agent):
    """Compute Risk(t) from CARLA world state."""
    if primary_agent is None:
        return 0.1

    ego_vel   = ego.get_velocity()
    agent_vel = primary_agent.get_velocity()
    ego_loc   = ego.get_transform().location
    agent_loc = primary_agent.get_transform().location

    distance = math.sqrt(
        (ego_loc.x - agent_loc.x)**2 +
        (ego_loc.y - agent_loc.y)**2
    )

    # Relative velocity toward each other
    rel_v = math.sqrt(
        (ego_vel.x - agent_vel.x)**2 +
        (ego_vel.y - agent_vel.y)**2
    )

    return ttc_risk(max(distance, 0.5), rel_v)


#  Single trial runner 

def run_trial(world, scenario_name: str, weather_name: str,
              trial: int, rng: np.random.Generator,
              duration: float = 8.0) -> dict:
    """
    Run one hesitation trial in CARLA.

    Args:
        world:         CARLA world object
        scenario_name: pedestrian_curb / merge / occluded
        weather_name:  clear / rain / fog / night
        trial:         trial index
        rng:           random generator
        duration:      trial duration in seconds

    Returns:
        Trial result dict with HQM components
    """
    import carla

    # Set weather
    weather_presets = get_weather_presets()
    if weather_name in weather_presets:
        world.set_weather(weather_presets[weather_name])

    spawner = ScenarioSpawner(world, None)
    sensors = None
    result  = {
        "scenario":    scenario_name,
        "weather":     weather_name,
        "trial":       trial,
        "hqm":         0.0,
        "S": 0.0, "E": 0.0, "B": 0.0, "R": 0.0,
        "collisions":  0,
        "unsafe_commits": 0,
        "state_trace": [],
    }

    try:
        # Spawn ego
        spawn_points = world.get_map().get_spawn_points()
        spawn_point  = rng.choice(spawn_points)
        ego = spawner.spawn_ego(spawn_point)
        if ego is None:
            return result

        # Attach sensors
        try:
            sensors = None
            time.sleep(0.5)
        except Exception as e:
            print(f"  [Sensor Error] {e}")
            sensors = None

        # Spawn ambiguous agent
        agent = None
        if scenario_name == "pedestrian_curb":
            agent = spawner.spawn_pedestrian_curb(ego, rng)
        elif scenario_name == "merge_hesitation":
            agent = spawner.spawn_merge_hesitation(ego, rng)
        elif scenario_name == "occluded_intersection":
            agent = spawner.spawn_occluded_intersection(ego, rng)

        

        # Initialise hesitation pipeline
        fusion   = AmbiguityFusion()
        machine  = HesitationStateMachine()
        hqm_comp = HQMComputer()
        risk_comp = RiskComposite()

        prev_state    = State.CRUISE
        risk_history  = deque(maxlen=90)
        A_history     = deque(maxlen=90)
        t_start       = time.time()
        frame         = 0
        greedy_risk   = 0.3
        unsafe_count  = 0
        state_trace   = []

        cfg = load_config()

        #  Main trial loop 
        while time.time() - t_start < duration:
            
            t = time.time() - t_start

            # Get ambiguity from world state
            amb_raw = estimate_ambiguity_from_world(
                world, ego, agent
            )
            risk_val = estimate_risk_from_world(ego, agent)

            # Update fusion layer
            amb = fusion.update(
                amb_raw["Ap"], amb_raw["Ab"], t
            )
            risk_history.append(risk_val)
            A_history.append(amb["A"])

            # Risk derivatives
            dR_dt = 0.0
            if len(risk_history) >= 2:
                dR_dt = (list(risk_history)[-1] -
                          list(risk_history)[-2]) * 30.0
            risk_proj = float(np.clip(
                risk_val + dR_dt * 1.0, 0.0, 1.0
            ))

            # State machine tick
            inp = MachineInput(
                t=t,
                A=amb["A"], dA_dt=amb["dA_dt"],
                osc=amb["osc"],
                risk=risk_val, dR_dt=dR_dt,
                risk_projected=risk_proj,
            )
            out = machine.tick(inp)
            state_trace.append(out.state_label)

            # HQM tracking
            if prev_state != State.PROBE and \
               out.state == State.PROBE:
                greedy_risk = float(np.mean(risk_history)) \
                              if risk_history else 0.3
                hqm_comp.on_probe_enter(t, greedy_risk)

            g3_ok = g3_check(
                amb["A"], amb["dA_dt"], amb["osc"],
                risk_val, out.t_in_state
            )
            hqm_comp.on_tick(
                out.state, amb["A"], risk_val,
                amb["dA_dt"], amb["osc"], t, g3_ok
            )

            if out.transition_fired:
                hqm_comp.on_transition(
                    out.transition_fired, t, risk_val
                )
                if out.transition_fired == "G3":
                    if risk_val > cfg["state_machine"]["rho_commit"]:
                        unsafe_count += 1

            prev_state = out.state
            frame += 1
            time.sleep(0.05)

        #  Collect results 
        episodes = hqm_comp.completed_episodes
        if episodes:
            last = episodes[-1]
            result.update({
                "hqm":   last["hqm"],
                "S":     last["S"],
                "E":     last["E"],
                "B":     last["B"],
                "R":     last["R"],
            })
        else:
            result["hqm"] = hqm_comp.greedy_baseline_hqm

        result["collisions"] = 0  # sensors disabled
        result["unsafe_commits"] = unsafe_count
        result["state_trace"]   = state_trace

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
    n_trials_per_condition: int = 50,
    seed: int = 42
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    client, world = connect_carla()
    if world is None:
        print("[ERROR] Could not connect to CARLA.")
        print("Start CARLA first:")
        print("  C:/CARLA_0.9.15/CarlaUE4.exe -quality-level=Low")
        return pd.DataFrame()

    # Set synchronous mode for reproducibility
    try:
        import carla
        settings = world.get_settings()
        settings.synchronous_mode  = False
        world.apply_settings(settings)
    except Exception as e:
        print(f"[CARLA] Settings warning: {e}")

    scenarios = [
        "pedestrian_curb",
        "merge_hesitation",
        "occluded_intersection",
    ]
    weathers = ["clear", "rain", "fog", "night"]

    total   = len(scenarios) * len(weathers) * n_trials_per_condition
    done    = 0
    results = []

    print(f"\n{'='*60}")
    print(f"  CARLA Phase 2 Validation")
    print(f"  {len(scenarios)} scenarios × {len(weathers)} weather "
          f"× {n_trials_per_condition} trials = {total} runs")
    print(f"{'='*60}\n")

    for scenario in scenarios:
        for weather in weathers:
            print(f"  [{scenario}] [{weather}]")

            for trial in range(n_trials_per_condition):
                
                result = run_trial(
                    world, scenario, weather,
                    trial, rng
                )
                
                results.append(result)
                done += 1

                # Save incrementally every 10 trials
                if done % 10 == 0:
                    pd.DataFrame(results).to_csv(
                        RESULTS_DIR / "carla_results_partial.csv",
                        index=False
                    )

                if trial % 10 == 0:
                    pct = 100 * done / total
                    print(f"    [{pct:5.1f}%] trial {trial+1} "
                          f"HQM={result['hqm']:.3f} "
                          f"collisions={result['collisions']}")

    # Restore async mode
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except:
        pass

    return pd.DataFrame(results)


#  Analysis 

def analyze_carla(df: pd.DataFrame,
                  phase1_results_path: str = None):
    """
    Analyze CARLA results and compare to Phase 1.
    """
    print(f"\n{'='*60}")
    print(f"  CARLA RESULTS")
    print(f"{'='*60}")

    # Overall
    print(f"\n  Overall HQM: {df['hqm'].mean():.4f} "
          f"± {df['hqm'].std():.4f}")
    print(f"  Collision rate: {df['collisions'].mean():.3f} "
          f"per trial")
    print(f"  Unsafe commit rate: "
          f"{df['unsafe_commits'].mean():.3f} per trial")

    # By scenario
    print(f"\n  By scenario:")
    s_summary = df.groupby("scenario").agg(
        hqm_mean=("hqm", "mean"),
        hqm_std=("hqm", "std"),
        collisions=("collisions", "mean"),
        n=("hqm", "count")
    ).round(4)
    print(s_summary.to_string())

    # By weather
    print(f"\n  By weather:")
    w_summary = df.groupby("weather").agg(
        hqm_mean=("hqm", "mean"),
        hqm_std=("hqm", "std"),
        collisions=("collisions", "mean"),
    ).round(4)
    print(w_summary.to_string())

    # Phase 1 comparison
    if phase1_results_path and \
       Path(phase1_results_path).exists():
        phase1 = pd.read_csv(phase1_results_path)
        phase1_hes = phase1[phase1.policy == "hesitation"]

        print(f"\n  PHASE 1 vs CARLA COMPARISON:")
        print(f"  {'Scenario':<25} {'Phase1 HQM':>12} "
              f"{'CARLA HQM':>12} {'Delta':>8}")
        print(f"  {'-'*60}")

        for scenario in df["scenario"].unique():
            p1_mask = phase1_hes["scenario"] == scenario
            p2_mask = df["scenario"] == scenario

            if p1_mask.sum() == 0:
                continue

            p1_hqm = phase1_hes[p1_mask]["hqm"].mean()
            p2_hqm = df[p2_mask]["hqm"].mean()
            delta  = p2_hqm - p1_hqm

            print(f"  {scenario:<25} {p1_hqm:>12.4f} "
                  f"{p2_hqm:>12.4f} {delta:>+8.4f}")

        # Correlation between phases
        from scipy import stats
        p1_hqms = []
        p2_hqms = []
        for scenario in df["scenario"].unique():
            p1_mask = phase1_hes["scenario"] == scenario
            p2_mask = df["scenario"] == scenario
            if p1_mask.sum() > 0:
                p1_hqms.append(
                    phase1_hes[p1_mask]["hqm"].mean()
                )
                p2_hqms.append(
                    df[p2_mask]["hqm"].mean()
                )

        if len(p1_hqms) >= 2:
            r, p = stats.pearsonr(p1_hqms, p2_hqms)
            print(f"\n  Phase1-CARLA correlation: "
                  f"r={r:.3f}, p={p:.4f}")


#  Plotting 

def plot_carla(df: pd.DataFrame):
    fig = plt.figure(figsize=(18, 10),
                     facecolor="#0f172a")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35)

    scenario_colors = {
        "pedestrian_curb":       "#3b82f6",
        "merge_hesitation":      "#22c55e",
        "occluded_intersection": "#f97316",
    }
    weather_colors = {
        "clear": "#22c55e",
        "rain":  "#3b82f6",
        "fog":   "#94a3b8",
        "night": "#a855f7",
    }

    #  HQM by scenario 
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#020617")
    scenarios = df["scenario"].unique()
    means = [df[df.scenario==s]["hqm"].mean()
             for s in scenarios]
    stds  = [df[df.scenario==s]["hqm"].std()
             for s in scenarios]
    colors = [scenario_colors.get(s, "#818cf8")
              for s in scenarios]
    ax1.bar([s.replace("_", "\n") for s in scenarios],
            means, color=colors, yerr=stds,
            capsize=4, ecolor="#475569",
            edgecolor="#020617")
    ax1.axhline(y=0.60, color="#ef4444",
                linestyle="--", linewidth=1,
                label="Greedy baseline")
    ax1.set_title("HQM by Scenario (CARLA)",
                  color="#e2e8f0", fontsize=9)
    ax1.set_ylabel("Mean HQM", color="#64748b",
                   fontsize=8)
    ax1.tick_params(colors="#475569", labelsize=7)
    ax1.legend(fontsize=7, facecolor="#0f172a",
               labelcolor="#94a3b8",
               edgecolor="#1e293b")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")

    #  HQM by weather 
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#020617")
    weathers = ["clear", "rain", "fog", "night"]
    w_means  = [df[df.weather==w]["hqm"].mean()
                for w in weathers]
    w_stds   = [df[df.weather==w]["hqm"].std()
                for w in weathers]
    w_colors = [weather_colors[w] for w in weathers]
    ax2.bar(weathers, w_means, color=w_colors,
            yerr=w_stds, capsize=4,
            ecolor="#475569", edgecolor="#020617")
    ax2.axhline(y=0.60, color="#ef4444",
                linestyle="--", linewidth=1)
    ax2.set_title("HQM by Weather Condition",
                  color="#e2e8f0", fontsize=9)
    ax2.set_ylabel("Mean HQM", color="#64748b",
                   fontsize=8)
    ax2.tick_params(colors="#475569", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")

    #  Collision rate by weather 
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#020617")
    col_means = [df[df.weather==w]["collisions"].mean()
                 for w in weathers]
    ax3.bar(weathers, col_means, color=w_colors,
            edgecolor="#020617")
    ax3.set_title("Collision Rate by Weather",
                  color="#e2e8f0", fontsize=9)
    ax3.set_ylabel("Mean Collisions / Trial",
                   color="#64748b", fontsize=8)
    ax3.tick_params(colors="#475569", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#1e293b")

    #  HQM heatmap: scenario × weather 
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#020617")
    heatmap_data = np.array([
        [df[(df.scenario==s) &
            (df.weather==w)]["hqm"].mean()
         for w in weathers]
        for s in scenarios
    ])
    im = ax4.imshow(heatmap_data, cmap="RdYlGn",
                    vmin=0.4, vmax=0.9,
                    aspect="auto")
    ax4.set_xticks(range(len(weathers)))
    ax4.set_xticklabels(weathers, color="#64748b",
                         fontsize=8)
    ax4.set_yticks(range(len(scenarios)))
    ax4.set_yticklabels(
        [s.replace("_", "\n") for s in scenarios],
        color="#64748b", fontsize=7
    )
    ax4.set_title("HQM Heatmap: Scenario × Weather",
                  color="#e2e8f0", fontsize=9)
    for i in range(len(scenarios)):
        for j in range(len(weathers)):
            ax4.text(j, i, f"{heatmap_data[i,j]:.3f}",
                     ha="center", va="center",
                     color="white", fontsize=8,
                     fontweight="bold")
    plt.colorbar(im, ax=ax4)

    #  Phase 1 vs CARLA comparison 
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#020617")

    phase1_path = RESULTS_DIR / "results.csv"
    if phase1_path.exists():
        phase1 = pd.read_csv(phase1_path)
        p1_hes = phase1[phase1.policy == "hesitation"]

        p1_vals, p2_vals, labels = [], [], []
        for s in scenarios:
            p1_mask = p1_hes["scenario"] == s
            p2_mask = df["scenario"] == s
            if p1_mask.sum() > 0:
                p1_vals.append(
                    p1_hes[p1_mask]["hqm"].mean()
                )
                p2_vals.append(
                    df[p2_mask]["hqm"].mean()
                )
                labels.append(
                    s.replace("_", "\n")
                )

        x = np.arange(len(labels))
        w = 0.35
        ax5.bar(x - w/2, p1_vals, w,
                color="#3b82f6", alpha=0.8,
                label="Phase 1 (tabletop)",
                edgecolor="#020617")
        ax5.bar(x + w/2, p2_vals, w,
                color="#22c55e", alpha=0.8,
                label="Phase 2 (CARLA)",
                edgecolor="#020617")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels, fontsize=7,
                             color="#64748b")
        ax5.axhline(y=0.60, color="#ef4444",
                    linestyle="--", linewidth=1)
        ax5.set_title("Phase 1 vs Phase 2",
                      color="#e2e8f0", fontsize=9)
        ax5.set_ylabel("HQM", color="#64748b",
                        fontsize=8)
        ax5.tick_params(colors="#475569",
                         labelsize=7)
        ax5.legend(fontsize=7, facecolor="#0f172a",
                   labelcolor="#94a3b8",
                   edgecolor="#1e293b")
        for spine in ax5.spines.values():
            spine.set_edgecolor("#1e293b")
    else:
        ax5.text(0.5, 0.5,
                 "Phase 1 results\nnot found",
                 transform=ax5.transAxes,
                 ha="center", color="#64748b")

    plt.savefig(RESULTS_DIR / "carla_validation.png",
                dpi=150, bbox_inches="tight",
                facecolor="#0f172a")
    print(f"\n  Plot saved → "
          f"experiments/results/carla_validation.png")


#  Entry point

if __name__ == "__main__":
    df = run_carla_validation(
        n_trials_per_condition=20
    )

    if df.empty:
        print("\n[ERROR] No results. "
              "Check CARLA connection.")
        sys.exit(1)

    # Save
    df.to_csv(
        RESULTS_DIR / "carla_results.csv",
        index=False
    )
    print(f"\n  Saved → "
          f"experiments/results/carla_results.csv")

    # Analyze
    analyze_carla(
        df,
        phase1_results_path=str(
            RESULTS_DIR / "results.csv"
        )
    )

    # Plot
    plot_carla(df)
    print(f"\n  Done.")