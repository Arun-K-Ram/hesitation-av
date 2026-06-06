"""
experiments/carla_spawn_demo.py

Interactive CARLA scenario visualizer for paper screenshots.

Spawns ego vehicle + ambiguous agents for each scenario class.
Use for taking screenshots for the paper figures.

Requirements:
  CARLA 0.9.15 running:
  C:/carla/WindowsNoEditor/CarlaUE4.exe -quality-level=Low

Run:
  python experiments/carla_spawn_demo.py
"""

import carla
import time
import sys
import math

#  Connect 

def connect():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world  = client.get_world()
    print(f"[CARLA] Connected. Map: {world.get_map().name}")
    return client, world


#  Cleanup 

def cleanup(actors):
    print("\n[CARLA] Cleaning up actors...")
    for actor in actors:
        try:
            actor.destroy()
        except:
            pass
    print("[CARLA] Done.")


#  Scenario 1: pedestrian_curb 

def spawn_pedestrian_curb(world):
    """
    Ego vehicle on road.
    Pedestrian standing near curb ahead with
    ambiguous crossing intent.
    """
    actors = []
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn ego
    ego_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    ego_bp.set_attribute('role_name', 'ego')
    ego_sp = spawn_points[0]
    ego = world.try_spawn_actor(ego_bp, ego_sp)
    if ego:
        actors.append(ego)
        print(f"  Ego spawned at {ego_sp.location}")

    # Spawn pedestrian 12m ahead near curb
    if ego:
        ego_loc = ego_sp.location
        fwd     = ego_sp.get_forward_vector()

        ped_loc = carla.Location(
            x = ego_loc.x + fwd.x * 12 + 2.5,
            y = ego_loc.y + fwd.y * 12,
            z = ego_loc.z + 0.5
        )
        ped_bp = bp_lib.filter('walker.pedestrian.0001')[0]
        ped = world.try_spawn_actor(
            ped_bp,
            carla.Transform(ped_loc)
        )
        if ped:
            # Slow walk toward curb — ambiguous intent
            control = carla.WalkerControl()
            control.speed = 0.5
            control.direction = carla.Vector3D(
                x=-fwd.y, y=fwd.x, z=0)
            ped.apply_control(control)
            actors.append(ped)
            print(f"  Pedestrian spawned near curb")

    return actors, ego


#  Scenario 2: merge_hesitation 

def spawn_merge_hesitation(world):
    """
    Ego vehicle on road.
    Second vehicle approaching from side lane
    at merge point — hesitating to yield.
    """
    actors = []
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn ego
    ego_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    ego_bp.set_attribute('role_name', 'ego')
    ego_sp = spawn_points[1] if len(spawn_points) > 1 \
             else spawn_points[0]
    ego = world.try_spawn_actor(ego_bp, ego_sp)
    if ego:
        actors.append(ego)
        print(f"  Ego spawned")

    # Spawn merging vehicle from side
    if ego:
        ego_loc = ego_sp.location
        fwd     = ego_sp.get_forward_vector()

        merge_loc = carla.Location(
            x = ego_loc.x + fwd.x * 18 + 4.0,
            y = ego_loc.y + fwd.y * 18 + 4.0,
            z = ego_loc.z
        )
        merge_bp = bp_lib.filter('vehicle.audi.a2')[0]
        merge_transform = carla.Transform(
            merge_loc,
            carla.Rotation(yaw=ego_sp.rotation.yaw + 90)
        )
        merge_vehicle = world.try_spawn_actor(
            merge_bp, merge_transform
        )
        if merge_vehicle:
            actors.append(merge_vehicle)
            print(f"  Merging vehicle spawned at side")

    return actors, ego


#  Scenario 3: occluded_intersection 

def spawn_occluded_intersection(world):
    actors = []
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn ego
    ego_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    ego_bp.set_attribute('role_name', 'ego')
    ego_sp = spawn_points[2] if len(spawn_points) > 2 \
             else spawn_points[0]
    ego = world.try_spawn_actor(ego_bp, ego_sp)
    if ego:
        actors.append(ego)

    # Spawn hidden vehicle at known offset
    # that places it behind building in Town10HD
    if ego:
        ego_loc = ego_sp.location
        fwd     = ego_sp.get_forward_vector()

        # Try multiple offsets until one looks occluded
        offsets = [
            (25, 8),   # far right
            (20, 10),  # far right farther
            (15, 12),  # sharp angle
            (30, 6),   # straight ahead offset
        ]

        hidden = None
        for fwd_dist, side_dist in offsets:
            hidden_loc = carla.Location(
                x = ego_loc.x + fwd.x * fwd_dist
                    + side_dist,
                y = ego_loc.y + fwd.y * fwd_dist
                    + side_dist,
                z = ego_loc.z
            )
            hidden_bp = bp_lib.filter(
                'vehicle.bmw.grandtourer')[0]
            hidden_transform = carla.Transform(
                hidden_loc,
                carla.Rotation(
                    yaw=ego_sp.rotation.yaw - 90)
            )
            hidden = world.try_spawn_actor(
                hidden_bp, hidden_transform)
            if hidden:
                actors.append(hidden)
                print(f"  Hidden vehicle spawned at "
                      f"offset ({fwd_dist}, {side_dist})")
                break

    return actors, ego

#  Spectator camera helper 

def position_spectator(world, ego, offset_back=8.0,
                        offset_up=4.0):
    """
    Move CARLA spectator camera behind and above ego
    for a good screenshot angle.
    """
    spectator   = world.get_spectator()
    ego_transform = ego.get_transform()
    fwd = ego_transform.get_forward_vector()

    cam_loc = carla.Location(
        x = ego_transform.location.x - fwd.x * offset_back,
        y = ego_transform.location.y - fwd.y * offset_back,
        z = ego_transform.location.z + offset_up
    )
    cam_rot = carla.Rotation(
        pitch = -15,
        yaw   = ego_transform.rotation.yaw,
        roll  = 0
    )
    spectator.set_transform(
        carla.Transform(cam_loc, cam_rot)
    )


#  Main 

def main():
    scenarios = {
        "1": ("pedestrian_curb",       spawn_pedestrian_curb),
        "2": ("merge_hesitation",      spawn_merge_hesitation),
        "3": ("occluded_intersection", spawn_occluded_intersection),
    }

    print("\n" + "="*55)
    print("  CARLA Scenario Visualizer — Paper Screenshots")
    print("="*55)
    print("\n  Scenarios:")
    print("  1 → pedestrian_curb")
    print("  2 → merge_hesitation")
    print("  3 → occluded_intersection")
    print("  q → quit\n")

    client, world = connect()

    # Set weather to clear for clean screenshots
    world.set_weather(carla.WeatherParameters.ClearNoon)

    current_actors = []

    while True:
        choice = input(
            "\n  Enter scenario (1/2/3) or q to quit: "
        ).strip().lower()

        if choice == 'q':
            cleanup(current_actors)
            break

        if choice not in scenarios:
            print("  Invalid choice. Enter 1, 2, or 3.")
            continue

        # Clean up previous scenario
        if current_actors:
            cleanup(current_actors)
            current_actors = []
            time.sleep(0.5)

        scenario_name, spawn_fn = scenarios[choice]
        print(f"\n  Spawning: {scenario_name}")
        print("  " + "-"*40)

        actors, ego = spawn_fn(world)
        current_actors = actors

        if ego:
            # Position camera behind ego
            time.sleep(0.5)
            position_spectator(world, ego)
            print(f"\n  ✓ Scene ready for screenshot")
            print(f"  Camera positioned behind ego vehicle")
            print(f"  ")
            print(f"  → In CARLA window:")
            print(f"    Use mouse to adjust view")
            print(f"    Press Windows+Shift+S to screenshot")
            print(f"    Save as: {scenario_name}.png")
            print(f"    Then upload to paper_figures/ in Overleaf")
        else:
            print("  [!] Ego failed to spawn.")
            print("      Try a different spawn point index.")


if __name__ == "__main__":
    main()