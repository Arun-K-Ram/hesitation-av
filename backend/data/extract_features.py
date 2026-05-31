"""
backend/data/extract_features.py

Extracts scene-level features from OnSiteVRU dataset
for training the context-adaptive weight MLP.

Input:  train_data_x.npy  (53703, 4, 5, 10)
Output: features.npy      (N, 8) scene feature vectors
        labels.npy        (N,)   ambiguity scores (computed)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Feature indices
X      = 0
Y      = 1
SPD_X  = 2
SPD_Y  = 3
ACC_X  = 4
ACC_Y  = 5
JRK_X  = 6
JRK_Y  = 7
ANGLE  = 8
VTYPE  = 9


def compute_speed(agent):
    """agent: (4, 10) - 4 timesteps, 10 features"""
    vx = agent[:, SPD_X]
    vy = agent[:, SPD_Y]
    return np.sqrt(vx**2 + vy**2)


def compute_motion_entropy(agent):
    """Shannon entropy over velocity direction histogram."""
    vx = agent[:, SPD_X]
    vy = agent[:, SPD_Y]
    angles = np.arctan2(vy, vx)
    
    # Bin into 8 heading bins
    bins   = np.linspace(-np.pi, np.pi, 9)
    counts = np.histogram(angles, bins=bins)[0].astype(float)
    
    if counts.sum() == 0:
        return 0.0
    
    probs   = counts / counts.sum()
    probs   = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy / np.log2(8))  # normalise to [0,1]


def compute_jerk_magnitude(agent):
    """Mean jerk magnitude - feeds CorrectionSeverity."""
    jx = agent[:, JRK_X]
    jy = agent[:, JRK_Y]
    return float(np.nanmean(np.sqrt(jx**2 + jy**2)))


def extract_scene_features(scene):
    """
    scene: (4, 5, 10) - 4 timesteps, 5 agents, 10 features
    
    Returns 8 scene-level features:
      0: agent_count
      1: mean_speed
      2: speed_variance
      3: mean_motion_entropy
      4: mean_jerk
      5: scene_density
      6: mean_heading_change
      7: vehicle_type_diversity
    """
    # Find valid agents (not all zeros/nan)
    valid_agents = []
    for a in range(5):
        agent = scene[:, a, :]
        if np.isnan(agent).all() or (agent == 0).all():
            continue
        # Replace NaN with 0 for computation
        agent = np.nan_to_num(agent, nan=0.0)
        valid_agents.append(agent)

    if not valid_agents:
        return np.zeros(8)

    n_agents = len(valid_agents)

    # Feature 0: agent count (normalised by max 5)
    agent_count = n_agents / 5.0

    # Feature 1: mean speed across agents
    speeds = [compute_speed(a).mean() for a in valid_agents]
    mean_speed = float(np.mean(speeds))

    # Feature 2: speed variance (unpredictability)
    speed_var = float(np.var(speeds)) if len(speeds) > 1 else 0.0

    # Feature 3: mean motion entropy (behavioral ambiguity proxy)
    entropies = [compute_motion_entropy(a) for a in valid_agents]
    mean_entropy = float(np.mean(entropies))

    # Feature 4: mean jerk magnitude
    jerks = [compute_jerk_magnitude(a) for a in valid_agents]
    mean_jerk = float(np.nanmean(jerks))

    # Feature 5: scene density
    # Spread of agent positions in last timestep
    positions = np.array([a[-1, :2] for a in valid_agents])
    if len(positions) > 1:
        spread = float(np.std(positions))
        scene_density = 1.0 / (1.0 + spread)
    else:
        scene_density = 1.0

    # Feature 6: mean heading change over observation
    heading_changes = []
    for a in valid_agents:
        vx = a[:, SPD_X]
        vy = a[:, SPD_Y]
        angles = np.arctan2(vy, vx)
        if len(angles) > 1:
            changes = np.abs(np.diff(angles))
            changes = np.minimum(changes, 2*np.pi - changes)
            heading_changes.append(changes.mean())
    mean_heading_change = float(np.mean(heading_changes)) if heading_changes else 0.0

    # Feature 7: vehicle type diversity
    vtypes = [a[0, VTYPE] for a in valid_agents]
    unique_types = len(set(vtypes))
    type_diversity = unique_types / 3.0  # normalised by max types

    features = np.array([
        agent_count,
        min(mean_speed / 10.0, 1.0),      # normalise speed
        min(speed_var / 5.0, 1.0),         # normalise variance
        mean_entropy,
        min(mean_jerk / 5.0, 1.0),         # normalise jerk
        scene_density,
        min(mean_heading_change / np.pi, 1.0),
        type_diversity,
    ], dtype=float)

    return features


def compute_ambiguity_label(scene):
    """
    Compute ground truth ambiguity score for a scene.
    This becomes the training target for our MLP.
    
    A(t) = α·Aₚ + β·A_b
    
    We proxy:
      Aₚ = detection uncertainty = speed variance (higher = harder to detect)
      A_b = behavioral uncertainty = motion entropy
    """
    features = extract_scene_features(scene)
    
    alpha = 0.45
    beta  = 0.55
    
    Ap = features[2]  # speed variance → perceptual proxy
    Ab = features[3]  # motion entropy → behavioral proxy
    
    A = alpha * Ap + beta * Ab
    return float(np.clip(A, 0.0, 1.0))


def main():
    print("Loading dataset...")
    train_x = np.load(DATA_DIR / "train_data_x.npy", allow_pickle=True)
    
    n_scenes  = train_x.shape[0]
    print(f"Processing {n_scenes} scenes...")

    features = np.zeros((n_scenes, 8), dtype=float)
    labels   = np.zeros(n_scenes, dtype=float)

    for i in range(n_scenes):
        if i % 5000 == 0:
            print(f"  [{i}/{n_scenes}] {100*i/n_scenes:.1f}%")

        scene         = train_x[i]  # (4, 5, 10)
        features[i]   = extract_scene_features(scene)
        labels[i]     = compute_ambiguity_label(scene)

    print(f"Done. Feature matrix: {features.shape}")
    print(f"Label distribution: min={labels.min():.3f} "
          f"max={labels.max():.3f} mean={labels.mean():.3f}")

    # Save
    np.save(DATA_DIR / "features.npy", features)
    np.save(DATA_DIR / "labels.npy",   labels)
    print(f"Saved features.npy and labels.npy to {DATA_DIR}")

    # Quick sanity check
    print("\nSample features (first 3 scenes):")
    cols = ["agent_count", "mean_speed", "speed_var", "entropy",
            "jerk", "density", "heading_change", "type_diversity"]
    for i in range(3):
        print(f"  Scene {i}: " + 
              " ".join(f"{c}={features[i,j]:.3f}" 
                      for j, c in enumerate(cols)))
        print(f"           ambiguity_label={labels[i]:.3f}")


if __name__ == "__main__":
    main()