"""
backend/ml/train_mlp.py

Context-adaptive weight MLP.
Learns to predict ambiguity score A(t) from 8 scene features.

Input:   backend/data/features.npy   (53703, 8)
Output:  backend/ml/model_weights.npy
         backend/ml/training_results.csv
         backend/ml/training_plot.png

Run:
  python backend/ml/train_mlp.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

DATA_DIR  = Path(__file__).parent.parent / "data"
ML_DIR    = Path(__file__).parent
ML_DIR.mkdir(exist_ok=True)


#  Data augmentation ─

def augment_features(X: np.ndarray, y: np.ndarray,
                     factor: int = 3) -> tuple:
    """
    Trajectory-space augmentation on extracted features.
    
    Transforms:
      1. Gaussian noise injection
      2. Speed scaling (slow/fast)
      3. Feature dropout (simulate missing data)
    
    Returns augmented (X, y) with factor x more samples.
    """
    X_aug = [X]
    y_aug = [y]

    rng = np.random.default_rng(42)

    for _ in range(factor):
        # Transform 1: Gaussian noise
        noise  = rng.normal(0, 0.02, X.shape)
        X_noisy = np.clip(X + noise, 0.0, 1.0)
        y_noisy = np.clip(y + rng.normal(0, 0.01, y.shape), 0.0, 1.0)
        X_aug.append(X_noisy)
        y_aug.append(y_noisy)

        # Transform 2: Speed scaling (multiply speed features by 0.5 or 1.5)
        X_scaled      = X.copy()
        scale         = rng.choice([0.5, 1.5])
        X_scaled[:, 1] = np.clip(X_scaled[:, 1] * scale, 0.0, 1.0)  # mean_speed
        X_scaled[:, 2] = np.clip(X_scaled[:, 2] * scale, 0.0, 1.0)  # speed_var
        X_aug.append(X_scaled)
        y_aug.append(y)

        # Transform 3: Random feature dropout (simulate sensor noise)
        X_drop         = X.copy()
        drop_mask      = rng.random(X.shape) < 0.05
        X_drop[drop_mask] = 0.0
        X_aug.append(X_drop)
        y_aug.append(y)

    X_out = np.vstack(X_aug)
    y_out = np.concatenate(y_aug)
    print(f"[Augment] {len(X)} → {len(X_out)} samples ({factor*3+1}x)")
    return X_out, y_out


#  MLP implementation (pure numpy - no pytorch/tensorflow needed) 

class MLP:
    """
    Lightweight 2-layer MLP in pure numpy.
    No external ML framework needed.
    Fully interpretable, fast to train.
    """

    def __init__(self, input_size=8, hidden_size=64,
                 output_size=1, dropout=0.1):
        self.hidden_size = hidden_size
        self.dropout     = dropout

        # Xavier initialisation
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b3 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X, training=False):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # Dropout
        if training and self.dropout > 0:
            self.mask1 = (np.random.rand(*self.a1.shape) > self.dropout)
            self.a1   *= self.mask1
        else:
            self.mask1 = np.ones_like(self.a1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        if training and self.dropout > 0:
            self.mask2 = (np.random.rand(*self.a2.shape) > self.dropout)
            self.a2   *= self.mask2
        else:
            self.mask2 = np.ones_like(self.a2)

        self.z3 = self.a2 @ self.W3 + self.b3
        out     = np.clip(self.z3, 0.0, 1.0)  # output in [0,1]
        return out

    def backward(self, X, y, lr):
        n   = X.shape[0]
        out = self.forward(X, training=True)

        # MSE loss gradient
        d_out = 2 * (out - y.reshape(-1, 1)) / n

        # Layer 3
        dW3 = self.a2.T @ d_out
        db3 = d_out.sum(axis=0)
        d2  = d_out @ self.W3.T * self.relu_grad(self.z2) * self.mask2

        # Layer 2
        dW2 = self.a1.T @ d2
        db2 = d2.sum(axis=0)
        d1  = d2 @ self.W2.T * self.relu_grad(self.z1) * self.mask1

        # Layer 1
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0)

        # Update
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        return float(np.mean((out - y.reshape(-1,1))**2))

    def predict(self, X):
        return self.forward(X, training=False).flatten()

    def save(self, path: Path):
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3,
                 hidden_size=np.array([self.hidden_size]),
                 dropout=np.array([self.dropout]))
        print(f"[Model] Saved to {path}")

    @classmethod
    def load(cls, path: Path):
        data   = np.load(path)
        hidden = int(data["hidden_size"][0])
        drop   = float(data["dropout"][0])
        model  = cls(hidden_size=hidden, dropout=drop)
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        return model


#  Training loop 

def train(model, X_train, y_train, X_val, y_val,
          lr=1e-3, epochs=50, batch_size=256):

    train_losses = []
    val_losses   = []
    best_val     = float("inf")
    best_weights = None

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Mini-batch SGD
        batch_losses = []
        for start in range(0, len(X_train), batch_size):
            Xb = X_train[start:start+batch_size]
            yb = y_train[start:start+batch_size]
            loss = model.backward(Xb, yb, lr)
            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses))
        val_pred   = model.predict(X_val)
        val_loss   = float(np.mean((val_pred - y_val)**2))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save best
        if val_loss < best_val:
            best_val     = val_loss
            best_weights = (model.W1.copy(), model.b1.copy(),
                            model.W2.copy(), model.b2.copy(),
                            model.W3.copy(), model.b3.copy())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: "
                  f"train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}")

    # Restore best weights
    if best_weights:
        (model.W1, model.b1, model.W2,
         model.b2, model.W3, model.b3) = best_weights

    return train_losses, val_losses


#  Hyperparameter sweep 

def hyperparameter_sweep(X_train, y_train, X_val, y_val):
    """Manual grid search over key hyperparameters."""

    param_grid = {
        "lr":          [1e-3, 5e-4, 1e-4],
        "hidden_size": [32, 64, 128],
        "dropout":     [0.1, 0.2, 0.3],
    }

    results = []
    best_val  = float("inf")
    best_cfg  = None

    combos = list(product(
        param_grid["lr"],
        param_grid["hidden_size"],
        param_grid["dropout"]
    ))

    print(f"\n[Sweep] {len(combos)} combinations\n")

    for i, (lr, hidden, dropout) in enumerate(combos):
        np.random.seed(42)
        model = MLP(hidden_size=hidden, dropout=dropout)
        _, val_losses = train(model, X_train.copy(), y_train.copy(),
                               X_val, y_val,
                               lr=lr, epochs=30, batch_size=256)

        final_val = val_losses[-1]
        results.append({
            "lr":          lr,
            "hidden_size": hidden,
            "dropout":     dropout,
            "val_loss":    round(final_val, 6),
        })

        print(f"[{i+1}/{len(combos)}] "
              f"lr={lr} hidden={hidden} dropout={dropout} "
              f"→ val_loss={final_val:.5f}")

        if final_val < best_val:
            best_val = final_val
            best_cfg = {"lr": lr, "hidden_size": hidden, "dropout": dropout}

    return results, best_cfg


#  Main─

def main():
    print("Loading features...")
    X = np.load(DATA_DIR / "features.npy")
    y = np.load(DATA_DIR / "labels.npy")
    print(f"Loaded: X={X.shape}, y={y.shape}")

    # Normalise features
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X      = (X - X_mean) / X_std
    np.save(ML_DIR / "feature_stats.npy",
            np.array([X_mean, X_std]))

    # Augment
    X_aug, y_aug = augment_features(X, y, factor=3)

    # Train/val/test split (70/15/15)
    n       = len(X_aug)
    idx     = np.random.default_rng(42).permutation(n)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    X_train = X_aug[idx[:train_end]]
    y_train = y_aug[idx[:train_end]]
    X_val   = X_aug[idx[train_end:val_end]]
    y_val   = y_aug[idx[train_end:val_end]]
    X_test  = X_aug[idx[val_end:]]
    y_test  = y_aug[idx[val_end:]]

    print(f"\nSplit: train={len(X_train)} "
          f"val={len(X_val)} test={len(X_test)}")

    #  Hyperparameter sweep 
    print("\n" + "="*50)
    print("  HYPERPARAMETER SWEEP")
    print("="*50)

    sweep_results, best_cfg = hyperparameter_sweep(
        X_train, y_train, X_val, y_val
    )

    sweep_df = pd.DataFrame(sweep_results).sort_values("val_loss")
    sweep_df.to_csv(ML_DIR / "sweep_results.csv", index=False)

    print(f"\nBest config: {best_cfg}")
    print(f"Best val_loss: {sweep_df.iloc[0]['val_loss']:.5f}")

    #  Final training with best config 
    print("\n" + "="*50)
    print("  FINAL TRAINING")
    print("="*50)

    np.random.seed(42)
    best_model = MLP(
        hidden_size=best_cfg["hidden_size"],
        dropout=best_cfg["dropout"]
    )
    train_losses, val_losses = train(
        best_model, X_train, y_train, X_val, y_val,
        lr=best_cfg["lr"], epochs=100, batch_size=256
    )

    #  Test evaluation ─
    test_pred = best_model.predict(X_test)
    test_mse  = float(np.mean((test_pred - y_test)**2))
    test_mae  = float(np.mean(np.abs(test_pred - y_test)))
    test_r2   = float(1 - np.sum((test_pred - y_test)**2) /
                          np.sum((y_test - y_test.mean())**2))

    print(f"\nTest Results:")
    print(f"  MSE:  {test_mse:.5f}")
    print(f"  MAE:  {test_mae:.5f}")
    print(f"  R²:   {test_r2:.4f}")

    #  Save model 
    best_model.save(ML_DIR / "model_weights.npz")

    #  Save training results
    results_df = pd.DataFrame({
        "epoch":      range(len(train_losses)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    })
    results_df.to_csv(ML_DIR / "training_results.csv", index=False)

    #  Plot─
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              facecolor="#0f172a")

    # Loss curves
    ax1 = axes[0]
    ax1.set_facecolor("#020617")
    ax1.plot(train_losses, color="#3b82f6", label="Train", linewidth=2)
    ax1.plot(val_losses,   color="#22c55e", label="Val",   linewidth=2)
    ax1.set_title("Training Loss", color="#e2e8f0", fontsize=11)
    ax1.set_xlabel("Epoch", color="#64748b")
    ax1.set_ylabel("MSE Loss", color="#64748b")
    ax1.tick_params(colors="#475569")
    ax1.legend(facecolor="#0f172a", labelcolor="#94a3b8",
                edgecolor="#1e293b")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1e293b")

    # Prediction vs ground truth
    ax2 = axes[1]
    ax2.set_facecolor("#020617")
    sample_idx = np.random.choice(len(X_test), 500, replace=False)
    ax2.scatter(y_test[sample_idx], test_pred[sample_idx],
                alpha=0.3, color="#818cf8", s=10)
    ax2.plot([0, 1], [0, 1], color="#ef4444",
             linestyle="--", linewidth=1, label="Perfect")
    ax2.set_title(f"Predicted vs Actual (R²={test_r2:.3f})",
                   color="#e2e8f0", fontsize=11)
    ax2.set_xlabel("Actual A(t)", color="#64748b")
    ax2.set_ylabel("Predicted A(t)", color="#64748b")
    ax2.tick_params(colors="#475569")
    ax2.legend(facecolor="#0f172a", labelcolor="#94a3b8",
                edgecolor="#1e293b")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")

    plt.tight_layout()
    plt.savefig(ML_DIR / "training_plot.png", dpi=150,
                bbox_inches="tight", facecolor="#0f172a")
    print(f"\nPlot saved → backend/ml/training_plot.png")

    #  Summary
    print("\n" + "="*50)
    print("  SUMMARY")
    print("="*50)
    print(f"  Dataset:     {len(X_aug):,} samples (after augmentation)")
    print(f"  Best config: lr={best_cfg['lr']} "
          f"hidden={best_cfg['hidden_size']} "
          f"dropout={best_cfg['dropout']}")
    print(f"  Test MSE:    {test_mse:.5f}")
    print(f"  Test MAE:    {test_mae:.5f}")
    print(f"  Test R²:     {test_r2:.4f}")
    print(f"\n  Model saved → backend/ml/model_weights.npz")
    print(f"  Sweep saved → backend/ml/sweep_results.csv")
    print(f"  Results saved → backend/ml/training_results.csv")


if __name__ == "__main__":
    main()