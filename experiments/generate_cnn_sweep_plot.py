# Save as generate_cnn_sweep_plot.py and run it

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

df = pd.read_csv("backend/ml/cnn_sweep_results.csv")

arch_colors = {
    "efficientnet_b2": "#3182ce",
    "efficientnet_b0": "#63b3ed",
    "mobilenet_v3":    "#38a169",
    "mobilenet_v2":    "#68d391",
    "convnext_tiny":   "#d69e2e",
    "resnet18":        "#dd6b20",
}
arch_labels = {
    "efficientnet_b2": "EfficientNetB2",
    "efficientnet_b0": "EfficientNetB0",
    "mobilenet_v3":    "MobileNetV3",
    "mobilenet_v2":    "MobileNetV2",
    "convnext_tiny":   "ConvNeXt-Tiny",
    "resnet18":        "ResNet18",
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

fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                          facecolor="white")
fig.suptitle(
    "CNN Architecture Sweep - 36 Configurations "
    "on HesitAV-1564",
    fontsize=12, color="#111111",
    fontweight="bold")

#  Plot 1: Val Acc by architecture (scatter)
ax = axes[0]
ax.set_facecolor("white")

for arch in df["arch"].unique():
    sdf = df[df.arch == arch]
    ax.scatter(
        range(len(sdf)),
        sdf["val_acc"],
        color=arch_colors.get(arch, "#718096"),
        label=arch_labels.get(arch, arch),
        s=60, alpha=0.85, zorder=3)

ax.axhline(y=99.81, color="#e53e3e",
            linestyle="--", linewidth=1,
            label="Best: 99.81%")
ax.set_title("Val Accuracy per Config",
              color="#222222", fontsize=9)
ax.set_xlabel("Configuration index",
               color="#444444", fontsize=8)
ax.set_ylabel("Validation Accuracy (%)",
               color="#444444", fontsize=8)
ax.set_ylim(98.5, 100.1)
ax.tick_params(colors="#444444", labelsize=7)
ax.grid(True, alpha=0.5)
ax.legend(fontsize=6, facecolor="white",
           edgecolor="#cccccc",
           loc="lower right")
for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")

#  Plot 2: Mean Val Acc by architecture (bar)
ax2 = axes[1]
ax2.set_facecolor("white")

arch_means = df.groupby("arch")["val_acc"]\
               .mean().sort_values(ascending=False)
arch_stds  = df.groupby("arch")["val_acc"].std()

bars = ax2.bar(
    [arch_labels.get(a, a)
     for a in arch_means.index],
    arch_means.values,
    yerr=[arch_stds.get(a, 0)
          for a in arch_means.index],
    color=[arch_colors.get(a, "#718096")
           for a in arch_means.index],
    capsize=4, ecolor="#888888",
    edgecolor="white", alpha=0.85)

ax2.set_title("Mean Val Accuracy by Architecture",
               color="#222222", fontsize=9)
ax2.set_ylabel("Mean Val Accuracy (%)",
                color="#444444", fontsize=8)
ax2.set_ylim(98.0, 100.2)
ax2.tick_params(colors="#444444", labelsize=7,
                 axis="x", rotation=20)
ax2.tick_params(colors="#444444", labelsize=7,
                 axis="y")
ax2.grid(True, axis="y", alpha=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor("#cccccc")

for bar, val in zip(bars, arch_means.values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.05,
        f"{val:.2f}%",
        ha="center", va="bottom",
        color="#111111", fontsize=7,
        fontweight="bold")

#  Plot 3: Val Acc by lr and dropout (grouped)
ax3 = axes[2]
ax3.set_facecolor("white")

lr_vals      = sorted(df["lr"].unique())
dropout_vals = sorted(df["dropout"].unique())
x            = np.arange(len(lr_vals))
w            = 0.25
dropout_colors = {
    0.1: "#3182ce",
    0.2: "#38a169",
    0.3: "#dd6b20",
}

for i, dropout in enumerate(dropout_vals):
    means = [
        df[(df.lr == lr) &
           (df.dropout == dropout)]["val_acc"].mean()
        for lr in lr_vals
    ]
    ax3.bar(x + i * w, means, w,
             label=f"dropout={dropout}",
             color=dropout_colors.get(
                 dropout, "#718096"),
             alpha=0.85, edgecolor="white")

ax3.set_title("Val Accuracy by LR and Dropout",
               color="#222222", fontsize=9)
ax3.set_xlabel("Learning Rate",
                color="#444444", fontsize=8)
ax3.set_ylabel("Mean Val Accuracy (%)",
                color="#444444", fontsize=8)
ax3.set_xticks(x + w)
ax3.set_xticklabels(
    [str(lr) for lr in lr_vals],
    fontsize=7, color="#444444")
ax3.set_ylim(98.0, 100.2)
ax3.tick_params(colors="#444444", labelsize=7)
ax3.grid(True, axis="y", alpha=0.5)
ax3.legend(fontsize=7, facecolor="white",
            edgecolor="#cccccc")
for spine in ax3.spines.values():
    spine.set_edgecolor("#cccccc")

plt.tight_layout()
plt.savefig("paper_figures/cnn_sweep_plot.png",
            dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved  paper_figures/cnn_sweep_plot.png")
