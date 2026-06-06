import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PAPER_FIGURES = Path("paper_figures")
PAPER_FIGURES.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n   = 240
t   = np.linspace(0, 8, n)
dt  = t[1] - t[0]

rise        = np.clip((t - 0.8) / 1.0, 0, 1)
oscillation = 0.10 * np.sin(2*np.pi*t*2.1) * \
              np.clip((t - 0.8) / 0.5, 0, 1)
decay       = np.clip(1.0 - (t - 2.8) / 0.8, 0, 1)
A = np.clip(0.58*rise*decay + oscillation +
            rng.normal(0, 0.018, n), 0, 1)

risk = np.clip(
    0.62 * np.exp(-((t-1.5)**2)/0.35) +
    0.08 * np.clip(1.0-(t-3.0)/1.5, 0, 1) +
    rng.normal(0, 0.012, n), 0, 1)

tau_l    = 0.35
t_greedy = t[(t > 1.2) & (A < tau_l)][0]
t_fixed  = 2.0
t_hesit  = t[(t > 3.0) & (A < tau_l * 0.8)][0] \
           if any((t > 3.0) & (A < tau_l * 0.8)) \
           else 3.4

gi = np.searchsorted(t, t_greedy)
fi = np.searchsorted(t, t_fixed)
hi = np.searchsorted(t, t_hesit)

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#333333",
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "text.color":        "#222222",
    "grid.color":        "#eeeeee",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.size":         10,
})

fig, ax = plt.subplots(figsize=(12, 5),
                        facecolor="white")
ax.set_facecolor("white")

ax.plot(t, A, color="#E24B4A", linewidth=2.0,
         label=r"Ambiguity $\mathcal{A}(t)$",
         zorder=3)
ax.plot(t, risk, color="#888780",
         linewidth=1.5, linestyle="--",
         label=r"Risk $\mathcal{R}(t)$",
         zorder=3)
ax.axhline(y=tau_l, color="#aaaaaa",
            linestyle=":", linewidth=1.2,
            label=r"Threshold $\tau_l = 0.35$",
            zorder=2)

# Shaded regions
ax.axvspan(0, t_greedy, alpha=0.04,
            color="#BA7517")
ax.axvspan(t_greedy, t_fixed, alpha=0.04,
            color="#639922")
ax.axvspan(t_fixed, t_hesit, alpha=0.04,
            color="#185FA5")

# Commit points
ax.scatter([t[gi]], [A[gi]], s=120,
            color="#BA7517", zorder=5,
            edgecolors="white", linewidths=1.5)
ax.scatter([t[fi]], [A[fi]], s=120,
            color="#639922", zorder=5,
            edgecolors="white", linewidths=1.5)
ax.scatter([t[hi]], [A[hi]], s=120,
            color="#185FA5", zorder=5,
            edgecolors="white", linewidths=1.5)

# Vertical drop lines
for idx, color in [(gi, "#BA7517"),
                   (fi, "#639922"),
                   (hi, "#185FA5")]:
    ax.plot([t[idx], t[idx]], [0, A[idx]],
             color=color, linewidth=1.0,
             linestyle="--", alpha=0.5,
             zorder=2)

# Annotations
ax.annotate(
    f"Greedy\nt={t[gi]:.1f}s\nHQM=0.600",
    xy=(t[gi], A[gi]),
    xytext=(t[gi]-0.1, A[gi]+0.18),
    fontsize=8.5, color="#854F0B",
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3",
              fc="#FAEEDA", ec="#BA7517",
              lw=0.8),
    arrowprops=dict(arrowstyle="->",
                    color="#BA7517",
                    lw=0.8))

ax.annotate(
    f"Fixed delay\nt={t_fixed:.1f}s\nHQM=0.436",
    xy=(t[fi], A[fi]),
    xytext=(t[fi]+0.05, A[fi]+0.20),
    fontsize=8.5, color="#3B6D11",
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3",
              fc="#EAF3DE", ec="#639922",
              lw=0.8),
    arrowprops=dict(arrowstyle="->",
                    color="#639922",
                    lw=0.8))

ax.annotate(
    f"Hesitation\nt={t[hi]:.1f}s\nHQM=0.747",
    xy=(t[hi], A[hi]),
    xytext=(t[hi]+0.15, A[hi]+0.22),
    fontsize=8.5, color="#0C447C",
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3",
              fc="#E6F1FB", ec="#185FA5",
              lw=0.8),
    arrowprops=dict(arrowstyle="->",
                    color="#185FA5",
                    lw=0.8))

ax.set_xlabel("Time (s)", fontsize=10,
               color="#333333")
ax.set_ylabel(r"$\mathcal{A}(t)$ / "
               r"$\mathcal{R}(t)$",
               fontsize=10, color="#333333")
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.5)
ax.legend(loc="upper right", fontsize=9,
           facecolor="white",
           edgecolor="#cccccc",
           framealpha=0.9)

for spine in ax.spines.values():
    spine.set_edgecolor("#cccccc")

plt.tight_layout()
out = PAPER_FIGURES / "visual_summary.png"
plt.savefig(out, dpi=150,
             bbox_inches="tight",
             facecolor="white")
print(f"Saved  {out}")