"""Generate horizontal bar plot of all experiments for Task 4 report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "Random\n(Hamza)",
    "MF-BPR\n(Hamza)",
    "Two-Tower basic\n(Hamza)",
    "Popularity\n(Hamza)",
    "Two-Tower enriched\n(OURS)",
    "TT enriched + LightGBM\n(OURS)",
]

recall_20 = [0.12, 0.002, 0.002, 16.99, 8.03, None]
ndcg_10  = [0.03, 0.0008, 0.0003, 5.46, 1.90, 30.11]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Color per model
colors = ["#95a5a6", "#95a5a6", "#95a5a6", "#e67e22", "#3498db", "#2ecc71"]
labels_colors = ["Hamza/Tan", "Hamza/Tan", "Hamza/Tan", "Hamza/Linh", "Malo", "Malo"]

# --- Recall@20 ---
y = np.arange(len(models))
valid_recall = [(r if r is not None else 0) for r in recall_20]
bars1 = ax1.barh(y, valid_recall, color=colors, edgecolor="white", height=0.6)
ax1.set_yticks(y)
ax1.set_yticklabels(models, fontsize=9)
ax1.set_xlabel("Recall@20 (%)", fontsize=11)
ax1.set_title("Retrieval Quality — Recall@20", fontsize=13, fontweight="bold")
ax1.invert_yaxis()

# Annotate N/A for LightGBM
for i, (r, color) in enumerate(zip(recall_20, colors)):
    if r is None:
        ax1.text(0.5, i, "N/A (ranking only)", ha="center", va="center", fontsize=8, color="#7f8c8d")
    else:
        ax1.text(r + 0.3, i, f"{r:.2f}%", va="center", fontsize=9, fontweight="bold")

ax1.set_xlim(0, 22)

# --- NDCG@10 ---
valid_ndcg = [(n if n is not None else 0) for n in ndcg_10]
bars2 = ax2.barh(y, valid_ndcg, color=colors, edgecolor="white", height=0.6)
ax2.set_yticks(y)
ax2.set_yticklabels(models, fontsize=9)
ax2.set_xlabel("NDCG@10 (%)", fontsize=11)
ax2.set_title("Ranking Quality — NDCG@10", fontsize=13, fontweight="bold")
ax2.invert_yaxis()

for i, (n, color) in enumerate(zip(ndcg_10, colors)):
    if n is not None:
        x_pos = n + 0.6
        ax2.text(x_pos, i, f"{n:.2f}%", va="center", fontsize=9, fontweight="bold",
                 color="#2c3e50" if n < 20 else "white",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.9) if n >= 20 else None)

ax2.set_xlim(0, 40)
# Break axis to show small values
ax2_inset = ax2.inset_axes([0.45, 0.55, 0.35, 0.35])
ax2_inset.barh(y[:3], valid_ndcg[:3], color=colors[:3], height=0.6)
ax2_inset.set_yticks(y[:3])
ax2_inset.set_yticklabels([m.replace("\n", " ") for m in models[:3]], fontsize=6)
ax2_inset.set_title("Zoom (bottom models)", fontsize=7)
ax2_inset.invert_yaxis()

# Labels inside bars for zero values
for i, n in enumerate(valid_ndcg[:3]):
    ax2_inset.text(n + 0.0003 if n > 0 else 0.0001, i, f"{ndcg_10[i]:.4f}%",
                   va="center", fontsize=6)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#95a5a6", label="Hamza / Kim Tan"),
    Patch(facecolor="#e67e22", label="Hamza / Linh Long"),
    Patch(facecolor="#3498db", label="Malo (retrieval)"),
    Patch(facecolor="#2ecc71", label="Malo (retrieval + ranking)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("/home/malo/Documents/ais3/resystem/steam-recsys-pipeline/reports/figures/all_experiments.png",
            dpi=150, bbox_inches="tight")
print("✓ Saved → reports/figures/all_experiments.png")
