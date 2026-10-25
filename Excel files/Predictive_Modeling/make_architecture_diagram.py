"""Draws architecture_diagram.png (12 -> 16 -> 8 -> 1) used in ANN_Architecture.docx."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

layers = [(12, "Input layer\n12 features", "#9aa0a6"),
          (16, "Hidden layer 1\n16 neurons, ReLU", "#1f77b4"),
          (8, "Hidden layer 2\n8 neurons, ReLU", "#1f77b4"),
          (1, "Output layer\n1 neuron, Sigmoid", "#d62728")]
xs = [0, 1.6, 3.2, 4.8]
H = 8.0
pos = [np.linspace(-H / 2, H / 2, n) if n > 1 else np.array([0.0]) for n, _, _ in layers]

fig, ax = plt.subplots(figsize=(9, 5.6))
for i in range(len(layers) - 1):
    for y1 in pos[i]:
        for y2 in pos[i + 1]:
            ax.plot([xs[i], xs[i + 1]], [y1, y2], c="#c8ccd0", lw=0.35, zorder=1)
for (n, label, col), x, ys in zip(layers, xs, pos):
    ax.scatter([x] * n, ys, s=170, c=col, edgecolor="white", zorder=2)
    ax.text(x, -H / 2 - 0.95, label, ha="center", va="top", fontsize=10)
ax.text(4.8, 0.65, "P(churn)", ha="center", fontsize=9, color="#d62728")
ax.set_xlim(-0.6, 5.4); ax.set_ylim(-H / 2 - 2.1, H / 2 + 0.6); ax.axis("off")
ax.set_title("ANN architecture: 12 -> 16 -> 8 -> 1  (353 trainable parameters)", fontsize=12)
plt.tight_layout(); plt.savefig("architecture_diagram.png", dpi=170); plt.close()
