"""Plot loss curves from three saved runs for comparison."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

files = [
    "saved_runs/2026-03-29_19-03-13.json",
    "saved_runs/2026-03-29_19-10-14.json",
    "saved_runs/2026-03-29_19-23-37.json",
    "saved_runs/2026-03-30_20-09-40.json"
]

fig, ax = plt.subplots(figsize=(10, 6))

for path in files:
    with open(path) as f:
        data = json.load(f)
    steps = data["steps"]
    losses = data["losses"]
    label = (
        f"{data['name']}  "
    )
    ax.plot(steps, losses, linewidth=1.5, label=label)

ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Loss Comparison — Bunny, 50 Ellipsoids, n=64")
ax.legend()
ax.grid(True, which="major", linewidth=0.6)
ax.grid(True, which="minor", linewidth=0.2, alpha=0.5)
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=150)
plt.show()