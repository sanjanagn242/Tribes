"""Plots a run's metrics.csv and eval.csv to <run dir>/metrics.png.

    python plot.py runs/actor_critic_gae/<run_name> [more runs ...]

With several run directories the curves are overlaid (e.g. lam=0 vs lam=0.95) and the figure is
written to the first run's directory as compare.png.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [
    ("Actor loss", [("actor loss", "actor_loss")], "metrics"),
    ("Critic loss (0.5 * MSE to return)", [("critic loss", "critic_loss")], "metrics"),
    ("Policy entropy (nats)", [("entropy", "entropy")], "metrics"),
    ("Mean |TD error|", [("|TD error|", "td_error_abs")], "metrics"),
    ("Critic value vs. return", [("mean U(s)", "value_mean"), ("mean return", "return_mean")], "metrics"),
    ("Training games (sampled policy)", [("win", "win_rate"), ("loss", "loss_rate")], "metrics"),
    ("Greedy policy vs random", [("win", "win_rate"), ("loss", "loss_rate")], "eval"),
    ("Agent decisions per game", [("decisions", "moves_per_game")], "metrics"),
]


def read(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in (rows[0].keys() if rows else [])}


def plot(runs):
    data = {r: {"metrics": read(os.path.join(r, "metrics.csv")), "eval": read(os.path.join(r, "eval.csv"))}
            for r in runs}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, (title, series, src) in zip(axes.flat, PANELS):
        for run in runs:
            d = data[run][src]
            if "iter" not in d:
                continue
            for label, key in series:
                name = label if len(runs) == 1 else f"{os.path.basename(run.rstrip('/'))}: {label}"
                ax.plot(d["iter"], d[key], label=name)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(runs[0], "metrics.png" if len(runs) == 1 else "compare.png")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("wrote", plot(sys.argv[1:]))
