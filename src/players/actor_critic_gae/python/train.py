"""Trains the GAE actor-critic on Tribes against a random opponent (tic-tac-toe's train.py, one variant).

    python train.py --classpath <compiled Tribes classes>:lib/json.jar --out <run dir> [--set lam=0.95 ...]

local/train.sh compiles Tribes and calls this for you. Writes to the run directory:
    config.yaml   the config used (+ training settings), so checkpoints can be reloaded
    metrics.csv   METRICS, averaged over every --log_every iterations
    eval.csv      greedy-policy win/draw/loss rates vs random, every --eval_every iterations
    ac_*.pkl      checkpoints; latest.pkl is the most recent (load with --init, or play it from Run)
    metrics.png   plot of the above, redrawn at every eval (skip with --no_plot)
"""
import argparse
import csv
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from agent import ActorCriticAgent, pad_batch
from networks import ActionScorer, MLP
from rollout import play_vs_random, to_batch
from tribes_env import REPO_DIR, TribesEnv

HERE = os.path.dirname(os.path.abspath(__file__))

METRICS = ["actor_loss", "critic_loss", "entropy", "value_mean", "return_mean",
           "td_error_abs", "win_rate", "draw_rate", "loss_rate", "moves_per_game"]


# ---------------------------------------------------------------- config

def load_config(path, overrides=()):
    """config.yaml with KEY=VALUE overrides applied (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        if key not in cfg:
            raise SystemExit(f"unknown config key {key!r}; keys: {', '.join(cfg)}")
        v = yaml.safe_load(value)
        if isinstance(v, str) and key not in ("game_mode",):
            try:
                v = float(v)            # YAML reads 5e-4 as a string
            except ValueError:
                pass
        if key == "tribes" and isinstance(v, str):
            v = [s.strip() for s in v.split(",")]
        cfg[key] = v
    return cfg


def build_agent(cfg, obs_size, act_size, seed=0):
    hidden = tuple(cfg["hidden"])
    return ActorCriticAgent(ActionScorer(hidden=hidden), MLP(1, hidden=hidden), obs_size, act_size,
                            gamma=cfg["gamma"], alpha=cfg["alpha"], alpha_phi=cfg["alpha_phi"],
                            beta=cfg.get("beta", 0.0), lam=cfg.get("lam", 0.95), seed=seed)


def save(agent, path, cfg, env):
    with open(path, "wb") as f:
        pickle.dump({"params": jax.device_get(agent.params), "config": cfg, "obs_size": env.obs_size,
                     "act_size": env.act_size, "board_size": env.board_size}, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- collect + diagnostics (__init__.py)

def collect(agent, env, rng, cfg):
    """Plays episodes_per_iter games vs random; returns (batch, pre-update stats)."""
    games = [play_vs_random(agent, env, rng) for _ in range(cfg["episodes_per_iter"])]
    batch = to_batch([t for g, _ in games for t in g])
    return batch, {**game_stats(games), **batch_stats(agent, batch)}


_stats_jit = {}


def batch_stats(agent, batch):
    """Diagnostics on a batch, using the parameters before this iteration's update.

        entropy       mean policy entropy over legal moves (falls as the policy sharpens)
        value_mean    mean critic U(s)
        return_mean   mean reward-to-go, the critic's target (U(s) should track it)
        td_error_abs  mean |r + gamma * U(s') * (1 - done) - U(s)|, the one-step TD error
    """
    if agent not in _stats_jit:
        def stats(params, b):
            logits = agent.actor.apply(params["theta"], b["obs"], b["acts"])
            logp = jax.nn.log_softmax(jnp.where(b["legal_mask"] > 0, logits, -1e9))
            p = jnp.exp(logp)
            entropy = -jnp.sum(jnp.where(b["legal_mask"] > 0, p * logp, 0.0), axis=-1)
            v = agent.critic.apply(params["phi"], b["obs"]).squeeze(-1)
            v_next = agent.critic.apply(params["phi"], b["next_obs"]).squeeze(-1)
            td = b["reward"] + agent.gamma * (1.0 - b["done"]) * v_next - v
            n = jnp.sum(b["mask"])
            mean = lambda x: jnp.sum(b["mask"] * x) / n
            return mean(entropy), mean(v), mean(jnp.abs(td))
        _stats_jit[agent] = jax.jit(stats)

    b = pad_batch({k: np.asarray(v) for k, v in batch.items()})
    entropy, v, td = _stats_jit[agent](agent.params, b)
    return {"entropy": float(entropy), "value_mean": float(v),
            "return_mean": float(np.mean(batch["rtg"])), "td_error_abs": float(td)}


def game_stats(games):
    """Outcome rates and length (agent decisions) of the training games (sampled, not greedy, policy)."""
    results = np.array([r for _, r in games])          # final reward = +1 / 0 / -1
    return {"win_rate": float(np.mean(results > 0)), "draw_rate": float(np.mean(results == 0)),
            "loss_rate": float(np.mean(results < 0)),
            "moves_per_game": float(np.mean([len(g) for g, _ in games]))}


def log_line(m):
    return (f"actor {m['actor_loss']:+.4f} critic {m['critic_loss']:.4f} "
            f"entropy {m['entropy']:.3f} |td| {m['td_error_abs']:.3f} | "
            f"train W/D/L {m['win_rate']:.2f}/{m['draw_rate']:.2f}/{m['loss_rate']:.2f} | "
            f"moves {m['moves_per_game']:.0f}")


def evaluate(agent, env, n, seed):
    """Greedy policy vs random, n games, seats alternating; the same games at every evaluation."""
    rng = np.random.default_rng(10_000 + seed)
    w = d = l = 0
    for i in range(n):
        _, r = play_vs_random(agent, env, rng, greedy=True, seat=i % 2, record=False)
        w, d, l = w + (r > 0), d + (r == 0), l + (r < 0)
    return w, d, l


# ---------------------------------------------------------------- main loop

class CsvLog:
    """Writes one row per call to a CSV file, starting a fresh file each run."""

    def __init__(self, path, columns):
        self.columns = ["iter"] + list(columns)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(self.columns)

    def write(self, it, row):
        self.w.writerow([it] + [row[c] for c in self.columns[1:]])
        self.f.flush()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classpath", required=True, help="compiled Tribes classes + lib/json.jar")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                    help="override config.yaml keys (may be repeated)")
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights (fresh Adam state)")
    ap.add_argument("--java_heap", default="2g")
    ap.add_argument("--repo", default=REPO_DIR, help="Tribes repo root (the game reads terrainProbs.json there)")
    ap.add_argument("--no_plot", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.yaml"), "w") as f:
        settings = {k: v for k, v in vars(args).items() if k not in ("set", "out", "no_plot", "classpath", "repo")}
        yaml.safe_dump({"agent": "actor_critic_gae", "config": cfg, "train": settings}, f, sort_keys=False)

    env = TribesEnv(args.classpath, cfg, heap=args.java_heap, repo_dir=args.repo)
    agent = build_agent(cfg, env.obs_size, env.act_size, seed=args.seed)
    if args.init:
        agent.params = load_checkpoint(args.init)["params"]
        agent.a_state = agent.actor_opt.init(agent.params["theta"])
        agent.c_state = agent.critic_opt.init(agent.params["phi"])
    rng = np.random.default_rng(args.seed)
    print(f"agent actor_critic_gae | config {cfg} | obs {env.obs_size} act {env.act_size} | out {args.out}",
          flush=True)

    metrics_log = CsvLog(os.path.join(args.out, "metrics.csv"), METRICS)
    eval_log = CsvLog(os.path.join(args.out, "eval.csv"), ["win_rate", "draw_rate", "loss_rate"])
    running, t0 = [], time.time()
    try:
        for it in range(1, args.iterations + 1):
            # 1. Collect experience
            batch, stats = collect(agent, env, rng, cfg)

            # 2. Update the agent
            running.append({**stats, **agent.update(batch)})

            # 3. Log, evaluate, checkpoint
            if it % args.log_every == 0:
                m = {k: float(np.mean([r[k] for r in running])) for k in METRICS}
                metrics_log.write(it, m)
                print(f"iter {it:6d} | {log_line(m)} | {time.time() - t0:.1f}s", flush=True)
                running = []

            if it % args.eval_every == 0 or it == args.iterations:
                n = args.eval_games
                wdl = evaluate(agent, env, n, args.seed)
                eval_log.write(it, dict(zip(["win_rate", "draw_rate", "loss_rate"], (x / n for x in wdl))))
                print(f"   eval vs random  W/D/L {tuple(int(x) for x in wdl)}", flush=True)
                save(agent, os.path.join(args.out, f"ac_{it:06d}.pkl"), cfg, env)
                save(agent, os.path.join(args.out, "latest.pkl"), cfg, env)
                if not args.no_plot:
                    try:
                        from plot import plot
                        plot([args.out])
                    except Exception as exc:   # a plotting problem shouldn't stop training
                        print(f"   plot failed: {exc}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
