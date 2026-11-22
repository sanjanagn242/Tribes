#!/usr/bin/env python3
"""
Pairwise round-robin for Tribes.

Takes the settings from a tournament config (default: tournament.json), and for every pair of
entries in its "Players" list runs a separate 1v1 Tournament with exactly those settings (seeds,
repetitions, search depth, shift tribes, ...). Only "Players" and "Tribes" are changed per pair;
the pair always uses the first two entries of the config's "Tribes" list.

Usage (from anywhere; it runs from the repo root):
    python3 round_robin.py                      # uses tournament.json
    python3 round_robin.py my_config.json
    python3 round_robin.py --jobs 4             # run up to 4 pairs in parallel
    python3 round_robin.py --compile            # recompile src/ into out/ first
    python3 round_robin.py --players MCTS EMCTS RHEA   # only these agents (overrides "Players")

Outputs go to results/round_robin_<timestamp>/:
    <A>_vs_<B>.json      config used for the pair
    <A>_vs_<B>.out       game output (FM: lines filtered out)
    <A>_vs_<B>.err       stderr
    summary.txt          win-rate matrix + per-player totals (also printed)
    summary.csv          one row per player per pair
"""
import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.abspath(__file__))
CLASSPATH = "out:lib/json.jar"
# Final results line printed by Tournament, e.g.
# [N:2];[%:100.00];[W:2];[S:12550.50];[T:24.00];[C:3.00];[P:28.50];[D:23.00];[R:120.50];[Player:1:EMCTS]
# Per-game line printed by Game when a game ends, e.g. "44; Game Results: " (44 = rounds/turns played)
GAME_END_RE = re.compile(r"^(\d+); Game Results:")
RESULT_RE = re.compile(r"^\[N:(\d+)\];\[%:([\d.]+)\];\[W:(\d+)\];\[S:([\d.\-]+)\];.*\[Player:(\d+):([^\]]+)\]")


def fmt_duration(secs):
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def compile_sources():
    srcs = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(REPO, "src")) for f in fs if f.endswith(".java")]
    os.makedirs(os.path.join(REPO, "out"), exist_ok=True)
    print("Compiling...", flush=True)
    subprocess.run(["javac", "-nowarn", "-cp", "lib/json.jar", "-d", "out"] + srcs, cwd=REPO, check=True)


def run_pair(i, j, a, b, base_cfg, outdir):
    name = f"{i}-{a}_vs_{j}-{b}"
    cfg = dict(base_cfg)
    cfg["Players"] = [a, b]
    cfg["Tribes"] = base_cfg["Tribes"][:2]
    cfg_path = os.path.join(outdir, name + ".json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[start] {a} vs {b}", flush=True)
    t0 = time.time()
    results = {}
    rounds = []  # rounds (turns) played in each completed game
    with open(os.path.join(outdir, name + ".out"), "w") as out, open(os.path.join(outdir, name + ".err"), "w") as err:
        proc = subprocess.Popen(["java", "-cp", CLASSPATH, "Tournament", cfg_path],
                                cwd=REPO, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1)
        for line in proc.stdout:
            if line.startswith("FM:"):
                continue
            out.write(line)
            out.flush()
            g = GAME_END_RE.match(line)
            if g:
                rounds.append(int(g.group(1)))
            m = RESULT_RE.match(line.strip())
            if m:
                n, pct, wins, score, pid, ptype = m.groups()
                results[int(pid)] = {"games": int(n), "win_pct": float(pct), "wins": int(wins), "score": float(score)}
        proc.wait()

    status = "ok" if len(results) == 2 else f"FAILED (exit {proc.returncode}, see {name}.err)"
    print(f"[done ] {a} vs {b}: {status} in {time.time() - t0:.0f}s"
          + (f" -> {a} {results[0]['wins']}/{results[0]['games']}, {b} {results[1]['wins']}/{results[1]['games']}"
             if status == "ok" else ""), flush=True)
    return (i, j, a, b, results, time.time() - t0, rounds)


def main():
    ap = argparse.ArgumentParser(description="Pairwise round-robin over the players in a Tribes tournament config.")
    ap.add_argument("config", nargs="?", default="tournament.json")
    ap.add_argument("--jobs", type=int, default=1, help="pairs to run in parallel (default 1)")
    ap.add_argument("--compile", action="store_true", help="recompile src/ into out/ before running")
    ap.add_argument("--players", nargs="+", metavar="NAME",
                    help='agents to include, overriding the config\'s "Players" (e.g. --players MCTS EMCTS "Rule Based")')
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(os.getcwd(), args.config)
    with open(cfg_path) as f:
        base_cfg = json.load(f)
    if args.players:
        base_cfg["Players"] = args.players

    players = base_cfg["Players"]
    if len(players) < 2:
        sys.exit("Need at least 2 players in \"Players\".")
    if len(base_cfg["Tribes"]) < 2:
        sys.exit("Need at least 2 tribes in \"Tribes\".")
    if len(set(base_cfg["Tribes"][:2])) < 2:
        sys.exit("The first two \"Tribes\" must be different (results are matched to players by tribe).")

    if args.compile:
        compile_sources()

    run_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    outdir = os.path.join(REPO, "results", "round_robin_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(outdir)
    pairs = list(itertools.combinations(range(len(players)), 2))
    n_games = len(base_cfg["Level Seeds"]) * base_cfg["Repetitions"]
    print(f"Players: {players}\n{len(pairs)} pairs x {n_games} games each, tribes {base_cfg['Tribes'][:2]}"
          f"\nOutput: {outdir}\n", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        done = list(ex.map(lambda p: run_pair(p[0], p[1], players[p[0]], players[p[1]], base_cfg, outdir), pairs))

    # ---- summary ----
    labels = [f"{p}#{k}" if players.count(p) > 1 else p for k, p in enumerate(players)]
    n = len(players)
    matrix = [["-" if r == c else "" for c in range(n)] for r in range(n)]
    totals = {k: {"wins": 0, "games": 0, "score_sum": 0.0, "rounds_sum": 0} for k in range(n)}
    rows = []
    pair_lines = []
    all_rounds = []
    for i, j, a, b, res, secs, rounds in done:
        all_rounds += rounds
        rnd = (f"{sum(rounds) / len(rounds):>11.1f}{min(rounds):>6}{max(rounds):>6}{sum(rounds):>9}"
               if rounds else f"{'-':>11}{'-':>6}{'-':>6}{0:>9}")
        pair_lines.append(f"{labels[i]} vs {labels[j]}".ljust(32) + f"{len(rounds):>6}" + rnd + "   "
                          + fmt_duration(secs) + ("" if len(res) == 2 else "  (FAILED)"))
        avg_rounds = round(sum(rounds) / len(rounds), 1) if rounds else ""
        if len(res) != 2:
            matrix[i][j] = matrix[j][i] = "ERR"
            continue
        for me, opp, pid in ((i, j, 0), (j, i, 1)):
            r = res[pid]
            matrix[me][opp] = f"{r['win_pct']:.0f}% ({r['wins']}/{r['games']})"
            totals[me]["wins"] += r["wins"]
            totals[me]["games"] += r["games"]
            totals[me]["score_sum"] += r["score"] * r["games"]
            totals[me]["rounds_sum"] += sum(rounds)
            rows.append([labels[me], labels[opp], r["games"], r["wins"], r["win_pct"], r["score"], avg_rounds, round(secs, 1)])

    w = max(14, max(len(l) for l in labels) + 2)
    cw = max([w] + [len(c) + 2 for row in matrix for c in row])
    lines = ["Win rate of ROW player against COLUMN player (wins/games):", "",
             "".ljust(w) + "".join(l.ljust(cw) for l in labels)]
    for r in range(n):
        lines.append(labels[r].ljust(w) + "".join(matrix[r][c].ljust(cw) for c in range(n)))
    lines += ["", "Overall (sorted by win rate):", f"{'Player'.ljust(w)}{'Wins':>8}{'Games':>8}{'Win %':>9}{'Avg score':>12}{'Avg rounds':>12}"]
    order = sorted(range(n), key=lambda k: -(totals[k]["wins"] / totals[k]["games"]) if totals[k]["games"] else 1)
    for k in order:
        t = totals[k]
        g = t["games"]
        lines.append(f"{labels[k].ljust(w)}{t['wins']:>8}{g:>8}{(100 * t['wins'] / g if g else 0):>8.1f}%"
                     f"{(t['score_sum'] / g if g else 0):>12.1f}{(t['rounds_sum'] / g if g else 0):>12.1f}")
    total = time.time() - run_start
    lines += ["", "Games, rounds (turns per game) and time per pair:",
              "Pair".ljust(32) + f"{'Games':>6}{'Avg rounds':>11}{'Min':>6}{'Max':>6}{'Total':>9}   Time"] + pair_lines + ["",
              f"Total games: {len(all_rounds)}, total rounds: {sum(all_rounds)}"
              + (f", avg rounds/game: {sum(all_rounds) / len(all_rounds):.1f}" if all_rounds else ""), "",
              f"Started:  {started_at}",
              f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"Total wall time: {fmt_duration(total)} ({len(pairs)} pairs, --jobs {max(1, args.jobs)})"]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["player", "opponent", "games", "wins", "win_pct", "avg_score", "avg_rounds", "pair_seconds"])
        wr.writerows(rows)
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
