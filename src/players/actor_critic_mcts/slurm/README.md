# Training pipeline on SLURM

These scripts run the whole AlphaZero-style loop on the cluster. Self-play is spread over many CPU nodes, and the network is trained on one GPU:

```
iteration k:  self-play array (SELFPLAY_NODES CPU tasks, n games each)  ──afterok──►  train (1 GPU)  ──afterok──►  iteration k+1
```

All games of an iteration are played before the network is updated. `submit.sh` submits the whole chain at once, and each job waits for the previous one to succeed (`--dependency=afterok`; for an array job, that means every task). If any job fails, SLURM cancels the rest of the chain (`--kill-on-invalid-dep=yes`).

| File | Role |
|------|------|
| [config.sh](config.sh) | All settings: games per iteration, nodes, search budget, training, SLURM resources |
| [submit.sh](submit.sh) | Creates the run folder and submits the job chain (new run or resume) |
| [selfplay.sbatch](selfplay.sbatch) | One array task: plays its share of the iteration's games |
| [train.sbatch](train.sbatch) | Checks the iteration's data is complete, then trains on a GPU (V100 by default) and saves the next model |
| [status.sh](status.sh) | Queued jobs, games played and model metrics per iteration |

## Quick start

```bash
cd /storage/scratch1/6/snayak89/Tribes
vim src/players/actor_critic_mcts/slurm/config.sh                    # set games per iteration, nodes, ...
DRY_RUN=1 src/players/actor_critic_mcts/slurm/submit.sh first_run    # check what would be submitted
rm -rf runs/first_run                                                # the dry run already created the folder
src/players/actor_critic_mcts/slurm/submit.sh first_run              # submit for real
src/players/actor_critic_mcts/slurm/status.sh first_run              # check progress
```

To change a setting for a single submission, set it in the environment, e.g. `ITERATIONS=3 GAMES_PER_ITER=40 submit.sh small_test`.

## Games per node

You choose `GAMES_PER_ITER` (games per iteration) and `SELFPLAY_NODES` (the array size). Each array task plays

```
n = ceil(GAMES_PER_ITER / SELFPLAY_NODES)
```

games, so an iteration has `n × SELFPLAY_NODES` games, rounded up from what you asked for. For example, 100 games over 10 nodes gives 10 games per node, and 100 games over 8 nodes gives 13 per node, 104 in total.

Task t plays game ids `(t-1)·n … t·n-1`. Each game's seeds come from `BASE_SEED`, the iteration and the game id, so no two nodes and no two iterations play the same game.

Within a task, the games are split over `SELFPLAY_WORKERS` parallel workers. Each worker is one JVM plus its own Python inference server, and gets `cpus / workers - 1` torch threads. With 8 CPUs and 2 workers, that's 3 threads each, plus a core for each JVM.

## Run folder

`runs/<run_name>/` is created by `submit.sh`:

```
run_config.sh            every setting, with its resolved value (paths to java / python included)
classes/  lib/  py/      frozen copies of the compiled Java, json.jar and the Python scripts
data/iter_XXX/           self-play data: games_AAAA-BBBB.jsonl.gz (one per worker) + schema.json
models/iter_XXX/         best.pt, last.pt, metrics.csv: the model that PLAYS iteration XXX
logs/                    SLURM logs (sp-<run>-<k>-<job>_<task>.out, train-<run>-<k>-<job>.out)
logs/selfplay_iter_XXX/  one log per self-play worker
jobs.txt                 job ids of every submitted iteration
```

Because the run uses frozen copies of the code, you can keep editing the code in the repository while a run is going.

**Naming:** `models/iter_001` is trained after self-play iteration 0, and is the model that plays iteration 1. Iteration 0 is played with `UniformNetwork` (uniform priors, heuristic leaf values). From iteration 1 on, the network gives the priors and the value.

## Training job

After the self-play array for iteration k finishes, `train.sbatch` runs these steps:
1. Checks that the game ranges in `data/iter_k/` add up to every game of the iteration. If any are missing, it fails, which stops the chain.
2. Trains on the latest `DATA_WINDOW` iterations (`iter_{k-DATA_WINDOW+1} … iter_k`).
3. Starts from `models/iter_k/best.pt` if it exists. The first model uses `TRAIN_HIDDEN` and `TRAIN_LAYERS` instead.
4. Saves the result to `models/iter_{k+1}/`.

It runs on `TRAIN_PARTITION` with `TRAIN_GRES`, by default one V100 on `gpu-v100` with 8 CPUs. That partition usually starts jobs sooner than `gpu-l40s`. To use other GPUs, change both settings, e.g. `TRAIN_PARTITION=gpu-l40s TRAIN_GRES=gpu:l40s:1 TRAIN_CPUS=4`. `gpu-l40s` allows at most 4 CPUs per GPU; check any partition's limits with `sbatch --test-only`.

## Self-play on GPU

The network's device during self-play is `SELFPLAY_DEVICE` (default `auto`: GPU if the node has one). By default, self-play runs on the CPU partition from your template. To run it on GPU nodes instead, set:

```bash
SELFPLAY_PARTITION=gpu-l40s SELFPLAY_CONSTRAINT= SELFPLAY_GRES=gpu:l40s:1 SELFPLAY_CPUS=4 SELFPLAY_WORKERS=1
```

Outside the pipeline, the same choice is the `-Dtribes.gnn.device=auto|cpu|cuda` Java property of `ExternalNetwork`.

## When something fails

- **A self-play task fails or hits its time limit:** its unfinished output stays as `*.part` and is never used for training. The training job then refuses to run, and the rest of the chain is cancelled. Resubmit just that task (workers whose file already exists are skipped):
  ```bash
  sbatch --array=<t> -A gts-mhale30 -q inferno -p cpu-gnr --constraint=graniterapids --cpus-per-task=8 --mem=48G -t 20:00:00 \
         --export=ALL,RUN_DIR=$PWD/runs/<run>,ITER=<k> src/players/actor_critic_mcts/slurm/selfplay.sbatch
  ```
  Then resubmit the rest of the run from the training step, or rerun `train.sbatch` by hand and resume at k+1.
- **Resume a run** from iteration k: `submit.sh <run> k [last]`. This needs `models/iter_k/best.pt`, and reuses the run's frozen config and code.
- **Cancel everything:** `scancel` the ids in `runs/<run>/jobs.txt`, or `squeue -u $USER` and cancel the `sp-<run>-*` and `train-<run>-*` jobs.

## Time and memory guide

These numbers come from local tests on 11×11 boards.
- **Iteration 0** (uniform network, 300 FM calls per move): a game takes 3–12 s.
- **With the network:** a game makes about 9,000 network calls at about 4 ms each (300 FM calls per move), so roughly 40 s per game. The number of calls grows with `FM_CALLS_PER_MOVE`, so at 1000 expect 2–3 min per game, and longer in `CAPITALS` mode (up to 50 turns).
- **Training memory:** about 16 MB per game. With 100 games per iteration and `DATA_WINDOW=3`, that's around 5 GB, well within `TRAIN_MEM=64G`.

Adjust `SELFPLAY_TIME` / `TRAIN_TIME` once you've seen the first iteration's logs. Each run's logs report the time per game and the mean network call time.
