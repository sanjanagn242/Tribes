# Settings for the actor-critic MCTS training pipeline (read by submit.sh, selfplay.sbatch and train.sbatch).
# submit.sh copies this file into the run folder, so a running pipeline isn't affected by later edits here.
# Override any value for one submission from the environment, e.g.  ITERATIONS=3 ./submit.sh my_run

# Defaults are sized for a first real run: ~200 games x ~400 samples = ~80k new samples per iteration.
# For a quick end-to-end check instead:
#   ITERATIONS=2 GAMES_PER_ITER=4 SELFPLAY_NODES=2 FM_CALLS_PER_MOVE=200 GAME_MODE=SCORE TRAIN_EPOCHS=1 ./submit.sh smoke_test

# ------------------------------------------------------------------ the loop
ITERATIONS=${ITERATIONS:-2}          # self-play -> train cycles to submit (resume later with submit.sh <run> <k>)
GAMES_PER_ITER=${GAMES_PER_ITER:-20} # requested games per iteration; rounded up to a multiple of SELFPLAY_NODES
SELFPLAY_NODES=${SELFPLAY_NODES:-20}  # SLURM array size -> n = ceil(200 / 20) = 10 games per task. Keep <= GAMES_PER_ITER
SELFPLAY_WORKERS=${SELFPLAY_WORKERS:-2} # parallel games per task: 2 x (JVM + 3 torch threads) fills 8 CPUs; 5 games each
DATA_WINDOW=${DATA_WINDOW:-3}         # train on the latest 3 iterations (~240k samples): enough data, not too stale

# ------------------------------------------------------------------ games and search (passed to SelfPlay)
TRIBES=${TRIBES:-XIN_XI,IMPERIUS}
GAME_MODE=${GAME_MODE:-SCORE}         # SCORE: 30 turns, always a winner, shorter games. CAPITALS: up to 50 turns
FM_CALLS_PER_MOVE=${FM_CALLS_PER_MOVE:-1000}  # search budget; ~3-5 min per game with the network (200 was only for testing)
MAX_ROLLOUT_LENGTH=${MAX_ROLLOUT_LENGTH:-20}  # tree depth limit
TEMPERATURE=${TEMPERATURE:-1.0}       # sample moves in proportion to visit counts ...
TEMP_MOVES=${TEMP_MOVES:-60}          # ... for each player's first 60 actions (~6-8 turns), then play the most visited
BASE_SEED=${BASE_SEED:-42}            # change it to get a different set of maps for a repeat run
SELFPLAY_DEVICE=${SELFPLAY_DEVICE:-auto}   # network device during self-play: auto (GPU if the node has one) | cpu | cuda

# ------------------------------------------------------------------ training (passed to train_gnn.py)
# Each sample is seen ~DATA_WINDOW x TRAIN_EPOCHS = 15 times over its lifetime. Keep epochs low: all samples of a game
# share one value target, so more passes mostly memorise games (the value loss overfits first; watch val_value).
TRAIN_EPOCHS=${TRAIN_EPOCHS:-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}   # fits easily on a 16 GB V100 at 11x11 / 14x14 boards
TRAIN_LR=${TRAIN_LR:-1e-3}            # peak of the one-cycle schedule, restarted each iteration from the warm start
TRAIN_HIDDEN=${TRAIN_HIDDEN:-128}     # only used for the first model; later ones keep the architecture they start from
TRAIN_LAYERS=${TRAIN_LAYERS:-4}
TRAIN_DEVICE=${TRAIN_DEVICE:-cuda}

# ------------------------------------------------------------------ SLURM resources
SLURM_ACCOUNT=${SLURM_ACCOUNT:-gts-mhale30}
SLURM_QOS=${SLURM_QOS:-inferno}

# Self-play array (CPU nodes)
SELFPLAY_PARTITION=${SELFPLAY_PARTITION:-cpu-gnr}
SELFPLAY_CONSTRAINT=${SELFPLAY_CONSTRAINT:-graniterapids}
SELFPLAY_GRES=${SELFPLAY_GRES:-}      # e.g. gpu:l40s:1 if you move self-play to a GPU partition
SELFPLAY_CPUS=${SELFPLAY_CPUS:-8}
SELFPLAY_MEM=${SELFPLAY_MEM:-48G}
SELFPLAY_TIME=${SELFPLAY_TIME:-06:00:00}   # ~5 games per worker x 3-5 min, with a wide margin; shorter limits start sooner

# Training (one GPU)
TRAIN_PARTITION=${TRAIN_PARTITION:-gpu-rtx6000}
TRAIN_GRES=${TRAIN_GRES:-gpu:rtx_6000:1}
TRAIN_CPUS=${TRAIN_CPUS:-6}         # gpu-rtx6000 allows at most 6 CPUs per GPU (gpu-v100 accepts 8, gpu-l40s at most 4)
TRAIN_MEM=${TRAIN_MEM:-64G}          # ~240k samples need ~15 GB in memory
TRAIN_TIME=${TRAIN_TIME:-03:00:00}    # ~5 epochs x 5-10 min on a V100 + a few minutes of data loading

# ------------------------------------------------------------------ tools and paths
JAVA=${JAVA:-$(command -v java)}      # resolved at submit time, so compute nodes use the same JDK
JAVAC=${JAVAC:-$(command -v javac)}
PYTHON=${PYTHON:-$(command -v python3)}
RUNS_DIR=${RUNS_DIR:-runs}            # relative to the repository root
