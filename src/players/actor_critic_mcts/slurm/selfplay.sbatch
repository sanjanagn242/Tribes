#!/bin/bash
#SBATCH -J acmcts-selfplay
#SBATCH -A gts-mhale30
#SBATCH -q inferno
#SBATCH -p cpu-gnr
#SBATCH --constraint=graniterapids
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 20:00:00
#SBATCH --array=1-10
#SBATCH --signal=B:TERM@120
#SBATCH -o logs/%x-%A_%a.out
#
# One self-play array task: plays GAMES_PER_TASK games of iteration ITER, split over SELFPLAY_WORKERS parallel
# workers (each = one JVM running SelfPlay + its Python inference server).
# Submitted by submit.sh, which overrides the #SBATCH defaults above from config.sh and passes RUN_DIR and ITER.
#
# Task t (1-based) plays game ids (t-1)*GAMES_PER_TASK ... t*GAMES_PER_TASK - 1, so tasks never overlap.
# A worker whose output file already exists is skipped, so a failed task can simply be resubmitted:
#   sbatch --array=<t> --export=ALL,RUN_DIR=<run dir>,ITER=<k> selfplay.sbatch

set -uo pipefail
: "${RUN_DIR:?RUN_DIR not set (submit with submit.sh)}" "${ITER:?ITER not set}"
source "$RUN_DIR/run_config.sh"

TASK=${SLURM_ARRAY_TASK_ID:-1}
CPUS=${SLURM_CPUS_PER_TASK:-$SELFPLAY_CPUS}
TAG=$(printf "%03d" "$ITER")
DATA_DIR="$RUN_DIR/data/iter_$TAG"
mkdir -p "$DATA_DIR" "$RUN_DIR/logs/selfplay_iter_$TAG"

MODEL_ARG=()
if (( ITER > 0 )); then
    MODEL="$RUN_DIR/models/iter_$TAG/best.pt"
    [[ -f "$MODEL" ]] || { echo "ERROR: model for iteration $ITER not found: $MODEL"; exit 1; }
    MODEL_ARG=("modelPath=$MODEL")
fi

echo "=== self-play iteration $ITER, task $TASK/$SELFPLAY_NODES on $(hostname), $CPUS CPUs, $(date)"
echo "    model: ${MODEL:-uniform}   games per task: $GAMES_PER_TASK   workers: $SELFPLAY_WORKERS"
command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null

# Split this task's games over the workers; give each worker an equal share of the CPUs.
FIRST_GAME=$(( (TASK - 1) * GAMES_PER_TASK ))
WORKERS=$(( SELFPLAY_WORKERS < GAMES_PER_TASK ? SELFPLAY_WORKERS : GAMES_PER_TASK ))
THREADS=$(( CPUS / WORKERS - 1 )); (( THREADS < 1 )) && THREADS=1   # torch threads; 1 CPU left for the JVM
MEM_GB=${SELFPLAY_MEM%[Gg]}
HEAP_GB=$(( MEM_GB / WORKERS - 3 )); (( HEAP_GB < 2 )) && HEAP_GB=2  # leave room for the Python server

cd "$REPO_DIR"   # the game loads level/terrain files relative to the repository root
pids=()
offset=$FIRST_GAME
for (( w = 0; w < WORKERS; w++ )); do
    n=$(( GAMES_PER_TASK / WORKERS + (w < GAMES_PER_TASK % WORKERS ? 1 : 0) ))
    out=$(printf "%s/games_%04d-%04d.jsonl.gz" "$DATA_DIR" "$offset" $(( offset + n - 1 )))
    log=$(printf "%s/logs/selfplay_iter_%s/task%03d_worker%d.log" "$RUN_DIR" "$TAG" "$TASK" "$w")
    if [[ -f "$out" ]]; then
        echo "worker $w: $out already exists, skipping"
    else
        echo "worker $w: games $offset-$(( offset + n - 1 )), $THREADS torch threads, heap ${HEAP_GB}g -> $log"
        "$JAVA" -Xmx${HEAP_GB}g \
            -Dtribes.python="$PYTHON" \
            -Dtribes.gnn.server="$RUN_DIR/py/serve_gnn.py" \
            -Dtribes.gnn.threads="$THREADS" \
            -Dtribes.gnn.device="$SELFPLAY_DEVICE" \
            -cp "$RUN_DIR/classes:$RUN_DIR/lib/json.jar" \
            players.actor_critic_mcts.selfplay.SelfPlay \
            iteration="$ITER" gameOffset="$offset" numGames="$n" outputDir="$RUN_DIR/data" \
            tribes="$TRIBES" gameMode="$GAME_MODE" fmCallsPerMove="$FM_CALLS_PER_MOVE" \
            maxRolloutLength="$MAX_ROLLOUT_LENGTH" temperature="$TEMPERATURE" tempMoves="$TEMP_MOVES" \
            baseSeed="$BASE_SEED" "${MODEL_ARG[@]}" \
            > "$log" 2>&1 &
        pids+=($!)
    fi
    offset=$(( offset + n ))
done

# SLURM sends TERM to this script 120 s before the time limit: pass it on to the workers.
# Their unfinished output stays as *.part and is ignored by training; resubmit the task to redo it.
trap 'echo "TERM received, stopping workers"; kill -TERM "${pids[@]}" 2>/dev/null; wait; exit 143' TERM

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "worker (pid $pid) failed with exit code $?"; status=1; }
done
grep -h "^Done\|^Network" "$RUN_DIR"/logs/selfplay_iter_$TAG/task$(printf %03d "$TASK")_worker*.log 2>/dev/null
echo "=== task $TASK finished with status $status at $(date)"
exit $status
