#!/bin/bash
# Submit the actor-critic MCTS training pipeline as a chain of SLURM jobs:
#
#   self-play iter 0 (array of SELFPLAY_NODES CPU tasks) -> train (1 GPU) -> self-play iter 1 -> train -> ...
#
# Each job starts only when the previous one finished successfully (--dependency=afterok), so all games of an
# iteration are played before the network is updated. If a job fails, the rest of the chain is cancelled.
#
# Usage (from anywhere):
#   src/players/actor_critic_mcts/slurm/submit.sh <run_name>                    new run: iterations 0 .. ITERATIONS-1
#   src/players/actor_critic_mcts/slurm/submit.sh <run_name> <first> [<last>]   resume an existing run at iteration <first>
#   DRY_RUN=1 src/players/actor_critic_mcts/slurm/submit.sh <run_name>          print the sbatch commands, submit nothing
#
# A new run gets its own folder <repo>/runs/<run_name>/ with a frozen copy of the config, the compiled Java classes
# and the Python scripts, so editing the code while the pipeline runs doesn't change it. Layout:
#   run_config.sh  classes/  lib/  py/  data/iter_XXX/  models/iter_XXX/best.pt  logs/  jobs.txt

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
RUN_NAME=${1:?"usage: $0 <run_name> [first_iter] [last_iter]"}
FIRST=${2:-0}
DRY_RUN=${DRY_RUN:-0}

source "$SCRIPT_DIR/config.sh"
[[ "$RUNS_DIR" = /* ]] || RUNS_DIR="$REPO_DIR/$RUNS_DIR"
RUN_DIR="$RUNS_DIR/$RUN_NAME"

if (( FIRST == 0 )); then
    # ------------------------------------------------------------ new run
    [[ -e "$RUN_DIR/run_config.sh" ]] && { echo "ERROR: $RUN_DIR already exists. Resume it with: $0 $RUN_NAME <first_iter>"; exit 1; }
    GAMES_PER_TASK=$(( (GAMES_PER_ITER + SELFPLAY_NODES - 1) / SELFPLAY_NODES ))   # n = ceil(games / nodes)
    mkdir -p "$RUN_DIR"/{classes,lib,py,data,models,logs}

    echo "Compiling Java into $RUN_DIR/classes ..."
    "$JAVAC" -nowarn -cp "$REPO_DIR/lib/json.jar" -d "$RUN_DIR/classes" $(find "$REPO_DIR/src" -name "*.java") 2>&1 | grep -v "^Note:" || true
    [[ -f "$RUN_DIR/classes/players/actor_critic_mcts/selfplay/SelfPlay.class" ]] || { echo "ERROR: compilation failed"; exit 1; }
    cp "$REPO_DIR/lib/json.jar" "$RUN_DIR/lib/"
    cp "$REPO_DIR"/py/actor_critic/*.py "$RUN_DIR/py/"

    # Freeze every setting with its resolved value (paths to java/python included).
    {
        echo "# Frozen settings for run $RUN_NAME, written by submit.sh on $(date)"
        for v in RUN_NAME REPO_DIR ITERATIONS GAMES_PER_ITER SELFPLAY_NODES GAMES_PER_TASK SELFPLAY_WORKERS DATA_WINDOW \
                 TRIBES GAME_MODE FM_CALLS_PER_MOVE MAX_ROLLOUT_LENGTH TEMPERATURE TEMP_MOVES BASE_SEED SELFPLAY_DEVICE \
                 TRAIN_EPOCHS TRAIN_BATCH_SIZE TRAIN_LR TRAIN_HIDDEN TRAIN_LAYERS TRAIN_DEVICE \
                 SLURM_ACCOUNT SLURM_QOS SELFPLAY_PARTITION SELFPLAY_CONSTRAINT SELFPLAY_GRES SELFPLAY_CPUS SELFPLAY_MEM \
                 SELFPLAY_TIME TRAIN_PARTITION TRAIN_GRES TRAIN_CPUS TRAIN_MEM TRAIN_TIME JAVA PYTHON; do
            printf '%s=%q\n' "$v" "${!v}"
        done
    } > "$RUN_DIR/run_config.sh"
else
    # ------------------------------------------------------------ resume: use the run's frozen settings
    [[ -f "$RUN_DIR/run_config.sh" ]] || { echo "ERROR: no run at $RUN_DIR"; exit 1; }
    source "$RUN_DIR/run_config.sh"
    MODEL="$RUN_DIR/models/iter_$(printf %03d "$FIRST")/best.pt"
    [[ -f "$MODEL" ]] || { echo "ERROR: iteration $FIRST needs $MODEL (trained after iteration $(( FIRST - 1 )))"; exit 1; }
fi
LAST=${3:-$(( ITERATIONS - 1 ))}

echo "Run $RUN_NAME: iterations $FIRST..$LAST, $SELFPLAY_NODES nodes x $GAMES_PER_TASK games = $(( SELFPLAY_NODES * GAMES_PER_TASK )) games per iteration"
echo "  self-play: $SELFPLAY_PARTITION, $SELFPLAY_CPUS CPUs, $SELFPLAY_WORKERS workers/node, $FM_CALLS_PER_MOVE FM calls/move"
echo "  training : $TRAIN_PARTITION ($TRAIN_GRES), $TRAIN_EPOCHS epochs on the last $DATA_WINDOW iterations"
if (( SELFPLAY_NODES * GAMES_PER_TASK > GAMES_PER_ITER )); then
    echo "  NOTE: $GAMES_PER_ITER games requested, but $SELFPLAY_NODES nodes x $GAMES_PER_TASK = $(( SELFPLAY_NODES * GAMES_PER_TASK )) will be played"
    (( SELFPLAY_NODES > GAMES_PER_ITER )) && echo "  WARNING: more nodes than games; use SELFPLAY_NODES <= GAMES_PER_ITER to avoid idle array tasks"
fi

submit() {   # prints the job id
    if (( DRY_RUN )); then
        echo "sbatch $*" >&2
        echo "DRYRUN$RANDOM"
    else
        sbatch --parsable "$@" | cut -d';' -f1
    fi
}

COMMON=(-A "$SLURM_ACCOUNT" -q "$SLURM_QOS" -N 1 -n 1 --kill-on-invalid-dep=yes)
SP_RES=(-p "$SELFPLAY_PARTITION" --cpus-per-task="$SELFPLAY_CPUS" --mem="$SELFPLAY_MEM" -t "$SELFPLAY_TIME")
[[ -n "$SELFPLAY_CONSTRAINT" ]] && SP_RES+=(--constraint="$SELFPLAY_CONSTRAINT")
[[ -n "$SELFPLAY_GRES" ]] && SP_RES+=(--gres="$SELFPLAY_GRES")
TR_RES=(-p "$TRAIN_PARTITION" --gres="$TRAIN_GRES" --cpus-per-task="$TRAIN_CPUS" --mem="$TRAIN_MEM" -t "$TRAIN_TIME")

prev=""
for (( k = FIRST; k <= LAST; k++ )); do
    dep=(); [[ -n "$prev" ]] && dep=(--dependency=afterok:"$prev")
    sp=$(submit "${COMMON[@]}" "${SP_RES[@]}" "${dep[@]}" \
            -J "sp-$RUN_NAME-$k" --array=1-"$SELFPLAY_NODES" --signal=B:TERM@120 \
            -o "$RUN_DIR/logs/%x-%A_%a.out" \
            --export=ALL,RUN_DIR="$RUN_DIR",ITER="$k" \
            "$SCRIPT_DIR/selfplay.sbatch")
    tr=$(submit "${COMMON[@]}" "${TR_RES[@]}" --dependency=afterok:"$sp" \
            -J "train-$RUN_NAME-$k" \
            -o "$RUN_DIR/logs/%x-%j.out" \
            --export=ALL,RUN_DIR="$RUN_DIR",ITER="$k" \
            "$SCRIPT_DIR/train.sbatch")
    printf "iteration %3d: self-play array %s -> train %s\n" "$k" "$sp" "$tr" | tee -a "$RUN_DIR/jobs.txt"
    prev=$tr
done

echo "Logs: $RUN_DIR/logs    Status: $SCRIPT_DIR/status.sh $RUN_NAME    Cancel all: scancel --name=... or scancel <ids in jobs.txt>"
