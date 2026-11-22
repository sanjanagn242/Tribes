#!/usr/bin/env bash
# Run the whole actor-critic MCTS training loop on one machine (e.g. a desktop with a GPU):
#
#   for k = 0 .. ITERATIONS-1:
#       self-play iteration k   (WORKERS parallel SelfPlay processes; model k, or the uniform network for k = 0)
#       train model k+1          (on the last DATA_WINDOW iterations, warm-started from model k)
#
# Usage, from anywhere:
#   src/players/actor_critic_mcts/local/run_loop.sh <run_name>
#   ITERATIONS=5 WORKERS=8 src/players/actor_critic_mcts/local/run_loop.sh <run_name>      # override settings
#   QUICK_TEST=1 src/players/actor_critic_mcts/local/run_loop.sh test_run                  # tiny end-to-end check
#
# Resuming: just run the same command again. Finished self-play files and trained models are kept, and the loop
# continues from the first missing piece (an interrupted self-play keeps its finished workers' files).
# A resumed run keeps the settings it started with (runs/<run_name>/run_config.sh); to run more iterations than
# planned, pass ITERATIONS_OVERRIDE=<n>.
# Ctrl-C stops everything cleanly; unfinished data stays as *.part and is ignored.
#
# Everything goes to <repo>/runs/<run_name>/ (same layout as the SLURM pipeline in ../slurm/):
#   run_config.sh  classes/ lib/ py/   frozen settings and code, so editing the repo doesn't affect a running loop
#   data/iter_XXX/                     self-play data (one file per worker) + schema.json
#   models/iter_XXX/best.pt            the model that plays iteration XXX (trained after iteration XXX-1)
#   logs/                              selfplay_iter_XXX_wN.log, train_iter_XXX.log
#   progress.csv                       one line per iteration: games, samples, minutes, validation metrics

set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
RUN_NAME=${1:?"usage: $0 <run_name>"}

# ------------------------------------------------------------------ settings (override from the environment)
ncpu() { nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }

ITERATIONS=${ITERATIONS:-20}
GAMES_PER_ITER=${GAMES_PER_ITER:-200}
WORKERS=${WORKERS:-$(( $(ncpu) / 4 > 0 ? $(ncpu) / 4 : 1 ))}   # parallel SelfPlay processes; each gets ~4 cores
DATA_WINDOW=${DATA_WINDOW:-3}

TRIBES=${TRIBES:-XIN_XI,IMPERIUS}
GAME_MODE=${GAME_MODE:-SCORE}
FM_CALLS_PER_MOVE=${FM_CALLS_PER_MOVE:-1000}
MAX_ROLLOUT_LENGTH=${MAX_ROLLOUT_LENGTH:-20}
TEMPERATURE=${TEMPERATURE:-1.0}
TEMP_MOVES=${TEMP_MOVES:-60}
BASE_SEED=${BASE_SEED:-42}
SELFPLAY_DEVICE=${SELFPLAY_DEVICE:-auto}      # network device during self-play: auto (GPU if available) | cpu | cuda
JAVA_HEAP=${JAVA_HEAP:-4g}                    # per SelfPlay process

TRAIN_EPOCHS=${TRAIN_EPOCHS:-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
TRAIN_LR=${TRAIN_LR:-1e-3}
TRAIN_HIDDEN=${TRAIN_HIDDEN:-128}
TRAIN_LAYERS=${TRAIN_LAYERS:-4}
TRAIN_DEVICE=${TRAIN_DEVICE:-auto}            # auto (GPU if available) | cpu | cuda

JAVA=${JAVA:-java}
JAVAC=${JAVAC:-javac}
PYTHON=${PYTHON:-python3}
RUNS_DIR=${RUNS_DIR:-$REPO_DIR/runs}

if [[ "${QUICK_TEST:-0}" == 1 ]]; then        # a few minutes, just to see the loop work end to end
    ITERATIONS=2 GAMES_PER_ITER=4 WORKERS=2 FM_CALLS_PER_MOVE=100 TRAIN_EPOCHS=1 TEMP_MOVES=10
fi

RUN_DIR="$RUNS_DIR/$RUN_NAME"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN_DIR/loop.log"; }
die() { log "ERROR: $*"; exit 1; }
tag() { printf "%03d" "$1"; }

# ------------------------------------------------------------------ set up the run (or reuse it)
if [[ -f "$RUN_DIR/run_config.sh" ]]; then
    source "$RUN_DIR/run_config.sh"           # resume with the settings the run started with
    [[ -n "${ITERATIONS_OVERRIDE:-}" ]] && ITERATIONS=$ITERATIONS_OVERRIDE
    log "Resuming run $RUN_NAME ($RUN_DIR)"
else
    command -v "$JAVA" >/dev/null && command -v "$JAVAC" >/dev/null || { echo "java/javac not found"; exit 1; }
    "$PYTHON" -c "import torch, numpy" 2>/dev/null || { echo "$PYTHON can't import torch and numpy"; exit 1; }
    mkdir -p "$RUN_DIR"/{classes,lib,py,data,models,logs}
    log "New run $RUN_NAME in $RUN_DIR: compiling Java ..."
    "$JAVAC" -nowarn -cp "$REPO_DIR/lib/json.jar" -d "$RUN_DIR/classes" $(find "$REPO_DIR/src" -name "*.java") 2>&1 | grep -v "^Note:"
    [[ -f "$RUN_DIR/classes/players/actor_critic_mcts/selfplay/SelfPlay.class" ]] || die "compilation failed"
    cp "$REPO_DIR/lib/json.jar" "$RUN_DIR/lib/"
    cp "$REPO_DIR"/py/actor_critic/*.py "$RUN_DIR/py/"
    {
        echo "# Frozen settings for run $RUN_NAME ($(date))"
        for v in ITERATIONS GAMES_PER_ITER WORKERS DATA_WINDOW TRIBES GAME_MODE FM_CALLS_PER_MOVE MAX_ROLLOUT_LENGTH \
                 TEMPERATURE TEMP_MOVES BASE_SEED SELFPLAY_DEVICE JAVA_HEAP TRAIN_EPOCHS TRAIN_BATCH_SIZE TRAIN_LR \
                 TRAIN_HIDDEN TRAIN_LAYERS TRAIN_DEVICE JAVA PYTHON; do
            printf '%s=%q\n' "$v" "${!v}"
        done
    } > "$RUN_DIR/run_config.sh"
    echo "iteration,games,samples,selfplay_min,train_min,val_kl,val_value_mse,val_winner_acc" > "$RUN_DIR/progress.csv"
fi

GPU=$("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>/dev/null)
THREADS=$(( $(ncpu) / WORKERS - 1 )); (( THREADS < 1 )) && THREADS=1
log "Settings: $ITERATIONS iterations x $GAMES_PER_ITER games, $WORKERS workers ($THREADS torch threads each), $FM_CALLS_PER_MOVE FM calls/move, $GAME_MODE, GPU: $GPU"

# Ctrl-C / kill: stop all workers and the trainer.
children=()
cleanup() { log "Interrupted, stopping child processes"; kill ${children[@]+"${children[@]}"} 2>/dev/null; wait; exit 130; }
trap cleanup INT TERM

games_in() {   # number of finished games in a data folder (from the file names games_AAAA-BBBB.jsonl.gz)
    local n=0 f r
    for f in "$1"/games_*.jsonl.gz; do
        [[ -e "$f" ]] || continue
        r=$(basename "$f" .jsonl.gz); r=${r#games_}
        n=$(( n + 10#${r#*-} - 10#${r%-*} + 1 ))
    done
    echo $n
}

cd "$REPO_DIR"   # the game loads level/terrain files relative to the repository root

# ------------------------------------------------------------------ the loop
for (( k = 0; k < ITERATIONS; k++ )); do
    T=$(tag "$k"); DATA_DIR="$RUN_DIR/data/iter_$T"; NEXT_MODEL="$RUN_DIR/models/iter_$(tag $((k + 1)))/best.pt"
    MODEL_ARG=()
    if (( k > 0 )); then
        MODEL="$RUN_DIR/models/iter_$T/best.pt"
        [[ -f "$MODEL" ]] || die "model for iteration $k missing: $MODEL"
        MODEL_ARG=("modelPath=$MODEL")
    fi

    # ---------------- self-play
    sp_min=0
    if (( $(games_in "$DATA_DIR") >= GAMES_PER_ITER )); then
        log "iteration $k: self-play already done ($(games_in "$DATA_DIR") games)"
    else
        log "iteration $k: self-play, $GAMES_PER_ITER games on $WORKERS workers, model: ${MODEL:-uniform}"
        t0=$(date +%s); children=(); offset=0
        for (( w = 0; w < WORKERS; w++ )); do
            n=$(( GAMES_PER_ITER / WORKERS + (w < GAMES_PER_ITER % WORKERS ? 1 : 0) ))
            (( n == 0 )) && continue
            out=$(printf "%s/games_%04d-%04d.jsonl.gz" "$DATA_DIR" "$offset" $(( offset + n - 1 )))
            if [[ ! -f "$out" ]]; then
                "$JAVA" -Xmx"$JAVA_HEAP" \
                    -Dtribes.python="$PYTHON" -Dtribes.gnn.server="$RUN_DIR/py/serve_gnn.py" \
                    -Dtribes.gnn.threads="$THREADS" -Dtribes.gnn.device="$SELFPLAY_DEVICE" \
                    -cp "$RUN_DIR/classes:$RUN_DIR/lib/json.jar" players.actor_critic_mcts.selfplay.SelfPlay \
                    iteration="$k" gameOffset="$offset" numGames="$n" outputDir="$RUN_DIR/data" \
                    tribes="$TRIBES" gameMode="$GAME_MODE" fmCallsPerMove="$FM_CALLS_PER_MOVE" \
                    maxRolloutLength="$MAX_ROLLOUT_LENGTH" temperature="$TEMPERATURE" tempMoves="$TEMP_MOVES" \
                    baseSeed="$BASE_SEED" ${MODEL_ARG[@]+"${MODEL_ARG[@]}"} \
                    > "$RUN_DIR/logs/selfplay_iter_${T}_w$w.log" 2>&1 &
                children+=($!)
            fi
            offset=$(( offset + n ))
        done
        failed=0
        for pid in ${children[@]+"${children[@]}"}; do wait "$pid" || failed=1; done
        children=()
        sp_min=$(( ($(date +%s) - t0 + 30) / 60 ))
        played=$(games_in "$DATA_DIR")
        (( failed == 0 && played >= GAMES_PER_ITER )) || die "self-play for iteration $k incomplete ($played/$GAMES_PER_ITER games); see $RUN_DIR/logs/selfplay_iter_${T}_w*.log"
        net=$(grep -h '^Network' "$RUN_DIR"/logs/selfplay_iter_${T}_w*.log 2>/dev/null | head -1)
        log "iteration $k: $played games in $sp_min min${net:+ ($net)}"
    fi

    # ---------------- training
    tr_min=0
    if [[ -f "$NEXT_MODEL" ]]; then
        log "iteration $k: model $(tag $((k + 1))) already trained"
    else
        DATA=(); for (( j = k - DATA_WINDOW + 1; j <= k; j++ )); do (( j >= 0 )) && DATA+=("$RUN_DIR/data/iter_$(tag "$j")"); done
        INIT=(--hidden "$TRAIN_HIDDEN" --layers "$TRAIN_LAYERS")
        (( k > 0 )) && INIT=(--init-from "$MODEL")
        log "iteration $k: training model $(tag $((k + 1))) on iterations $(( k - ${#DATA[@]} + 1 ))..$k"
        t0=$(date +%s)
        # Train into a temporary folder and rename it when training has finished: train_gnn.py saves best.pt after
        # every improving epoch, so an interrupted run must not leave a half-trained model under the final name.
        OUT_DIR=$(dirname "$NEXT_MODEL"); TMP_DIR="$OUT_DIR.partial"
        rm -rf "$TMP_DIR"
        "$PYTHON" "$RUN_DIR/py/train_gnn.py" --data "${DATA[@]}" --out "$TMP_DIR" \
            --device "$TRAIN_DEVICE" --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BATCH_SIZE" --lr "$TRAIN_LR" \
            --seed "$k" "${INIT[@]}" > "$RUN_DIR/logs/train_iter_$T.log" 2>&1 &
        children=($!)
        wait "${children[0]}" || die "training after iteration $k failed; see $RUN_DIR/logs/train_iter_$T.log"
        children=()
        rm -rf "$OUT_DIR" && mv "$TMP_DIR" "$OUT_DIR"
        tr_min=$(( ($(date +%s) - t0 + 30) / 60 ))
        log "iteration $k: $(grep '^Done' "$RUN_DIR/logs/train_iter_$T.log") ($tr_min min)"

        # One summary line per iteration: games, samples, minutes, best epoch's validation metrics.
        samples=$(grep -h '^Done:' "$RUN_DIR"/logs/selfplay_iter_${T}_w*.log 2>/dev/null | awk '{s += $2} END {print s + 0}')
        best=$(awk -F, '{sub(/\r$/, "")} NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $h["val_loss"]!="" && (b=="" || $h["val_loss"]<b){b=$h["val_loss"]; r=$h["val_kl"]","$h["val_value"]","$h["val_value_sign"]} END{print r}' \
               "$(dirname "$NEXT_MODEL")/metrics.csv")
        echo "$k,$(games_in "$DATA_DIR"),$samples,$sp_min,$tr_min,$best" >> "$RUN_DIR/progress.csv"
    fi
done

log "Finished $ITERATIONS iterations. Latest model: $RUN_DIR/models/iter_$(tag "$ITERATIONS")/best.pt"
column -s, -t "$RUN_DIR/progress.csv" 2>/dev/null || cat "$RUN_DIR/progress.csv"
