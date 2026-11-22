#!/bin/bash
# Show the progress of a pipeline run: queued/running jobs, games played and models trained per iteration.
#   src/players/actor_critic_mcts/slurm/status.sh <run_name>
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
RUN_NAME=${1:?"usage: $0 <run_name>"}
source "$SCRIPT_DIR/config.sh"
[[ "$RUNS_DIR" = /* ]] || RUNS_DIR="$REPO_DIR/$RUNS_DIR"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
[[ -f "$RUN_DIR/run_config.sh" ]] || { echo "No run at $RUN_DIR"; exit 1; }
source "$RUN_DIR/run_config.sh"

echo "Run $RUN_NAME ($RUN_DIR)"
echo "--- jobs in the queue"
squeue -u "$USER" -h -o "%.12i %.28j %.9T %.10M %R" | grep -E "(sp|train)-$RUN_NAME-" || echo "(none)"

echo "--- per iteration (expected $(( GAMES_PER_TASK * SELFPLAY_NODES )) games)"
for (( k = 0; k <= ITERATIONS; k++ )); do
    tag=$(printf %03d "$k")
    games=0
    for f in "$RUN_DIR/data/iter_$tag"/games_*.jsonl.gz; do
        [[ -e "$f" ]] || continue
        r=$(basename "$f" .jsonl.gz); r=${r#games_}
        games=$(( games + 10#${r#*-} - 10#${r%-*} + 1 ))
    done
    model="$RUN_DIR/models/iter_$tag/best.pt"
    metrics="$RUN_DIR/models/iter_$tag/metrics.csv"
    [[ $games -eq 0 && ! -f "$model" ]] && continue
    line=$(printf "iter %3d: %4d games played" "$k" "$games")
    if [[ -f "$metrics" ]]; then
        # best epoch's validation policy KL, value MSE and winner accuracy
        best=$(awk -F, '{sub(/\r$/, "")} NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $h["val_loss"]!="" && (b=="" || $h["val_loss"]<b){b=$h["val_loss"]; r=sprintf("val kl %.3f, value mse %.3f, winner acc %.2f", $h["val_kl"], $h["val_value"], $h["val_value_sign"])} END{print r}' "$metrics")
        line="$line | model trained on iterations up to $(( k - 1 )): $best"
    fi
    echo "$line"
done
