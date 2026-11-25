#!/bin/bash
#SBATCH -J acmcts-train
#SBATCH -A gts-mhale30
#SBATCH -q inferno
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#
# Train the network on GPU after self-play iteration ITER: data from the last DATA_WINDOW iterations,
# warm-started from the model that played iteration ITER, saved as the model for iteration ITER+1.
# Submitted by submit.sh (with --dependency=afterok:<self-play array>), which passes RUN_DIR and ITER.

set -euo pipefail
: "${RUN_DIR:?RUN_DIR not set (submit with submit.sh)}" "${ITER:?ITER not set}"
source "$RUN_DIR/run_config.sh"

TAG=$(printf "%03d" "$ITER")
NEXT=$(printf "%03d" $(( ITER + 1 )))
OUT="$RUN_DIR/models/iter_$NEXT"
echo "=== training after iteration $ITER on $(hostname), $(date)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check this iteration's data is complete: the game ranges in the file names must cover every game.
expected=$(( GAMES_PER_TASK * SELFPLAY_NODES ))
found=0
for f in "$RUN_DIR/data/iter_$TAG"/games_*.jsonl.gz; do
    [[ -e "$f" ]] || continue
    range=$(basename "$f" .jsonl.gz); range=${range#games_}
    found=$(( found + 10#${range#*-} - 10#${range%-*} + 1 ))
done
parts=$(find "$RUN_DIR/data/iter_$TAG" -maxdepth 1 -name "*.part" | wc -l)
echo "iteration $ITER data: $found / $expected games ($parts unfinished .part files, ignored)"
(( found == expected )) || { echo "ERROR: self-play data for iteration $ITER is incomplete"; exit 1; }

DATA=()
for (( j = ITER - DATA_WINDOW + 1; j <= ITER; j++ )); do
    (( j < 0 )) && continue
    DATA+=("$RUN_DIR/data/iter_$(printf "%03d" "$j")")
done
echo "training data: ${DATA[*]}"

INIT=()
if [[ -f "$RUN_DIR/models/iter_$TAG/best.pt" ]]; then
    INIT=(--init-from "$RUN_DIR/models/iter_$TAG/best.pt")
else
    INIT=(--hidden "$TRAIN_HIDDEN" --layers "$TRAIN_LAYERS")
fi

if [[ "$TRAIN_DEVICE" == cuda* ]]; then
    "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'ERROR: no CUDA device visible to torch')"
fi

# Train into a temporary folder and rename it at the end: train_gnn.py saves best.pt after every improving epoch, so
# a job killed at its time limit must not leave a half-trained model where the next self-play (or a resume) finds it.
rm -rf "$OUT.partial"
"$PYTHON" "$RUN_DIR/py/train_gnn.py" \
    --data "${DATA[@]}" \
    --out "$OUT.partial" \
    --device "$TRAIN_DEVICE" \
    --epochs "$TRAIN_EPOCHS" \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --lr "$TRAIN_LR" \
    --threads "${SLURM_CPUS_PER_TASK:-$TRAIN_CPUS}" \
    --seed "$ITER" \
    "${INIT[@]}"
rm -rf "$OUT" && mv "$OUT.partial" "$OUT"

echo "=== model for iteration $(( ITER + 1 )): $OUT/best.pt ($(date))"
