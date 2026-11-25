#!/usr/bin/env bash
# Train the GAE actor-critic (players.actor_critic_gae) against a random opponent on this machine.
# The networks and training are Python (JAX/flax/optax, python/); Java only runs the Tribes games
# (EnvServer.java, started by python/tribes_env.py).
#
# Usage, from anywhere:
#   src/players/actor_critic_gae/local/train.sh <run_name> [train.py options]
#
#   src/players/actor_critic_gae/local/train.sh simple_ac                              # config.yaml (lam=0)
#   src/players/actor_critic_gae/local/train.sh gae --set lam=0.95                     # GAE actor-critic
#   src/players/actor_critic_gae/local/train.sh b01 --set beta=0.01 alpha=5e-4 --iterations 5000
#   QUICK_TEST=1 src/players/actor_critic_gae/local/train.sh quick                     # ~1 minute check
#   nohup src/players/actor_critic_gae/local/train.sh gae --set lam=0.95 > /dev/null 2>&1 &
#
# train.py options: --set KEY=VALUE ... (keys of python/config.yaml), --iterations, --log_every,
#   --eval_every, --eval_games, --seed, --init <ckpt.pkl>, --no_plot   (python train.py --help)
#
# Environment:
#   PYTHON      Python with jax, flax, optax, pyyaml, matplotlib
#               (default: the tic-tac-toe venv if it exists, else python3)
#   RUNS_DIR    where runs go                        (default <repo>/runs/actor_critic_gae)
#   JAVA_HEAP   heap of the Java game process        (default 2g)
#   OVERWRITE=1 replace an existing run with the same name
#   QUICK_TEST=1 tiny run: 4 iterations of 2 games, eval on 4 games
#
# Output, in $RUNS_DIR/<run_name>/:
#   classes/ python/   frozen copies of the Java classes and Python code this run uses
#   config.yaml        the config and training settings
#   metrics.csv        actor/critic loss, entropy, value/return means, |TD error|, train W/D/L, decisions/game
#   eval.csv           greedy policy vs random: win/draw/loss rates
#   ac_XXXXXX.pkl, latest.pkl    checkpoints ({"params": {"theta", "phi"}, "config", ...})
#   train.log, metrics.png

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AGENT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../../../.." && pwd)
RUN_NAME=${1:?"usage: $0 <run_name> [train.py options]"}
shift

DEFAULT_VENV=/storage/scratch1/6/snayak89/tic-tac-toe/venv/bin/python
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$DEFAULT_VENV" ]]; then PYTHON=$DEFAULT_VENV; else PYTHON=python3; fi
fi
RUNS_DIR=${RUNS_DIR:-$REPO_DIR/runs/actor_critic_gae}
JAVA_HEAP=${JAVA_HEAP:-2g}
JAVAC=${JAVAC:-javac}
RUN_DIR="$RUNS_DIR/$RUN_NAME"

EXTRA=()
if [[ "${QUICK_TEST:-0}" == 1 ]]; then
    EXTRA=(--iterations 4 --log_every 1 --eval_every 2 --eval_games 4)
    set -- --set episodes_per_iter=2 "$@"
fi

if [[ -f "$RUN_DIR/metrics.csv" && "${OVERWRITE:-0}" != 1 ]]; then
    echo "run '$RUN_NAME' already exists at $RUN_DIR"
    echo "pick another name, or set OVERWRITE=1 to replace it (or continue from it: --init $RUN_DIR/latest.pkl)"
    exit 1
fi
command -v java >/dev/null && command -v "$JAVAC" >/dev/null || { echo "java/javac not found"; exit 1; }
"$PYTHON" -c "import jax, flax, optax, yaml" 2>/dev/null \
    || { echo "$PYTHON can't import jax, flax, optax and yaml; set PYTHON=<python with them>"; exit 1; }

mkdir -p "$RUN_DIR"
rm -rf "$RUN_DIR/classes" "$RUN_DIR/python"
echo "compiling Tribes into $RUN_DIR/classes ..."
"$JAVAC" -nowarn -cp "$REPO_DIR/lib/json.jar" -d "$RUN_DIR/classes" $(find "$REPO_DIR/src" -name "*.java") 2>&1 | grep -v "^Note:" || true
[[ -f "$RUN_DIR/classes/players/actor_critic_gae/EnvServer.class" ]] || { echo "compilation failed"; exit 1; }
cp -r "$AGENT_DIR/python" "$RUN_DIR/python"
rm -rf "$RUN_DIR/python/__pycache__"

# The frozen python/ copy reads config.yaml from itself, so edits to the repo don't affect this run.
"$PYTHON" -u "$RUN_DIR/python/train.py" \
    --classpath "$RUN_DIR/classes:$REPO_DIR/lib/json.jar" --out "$RUN_DIR" --repo "$REPO_DIR" --java_heap "$JAVA_HEAP" \
    "${EXTRA[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
exit "${PIPESTATUS[0]}"
