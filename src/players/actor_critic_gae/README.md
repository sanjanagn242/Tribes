# GAE Actor-Critic (`players.actor_critic_gae`)

The tic-tac-toe agent from `tic-tac-toe/actor-critic/gae-actor-critic-with-random`, playing Tribes against `players.RandomAgent`.

The work is split by language:

- **Python** (`python/`, JAX + flax + optax) holds the networks, the agent and the training. `agent.py` is the tic-tac-toe `agent.py`, changed only where the action space differs.
- **Java** runs the game. [`EnvServer`](EnvServer.java) is the environment Python plays in. [`ActorCriticGAEPlayer`](ActorCriticGAEPlayer.java) lets a trained model play from `Play`/`Tournament`.

```
actor    pi_theta(a|s)  ActionScorer: MLP([obs(s) ; act(s,a)]) -> one logit per legal action, softmax over them
critic   U_phi(s)       MLP(obs(s)) -> scalar, regressed on the Monte Carlo reward-to-go G_t
actor    loss           -( mean gamma^k * A_t * log pi(a_t|s_t)  +  beta * mean H(pi(.|s_t)) )
         A_t (GAE)      sum_l (gamma*lam)^l delta_{t+l} up to the end of the game,
                        delta_t = r_t + gamma (1 - done_t) U(s'_t) - U(s_t)       (stop-gradient)
critic   loss           0.5 * mean (U(s_t) - G_t)^2
```

**`lam: 0` (the default in `config.yaml`) is the simple actor-critic**: the advantage is the one-step TD error. Use `lam: 0.95` for GAE and `lam: 1` for the Monte Carlo advantage `G_t - U(s_t)`.

## Running

From the repo root:

```bash
src/players/actor_critic_gae/local/train.sh simple_ac                         # config.yaml as is (lam = 0)
src/players/actor_critic_gae/local/train.sh gae --set lam=0.95                # GAE actor-critic
src/players/actor_critic_gae/local/train.sh b01 --set beta=0.01 --iterations 5000
QUICK_TEST=1 src/players/actor_critic_gae/local/train.sh quick                # under a minute, end-to-end check
nohup src/players/actor_critic_gae/local/train.sh gae --set lam=0.95 > /dev/null 2>&1 &
python src/players/actor_critic_gae/python/plot.py runs/actor_critic_gae/simple_ac runs/actor_critic_gae/gae
```

The script does three things:

1. It compiles Tribes into `runs/actor_critic_gae/<name>/classes/` and copies `python/` next to it, so you can edit the repo while a run is going.
2. It runs `python/train.py` with the Python in `$PYTHON`. The default is the tic-tac-toe venv, which has jax, flax, optax, pyyaml and matplotlib.
3. `train.py` starts the Java game as a subprocess.

The run writes these files to `runs/actor_critic_gae/<name>/`:

- `config.yaml`
- `metrics.csv` and `eval.csv`, the greedy policy against random
- `ac_XXXXXX.pkl` and `latest.pkl`, which hold `{"params": {"theta", "phi"}, "config", ...}`
- `train.log`
- `metrics.png`

To continue from a checkpoint, pass `--init <ckpt.pkl>`. The weights are loaded, but the Adam state starts fresh.

### Playing a trained model in `Play` or `Tournament`

The player is registered as `"ACGAE"`. It starts `python/serve.py` with the checkpoint, so it needs a Python that has JAX:

```bash
TRIBES_PYTHON=/storage/scratch1/6/snayak89/tic-tac-toe/venv/bin/python \
java -Dtribes.acgae.model=runs/actor_critic_gae/simple_ac/latest.pkl -cp out:lib/json.jar Tournament tournament.json
```

Add `-Dtribes.acgae.greedy=false` to sample from pi. The default is argmax.

## Python files (`python/`)

| File | Role (tic-tac-toe counterpart) |
|---|---|
| [config.yaml](python/config.yaml) | `config.yaml`, plus the Tribes setup (`tribes`, `game_mode`, `level_seed`) |
| [networks.py](python/networks.py) | `networks.py`: `MLP` (critic) and `ActionScorer` (actor) |
| [agent.py](python/agent.py) | `agent.py`: `act`, `gae`, `actor_loss`, `critic_loss`, `update` |
| [rollout.py](python/rollout.py) | `rollout.py`: `play_vs_random`, `reward_to_go`, `to_batch` |
| [tribes_env.py](python/tribes_env.py) | `game.py` + `env_step`: `reset(seed)` / `step(action)` against the Java game |
| [train.py](python/train.py) | `train.py` + the variant's `__init__.py` (`collect`, `batch_stats`, `game_stats`) |
| [plot.py](python/plot.py) | `plot.py` |
| [serve.py](python/serve.py) | serves pi(.\|s) to the Java player for tournaments |
| [tribes_io.py](python/tribes_io.py) | reads and writes the binary pipe messages |

## Java files

| File | Role |
|---|---|
| [EnvServer.java](EnvServer.java) | The environment. Tribes' `Game` loop plays agent vs `RandomAgent` in a thread. The agent's seat hands each state to Python and waits for the chosen index. |
| [Features.java](Features.java) | State and action features, built from `actor_critic_mcts.encoding` and sent as sparse vectors |
| [ActorCriticGAEPlayer.java](ActorCriticGAEPlayer.java), [PythonPolicy.java](PythonPolicy.java) | `"ACGAE"` in `Run`: pi from `serve.py`, then argmax or sample |
| [ActorCriticGAEParams.java](ActorCriticGAEParams.java) | Java-side settings: greedy flag, model path, tribes, game mode, level seed |

## Mapping from tic-tac-toe to Tribes

| tic-tac-toe | here |
|---|---|
| a move | one Tribes action. Tribes asks for one action at a time until `END_TURN`, so a turn is several moves. |
| `env_step`: opponent replies at once | `env.step` plays on until the agent's next decision, including `RandomAgent`'s whole turn |
| agent is X or O at random | agent takes seat 0 or 1 at random (and so one of `tribes`) |
| `game.observation` (27 values) | `obs`: 121 tiles × 104 node features + 80 globals = 12,664 values, from the agent's point of view |
| 9 logits + `legal_mask` | `acts`: 316 features per legal action (A changes from state to state). Action lists are zero-padded and `legal_mask` marks the real ones, so the masking code is unchanged. |
| reward +1 / 0 / -1 at the end | WIN +1, LOSS -1 (otherwise 0). In SCORE mode the higher score at turn 30 wins. |

The episode is played by Tribes' own `Game` loop, which drives the forward model. `GameState.advance()` alone can't play a whole game: it passes the turn on but never increments the tick (`incTick` is package-private and only `Game` calls it), so the turn limit and per-turn income would never trigger.

### Things that behave differently because Tribes is bigger

- **`gamma^k` weighting.** `k` counts the agent's decisions, about 50 to 200 per game, and `0.99^150 ≈ 0.22`. Late decisions therefore get much smaller actor gradients than in tic-tac-toe, where k ≤ 4. This port keeps that weighting exactly.
- **Large action lists.** `SEND_STARS` and `DECLARE_WAR` make up a big share of the legal actions (see `actor_critic_mcts/encoding/README.md`), so an untrained policy picks them often.
- **Padded shapes.** Batches are padded to a power of two in both the transition and action dimensions. JAX compiles once per new shape, so the first iterations are slower.
