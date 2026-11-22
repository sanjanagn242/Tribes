# Actor-Critic MCTS (exercise skeleton)

`players.actor_critic_mcts` is a copy of [`players.mcts`](../mcts/README.md) with hooks for an actor-critic network:

- the **actor** (policy head) gives priors P(s,a) and turns UCT into **PUCT**;
- the **critic** (value head) evaluates leaves instead of the heuristic or a rollout;
- **self-play** records (state, search policy, final outcome) samples to train the network.

The copied MCTS code works unchanged. Every new function is an empty stub marked `TODO`. Non-void stubs throw `UnsupportedOperationException`, so you'll know straight away if you call one before writing it. When all the new flags in `ACMCTSParams` are `false` (the default), the agent behaves exactly like plain MCTS.

## Layout

| File | Role |
|------|------|
| [ACMCTSPlayer.java](ACMCTSPlayer.java) | Agent. Copy of `MCTSPlayer` plus `setNetwork`, `startRecording`, `recordSample`, `sampleActionByVisits` and `result` |
| [ACMCTSParams.java](ACMCTSParams.java) | Copy of `MCTSParams` plus `USE_PRIORS`, `USE_VALUE_NET`, `C_PUCT`, Dirichlet noise, temperature and `RECORD_SAMPLES` |
| [ACTreeNode.java](ACTreeNode.java) | Copy of `SingleTreeNode` plus `priors[]`, `initRootPriors`, `computePriors`, `puctValue`, `evaluateWithCritic`, `getRootVisitCounts` and `getRootValue` |
| [search/PUCT.java](search/PUCT.java) | Stateless maths: PUCT score, Dirichlet noise, visit distribution and sampling |
| [network/ActorCriticNetwork.java](network/ActorCriticNetwork.java) | Interface: `evaluate(gs, actions, playerID) -> PolicyValue` |
| [network/PolicyValue.java](network/PolicyValue.java) | `priors[]` + `value` |
| [network/UniformNetwork.java](network/UniformNetwork.java) | Baseline: uniform priors and a neutral value |
| [network/ExternalNetwork.java](network/ExternalNetwork.java) | The trained GNN. It starts `py/actor_critic/serve_gnn.py` as a subprocess and talks to it over stdin/stdout. See [Using a trained model](#using-a-trained-model) |
| [encoding/StateEncoder.java](encoding/StateEncoder.java) | `GameState` → graph: one node per tile, 8-neighbour edges, global vector. See [encoding/README.md](encoding/README.md) |
| [encoding/GraphState.java](encoding/GraphState.java) | Holds the encoded graph (node, edge and global features) |
| [encoding/ActionEncoder.java](encoding/ActionEncoder.java) | `Action` → sparse feature vector that points at graph nodes. See [encoding/README.md](encoding/README.md) |
| [records/TrainingSample.java](records/TrainingSample.java) | One training example, with `toJson()` |
| [records/GameRecord.java](records/GameRecord.java) | One player's samples for one game. `assignOutcome` fills in the value targets |
| [records/RecordWriter.java](records/RecordWriter.java) | Writes records as JSON Lines |
| [selfplay/SelfPlayConfig.java](selfplay/SelfPlayConfig.java) | Number of games, seeds, tribes, budget and paths |
| [selfplay/SelfPlay.java](selfplay/SelfPlay.java) | `main`: build the network, play games headless and write the records |

## Encoding

The state is encoded as a graph and each action as a sparse feature vector. The full feature layout and how to rebuild the tensors in Python are in [encoding/README.md](encoding/README.md).

## Data flow

```
SelfPlay.playGame
  └─ Game.run(null,null)
       └─ ACMCTSPlayer.act ──► ACTreeNode.initRootPriors ──► network.evaluate (actor)
            │                  ACTreeNode.mctsSearch
            │                     ├─ uct()   → puctValue  (USE_PRIORS)
            │                     ├─ expand  → computePriors
            │                     └─ rollOut → evaluateWithCritic (USE_VALUE_NET)
            ├─ recordSample → GameRecord.add(TrainingSample{state, pi = visits, ...})
            └─ sampleActionByVisits (temperature)
       └─ Agent.result ──► GameRecord.assignOutcome (value target z)
  └─ RecordWriter.writeAll ──► data/selfplay/iter_XXX/*.jsonl ──► (train in Python) ──► model_{k+1}
```

## Suggested order

1. `search/PUCT.java`: pure functions, so you can unit-test them on their own.
2. `network/UniformNetwork.java`, then `ACTreeNode` (`computePriors`, `initRootPriors` and `puctValue`). Turn on `USE_PRIORS` with the uniform network and check that it plays about as well as plain MCTS. Also work out how PUCT should handle opponent nodes; the copied code minimises there (see the comment on `puctValue`).
3. `encoding/StateEncoder.java` and `encoding/ActionEncoder.java`: choose scheme A or B.
4. `records/*`, then `ACMCTSPlayer` (`startRecording`, `recordSample`, `sampleActionByVisits` and `result`).
5. `selfplay/*`: generate iteration 0 with `UniformNetwork`.
6. Write a Python training script for the JSON Lines files. Loss = cross-entropy(pi, policy) + MSE(z, value).
7. `network/ExternalNetwork.java` and `ACTreeNode.evaluateWithCritic`, then iterate: self-play → train → evaluate against MCTS.

## Running self-play

```bash
javac -cp lib/json.jar -d out/classes $(find src -name "*.java")
java -cp out/classes:lib/json.jar players.actor_critic_mcts.selfplay.SelfPlay [config.json] [key=value ...]
```

With no arguments, it plays 100 two-player games (Xin-Xi vs Imperius, capitals mode, 2000 forward-model calls per move) with `UniformNetwork`, and writes the results to `data/selfplay/iter_000/`.

To change a setting, pass `key=value` pairs, a JSON file with the same keys, or both. Later values override earlier ones. All keys are listed in [SelfPlayConfig.java](selfplay/SelfPlayConfig.java). For example:

```bash
# a quick test run
... SelfPlay numGames=3 gameMode=SCORE fmCallsPerMove=300
# iteration 1, with a trained model and three tribes
... SelfPlay iteration=1 modelPath=models/model_1.onnx tribes=XIN_XI,IMPERIUS,BARDUR
```

**Search settings during self-play**
- `PRIORITIZE_ROOT` is off, so the policy target covers every legal action.
- Priors are on, and Dirichlet noise is added at the root.
- Moves are sampled from the visit counts (temperature `temperature`) for the first `tempMoves` actions of each game. After that, the most-visited action is played.
- The critic (`USE_VALUE_NET`) is used only when `modelPath` is set. With `UniformNetwork`, leaves are evaluated with the DIFF heuristic instead.

**Output files** (in `<outputDir>/iter_<NNN>/`)
- `games_0000-0099.jsonl.gz`: one sample per line, gzip-compressed (about 2 KB per sample). Set `compress=false` for plain `.jsonl`. Each line has these keys:
  - `gameId`, `playerID`, `tick`, `boardSize`;
  - `stateFeatures`: the flat graph (see [encoding/README.md](encoding/README.md));
  - `actionFeatures`: one row per root action;
  - `searchPolicy`: the root visit proportions, the policy target;
  - `valueTarget`: the value target;
  - `rootValue`: the mean backed-up value at the root.
- `schema.json`: the config used, feature sizes, the action node-id columns, and the name of every feature column.

**Value targets** are on a [-1, 1] scale: a win is 1 and a loss is -1. If a game ends without a result for a player, the value is set by score rank, from 0.5 (best) to -0.5 (worst). Every sample in a game gets that game's final outcome, with no discounting.

**`rootValue`** is on the critic's scale only when `USE_VALUE_NET` is on. In iteration 0 it is a raw DIFF-heuristic value (for example -5 to 34), so normalise it before mixing it into the value target.

**Speed:** a 300-call test game takes 3–12 s. At the default 2000 calls, expect several times longer per game.

**Training:** [py/actor_critic/](../../../py/actor_critic/README.md) has the GNN training script for this data.

## Training on the cluster

[slurm/](slurm/README.md) runs the whole loop on SLURM: each iteration's self-play is a CPU array job (`n = ceil(games / nodes)` games per node), then the network is trained on a GPU, and the next iteration starts. One command submits the chain: `slurm/submit.sh <run_name>`.

## Training on a local machine

[local/run_loop.sh](local/run_loop.sh) runs the same loop on one machine, such as a desktop with a GPU. Each iteration runs self-play as several parallel `SelfPlay` processes, then trains on the GPU. If it's interrupted, rerun the same command and it continues where it stopped. Each iteration adds a line to `runs/<name>/progress.csv`. Commands are in `my_cmds.txt`, section 6c.

## Using a trained model

Pass a checkpoint from `train_gnn.py` as `modelPath`, and run from the repository root:

```bash
java -cp out/classes:lib/json.jar players.actor_critic_mcts.selfplay.SelfPlay iteration=1 modelPath=models/iter_001/best.pt
```

`ExternalNetwork.load()` starts `python3 py/actor_critic/serve_gnn.py --checkpoint <modelPath>`. That process loads the same PyTorch model used in training and answers each `evaluate()` call over stdin/stdout. When it starts, it reports the feature layout the model was trained on, and `load()` refuses a model whose layout doesn't match the current encoders. The server's logs and any Python errors appear in Java's stderr.

**Settings** (Java system properties):

| Property | Default | Meaning |
|---|---|---|
| `tribes.python` | `$TRIBES_PYTHON`, else `python3` | Python executable |
| `tribes.gnn.server` | `py/actor_critic/serve_gnn.py` | path to the server script |
| `tribes.gnn.threads` | 4 | torch CPU threads |
| `tribes.gnn.device` | `cpu` | torch device |

**Perspective.** Every training sample was recorded on the acting player's own turn, so `evaluate()` encodes each state from the acting tribe's point of view, including at opponent nodes in the tree. If the acting tribe isn't the searching player, the value is negated to give the searching player's point of view. That's exact for two players and an approximation with three or more.

**Speed.** One call takes about 4 ms (encoding plus the round trip to Python, 4 threads, 11×11 board). In a test at 300 forward-model calls per move, a game made about 9,000 calls and took about 40 s, compared with 3–5 s using `UniformNetwork`. `SelfPlay` prints the number of calls and their mean time at the end of a run.

The agent is also registered in `Run` as `AC_MCTS`, for `play.json` and `tournament.json`. There it still uses `UniformNetwork`; `Run` doesn't load a model yet.
