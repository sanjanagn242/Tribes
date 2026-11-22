# Actor-critic GNN training

These scripts train the graph network that guides `players.actor_critic_mcts`, using the self-play data written by `SelfPlay` ([Java README](../../src/players/actor_critic_mcts/README.md)). For the feature layout, see the [encoding README](../../src/players/actor_critic_mcts/encoding/README.md).

| File | Role |
|------|------|
| [train_gnn.py](train_gnn.py) | Training script: loss, optimiser, validation, checkpoints, metrics |
| [gnn_model.py](gnn_model.py) | `TribesGNN`, the model (plain PyTorch) |
| [selfplay_data.py](selfplay_data.py) | Reads `games_*.jsonl[.gz]` and `schema.json`, rebuilds the graphs, and makes batches |
| [serve_gnn.py](serve_gnn.py) | Inference server that the Java `ExternalNetwork` starts as a subprocess. Binary protocol over stdin/stdout, documented at the top of the file |

Requirements: `torch` and `numpy` (PyTorch Geometric isn't needed).

## Usage

```bash
# iteration 0: train on the uniform-network self-play data
python py/actor_critic/train_gnn.py --data data/selfplay/iter_000 --out models/iter_001

# later iterations: warm-start from the previous model and train on the last few iterations
python py/actor_critic/train_gnn.py --data data/selfplay/iter_001 data/selfplay/iter_002 \
    --init-from models/iter_002/best.pt --out models/iter_003
```

Main options:

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 20 | number of passes over the training data |
| `--batch-size` | 64 | samples per batch |
| `--lr` | 1e-3 | peak learning rate |
| `--hidden` | 128 | embedding size |
| `--layers` | 4 | number of message-passing rounds |
| `--value-weight` | 1.0 | weight of the value loss |
| `--val-frac` | 0.1 | fraction of games held out for validation |
| `--device` | auto | `cpu`, `cuda`, or `auto` (uses a GPU if there is one) |
| `--threads` | torch default | CPU threads |
| `--no-cache` | off | don't read or write the `.npz` caches |

Run `--help` for the full list.

## The model

1. **Embed:** each tile, edge and the global vector go through their own small MLP, giving `hidden`-sized embeddings.
2. **Message passing:** this is repeated `--layers` times.
   - Each edge s → t builds a message from (tile s, tile t, edge features).
   - Each tile averages its incoming messages, then updates itself using them and the global vector.
   - The global vector is updated from the average tile.
   - Every update is residual, followed by LayerNorm.
3. **Value:** a graph embedding is built from the mean tile, the max tile and the global vector. An MLP with tanh turns it into a value in [-1, 1].
4. **Policy:** each action is scored by an MLP over four inputs:
   - the embedding of its source tile;
   - the embedding of its target tile;
   - its own features, minus the two node-id columns;
   - the graph embedding.

   Actions without a tile get zeros in place of that tile's embedding. Padding actions get a logit of -inf, so softmax gives them probability 0.

All graphs of the same board size share one edge list, so each batch holds a single board size. The data loader groups samples by size, so you can mix 2-player and 3-player data.

## Loss and metrics

The loss is `CE(pi, softmax(logits)) + value_weight * MSE(z, value)`.

- `pi` is the search policy: the root visit proportions.
- `z` is the game outcome for the recorded player, in [-1, 1].

Columns in `metrics.csv`, given for both train and val:

| Column | Meaning |
|---|---|
| `policy` | cross-entropy against `pi` |
| `kl` | policy cross-entropy minus the entropy of `pi`. 0 means the policy matches `pi` exactly; `policy` alone can never go below the entropy of `pi` |
| `value` | MSE of the value against `z` |
| `top1` | how often the top-scored action is the most-visited one |
| `value_sign` | how often the value predicts the right winner |

The validation split holds out **whole games**. Samples from one game share the same `z` and very similar states, so splitting by sample would leak answers into validation.

## Outputs

These go in `--out`:
- `best.pt`: the epoch with the lowest validation loss;
- `last.pt`: the final epoch;
- `metrics.csv`: one row per epoch.

A checkpoint holds the weights, `model_config` (enough to rebuild `TribesGNN`), the data's `schema.json` and the training arguments. To reload one:

```python
ckpt = torch.load("models/iter_001/best.pt", weights_only=False)
model = TribesGNN(**ckpt["model_config"]); model.load_state_dict(ckpt["model_state"]); model.eval()
```

## Notes

- **Cache files:** the first time a data file is read, a `<file>.npz` cache is written next to it, with states stored as float16. Later runs load the cache instead of parsing JSON again. Delete the caches if you regenerate the data in place (they're refreshed automatically when the data file is newer).
- **Memory:** on an 11×11 board, the state is about 40 KB per sample in memory, so 100 games (about 40k samples) need around 1.6 GB.
- **Speed:** on CPU, an epoch over 2,000 samples took about 30 s with 8 threads. For real datasets, run on a GPU node.
- **Feature layout:** the training script refuses to mix data folders whose feature layouts differ, and to warm-start a model trained on a different layout. If you change either encoder, regenerate the self-play data.
- **Still TODO:** `rootValue` isn't used yet; it's only on the critic's scale once `USE_VALUE_NET` is on.
- **Using a model from Java:** pass `modelPath=<checkpoint>` to `SelfPlay`. `ExternalNetwork` runs `serve_gnn.py` for you; see the Java README's "Using a trained model" section.
