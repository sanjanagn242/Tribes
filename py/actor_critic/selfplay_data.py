"""
Loading self-play data written by players.actor_critic_mcts.selfplay.SelfPlay.

Each data folder (e.g. data/selfplay/iter_000/) holds:
    games_*.jsonl[.gz] : one TrainingSample per line
    schema.json        : feature sizes, column names and the node-id columns of the action features

A sample's flat `stateFeatures` = node features (N x F, row-major) + edge features (E x Fe) + globals (G),
where N = boardSize^2 and the edges are the 8-neighbour grid built by `edge_index(boardSize)`, in the same
order as StateEncoder.buildEdgeIndex in Java.

Parsed files are cached next to the data as <file>.npz (state features stored as float16), so a second
training run on the same data skips the JSON parsing.
"""

import glob
import gzip
import json
import os
from dataclasses import dataclass

import numpy as np
import torch

# Same order as StateEncoder.NEIGHBOURS in Java: edge ids depend on it.
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

_EDGE_CACHE = {}


def edge_index(size):
    """[2, E] int64 edge list (source row, target row) for a size x size board, node id = x * size + y."""
    if size not in _EDGE_CACHE:
        src, tgt = [], []
        for x in range(size):
            for y in range(size):
                for dx, dy in NEIGHBOURS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size:
                        src.append(x * size + y)
                        tgt.append(nx * size + ny)
        _EDGE_CACHE[size] = np.array([src, tgt], dtype=np.int64)
    return _EDGE_CACHE[size]


# --------------------------------------------------------------------------------------------- schema

@dataclass
class Schema:
    node_features: int
    edge_features: int
    global_features: int
    action_features: int
    src_col: int
    tgt_col: int
    max_players: int
    raw: dict

    @staticmethod
    def load(folder):
        with open(os.path.join(folder, "schema.json")) as f:
            return Schema.from_dict(json.load(f))

    @staticmethod
    def from_dict(s):
        """From the parsed schema.json (also stored in every checkpoint as ckpt["schema"])."""
        return Schema(s["nodeFeatureSize"], s["edgeFeatureSize"], s["globalFeatureSize"], s["actionFeatureSize"],
                      s["actionSrcNodeColumn"], s["actionTgtNodeColumn"], s["maxPlayers"], s)

    def split_state(self, flat, size):
        """flat [B, L] state features -> (x [B,N,F], edge_attr [B,E,Fe], u [B,G]) for a size x size board."""
        n, e = size * size, edge_index(size).shape[1]
        nf, ef = n * self.node_features, e * self.edge_features
        b = flat.shape[0]
        return (flat[:, :nf].reshape(b, n, self.node_features),
                flat[:, nf:nf + ef].reshape(b, e, self.edge_features),
                flat[:, nf + ef:])

    def layout_key(self):
        """Everything that must match between data folders (and between data and a trained model)."""
        return (self.node_features, self.edge_features, self.global_features, self.action_features,
                self.src_col, self.tgt_col, self.max_players)

    def state_length(self, size):
        n, e = size * size, edge_index(size).shape[1]
        return n * self.node_features + e * self.edge_features + self.global_features


# --------------------------------------------------------------------------------------------- files

def find_files(paths):
    """Expand files / folders into a sorted list of .jsonl / .jsonl.gz files (folders are searched recursively)."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)
            files += glob.glob(os.path.join(p, "**", "*.jsonl.gz"), recursive=True)
        elif os.path.isfile(p):
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return sorted(set(files))


def parse_file(path, use_cache=True):
    """Read one .jsonl[.gz] file into numpy arrays (cached as <path>.npz)."""
    cache = path + ".npz"
    if use_cache and os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        with np.load(cache) as d:
            return {k: d[k] for k in d.files}

    opener = gzip.open if path.endswith(".gz") else open
    states, sizes, acts, act_off, pis, z, root_v, game_id, player_id, tick = [], [], [], [0], [], [], [], [], [], []
    with opener(path, "rt") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if s["valueTarget"] is None:
                raise ValueError(f"{path}: sample without valueTarget (game {s['gameId']}, tick {s['tick']})")
            states.append(np.asarray(s["stateFeatures"], dtype=np.float16))
            sizes.append(s["boardSize"])
            a = np.asarray(s["actionFeatures"], dtype=np.float32)
            acts.append(a)
            act_off.append(act_off[-1] + len(a))
            pis.append(np.asarray(s["searchPolicy"], dtype=np.float32))
            z.append(s["valueTarget"])
            root_v.append(np.nan if s["rootValue"] is None else s["rootValue"])
            game_id.append(s["gameId"])
            player_id.append(s["playerID"])
            tick.append(s["tick"])

    if not states:
        raise ValueError(f"{path}: no samples")
    if len(set(len(x) for x in states)) != 1:
        raise ValueError(f"{path}: samples have different state lengths (mixed board sizes in one file)")

    out = dict(states=np.stack(states), board_size=np.asarray(sizes, np.int32),
               actions=np.concatenate(acts), act_off=np.asarray(act_off, np.int64),
               pi=np.concatenate(pis), z=np.asarray(z, np.float32), root_value=np.asarray(root_v, np.float32),
               game_id=np.asarray(game_id, np.int64), player_id=np.asarray(player_id, np.int32),
               tick=np.asarray(tick, np.int32))
    if use_cache:
        np.savez(cache, **out)
    return out


# --------------------------------------------------------------------------------------------- dataset

class Bucket:
    """All samples of one board size (they share a graph topology, so they can be batched as dense tensors)."""

    def __init__(self, size, parts):
        self.size = size
        self.states = np.concatenate([p["states"] for p in parts])
        self.z = np.concatenate([p["z"] for p in parts])
        self.root_value = np.concatenate([p["root_value"] for p in parts])
        self.game_key = np.concatenate([p["game_key"] for p in parts])
        # Re-base the per-file action offsets onto one concatenated action array.
        acts, pis, offs, base = [], [], [0], 0
        for p in parts:
            acts.append(p["actions"])
            pis.append(p["pi"])
            offs.append(p["act_off"][1:] + base)
            base += p["act_off"][-1]
        self.actions = np.concatenate(acts)
        self.pi = np.concatenate(pis)
        self.act_off = np.concatenate([np.array([0], np.int64)] + [o for o in offs[1:]])

    def __len__(self):
        return len(self.z)


class SelfPlayDataset:
    """Samples from one or more self-play folders/files, grouped by board size."""

    def __init__(self, paths, use_cache=True, verbose=True):
        files = find_files(paths)
        if not files:
            raise FileNotFoundError(f"No .jsonl / .jsonl.gz files under {paths}")

        self.schema = None
        by_size = {}
        for file_idx, path in enumerate(files):
            schema = Schema.load(os.path.dirname(path))
            if self.schema is None:
                self.schema = schema
            elif schema.layout_key() != self.schema.layout_key():
                raise ValueError(f"{path}: feature layout differs from the first data folder; "
                                 f"don't mix data from different encoder versions")
            d = parse_file(path, use_cache)
            size = int(d["board_size"][0])
            if d["states"].shape[1] != schema.state_length(size):
                raise ValueError(f"{path}: state length {d['states'].shape[1]} != expected "
                                 f"{schema.state_length(size)} for board {size}")
            # Unique id per (file, game) so the train/val split never splits one game.
            d["game_key"] = file_idx * 1_000_000 + d["game_id"]
            by_size.setdefault(size, []).append(d)
            if verbose:
                print(f"  {path}: {len(d['z'])} samples, board {size}x{size}")

        self.buckets = {size: Bucket(size, parts) for size, parts in sorted(by_size.items())}

    def __len__(self):
        return sum(len(b) for b in self.buckets.values())

    def split(self, val_frac, seed=0):
        """Hold out whole games: returns (train, val) lists of (size, sample indices)."""
        rng = np.random.default_rng(seed)
        train, val = [], []
        for size, b in self.buckets.items():
            games = np.unique(b.game_key)
            rng.shuffle(games)
            n_val = int(round(len(games) * val_frac))
            if val_frac > 0 and n_val == 0 and len(games) > 1:
                n_val = 1
            val_games = set(games[:n_val].tolist())
            is_val = np.array([g in val_games for g in b.game_key])
            train.append((size, np.nonzero(~is_val)[0]))
            val.append((size, np.nonzero(is_val)[0]))
        return train, val

    def batches(self, split, batch_size, shuffle, rng=None):
        """Yield dict batches; every batch holds samples of a single board size."""
        plan = []
        for size, idx in split:
            idx = rng.permutation(idx) if shuffle else idx
            plan += [(size, idx[i:i + batch_size]) for i in range(0, len(idx), batch_size)]
        if shuffle:
            plan = [plan[i] for i in rng.permutation(len(plan))]
        for size, idx in plan:
            yield self.make_batch(size, idx)

    def make_batch(self, size, idx):
        b, s = self.buckets[size], self.schema
        flat = torch.from_numpy(b.states[idx].astype(np.float32))
        x, edge_attr, u = s.split_state(flat, size)

        counts = b.act_off[idx + 1] - b.act_off[idx]
        a_max = int(counts.max())
        actions = np.zeros((len(idx), a_max, s.action_features), np.float32)
        pi = np.zeros((len(idx), a_max), np.float32)
        mask = np.zeros((len(idx), a_max), bool)
        for row, (i, c) in enumerate(zip(idx, counts)):
            lo = b.act_off[i]
            actions[row, :c] = b.actions[lo:lo + c]
            pi[row, :c] = b.pi[lo:lo + c]
            mask[row, :c] = True

        return dict(
            x=x, edge_attr=edge_attr, u=u,
            actions=torch.from_numpy(actions),
            action_mask=torch.from_numpy(mask),
            pi=torch.from_numpy(pi),
            z=torch.from_numpy(b.z[idx]),
        )
