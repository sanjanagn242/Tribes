# Tribes Actor-Critic GNN: Design Notes

These notes describe the graph neural network that guides `players.actor_critic_mcts`: how a game state and its legal actions become tensors, what every layer does, every parameter (with the reasoning behind its value), how the model is trained, and how Java uses it during search.

Source files:

| Part | File |
|---|---|
| State → graph | [src/players/actor_critic_mcts/encoding/StateEncoder.java](../../src/players/actor_critic_mcts/encoding/StateEncoder.java) |
| Graph container / flattening | [src/players/actor_critic_mcts/encoding/GraphState.java](../../src/players/actor_critic_mcts/encoding/GraphState.java) |
| Action → feature vector | [src/players/actor_critic_mcts/encoding/ActionEncoder.java](../../src/players/actor_critic_mcts/encoding/ActionEncoder.java) |
| Model | [py/actor_critic/gnn_model.py](gnn_model.py) |
| Data loading / batching | [py/actor_critic/selfplay_data.py](selfplay_data.py) |
| Training | [py/actor_critic/train_gnn.py](train_gnn.py) |
| Inference server | [py/actor_critic/serve_gnn.py](serve_gnn.py) + [ExternalNetwork.java](../../src/players/actor_critic_mcts/network/ExternalNetwork.java) |
| Value targets | [src/players/actor_critic_mcts/records/GameRecord.java](../../src/players/actor_critic_mcts/records/GameRecord.java) |

All sizes below were dumped from the actual encoders (`nodeFeatureNames()` etc.), not copied from comments. Parameter counts are computed for the default `hidden=128, layers=4`.

---

## 1. The big picture

```
                 Java (game engine)                                   Python (PyTorch)
 ┌────────────────────────────────────────────┐        ┌───────────────────────────────────────────────┐
 │ GameState ──StateEncoder──► GraphState      │        │                                               │
 │   nodes  [N,104]  (one per tile)            │ flat   │  split_state ─► x [B,N,104]                    │
 │   edges  [E,9]    (8-neighbour grid)        │ float  │                 edge_attr [B,E,9]              │
 │   global [80]     (tribe-level)             ├───────►│                 u [B,80]                       │
 │                                             │ vector │                                               │
 │ legal actions ──ActionEncoder──► [A,110]    ├───────►│  actions [B,A,110], action_mask [B,A]          │
 │                                             │        │                    │                          │
 │                                             │        │               TribesGNN                       │
 │  ACTreeNode (MCTS)                          │◄───────┤  logits [B,A]  (one per legal action)         │
 │    priors = softmax(logits)  -> PUCT        │        │  value  [B]    (tanh, in [-1,1])               │
 │    value  -> leaf evaluation                │        │                                               │
 └────────────────────────────────────────────┘        └───────────────────────────────────────────────┘
```

The network is AlphaZero-style: one shared trunk, two heads.

- **Actor (policy head):** a score for each legal action. Softmax over them gives the search priors P(s,a).
- **Critic (value head):** one number in [-1, 1], the expected outcome for the player being encoded.

It is trained on self-play data: the policy target is the MCTS root visit distribution π, and the value target is the final game result z.

---

## 2. Why a graph network?

The state is a square tile grid, so a CNN would be the obvious choice. A GNN was picked for these reasons:

1. **Board size varies** (8×8, 11×11, 14×14, 16×16 depending on the number of tribes and the level). Message passing uses the same weights on any graph, so one model covers every size with no padding or resizing. The grid topology is rebuilt from the board size alone.
2. **Relations between tiles matter and change during a game:** road links, territory borders, coastlines (embark/disembark), "unit can step here". These are naturally **edge features**. A CNN would have to infer them from pairs of channels.
3. **Actions point at tiles.** A MOVE goes from tile s to tile t; a SPAWN happens at a city tile. With one embedding per tile, the policy head can **gather** the embeddings of exactly the tiles an action touches (a pointer-network idea). This is what makes a variable-length, state-dependent action list workable (section 6).
4. **Tribe-level information** (stars, techs, diplomacy) belongs to no tile. It lives in a separate **global vector** that every node reads and writes on every layer.

It is written in plain PyTorch (no `torch_geometric`). The grid is regular and every graph in a batch has the same topology, so dense tensors `[B, N, H]` and an `index_add` are enough. This keeps the dependencies down to `torch` + `numpy`.

---

## 3. State encoding (Java → graph)

### 3.1 Perspective: relative tribe slots

Everything is encoded **from one player's point of view** (`playerID`):

```
slot 0 = playerID itself ("self")
slot k = tribe (playerID + k) mod numTribes   ("opp1", "opp2", "opp3")
```

**Why:** the network never needs to know whether it is "tribe 0" or "tribe 2". "Mine" vs "theirs" is always in the same columns, so one network can play any seat, and data from every seat trains the same weights. The layout reserves `DEFAULT_MAX_PLAYERS = 4` slots; games with fewer tribes leave the extra slots at zero. This number must stay fixed for a training run because it changes the feature sizes.

### 3.2 Normalisation

Every count or stat is divided by a scale and **clipped to [0, 1]**. One-hots and flags are already 0/1. The exceptions are allegiances (in [-1, 1]) and edge directions (in {-1, 0, 1}).

| Constant | Value | Applied to | Reasoning |
|---|---|---|---|
| `TURN_SCALE` | 50 | turn number | CAPITALS games last up to 50 turns. SCORE games (30 turns) only reach 0.6 |
| `STARS_SCALE` | 100 | own stars | stars rarely exceed 100 in a 30-turn game |
| `SCORE_SCALE` | 10000 | own / opponent score | typical end scores are a few thousand |
| `PRODUCTION_SCALE` | 30 | stars per turn | |
| `COUNT_CITIES_SCALE` | 15 | own / seen-opponent city count | |
| `COUNT_UNITS_SCALE` | 30 | own unit count | |
| `KILLS_SCALE` | 20 | tribe kills | |
| `UNIT_STAT_SCALE` | 5 | ATK, DEF, MOV, RANGE | every unit's stats are ≤ 5 (the super unit's ATK 5 is the max), so nothing is clipped |
| `UNIT_HP_SCALE` | 40 | max HP | super unit has 40 |
| `UNIT_KILLS_SCALE` | 3 | unit kills | 3 kills → can become veteran, so 1.0 means "eligible" |
| `CITY_LEVEL_SCALE` | 10 | city level, population | |
| `CITY_BOUND_SCALE` | 3 | city border radius | |
| `CITY_UNITS_SCALE` | 10 | units supported by a city | |
| `CITY_POINTS_SCALE` | 1000 | points a city is worth | |

**Why clip to [0, 1]:** features on a similar scale keep the first Linear layers well-conditioned, and clipping stops a rare extreme value (a 200-star hoard) from blowing up activations. It also means the cached states can be stored as **float16** safely (section 8.2).

### 3.3 Nodes: 104 features per tile

Node id = `x * boardSize + y`, the same `[x][y]` indexing as `core.game.Board`. An 11×11 board has N = 121 nodes.

| Cols | Block | Size | Contents |
|---|---|---|---|
| 0–1 | `pos` | 2 | x/(size−1), y/(size−1) |
| 2 | `visible` | 1 | 1 if the tile is not fogged |
| 3–10 | `terrain` | 8 | one-hot: PLAIN, SHALLOW_WATER, DEEP_WATER, MOUNTAIN, VILLAGE, CITY, FOREST, **FOG** |
| 11–17 | `resource` | 7 | one-hot: FISH, FRUIT, ANIMAL, WHALES, ORE, CROPS, RUINS |
| 18–36 | `building` | 19 | one-hot: PORT, MINE, FORGE, FARM, WINDMILL, CUSTOMS_HOUSE, LUMBER_HUT, SAWMILL, 4 temples, 7 monuments |
| 37 | `has_building` | 1 | any building present |
| 38 | `road` | 1 | road on the tile |
| 39 | `trade_network` | 1 | on the trade network (road / city / port) |
| 40–45 | `territory_owner` | 6 | one-hot: none, self, opp1, opp2, opp3, **unknown** (owned, but the owning city isn't visible) |
| 46–54 | `city` | 9 | is_city_centre, level, pop_progress, population, is_capital, has_walls, bound, num_units, points_worth |
| 55–58 | `city_owner` | 4 | slot one-hot (city centres only) |
| 59 | `unit_present` | 1 | |
| 60–63 | `unit_owner` | 4 | slot one-hot |
| 64–75 | `unit_type` | 12 | WARRIOR … SUPERUNIT |
| 76–85 | `unit` | 10 | hp_frac, max_hp, atk, def, mov, range, kills, veteran, is_ranged, is_water |
| 86–91 | `unit_status` | 6 | FRESH, MOVED, ATTACKED, MOVED_AND_ATTACKED, PUSHED, FINISHED |
| 92–103 | `unit_carried` | 12 | land unit inside a BOAT/SHIP/BATTLESHIP |

Design notes:

- **Position is included** even though GNNs are usually position-agnostic. In Tribes, distance to the edge and corners matters (expansion room, map edge as a wall), and border tiles have fewer neighbours. With mean aggregation (section 5.2), the degree is otherwise invisible. Coordinates are normalised by `size − 1`, so they mean "relative position" on any board size.
- **One-hots for every enum**, not ordinal integers. Unit types and buildings have no natural order, and an integer code would imply one.
- **`unit_status`** matters a lot for the policy: a FINISHED unit has no actions left, and a MOVED unit can still attack.
- **`unit_carried`**: a boat's real threat is the land unit inside it. Without this block, a boat with a knight would look identical to a boat with a warrior.
- **`has_building`** duplicates "any of cols 18–36". It costs one column and gives the network a direct presence flag.

### 3.4 Fog of war

The `GameState` an agent receives is already a partial-observation copy (unseen terrain = FOG, enemy stars/techs zeroed). On a fogged tile, the encoder writes only the **position and the FOG one-hot**, then returns. Everything else stays 0, so hidden information can't leak in. Other hidden-information choices:

- `territory_owner:unknown` separates "unowned" from "owned by someone I can't see".
- Enemy `city:num_units` is 0 (the engine hides it).
- Opponent `num_cities_seen` counts only visible cities.

Note: `Constants.PLAY_WITH_FULL_OBS` is `true` by default, so in current runs the FOG features are always 0. Set it to `false` to train under fog.

### 3.5 Edges: 9 features per directed edge

Edges link each tile to its **8 neighbours** (Tribes movement and vision use the 3×3 neighbourhood). They are **directed**, so (u→v) and (v→u) are two edges with their own features. The topology depends only on the board size:

| Board | N | E |
|---|---|---|
| 8×8 | 64 | 420 |
| 11×11 | 121 | 840 |
| 14×14 | 196 | 1,404 |
| 16×16 | 256 | 1,860 |

Edge ordering (must match between Java and Python, since edge features are stored by position):

```
for x in 0..size-1:
  for y in 0..size-1:
    for (dx,dy) in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
      if (x+dx, y+dy) on board: edge (x*size+y) -> ((x+dx)*size+(y+dy))
```

```
     neighbourhood of tile t          edge features of s→t
   ┌─────┬─────┬─────┐
   │ ↘   │  ↓  │   ↙ │       diagonal      : dx≠0 and dy≠0
   ├─────┼─────┼─────┤       road_link     : both on trade network
   │ →   │  t  │   ← │       land_water    : one water, one land
   ├─────┼─────┼─────┤       same_city     : both inside one city's borders
   │ ↗   │  ↑  │   ↖ │       border        : territory owner differs
   └─────┴─────┴─────┘       touches_fog   : either end unseen
   (8 incoming edges)         unit_to_free  : unit on s, t visible, empty, terrain-compatible
                              dx, dy        : direction, raw in {-1,0,1}
```

| Col | Feature | Reasoning |
|---|---|---|
| 0 | `diagonal` | redundant with |dx|·|dy|, but a direct flag is cheap |
| 1 | `road_link` | roads double movement; a "fast lane" the network should see directly |
| 2 | `land_water` | coastline crossing = embark/disembark, which changes unit type and movement |
| 3 | `same_city` | groups tiles of one city, so messages can spread city-level information |
| 4 | `border` | frontier between territories: where conflict happens |
| 5 | `touches_fog` | marks the edge of known space; all other features are skipped for fogged edges |
| 6 | `unit_to_free` | approximate "a unit on s could move to t" (land→land, water→water, or into a city). It hints at the action space without enumerating it |
| 7–8 | `dx`, `dy` | direction, so messages can be anisotropic (e.g. "enemy to the north-east") |

**Why edge features at all**, when the node features at both ends are already in the message: most of these are functions of the two nodes, so in principle the message MLP could learn them. Giving them explicitly saves the network from learning pairwise logic (owner A ≠ owner B, water XOR land) from scratch, which helps a lot with limited self-play data.

### 3.6 Globals: 80 features

| Cols | Block | Size | Contents |
|---|---|---|---|
| 0 | `turn` | 1 | tick / 50 |
| 1–2 | `game_mode` | 2 | CAPITALS, SCORE |
| 3 | `leveling_up` | 1 | a city is choosing a level-up bonus |
| 4–7 | `active_tribe` | 4 | slot of the tribe whose turn it is |
| 8 | `num_tribes` | 1 | nTribes / 4 |
| 9–15 | `own` | 7 | stars, score, production, num_cities, num_units, kills, num_techs (/24) |
| 16–39 | `tech` | 24 | one flag per researched technology |
| 40–51 | `tribe_type` | 12 | XIN_XI … YADAKK (each tribe has a different starting tech/unit) |
| 52–58 | `monument` | 7 | 0 unavailable, 0.5 available, 1 built |
| 59–65 | `opp1` | 7 | present, alive, met, score, num_cities_seen, allegiance me→them, them→me |
| 66–72 | `opp2` | 7 | same |
| 73–79 | `opp3` | 7 | same |

**Why:** stars and techs decide which actions are possible and good, but they live on no tile. The `present` flag lets the network tell "opponent slot unused" apart from "opponent with zero score".

### 3.7 Flat wire / storage format

`GraphState.flatten()` = node features row by row, then edge features row by row, then globals:

```
| x: N×104 | edge_attr: E×9 | u: 80 |
```

Length on 11×11 = 121·104 + 840·9 + 80 = **20,224 floats**. (8×8: 10,516; 14×14: 33,100; 16×16: 43,444.) `edge_index` is **not** stored; Python rebuilds it from the board size (`selfplay_data.edge_index`). Each sample is about 40 KB in memory as float16.

---

## 4. Action encoding (scheme B: action features)

### 4.1 Why per-action features instead of a fixed output head

Tribes has a **variable, state-dependent** list of legal actions (tens to hundreds per state). Its parameters include units, cities, tiles, techs, building types, star amounts and target tribes. A fixed index space ("scheme A": one logit per (type, tile, target, extra) combination) would be huge and mostly masked, and it would change with the board size. Instead, each legal action gets a **fixed-length feature vector**, and the network scores every (state, action) pair. Output size = number of legal actions, whatever that is. Scheme A's `toIndex` / `actionSpaceSize` are still TODO stubs and are not used.

### 4.2 Layout: 110 features per action

| Cols | Block | Size | Filled for |
|---|---|---|---|
| 0–24 | `action_type` | 25 | every action (BUILD … UPGRADE_SHIP) |
| 25 | `has_src` | 1 | 1 if the action has a source tile |
| **26** | `src_node` | 1 | **source node id** (−1 if none). `srcNodeColumn()` |
| 27 | `has_tgt` | 1 | 1 if the action has a target tile |
| **28** | `tgt_node` | 1 | **target node id** (−1 if none). `tgtNodeColumn()` |
| 29–30 | `dx`, `dy` | 2 | (target − source) / (size − 1), if both exist |
| 31 | `star_cost` | 1 | stars the action costs / 20 |
| 32–50 | `building` | 19 | BUILD |
| 51–57 | `resource` | 7 | RESOURCE_GATHERING |
| 58–69 | `unit_type` | 12 | SPAWN |
| 70–77 | `level_up` | 8 | LEVEL_UP (WORKSHOP, EXPLORER, CITY_WALL, RESOURCES, POP_GROWTH, BORDER_GROWTH, PARK, SUPERUNIT) |
| 78–101 | `tech` | 24 | RESEARCH_TECH |
| 102 | `tech_tier` | 1 | tier / 3 |
| 103–106 | `target_tribe` | 4 | DECLARE_WAR, SEND_STARS (relative slot) |
| 107 | `stars_sent` | 1 | SEND_STARS, / 50 |
| 108–109 | `capture` | 2 | CAPTURE: city / village |

The layout is **sparse**: blocks that don't apply stay 0, and the `action_type` one-hot tells the network which blocks are live. Two actions of the same type from the same tile (e.g. `SPAWN(Warrior)` vs `SPAWN(Archer)`) differ only in their parameter block and `star_cost`.

`CLIMB_MOUNTAIN` exists in `Types.ACTION`, but the engine never generates it (climbing is a normal MOVE once CLIMBING is researched), so column 22 is always 0.

### 4.3 Which tiles an action points at

| Action(s) | Source | Target |
|---|---|---|
| City actions (BUILD, RESOURCE_GATHERING, BURN/CLEAR/GROW_FOREST, DESTROY) | city tile | `targetPos` |
| SPAWN, LEVEL_UP | city tile | city tile |
| MOVE | unit tile | destination |
| ATTACK, CONVERT | unit tile | target unit's tile |
| Other unit actions (CAPTURE, EXAMINE, RECOVER, HEAL_OTHERS, DISBAND, MAKE_VETERAN, UPGRADE_*) | unit tile | same tile |
| BUILD_ROAD | — | road tile |
| RESEARCH_TECH, END_TURN, DECLARE_WAR, SEND_STARS | — | — |

**Why the pointers:** the node ids are not features to learn from. They are **indices** used to gather tile embeddings from the GNN output (section 6). That's why they're stored as raw floats (e.g. 57.0) and are **removed** before the action MLP. Feeding a raw id like 57 into a Linear layer would be meaningless and badly scaled. The `has_src` / `has_tgt` flags stay in the MLP input, so the network still knows whether a tile is attached.

`dx`, `dy` are normalised by `size − 1` here, unlike edge directions, because a MOVE can jump several tiles (roads, riders) and an ATTACK can reach range 2–3.

### 4.4 Size of the action list

`SendStarsFactory` creates one SEND_STARS action per (opponent, star amount), for amounts 1 … `min(stars, 15) − 1`, so up to 14 per opponent (42 in a 4-player game). It doesn't check whether the target has been met, and the only other limit is 30 stars sent per turn.

Measured on 4,102 states from `runs/score_based_1000fm_0` (2 players, 11×11, avg 32.2 legal actions per state):

| Action | Share of legal actions | Share of MCTS visit mass |
|---|---:|---:|
| MOVE | 35.8% | 30.4% |
| BUILD_ROAD | 19.5% | 8.9% |
| SEND_STARS | 13.3% | 10.5% |
| RESEARCH_TECH | 8.1% | 12.6% |
| END_TURN | 3.0% | 14.0% |

SEND_STARS appears in 70% of states (about 6 per such state). In a 2-player game, giving stars to the only opponent is almost never good, yet it takes ~10% of the search visits, because unvisited children get q = 1.0 and early priors are uniform. Those visits then end up in π and teach the policy that the move has value. Filtering SEND_STARS (and DECLARE_WAR) out of 2-player self-play would free that budget. (An earlier estimate of 27% came from test games, not from this run.)

---

## 5. Model architecture (`TribesGNN`)

### 5.1 Building block

Every MLP in the model (except the policy head) is a 2-layer MLP:

```
mlp(d_in, d_hidden, d_out) = Linear(d_in, d_hidden) → SiLU → Linear(d_hidden, d_out)
```

**SiLU** (x·sigmoid(x)) is smooth and doesn't die like ReLU. It's a common default in modern GNNs and transformers. There is **no dropout**: the model is small relative to the data, and self-play data is refreshed every iteration.

### 5.2 Full forward pass

Shapes are for batch size B, N nodes, E edges, A actions, H = `hidden` = 128.

```
 x [B,N,104]         edge_attr [B,E,9]          u [B,80]          actions [B,A,110]
     │                     │                       │                     │
 node_in: mlp(104→128→128) edge_in: mlp(9→128→128) glob_in: mlp(80→128→128)
     │                     │                       │                     │
 h [B,N,128]           e [B,E,128]             g [B,128]                 │
     │                     │ (fixed for all layers)│                     │
     ▼                     ▼                       ▼                     │
 ┌──────────────────────────────────────────────────────────┐            │
 │  GraphLayer × 4   (h, g) ← layer(h, e, g)                │            │
 └──────────────────────────────────────────────────────────┘            │
     │                                             │                     │
     h [B,N,128]                                   g [B,128]             │
     │                                             │                     │
     ├── mean over nodes ──┐                       │                     │
     ├── max over nodes  ──┼── concat [B,384] ─────┘                     │
     │                     ▼                                             │
     │           graph_out: mlp(384→128→128)                             │
     │                     │                                             │
     │               graph [B,128] ──────────────┐                       │
     │                     │                     │                       │
     │          value_head: mlp(128→128→1)       │                       │
     │                     │                     │                       │
     │                  tanh → value [B]         │                       │
     │                                           │                       │
     │   gather h[src_node] → h_src [B,A,128] ◄──┼───── cols 26, 28 ─────┤
     │   gather h[tgt_node] → h_tgt [B,A,128]    │                       │
     │                                           │      other 108 cols   │
     │                         ctx = graph broadcast [B,A,128]           │
     │                                           │                       ▼
     │                                           │    action_in: mlp(108→128→128)
     │                                           │             a [B,A,128]
     │                                           ▼                       │
     └──────────────► concat [h_src, h_tgt, a, ctx]  [B,A,512] ◄─────────┘
                                     │
                      policy_head: 512 → 256 → 128 → 1  (SiLU between)
                                     │
                      logits [B,A];  padding → −inf
```

### 5.3 One message-passing layer (`GraphLayer`)

```
                    for every directed edge s → t:
   h_s [H] ─┐
   h_t [H] ─┼─ concat [3H] ── message: mlp(384→128→128) ──► m_st [H]
   e_st[H] ─┘

                    for every node t:
   agg_t = (1 / in_degree(t)) · Σ_{s→t} m_st          (mean of incoming messages)

   h_t' = LayerNorm( h_t + update( concat[h_t, agg_t, g] ) )      update: mlp(384→128→128)

                    graph-level:
   g'   = LayerNorm( g   + glob( concat[g, mean_t h_t'] ) )        glob:   mlp(256→128→128)
```

In code ([gnn_model.py:48-54](gnn_model.py#L48-L54)):

```python
m   = self.message(torch.cat([h[:, src], h[:, tgt], e], dim=-1))          # [B,E,H]
agg = torch.zeros_like(h).index_add(1, tgt, m) * inv_deg[None, :, None]   # mean over incoming edges
h   = self.norm_h(h + self.update(torch.cat([h, agg, g[:, None, :].expand_as(h)], dim=-1)))
g   = self.norm_g(g + self.glob(torch.cat([g, h.mean(dim=1)], dim=-1)))
```

Design choices and reasoning:

| Choice | Reasoning |
|---|---|
| **Message uses both endpoints + edge** `(h_s, h_t, e_st)` | an edge-conditioned message (as in MPNN / GN blocks). The receiver's state matters too ("enemy next to *my* city" ≠ "enemy next to an empty plain") |
| **Mean aggregation** (not sum) | border tiles have 3 or 5 neighbours, interior tiles 8. Sum would give corners systematically smaller inputs; mean keeps the scale the same. Degree information is recovered from the `pos` features |
| **Mean** (not max / attention) | simplest stable choice. Attention would add parameters and cost for a regular grid where every neighbour matters |
| **Global vector into every node update** | every tile sees tribe-level context (stars, techs) on every layer, not just at the end |
| **Global vector updated from mean node** | gives a whole-board "summary" channel. Information can cross the board in one hop via g, instead of needing ~size layers |
| **Residual connections** `h + f(...)` | stable training of a stack of layers; each layer learns a correction |
| **Post-LayerNorm** | keeps embeddings on a stable scale across layers and batches. LayerNorm (not BatchNorm) because batches are small and mix board sizes across steps |
| **Edge embeddings are not updated** | `e` is computed once by `edge_in` and reused by every layer. Edge features are mostly static relational facts, so updating them would add an MLP over E (≈7× N) per layer for little benefit. This is a candidate extension if the model is under-fitting |
| **Weights not shared across layers** | each of the 4 layers has its own parameters (more capacity for a small parameter cost) |

### 5.4 Receptive field

With 8-neighbour edges, each layer extends a tile's view by one ring (Chebyshev distance):

```
layer 1: 3×3     layer 2: 5×5     layer 3: 7×7     layer 4: 9×9 window around each tile
```

On an 11×11 board, 4 layers let a central tile see almost the whole board directly. On 14×14 / 16×16 boards the global vector carries the rest. **Why 4 layers:** units move 1–3 tiles per turn (more on roads), so threats and opportunities within about 4 tiles are what matters for one turn's decisions. More layers cost time linearly in every MCTS evaluation, and GNNs tend to over-smooth with depth.

### 5.5 Readout and value head

```
graph = graph_out( concat[ mean_n h_n,  max_n h_n,  g ] )      # [B,128]
value = tanh( value_head(graph) )                              # [B]
```

- **Mean + max pooling:** mean captures overall balance (territory, army size); max picks out single critical features (a unit next to a capital, a ready super unit) that a mean would dilute.
- **Plus g:** the global vector already holds tribe-level information (stars, score, techs).
- **tanh:** matches the value target range [-1, 1] exactly (section 7.2).

### 5.6 Policy head (action scoring)

For each action i:

```
h_src_i = h[src_node_i]  if src_node_i ≥ 0 else 0
h_tgt_i = h[tgt_node_i]  if tgt_node_i ≥ 0 else 0
a_i     = action_in(action_features_i without cols 26, 28)
logit_i = policy_head( concat[ h_src_i, h_tgt_i, a_i, graph ] )
```

- **policy_head** is deeper than the others: `Linear(512,256) → SiLU → Linear(256,128) → SiLU → Linear(128,1)`. It has to combine four different sources (two tiles, the action's own parameters and global context) into a fine-grained ranking, and this is where most of the decision is made.
- **Source and target tile embeddings** give each action full local context. "Move this knight onto that tile" is judged with both tiles' 4-hop neighbourhoods.
- **Graph context** lets tile-less actions (RESEARCH_TECH, END_TURN, SEND_STARS) be scored from the global state: their h_src / h_tgt are zeros.
- **Missing tiles → zero vectors** (`_gather_nodes` multiplies by a presence mask). The `has_src` / `has_tgt` flags in `a_i` tell the network these zeros mean "no tile", not "a tile with a zero embedding".
- **Padding:** batches pad action lists to the longest one. Padded slots get logit −inf, so softmax gives them probability exactly 0.
- **Actions are scored independently.** There's no attention between actions. Their interaction comes only through the shared state context. This is simple and O(A) per state.

### 5.7 Every parameter (hidden = 128, layers = 4)

`Linear(i, o)` has i·o + o parameters; `LayerNorm(H)` has 2H.

| Module | Structure | Parameters |
|---|---|---:|
| `node_in` | Linear(104,128) → SiLU → Linear(128,128) | 29,952 |
| `edge_in` | Linear(9,128) → SiLU → Linear(128,128) | 17,792 |
| `glob_in` | Linear(80,128) → SiLU → Linear(128,128) | 26,880 |
| `layers[k].message` (each layer) | Linear(384,128) → SiLU → Linear(128,128) | 65,792 |
| `layers[k].update` (each layer) | Linear(384,128) → SiLU → Linear(128,128) | 65,792 |
| `layers[k].glob` (each layer) | Linear(256,128) → SiLU → Linear(128,128) | 49,408 |
| `layers[k].norm_h`, `norm_g` (each layer) | LayerNorm(128) × 2 | 512 |
| **one GraphLayer** | | **181,504** |
| **4 GraphLayers** | | **726,016** |
| `graph_out` | Linear(384,128) → SiLU → Linear(128,128) | 65,792 |
| `value_head` | Linear(128,128) → SiLU → Linear(128,1) | 16,641 |
| `action_in` | Linear(108,128) → SiLU → Linear(128,128) | 30,464 |
| `policy_head` | Linear(512,256) → SiLU → Linear(256,128) → SiLU → Linear(128,1) | 164,353 |
| **Total** | | **1,077,890** |

About two-thirds of the parameters are in message passing, and about 15% in the policy head. At ~1.1 M parameters, the model is small enough for a CPU forward pass in ~3–9 ms, which matters because MCTS calls it on every expansion.

Constructor arguments ([gnn_model.py:58](gnn_model.py#L58)), saved in every checkpoint as `model_config`:

| Argument | Value | Source |
|---|---|---|
| `node_features` | 104 | `schema.json` (from `StateEncoder`) |
| `edge_features` | 9 | `schema.json` |
| `global_features` | 80 | `schema.json` |
| `action_features` | 110 | `schema.json` (from `ActionEncoder`) |
| `src_col` | 26 | `ActionEncoder.srcNodeColumn()` |
| `tgt_col` | 28 | `ActionEncoder.tgtNodeColumn()` |
| `hidden` | 128 | `--hidden` |
| `layers` | 4 | `--layers` |

**Why hidden = 128:** big enough to hold a tile's terrain, unit, ownership and neighbourhood, small enough for fast CPU inference inside the search. Doubling it would roughly quadruple most layer costs.

### 5.8 Graph caching

`_graph(n, device)` builds the edge list and 1/in-degree once per (board size, device) and caches it. N must be a perfect square (the assert catches mismatched data). Every graph in a batch shares this topology, which is why batches contain **one board size only**.

---

## 6. How the pieces fit: an example

A MOVE of a knight from (3,4) to (5,4) on an 11×11 board:

```
action features:  action_type:MOVE = 1
                  has_src = 1,  src_node = 3*11+4 = 37
                  has_tgt = 1,  tgt_node = 5*11+4 = 59
                  dx = 2/10 = 0.2, dy = 0
                  everything else 0

h_src = h[37]   ← knight's tile after 4 layers: knows nearby enemies, roads, its own HP/status
h_tgt = h[59]   ← destination: what's around it, who owns it, is it a border tile
a     = action_in([MOVE one-hot, has_src, has_tgt, dx, dy, ...])   (cols 26 and 28 removed)
ctx   = graph   ← overall situation (stars, turn, score race)
logit = policy_head([h_src, h_tgt, a, ctx])
```

---

## 7. Training

### 7.1 Data

Each self-play sample (`TrainingSample`, one JSON line) holds:

| Field | Meaning |
|---|---|
| `stateFeatures` | flat state, encoded from the **acting player's** view |
| `actionFeatures` | [A × 110], in the same order as the tree's root children |
| `searchPolicy` | π = root visit proportions, `visitDistribution(visits, T=1)` |
| `valueTarget` | z, the final outcome for that player |
| `rootValue` | root value estimate (recorded, not used yet) |
| `boardSize`, `gameId`, `playerID`, `tick` | bookkeeping |

`schema.json` next to the data records the feature sizes and id columns. The loader refuses to mix folders with different layouts, and refuses to warm-start a model trained on a different layout.

### 7.2 Value target z

From `GameRecord.outcomeToValue`:

| Result | z |
|---|---|
| WIN | +1 |
| LOSS | −1 |
| INCOMPLETE (no winner) | score rank squeezed into [−0.5, +0.5]: best +0.5, worst −0.5, ties share the better rank |

Every sample of a game gets the same z (no discounting, as in AlphaZero). The [−0.5, 0.5] squeeze means an unfinished game never looks like a real win or loss. In SCORE mode there is always a winner, so z is ±1.

### 7.3 Loss

```
L = CE(π, softmax(logits)) + value_weight · MSE(value, z)

CE = − Σ_a π(a) · log_softmax(logits)(a)      (padding: log p set to 0, so 0 · −inf doesn't give NaN)
```

- **Cross-entropy to the full visit distribution**, not just the most-visited action: π carries the search's confidence across all actions.
- **MSE for value**: standard for a [−1, 1] regression target.
- **value_weight = 1.0**: AlphaZero's equal weighting. Lower it if the value loss overfits first (it usually does, see 7.6).

### 7.4 Every training hyperparameter

| Flag | Default in `train_gnn.py` | Used by the pipeline (`slurm/config.sh`, `local/run_loop.sh`) | Reasoning |
|---|---|---|---|
| `--epochs` | 20 | **5** | each sample stays in a 3-iteration window, so it's seen ~15 times overall. More passes just memorise games |
| `--batch-size` | 64 | **128** | fits easily on a 16 GB V100 at 11×11 / 14×14 |
| `--lr` | 1e-3 | 1e-3 | standard Adam peak; peak of the one-cycle schedule |
| `--weight-decay` | 1e-4 | 1e-4 | mild L2-style regularisation (AdamW, decoupled) |
| `--hidden` | 128 | 128 | first model only; warm-started models keep their architecture |
| `--layers` | 4 | 4 | same |
| `--value-weight` | 1.0 | 1.0 | |
| `--val-frac` | 0.1 | 0.1 | 10% of **games** held out |
| `--grad-clip` | 1.0 | 1.0 | global grad-norm clip; caps the step size when an outlier batch (e.g. a near-one-hot π the model gives tiny probability) produces a huge gradient |
| `--init-from` | none | previous iteration's `best.pt` | warm start, so each iteration fine-tunes instead of relearning |
| `--device` | auto | cuda (training), auto (self-play) | |
| `--threads` | torch default | | CPU only |
| `--no-cache` | off | off | |
| `--seed` | 0 | 0 | torch + numpy RNG, and the train/val split |

Pipeline-level settings that affect what the GNN learns from:

| Setting | Value | Meaning |
|---|---|---|
| `DATA_WINDOW` | 3 | train on the last 3 iterations (~240k samples) |
| `GAMES_PER_ITER` | 200 | ~300–400 samples per game |
| `ITERATIONS` | 20 | self-play → train cycles |
| `FM_CALLS_PER_MOVE` | 1000 | MCTS budget per decision; sets how sharp π is |
| `TRIBES` / `GAME_MODE` | XIN_XI, IMPERIUS / SCORE | 2 players, 30 turns, always a winner |

### 7.5 Optimiser and schedule

- **AdamW** (`lr=1e-3, weight_decay=1e-4`). Weight decay applies to all parameters, including biases and LayerNorm gains. That's a common simplification; excluding them is a possible tweak.
- **OneCycleLR** with `max_lr = lr`, `pct_start = 0.1`, `total_steps = epochs × steps_per_epoch`, and PyTorch defaults for the rest: cosine annealing, start LR = max/25 = 4e-5, final LR = start/1e4 ≈ 4e-9, Adam β₁ cycled between 0.85 and 0.95. The schedule is stepped **per batch**. A short warm-up stops the first steps of a fresh (or warm-started on new data) model from overshooting; the cosine decay gives a well-settled final model. The schedule restarts every iteration.
- **Checkpoint selection:** `best.pt` = epoch with the lowest validation loss; `last.pt` = final epoch. The pipeline uses `best.pt`.

### 7.6 Validation by game

The split holds out **whole games** (unique key = file index × 10⁶ + gameId). All samples of a game share the same z and have very similar states, so a per-sample split would leak the answer into validation and make the value loss look much better than it is.

Logged metrics (train and val, `metrics.csv`):

| Metric | Meaning |
|---|---|
| `loss` | total loss |
| `policy` | CE(π, p). Can never go below H(π) |
| `kl` | CE − H(π) = KL(π ‖ p). 0 means a perfect match; this is the real "distance to go" |
| `value` | MSE(value, z) |
| `top1` | how often the top-scored action is the most-visited one |
| `value_sign` | how often sign(value) = sign(z), i.e. it predicts the winner |

### 7.7 Batching

`SelfPlayDataset` groups samples into **buckets by board size**. Each batch comes from one bucket (shared topology); the order of batches is shuffled across buckets. Action lists are padded to the longest in the batch, with `action_mask` marking real actions. Parsed files are cached as `<file>.npz` with states in **float16**. That's safe because all state features are in [−1, 1] with 0/1 flags. Action features stay float32: node ids up to 255 are exact in either, but there's no need to risk it.

---

## 8. Inference: how Java calls the GNN

### 8.1 Server

`ExternalNetwork.load()` starts `serve_gnn.py --checkpoint <best.pt>` as a subprocess and talks to it over **stdin/stdout in big-endian binary** (Java's `DataOutputStream` format).

```
handshake  (server → Java): MAGIC 0x54474E4E "TGNN", VERSION 1,
                            nodeF, edgeF, globalF, actionF, srcCol, tgtCol, maxPlayers
           Java checks these against its own encoders → fails fast on a layout mismatch

request    (Java → server): op (1 = evaluate, 0 = quit), boardSize, stateLength, numActions, actionF,
                            float32[stateLength], float32[numActions × actionF]

response   (server → Java): status 0 → float32 value, float32[numActions] logits
                            status 1 → error message (so Python errors show up in Java instead of a hang)
```

**Why a subprocess:** Java runs the exact PyTorch model used in training, with no export step, no second implementation to keep in sync, and no extra Java libraries. A CPU forward pass takes ~3–9 ms. Batch size is 1; with zero legal actions, the server passes one masked dummy action.

Settings (Java system properties): `tribes.python`, `tribes.gnn.server`, `tribes.gnn.threads` (default 4), `tribes.gnn.device` (auto / cpu / cuda).

### 8.2 Perspective at inference

Training samples are always recorded on the acting player's own turn. So `evaluate()` encodes the state and actions from **`gs.getActiveTribeID()`**, then converts the value to the searching player's view: unchanged if they're acting, **negated** otherwise. Negation is exact for 2-player zero-sum values and only an approximation with more players.

### 8.3 Use in the search (`ACTreeNode`)

| Output | Use | Parameter |
|---|---|---|
| `softmax(logits)` (done in Java) | priors P(s,a) for PUCT: `Q + C_PUCT · P · √N(s) / (1 + N(s,a))` | `C_PUCT = 1.5`, `USE_PRIORS` (on in self-play) |
| root priors in self-play | `P' = (1 − ε)·P + ε·Dir(α)`, Gamma via Marsaglia–Tsang | `DIRICHLET_ALPHA = 0.3`, `DIRICHLET_EPS = 0.25`, `ROOT_NOISE` |
| `value` | leaf evaluation instead of the DIFF heuristic, mixed as λ·critic + (1−λ)·heuristic | `USE_VALUE_NET` (on once a model is loaded), `VALUE_MIX_LAMBDA = 1.0` |
| move choice in self-play | sample from visits^(1/T) for each player's first `TEMP_MOVES` actions, then argmax | `TEMPERATURE = 1.0`, `TEMP_MOVES = 60` |

In iteration 0, there is no model yet: `UniformNetwork` gives uniform priors and value 0, and leaves are evaluated with the DIFF heuristic.

Notes from reading the code:

- PUCT and the Dirichlet noise are implemented **inline in `ACTreeNode`**. The `PUCT.score`, `addDirichletNoise` and `sampleDirichlet` methods in `search/PUCT.java` are still TODO stubs that throw, but nothing calls them, so they're dead code.
- Unvisited children get **q = 1.0** (optimistic first-play urgency), so every child is tried once before priors start to dominate. With hundreds of legal actions and 1000 FM calls, a large share of the budget goes to trying each action once, which weakens the priors' effect. A lower first-play value (e.g. the parent's Q) would let the actor steer the search more.

---

## 9. Known limitations and possible extensions

| Area | Current state | Possible change |
|---|---|---|
| Edge embeddings | computed once, not updated per layer | add an edge-update MLP per layer (full GN block) |
| Aggregation | mean | add a sum or max channel (PNA-style) for count-sensitive signals ("3 enemies adjacent" vs 1) |
| Action interactions | each action scored independently | self-attention over the action set |
| Action list size | SEND_STARS = 13% of legal actions and ~10% of visits in the 2-player run | filter them in 2-player self-play, or merge the amounts into a few buckets |
| `rootValue` | recorded, unused | blend into the value target (e.g. z/q mixing, as in some AlphaZero reproductions) |
| Multi-player value | negation for non-acting player | per-slot value vector (one value per tribe) |
| Weight decay | applied to biases and LayerNorm too | exclude them |
| Fog | `PLAY_WITH_FULL_OBS = true`, fog features always 0 | train with fog on, once the full-obs model works |
| Scheme A action indexing | TODO stubs | not needed with scheme B |
| Value overfitting | all samples of a game share z | few epochs (5), game-level split, watch `val_value` |

---

## 10. Quick reference: shapes on an 11×11, 2-player board

| Tensor | Shape |
|---|---|
| `x` | [B, 121, 104] |
| `edge_attr` | [B, 840, 9] |
| `u` | [B, 80] |
| `actions` | [B, A, 110] (A = longest action list in the batch) |
| `action_mask` | [B, A] |
| `h` after embedding / each layer | [B, 121, 128] |
| `e` | [B, 840, 128] |
| `g` | [B, 128] |
| messages per layer | [B, 840, 128] |
| `graph` | [B, 128] |
| `logits` | [B, A] |
| `value` | [B] |
| flat state per sample | 20,224 floats |
| model parameters | 1,077,890 |
