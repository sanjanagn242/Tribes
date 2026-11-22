# State and action encoding

This package turns a Tribes `GameState` and its legal `Action`s into numbers for the actor-critic network.

- The **state** becomes a graph: one node per tile, edges between neighbouring tiles, and a vector of tribe-level features.
- Each **action** becomes a fixed-length, sparse vector. It points at the graph nodes it involves.

| File | Role |
|------|------|
| [StateEncoder.java](StateEncoder.java) | `GameState` → `GraphState` |
| [GraphState.java](GraphState.java) | Holds the encoded graph: `nodeFeatures [N,F]`, `edgeIndex [2,E]`, `edgeFeatures [E,Fe]`, `globalFeatures [G]` |
| [ActionEncoder.java](ActionEncoder.java) | `Action` → feature vector (scheme B). The scheme A stubs (`toIndex`, `actionSpaceSize`) are still TODO |

Both encoders can list every column of their output: `nodeFeatureNames()`, `edgeFeatureNames()` and `globalFeatureNames()` on `StateEncoder`, and `featureNames()` on `ActionEncoder`. Save these lists next to your training data, so you can check the Python side reads the columns you think it does.

## Perspective

Both encoders take a `playerID`, and **you must pass the same one to both**. Tribes go into relative slots:

- slot 0 is `playerID` itself;
- slot k is tribe `(playerID + k) % numTribes`.

This lets one network play any seat. The layout reserves `StateEncoder.DEFAULT_MAX_PLAYERS = 4` slots. To change that, pass the same value to both constructors, and keep it fixed for a whole training run.

## State encoding (graph)

`StateEncoder.encodeGraph(gs, playerID)` returns a `GraphState`. In PyTorch Geometric terms it gives you `x`, `edge_index`, `edge_attr` and a global vector `u`.

### Nodes (104 features per tile)

Node id is `x * boardSize + y`, using the same `[x][y]` indexing as `core.game.Board`.

| Block | Contents |
|---|---|
| tile | position (x, y), visible flag, terrain one-hot (includes FOG), resource one-hot, building one-hot, has building, road, trade network |
| territory owner | `[none, self, opp1..3, unknown]` (`unknown` = the tile is owned but the owning city isn't visible) |
| city centre | level, population progress, population, capital, walls, border size, number of units, points, owner slot |
| unit | owner slot, type, HP fraction, max HP, ATK/DEF/MOV/RANGE, kills, veteran, ranged, water unit, turn status (fresh / moved / attacked / ...), and the land unit carried by a boat, ship or battleship |

### Edges (9 features per edge)

Edges are directed, from each tile to its 8 neighbours, because Tribes movement and vision use the 3×3 neighbourhood. An 11×11 board has 840 edges. Which tiles are connected depends only on the board size, but the edge features change every state.

| Feature | Meaning |
|---|---|
| `diagonal` | the neighbour is diagonal |
| `road_link` | both ends are on the trade network (road, city or port) |
| `land_water` | the edge crosses the coast (embark or disembark) |
| `same_city` | both ends are inside the same city's borders |
| `border` | the territory owner differs between the two ends |
| `touches_fog` | at least one end is unseen |
| `unit_to_free` | a unit is on the source tile, and the target is visible, empty and the right terrain for it (land or water) |
| `dx`, `dy` | direction of the edge |

### Globals (80 features)

- **Game:** turn, game mode, a city-is-levelling-up flag, the active tribe's slot, number of tribes.
- **Own tribe:** stars, score, production, cities, units, kills, number of techs; each of the 24 techs; tribe type; status of each monument.
- **Each opponent:** present, alive, met, score, cities seen, allegiance in both directions.

### Fog of war

Fogged tiles only get their coordinates and the FOG flag. Hidden enemy stars and techs are never encoded.

`Constants.PLAY_WITH_FULL_OBS` is `true` by default ([Constants.java](../../../core/Constants.java)). With that setting, agents see the whole map and the fog features stay zero. Set it to `false` to train under fog of war.

### Flat form

`encode()` returns `GraphState.flatten()`. This is what goes in `TrainingSample.stateFeatures`: node features row by row, then edge features row by row, then globals. Its length is `featureSize(boardSize)`, which is 20,224 on an 11×11 board.

To rebuild the graph in Python, with N = size², F = 104, E = number of edges and Fe = 9:

```python
x         = flat[:N*F].view(N, F)
edge_attr = flat[N*F : N*F + E*Fe].view(E, Fe)
u         = flat[N*F + E*Fe:]
```

`edge_index` is not stored. Build it the way `StateEncoder.buildEdgeIndex` does: for each x, then each y, visit the neighbours in this order and keep those inside the board:
`(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)`.

## Action encoding (scheme B)

`ActionEncoder.toFeatures(actions, gs, playerID)` returns one row per action, in the same order as `actions`. That's the order the tree node uses, and the format `TrainingSample.actionFeatures` expects. Each row has 110 values.

### Layout

| Block | Size | Filled by |
|---|---|---|
| action type one-hot | 25 | every action |
| `has_src`, `src_node`, `has_tgt`, `tgt_node` | 4 | node ids into the state graph, or -1 when there's no tile |
| `dx`, `dy` (target − source, normalised) | 2 | actions with both a source and a target tile |
| star cost (/20) | 1 | BUILD, SPAWN, RESOURCE_GATHERING, RESEARCH_TECH, BUILD_ROAD, UPGRADE_*, SEND_STARS |
| building one-hot | 19 | BUILD |
| resource one-hot | 7 | RESOURCE_GATHERING |
| unit type one-hot | 12 | SPAWN |
| level-up bonus one-hot | 8 | LEVEL_UP |
| tech one-hot + tier | 24 + 1 | RESEARCH_TECH |
| target tribe (relative slot) | 4 | DECLARE_WAR, SEND_STARS |
| stars sent | 1 | SEND_STARS |
| capture city / village | 2 | CAPTURE |

The layout is **sparse**: blocks that don't apply to an action stay 0, and the action-type one-hot tells the network which blocks are live. For example, a LEVEL_UP action has an all-zero unit-type block. Its EXPLORER and SUPERUNIT bonuses go in the level-up block, because they are city bonuses, not units bought with stars.

Two actions of the same type differ only in their parameter blocks. For example, `SPAWN(Warrior)` and `SPAWN(Archer)` from the same city have the same type and the same node ids. They differ in the unit-type one-hot and the star cost.

### Source and target tiles

| Action(s) | Source tile | Target tile |
|---|---|---|
| city actions | the city's tile | `targetPos` (for SPAWN and LEVEL_UP, the city tile) |
| MOVE | the unit's tile | the destination |
| ATTACK, CONVERT | the unit's tile | the enemy unit's tile |
| other unit actions (CAPTURE, EXAMINE, RECOVER, HEAL_OTHERS, DISBAND, MAKE_VETERAN, UPGRADE_*) | the unit's tile | the same tile |
| BUILD_ROAD | none | the road tile |
| RESEARCH_TECH, END_TURN, DECLARE_WAR, SEND_STARS | none | none |

### Using the node ids in Python

`srcNodeColumn()` and `tgtNodeColumn()` give the positions of the two node-id columns. Use them to fetch tile embeddings from the GNN output `h`, then score each action:

```python
src = feats[:, SRC_COL].long()
tgt = feats[:, TGT_COL].long()
h_src = torch.where((src >= 0)[:, None], h[src.clamp(min=0)], 0)
h_tgt = torch.where((tgt >= 0)[:, None], h[tgt.clamp(min=0)], 0)
logits = mlp(torch.cat([h_src, h_tgt, feats_without_id_cols, u_emb.expand(len(feats), -1)], dim=1))
```

The two id columns hold raw ids, not values in [0, 1], so don't feed them into the MLP.

### Notes

- **Unused action types:** `CLIMB_MOUNTAIN` is in `Types.ACTION`, but the game never creates it, so its column is always 0. Climbing is handled by ordinary MOVE actions once the tribe has researched CLIMBING. This comes from the original Tribes code, not this package.
- **Large action lists:** SEND_STARS adds one action per star amount per tribe. In test games it made up about 27% of all legal actions, and DECLARE_WAR another 4%. Consider filtering them out in self-play if they slow down the search.
