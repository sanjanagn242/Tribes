# Portfolio MCTS: MCTS over scripted action assignments

`players.portfolioMCTS` contains a closed-loop UCT Monte Carlo Tree Search agent (`"pMCTS"`) that searches over **portfolio action assignments** instead of raw actions. Each edge in the tree is an `ActionAssignment` from the [portfolio](../portfolio/README.md): an actor paired with a script that picked one concrete action for it. Duplicate choices are merged, so nodes have fewer children than in [plain MCTS](../mcts/README.md). On top of plain UCT, the agent can **prune** a node's children down to the best-ranked few and then **unprune** them one at a time as visits grow (progressive widening), and it can add a **progressive bias** term from the same ranking heuristic. Like plain MCTS, each node is the state after one action, the tree runs into opponents' turns, and opponents minimise our heuristic (paranoid search).

| File | Purpose |
|------|---------|
| [PortfolioMCTSPlayer.java](PortfolioMCTSPlayer.java) | The `Agent`: builds the root, returns the chosen action, records branching-factor stats |
| [PortfolioMCTSParams.java](PortfolioMCTSParams.java) | Parameters (extends `AlgParams`): UCT constant, pruning schedule, portfolio and prune heuristic |
| [PortfolioTreeNode.java](PortfolioTreeNode.java) | Tree node with the search loop: selection with pruning, expansion, rollout and backup |
| [../heuristics/PruneHeuristic.java](../heuristics/PruneHeuristic.java) | Interface: `evaluatePrune`, `prune`, `unprune` (all default to no-ops) |
| [../heuristics/PrunePortfolioHeuristic.java](../heuristics/PrunePortfolioHeuristic.java) | The implementation used: scores an assignment as `weight × script value` |

---

## How it works

### `PortfolioMCTSPlayer.act(gs, ect)`

1. **Shortcut.** If `gs.getAllAvailableActions()` has only one action (normally `END_TURN`), return it without searching.
2. Create a root `PortfolioTreeNode`. `setRootGameState` sets up the state heuristic (`params.getStateHeuristic(playerID, allPlayerIDs)`), gets the root's options with `portfolio.produceActionAssignments(gs)`, and fetches the prune heuristic.
3. Run `mctsSearch(ect)` until the budget is used up.
4. Return the action of the root child with the **highest mean value** (`bestAction()`). `mostVisitedAction()` exists but its call is commented out.
5. Record the branching factor (assignments per actor and in total) in `AIStats`. These are printed in `result()` at the end of the game.

There is no root action-group restriction: the `PRIORITIZE_ROOT` code is commented out. The tree is **rebuilt from scratch on every `act()` call**.

### Search iteration (`PortfolioTreeNode.mctsSearch`)

1. **Tree policy (`treePolicy`).** Starting at the root, and while the game isn't over and `depth < ROLLOUT_LENGTH`:
   - if the node has an unexpanded child, **expand** it and stop;
   - otherwise move to the child picked by **`uct()`**.
2. **Expansion (`expand`).** Pick a random unexpanded assignment. Copy the state, apply `assignment.getAction()` (one forward-model call), then call `produceActionAssignments` on the new state to get the child's options. There is no forced `END_TURN` (`FORCE_TURN_END` is never read).
3. **Pruning / unpruning (start of `uct`).** Only when `PRUNING` is on. With `n` = number of children and `k⁺` = `k_plus` (starts at 0):
   ```
   T      = ⌊n · T_mult⌋                     // visits before the first prune
   k_init = max(1, ⌊n · K_init_mult⌋)        // children kept by the first prune
   A      = ⌊T · A_mult⌋
   limit(k⁺) = ⌊A · B^k⁺⌋                    // visits needed for the next unprune
   ```
   - When `nVisits >= T` and `k⁺ == 0`, `prune()` keeps the `k_init` children with the highest `evaluatePrune` score, marks the others as pruned, and sets `k⁺ = 1`.
   - After that, whenever `nVisits > limit(k⁺)`, `unprune()` brings back the pruned child with the highest score (random tie-break) and increments `k⁺`.

   For example, with `n = 10` and the default multipliers: prune to 5 children at 20 visits, then unprune at more than 39, 50, 65, 85, 111, … visits.
4. **UCT selection (`uct`).** For each child that is **not pruned** (see the quirks for what happens before pruning starts):
   ```
   Q        = normalise(totValue / (nVisits + ε), bounds_min, bounds_max)   // parent's bounds
   explore  = sqrt( ln(parentVisits + 1) / (childVisits + ε) )
   progBias = evaluatePrune(state, assignment) / (childVisits + ε)
   UCT      = Q + C · explore  (+ progBias if PROGBIAS)
   UCT      = (UCT + ε) · (1 + ε · (rand − 0.5))                            // tie-break noise
   ```
   Pruned children keep the default value `0`. On our move, take the child with the highest value. On an opponent's move, take the lowest. The tree is **closed-loop**: each node stores its own `GameState`.
5. **Rollout (`rollOut`).**
   - With `ROLOUTS_ENABLED` on (always the case from `Run.java`), copy the leaf state and repeatedly call `produceActionAssignments`, pick a **uniformly random assignment**, and apply it, until `depth >= ROLLOUT_LENGTH` or the game ends.
   - With it off, evaluate the leaf directly.
   - Evaluation is `heuristic.evaluateState(rootState, state)`.
6. **Backup (`backUp`).** From the leaf up to the root, increment `nVisits`, add the value to `totValue`, and widen each node's `[min, max]` bounds.

### The prune heuristic (`PrunePortfolioHeuristic`)

```
evaluatePrune(aas) = w(aas) · aas.getValue()
w(aas) = weights[actionType]                 // 0.5 for every type by default
       = weightScripts[script class]         // if weights[actionType] == -1
```

`setWeights(double[])` sets per-script weights for the 6 `SPAWN` scripts and then the 5 `RESEARCH_TECH` scripts (11 values, in `SimplePortfolio` order). It marks those two action types with weight `-1` so the per-script table is used. Weights for other action types stay at `0.5`. `RunElites.java` evolves these 11 weights with MAP-Elites.

With the default weights, ranking is just by the script's own value, which is not on a common scale across scripts (see the [portfolio quirks](../portfolio/README.md#known-quirks-and-limitations)).

### Final action choice (`bestAction`)

- Pick the expanded root child with the highest `totValue / (nVisits + ε)`, with tie-break noise.
- If no child was expanded, print `"Unexpected selection!"` and return index 0.

### Comparison with plain MCTS

| | [MCTS](../mcts/README.md) | Portfolio MCTS |
|---|---|---|
| Edge | One raw `Action` | One `ActionAssignment` (actor + script → action), deduplicated |
| Actions reachable | All actions | Only what some script picks. Never `DESTROY`, `DISBAND`, `DECLARE_WAR`, `SEND_STARS`, or monuments and temples (see the portfolio quirks) |
| Root restriction | Optional `PRIORITIZE_ROOT` | None |
| Forced turn end | `FORCE_TURN_END` | None |
| Selection | UCT | UCT on unpruned children, plus progressive widening and optional progressive bias |
| Rollout policy | Random raw actions | Random portfolio assignments |
| Default `ROLLOUT_LENGTH` | 10 | 20 |
| Rollouts from `Run.java` | JSON `"Rollouts"` | Always on (JSON `"Rollouts"` ignored) |
| Final choice | Most visited (best mean if all tied) | Best mean |
| Exploration constant | `K = √2` | `C = √2` |

Both are closed-loop, paranoid, rebuild the tree on each call, and count UCT descents as forward-model calls.

### Forward-model accounting

The root's `fmCallsCount` goes up:
- once per `GameState.advance()` (in expansion and in each rollout step), **and**
- once per UCT descent step (`uct()`), even though no state is advanced there.

`produceActionAssignments` runs every script on every actor. It is called once per expansion and **twice** per rollout step: once to choose the action, and again inside `advance()`, where the result is thrown away. None of these calls count toward `num_fmcalls`, so the real CPU cost per FM call is much higher than for plain MCTS.

---

## Parameters

### Portfolio-MCTS-specific (`PortfolioMCTSParams`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `C` | `double` | `√2 ≈ 1.414` | UCT exploration constant. |
| `ROLLOUT_LENGTH` | `int` | `20` | Search **horizon** in actions from the root. It caps tree depth and is the depth at which rollouts stop. |
| `ROLOUTS_ENABLED` | `boolean` | `true` | Random portfolio rollouts from each new leaf. If `false`, the leaf is evaluated directly. (The field name has a typo.) |
| `PRUNING` | `boolean` | `true` | Turns on pruning and progressive unpruning. |
| `PROGBIAS` | `boolean` | `true` | Adds `evaluatePrune / (childVisits + ε)` to UCT. Only takes effect once a node has been pruned. |
| `T_mult` | `double` | `2.0` | `T = ⌊n · T_mult⌋` visits before the first prune. |
| `K_init_mult` | `double` | `0.5` | `k_init = max(1, ⌊n · K_init_mult⌋)` children kept at the first prune. |
| `A_mult` | `double` | `1.5` | `A = ⌊T · A_mult⌋`, the base of the unprune schedule. |
| `B` | `double` | `1.3` | Growth factor: the next unprune happens when `nVisits > ⌊A · B^k⁺⌋`. |
| `portfolio` (private) | `Portfolio` | `null` | Set with `setPortfolio()`. **Required.** |
| `pruneHeuristic` | `PrunePortfolioHeuristic` | `null` | Returned by `getPruneHeuristic()`. Required when `PRUNING` is on. |

`printPruneLine(n)` prints the prune and unprune schedule for `n` children. `main()` is leftover test code that only prints a `LinearRegression`.

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

| Field | Default | Used by pMCTS? | Description |
|-------|---------|----------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ✅ | Which budget ends the search. |
| `num_iterations` | `200` | ✅ (`STOP_ITERATIONS`) | Iterations per `act()`. |
| `num_fmcalls` | `2000` | ✅ (`STOP_FMCALLS`) | Forward-model call budget per `act()`. The search stops when `fmCallsCount > num_fmcalls`. This is checked only after a full iteration, so it can go over budget. |
| `num_time` | `40` | ❌ | Not read. `STOP_TIME` uses the `ElapsedCpuTimer` passed to `act()`. |
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | `ENTROPY_HEURISTIC` (0), `SIMPLE_HEURISTIC` (1) or `DIFF_HEURISTIC` (2). |
| `epsilon` | `1e-6` | ✅ | Avoids division by zero in Q, UCT and progressive bias, and sets the size of the tie-break noise. |
| `FORCE_TURN_END` | `5` | ❌ | Never read. |
| `PRIORITIZE_ROOT` | `false` | ❌ | Never read (the code is commented out in `act()`). |
| `getPruneHeuristic()` | `null` | ✅ (overridden) | Returns `pruneHeuristic`. |

**Stop-type behaviour**

| Constant | Value | Stops when |
|----------|-------|------------|
| `STOP_TIME` | `0` | the remaining time is `<= 2 × average iteration time` or `<= 5 ms` |
| `STOP_ITERATIONS` | `1` | `numIters >= num_iterations` |
| `STOP_FMCALLS` | `2` | `fmCallsCount > num_fmcalls` |

### Tuning interface

| Method | Keys |
|--------|------|
| `setParameterValue` | `"C"`, `"ROLLOUT_LENGTH"`, `"heuristic_method"` |
| `getParameterValue` | `"K"` (returns `C`), `"ROLLOUT_LENGTH"`, `"heuristic_method"` |
| `getParameters` | `"K"`, `"rollout_depth"`, `"heuristic_method"` |
| `getParameterValues` | `K ∈ {1.0, √2, 2.0}`, `rollout_depth ∈ {5, 8, 10, 12, 15}`, `heuristic_method ∈ {0, 1, 2}` |

The pruning parameters are not part of the tuning interface. They are set from JSON, and the prune weights are tuned separately by `RunElites`.

---

## Node properties (`PortfolioTreeNode`)

| Field | Meaning |
|-------|---------|
| `params` | Shared `PortfolioMCTSParams`. |
| `root` / `parent` / `children[]` | Tree links. `children` has one slot per assignment in `actions`, and `null` means not yet expanded. |
| `actions` | The `ActionAssignment`s available at this node, from `produceActionAssignments(state)`. |
| `pruned[]` | `null` until the first prune. After that, `true` marks a pruned child. |
| `k_plus` | `0` before pruning, then `1 +` number of unprune steps. Read with `getPruneK()`. |
| `totValue` | Sum of the values backed up through this node. |
| `nVisits` | Visit count. |
| `bounds[2]` | Smallest and largest values backed up through this node. Used to normalise Q. |
| `m_depth` | Depth from the root (root = 0). |
| `state` | The `GameState` at this node (closed loop). |
| `rootState` | Root only: the state at the root, used as the reference for the heuristic. |
| `rootStateHeuristic` | The `StateHeuristic`, shared by all nodes. |
| `rootPruneHeuristic` | The `PruneHeuristic`, shared by all nodes. |
| `fmCallsCount` | Forward-model call counter. Only the root's counter is used. |
| `playerID` | Our tribe ID. Used to tell maximising nodes from minimising ones. |
| `m_rnd` | Shared RNG, seeded from the agent seed. |

---

## Usage

### Default constructor

`new PortfolioMCTSPlayer(seed)` uses a plain `PortfolioMCTSParams` with **no portfolio and no prune heuristic**, so the first `act()` that searches throws a `NullPointerException`. Always pass params with a portfolio set.

### From `Run.java`, `Tournament.java` and `Play.java`

Player type `"pMCTS"`. [`Run.getAgent`](../../Run.java) builds it with:

```java
PortfolioMCTSParams p = new PortfolioMCTSParams();
p.stop_type        = p.STOP_FMCALLS;     // 2000 FM calls per action
p.heuristic_method = p.DIFF_HEURISTIC;
p.PRIORITIZE_ROOT  = false;              // unused anyway
p.ROLLOUT_LENGTH   = MAX_LENGTH;         // JSON "Search Depth"
p.PRUNING          = PRUNING;            // JSON "Pruning"
p.PROGBIAS         = PROGBIAS;           // JSON "Progressive Bias"
p.K_init_mult      = K_INIT_MULT;        // JSON "K init mult"
p.A_mult           = A_MULT;             // JSON "A mult"
p.B                = B;                  // JSON "B"
p.T_mult           = T_MULT;             // JSON "T mult"
Portfolio portfolio = new SimplePortfolio(agentSeed);
p.setPortfolio(portfolio);
p.pruneHeuristic = new PrunePortfolioHeuristic(portfolio);
if (Run.pMCTSweights != null)
    p.pruneHeuristic.setWeights(Run.pMCTSweights);   // JSON "pMCTS Weights"
return new PortfolioMCTSPlayer(agentSeed, p);
```

These JSON config keys affect pMCTS:

| JSON key | Maps to | Effect |
|----------|---------|--------|
| `"Search Depth"` | `ROLLOUT_LENGTH` | Search horizon |
| `"Pruning"` | `PRUNING` | Turns pruning and unpruning on or off |
| `"Progressive Bias"` | `PROGBIAS` | Adds the progressive-bias term |
| `"K init mult"`, `"T mult"`, `"A mult"`, `"B"` | `K_init_mult`, `T_mult`, `A_mult`, `B` | Pruning schedule |
| `"pMCTS Weights"` (optional; read by `Tournament` and `Play`) | `PrunePortfolioHeuristic.setWeights` | 11 per-script weights (6 `SPAWN`, then 5 `RESEARCH_TECH`) |

`"Rollouts"` and `"Force End"` are **not** applied to pMCTS, so rollouts are always on and there is no forced turn end. `C` and `num_fmcalls` cannot be set from JSON (they stay at √2 and 2000). `Tournament` and `Play` read the pruning keys with `getBoolean`/`getDouble`, so they must be present in the JSON even when pMCTS isn't playing.

For example, the current `tournament.json` sets `"Search Depth": 20`, `"Pruning": true`, `"Progressive Bias": true` and the default multipliers (0.5 / 2.0 / 1.5 / 1.3). That gives a tree up to 20 plies deep with random portfolio rollouts to depth 20. pMCTS is not in its current `"Players"` list.

### Custom configuration

```java
PortfolioMCTSParams p = new PortfolioMCTSParams();
p.stop_type = p.STOP_FMCALLS;
p.num_fmcalls = 5000;
p.C = 1.0;
p.ROLLOUT_LENGTH = 10;
p.heuristic_method = p.DIFF_HEURISTIC;
Portfolio portfolio = new SimplePortfolio(seed);
p.setPortfolio(portfolio);
p.pruneHeuristic = new PrunePortfolioHeuristic(portfolio);
Agent pmcts = new PortfolioMCTSPlayer(seed, p);
```

---

## Known quirks and limitations

These describe the current code.

- **Before pruning, UCT always picks child 0.** `uct()` only computes values for children where `pruned != null && !pruned[i]`. Until the first prune (`nVisits < T`), or always when `PRUNING` is `false`, `pruned` is `null`, so every value stays `0`, and both the max and min loops pick index 0. With the defaults, each node sends visits `n` to `2n` all to its first child. With `"Pruning": false`, the search below each fully expanded node goes down a single path, and `PROGBIAS` never applies.
- **Opponent nodes prefer pruned children.** Pruned children keep the value `0`, while unpruned ones normally have a positive UCT value. At opponent nodes the agent takes the **minimum**, so it goes into the first pruned child instead of the ones pruning kept.
- **Opponent-node exploration is inverted.** As in plain MCTS, the exploration bonus (and progressive bias) is added before minimising, so opponent nodes prefer well-visited children.
- **Q is normalised with the parent's bounds**, and `normalise(value, 0, 1)` in `rollOut` changes nothing. Both are the same as in plain MCTS.
- **The prune ranking depends on script values that aren't comparable.** With default weights (0.5 for every type), the ranking is just the script's own value: maximising attribute scripts usually give `1.0`, wrappers like `END_TURN` give `0.5`, research gives `(4 − tier)/4`, and so on. `setWeights` only changes `SPAWN` and `RESEARCH_TECH`. It also assumes exactly 11 values (otherwise an `ArrayIndexOutOfBoundsException`). `getWeight()` ignores the per-script table.
- **Ties in `prune()` are broken arbitrarily.** The sorting comparator never returns 0 (`-1` on equal values), so among equal scores the kept children depend on `TreeSet` insertion order.
- **The search can't reach some actions.** Whatever the portfolio can't produce is never searched, including `DESTROY`, `DISBAND`, monuments and temples (see the [portfolio quirks](../portfolio/README.md#known-quirks-and-limitations)). The `act()` shortcut checks the raw action count, so with exactly one portfolio assignment the agent still runs a full search.
- **Rollouts are expensive and not counted.** Each rollout step evaluates the whole portfolio twice, and the second result is thrown away. This CPU cost is not counted in `num_fmcalls`.
- **JSON `"Rollouts"` and `"Force End"` are ignored**, and `FORCE_TURN_END` / `PRIORITIZE_ROOT` are never read.
- **Tuning keys don't match.** `getParameters()` lists `"K"` and `"rollout_depth"`, but `setParameterValue()` only accepts `"C"` and `"ROLLOUT_LENGTH"`, so tuning either does nothing.
- **The default constructor is unusable.** It leaves `portfolio` and `pruneHeuristic` as `null` (see *Usage*).
- **`AIStats` uses the wrong player ID.** It is created in the constructor, before `setPlayerIDs()`, so the stats printed by `result()` are always labelled player 0. Those stats are printed to stdout at the end of every game.
- **`copy()` returns `null`.** pMCTS can't be nested inside or cloned by other agents.
- **No reuse between calls.** The tree is thrown away after each `act()`.
