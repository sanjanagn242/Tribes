# MCTS: Monte Carlo Tree Search

`players.mcts` contains a closed-loop UCT Monte Carlo Tree Search agent for Tribes. Unlike EMCTS, where each node is a whole-turn plan, each MCTS node is the game state after **one action**. The tree can run past `END_TURN` into opponents' turns, and opponents are modelled as minimising our heuristic (paranoid search).

| File | Purpose |
|------|---------|
| [MCTSPlayer.java](MCTSPlayer.java) | The `Agent`: builds the root and returns the chosen action |
| [MCTSParams.java](MCTSParams.java) | MCTS-specific parameters (extends `AlgParams`) |
| [SingleTreeNode.java](SingleTreeNode.java) | Tree node with the search loop: selection, expansion, rollout and backup |

---

## How it works

### `MCTSPlayer.act(gs, ect)`

1. **Shortcut.** If only one action is available (normally `END_TURN`), return it without searching.
2. **Pick the root actions.**
   - If `PRIORITIZE_ROOT` is `true`, `Agent.determineActionGroup()` picks one group **at random** from *city*, *unit* and *tribe* actions, and the root only considers that group. The tribe group needs more than just `END_TURN`, and city `DESTROY` actions are left out. If no group is available, the agent returns `new EndTurn()`.
   - Otherwise the root considers every available action.
3. Create a root `SingleTreeNode` and set up the state heuristic with `params.getStateHeuristic(playerID, allPlayerIDs)`.
4. Run `mctsSearch(ect)` until the budget is used up.
5. Return the root action with the most visits (see *Final action choice*).

The tree is **rebuilt from scratch on every `act()` call**. Nothing is reused between calls.

### Search iteration (`SingleTreeNode.mctsSearch`)

Each iteration runs these steps:

1. **Tree policy (`treePolicy`).** Starting at the root, and while the game isn't over and `depth < ROLLOUT_LENGTH`:
   - if the current node has an unexpanded child, **expand** it and stop;
   - otherwise move to the child picked by **UCT**.
2. **Expansion (`expand`).**
   - If a forced turn end applies (see `FORCE_TURN_END`), pick `END_TURN`.
   - Otherwise pick a random unexpanded action.
   - Copy the state, apply the action (one forward-model call), and create a child holding the resulting state and its number of available actions.
   - At depth 0 with `PRIORITIZE_ROOT` on, actions come from the restricted root list. Deeper nodes use `getAllAvailableActions()`, which includes every action type.
3. **UCT selection (`uct`).** For each child:
   ```
   Q     = normalise(totValue / (nVisits + ε), bounds_min, bounds_max)
   UCT   = Q + K * sqrt( ln(parentVisits + 1) / (childVisits + ε) )
   UCT   = (UCT + ε) * (1 + ε * (rand - 0.5))     // random tie-break noise
   ```
   - If it is **our** tribe's move, take the child with the highest UCT.
   - If it is an **opponent's** move, take the child with the lowest UCT.
   - The tree is **closed-loop**: each node stores its own `GameState`, so descending doesn't advance the state again. A comment in the code explains why: random events such as examining ruins make open-loop replays inconsistent.
4. **Rollout (`rollOut`).**
   - With `ROLOUTS_ENABLED` on, copy the leaf state and play random actions, forcing `END_TURN` where it applies, until `depth >= ROLLOUT_LENGTH` or the game ends. Then evaluate the final state.
   - With `ROLOUTS_ENABLED` off, evaluate the leaf state directly.
   - Evaluation is `heuristic.evaluateState(rootState, state)`. It is relative to the state at the root, which suits `DIFF_HEURISTIC`.
5. **Backup (`backUp`).** From the leaf up to the root, increment `nVisits`, add the value to `totValue`, and widen each node's `[min, max]` value bounds. These bounds are used to normalise Q.

### Final action choice (`mostVisitedAction`)

- Pick the expanded root child with the most visits, breaking ties with small random noise.
- If **every** expanded child has the same visit count, pick the one with the highest mean value (`bestAction`).
- If no child was expanded, return index 0.

### Forward-model accounting

The root's `fmCallsCount` goes up:
- once per `GameState.advance()` (in expansion and in each rollout step), **and**
- once per UCT descent step (`uct()`), even though no state is advanced there.

So the `num_fmcalls` budget counts tree traversal steps as well as real simulation steps.

---

## Parameters

### MCTS-specific (`MCTSParams`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `K` | `double` | `√2 ≈ 1.414` | UCT exploration constant. Higher values explore more. |
| `ROLLOUT_LENGTH` | `int` | `10` | Search **horizon**, counted in actions from the root. It caps tree depth in `treePolicy` and is the depth at which rollouts stop. With rollouts off, it is just the maximum tree depth. |
| `ROLOUTS_ENABLED` | `boolean` | `true` | If `true`, random rollouts run from each new leaf up to `ROLLOUT_LENGTH`. If `false`, the leaf state is evaluated directly. (The field name has a typo.) |

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

| Field | Default | Used by MCTS? | Description |
|-------|---------|---------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ✅ | Which budget ends the search. |
| `num_iterations` | `200` | ✅ (`STOP_ITERATIONS`) | Iterations per `act()`. |
| `num_fmcalls` | `2000` | ✅ (`STOP_FMCALLS`) | Forward-model call budget per `act()`. The search stops when `fmCallsCount > num_fmcalls`. This is checked only after a full iteration, so the search can go over budget. |
| `num_time` | `40` | ❌ | Not read. `STOP_TIME` uses the `ElapsedCpuTimer` passed to `act()`. |
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | `ENTROPY_HEURISTIC` (0), `SIMPLE_HEURISTIC` (1) or `DIFF_HEURISTIC` (2). |
| `epsilon` | `1e-6` | ✅ | Avoids division by zero in Q and UCT, and sets the size of the tie-break noise. |
| `FORCE_TURN_END` | `5` | ✅ | At any depth `d > 0` with `d % FORCE_TURN_END == 0`, if `END_TURN` is feasible, expansion, UCT and rollouts all take `END_TURN`. Setting it above `ROLLOUT_LENGTH` turns this off. |
| `PRIORITIZE_ROOT` | `false` | ✅ | Limits the root to one randomly chosen action group (city, unit or tribe). See above. |

**Stop-type behaviour**

| Constant | Value | Stops when |
|----------|-------|------------|
| `STOP_TIME` | `0` | the remaining time is `<= 2 × average iteration time` or `<= 5 ms` |
| `STOP_ITERATIONS` | `1` | `numIters >= num_iterations` |
| `STOP_FMCALLS` | `2` | `fmCallsCount > num_fmcalls` |

### Tuning interface

`MCTSParams` overrides the tuning hooks:

| Method | Keys |
|--------|------|
| `setParameterValue` / `getParameterValue` | `"K"`, `"ROLLOUT_LENGTH"`, `"heuristic_method"` |
| `getParameters` | `"K"`, `"rollout_depth"`, `"heuristic_method"` |
| `getParameterValues` | `K ∈ {1.0, √2, 2.0}`, `rollout_depth ∈ {5, 8, 10, 12, 15}`, `heuristic_method ∈ {0, 1, 2}` |

---

## Node properties (`SingleTreeNode`)

| Field | Meaning |
|-------|---------|
| `params` | Shared `MCTSParams`. |
| `root` / `parent` / `children[]` | Tree links. `children` has one slot per available action, and `null` means not yet expanded. |
| `totValue` | Sum of the values backed up through this node. |
| `nVisits` | Visit count. |
| `bounds[2]` | Smallest and largest values backed up through this node. Used to normalise Q in UCT. |
| `m_depth` | Depth from the root (root = 0). |
| `state` | The `GameState` at this node (closed loop). |
| `actions` | Root only: the restricted action list when `PRIORITIZE_ROOT` is on. |
| `rootState` | The state at the root, used as the reference for the heuristic. |
| `rootStateHeuristic` | The `StateHeuristic` instance, shared by all nodes. |
| `fmCallsCount` | Forward-model call counter. Only the root's counter is used. |
| `playerID` | Our tribe ID. Used to tell maximising nodes from minimising ones. |
| `m_rnd` | Shared RNG, seeded from the agent seed. |

---

## Usage

### Default constructor

`new MCTSPlayer(seed)` uses the plain `MCTSParams` defaults: `STOP_TIME`, `SIMPLE_HEURISTIC`, `K = √2`, `ROLLOUT_LENGTH = 10`, rollouts on, `FORCE_TURN_END = 5`, `PRIORITIZE_ROOT = false`.

### From `Run.java`, `Tournament.java` and `Play.java`

Player type `"MCTS"`. [`Run.getAgent`](../../Run.java) builds it with:

```java
MCTSParams mctsParams = new MCTSParams();
mctsParams.stop_type        = mctsParams.STOP_FMCALLS;   // 2000 FM calls per action
mctsParams.heuristic_method = mctsParams.DIFF_HEURISTIC;
mctsParams.PRIORITIZE_ROOT  = true;
mctsParams.ROLLOUT_LENGTH   = MAX_LENGTH;                // JSON "Search Depth"
mctsParams.FORCE_TURN_END   = FORCE_TURN_END ? 5 : mctsParams.ROLLOUT_LENGTH + 1; // JSON "Force End"
mctsParams.ROLOUTS_ENABLED  = MCTS_ROLLOUTS;             // JSON "Rollouts"
return new MCTSPlayer(agentSeed, mctsParams);
```

These JSON config keys affect MCTS:

| JSON key | Maps to | Effect |
|----------|---------|--------|
| `"Search Depth"` | `ROLLOUT_LENGTH` | Search horizon |
| `"Force End"` | `FORCE_TURN_END` | `true` forces `END_TURN` every 5 plies. `false` sets it to `ROLLOUT_LENGTH + 1`, which turns forcing off. |
| `"Rollouts"` | `ROLOUTS_ENABLED` | Turns random rollouts on or off |

`K` and `num_fmcalls` cannot be set from JSON. They keep their defaults (√2 and 2000).

For example, the current `tournament.json` sets `"Search Depth": 20`, `"Force End": false` and `"Rollouts": false`. That gives a tree up to 20 plies deep, scored by the diff heuristic at each leaf, with no forced turn ends.

### Custom configuration

```java
MCTSParams p = new MCTSParams();
p.stop_type = p.STOP_FMCALLS;
p.num_fmcalls = 5000;
p.K = 1.0;
p.ROLLOUT_LENGTH = 15;
p.ROLOUTS_ENABLED = true;
p.heuristic_method = p.DIFF_HEURISTIC;
Agent mcts = new MCTSPlayer(seed, p);
```

---

## Known quirks and limitations

These describe the current code.

- **`copy()` returns `null`.** MCTS can't be nested inside or cloned by other agents.
- **The budget counts UCT descents.** `uct()` increments `fmCallsCount` without advancing a state, so deep trees use up the budget faster than the number of real simulations would suggest.
- **Opponent-node exploration is inverted.** At opponent nodes the agent picks the lowest `Q + exploration` value. The exploration bonus is still added, so these nodes prefer *well-visited* children instead of exploring under-visited ones.
- **Forced-end nodes get rebuilt.** When `FORCE_TURN_END` applies, only the `END_TURN` child is ever created, so the node never counts as fully expanded. Each visit calls `expand()` again and **replaces** the `END_TURN` child, throwing away its statistics. This only happens when forcing is on (`"Force End": true`).
- **Q is normalised with the parent's bounds.** UCT normalises each child's mean using the parent's `bounds`, not the child's own.
- **`normalise(value, 0, 1)` changes nothing.** In `rollOut` it just returns the raw heuristic value.
- **The root restriction is random.** With `PRIORITIZE_ROOT`, the action group (city, unit or tribe) is picked uniformly at random each call, not by value. In `determineActionGroup`, the unit group keeps `DISBAND` actions, while the city group removes `DESTROY`.
- **Tuning keys don't match.** `getParameters()` and `getParameterValues()` use `"rollout_depth"`, but `setParameterValue()` only accepts `"ROLLOUT_LENGTH"`, so tuning `rollout_depth` does nothing.
- **No reuse between calls.** The tree is thrown away after each `act()`.
