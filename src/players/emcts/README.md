# EMCTS: Evolutionary Monte Carlo Tree Search

`players.emcts` contains an EMCTS agent for Tribes. EMCTS is a hybrid of evolutionary search and MCTS. Each node in the tree is a **whole turn**: a sequence of actions that ends with `END_TURN`. A child node is a **mutation** of its parent's sequence. The tree is searched with a UCB-style score, and each candidate sequence is scored by a state heuristic.

| File | Purpose |
|------|---------|
| [EMCTSAgent.java](EMCTSAgent.java) | The agent: search loop, mutation, repair and evaluation |
| [EMCTSParams.java](EMCTSParams.java) | EMCTS-specific parameters (extends `AlgParams`) |
| [EMCTSTreeNode.java](EMCTSTreeNode.java) | A tree node holding an action sequence, its value and its exploration score |

---

## How it works

Each call to `act(gs, ect)` does the following:

1. **Build the root.** `randomActions()` makes a random sequence of up to `NODE_SIZE - 1` actions from `allGoodActions()` (all city, unit and tribe actions, except `DESTROY` and `DISBAND`).
   - `END_TURN` is only taken when it is the only legal action.
   - If the sequence reaches its length cap, an `END_TURN` is appended.
   - The root is evaluated, and its resulting `GameState` is cached on the node.
2. **Run search iterations** until the budget is used up (see `stop_type`):
   1. **Select.** `nodeToExpand()` walks down from the root. At each level it moves to a child whose `score` is higher than the current node's, and it stops at a leaf or when no child scores higher.
   2. **Mutate in a chain, `depth` times.** Starting from the selected node:
      - `mutate()` picks a random index `i` in the sequence, replays actions `0..i-1`, and replaces action `i` with a random legal action. It keeps the rest of the sequence.
      - `repair()` replays the new sequence on a fresh copy of the state. Any action that is no longer feasible, or that throws, is swapped for a random legal action. Replay stops early if the tribe's turn ends. An `END_TURN` is appended if the sequence doesn't end with one.
      - The new child is evaluated with the heuristic. If its value beats the best found so far, it becomes `bestNode`.
      - The child becomes the parent for the next mutation, so each iteration adds a chain `depth` nodes deep.
   3. **Backpropagate.** Going back up the chain for `depth` levels, each node calls `refreshScore(bias)`:
      ```
      score = value + bias * sqrt( ln( parentVisits / visits ) )
      ```
3. **Return** the first action of `bestNode`'s sequence. `returnNext()` also removes that action from the node's sequence.

The tree is **rebuilt from scratch on every `act()` call**. Nothing is reused between calls.

### Evaluation

`eval()` replays a node's sequence from the current state if the node has no cached state. It then calls:

```java
heuristic.evaluateState(old_gs, resultingState)
```

`old_gs` is a copy of the state at the start of `act()`, so diff-style heuristics measure the change made during this turn.

### Forward-model accounting

Every `GameState.advance()` goes through the private `advance()` method, which increments two counters:
- `fmCallsCount`: total forward-model calls during the current `act()`.
- `fmCallsRun`: forward-model calls during the current iteration.

These counters drive the `STOP_FMCALLS` budget.

---

## Parameters

### EMCTS-specific (`EMCTSParams`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bias` | `float` | `1f` | Exploration constant `C` in the node score `value + C·sqrt(ln(N_parent / N))`. Higher values favour less-visited nodes. |
| `depth` | `int` | `10` | Number of chained mutations per iteration, which is how many levels the tree can grow per iteration. Also the number of levels updated during backpropagation. |
| `NODE_SIZE` | `int` | `20` | Maximum length of the **initial random** sequence: `NODE_SIZE - 1` actions plus the closing `END_TURN`. Mutation and repair keep the length of the sequence they start from, plus a possible appended `END_TURN`. |

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

**Budget / stopping**

| Field | Default | Used by EMCTS? | Description |
|-------|---------|----------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ✅ | Which budget ends the search. Set it to one of the constants below. |
| `num_iterations` | `200` | ✅ (with `STOP_ITERATIONS`) | Maximum number of search iterations per `act()`. |
| `num_fmcalls` | `2000` | ✅ (with `STOP_FMCALLS`) | Forward-model call budget per `act()`. Search stops when `fmCallsCount > num_fmcalls`, or when the last iteration's calls are more than the budget left. |
| `num_time` | `40` | ❌ | Not read. `STOP_TIME` uses the `ElapsedCpuTimer` passed to `act()` instead. |

**Stop-type constants**

| Constant | Value | Behaviour in EMCTS |
|----------|-------|--------------------|
| `STOP_TIME` | `0` | Stops when the remaining time is `<= 2 × average iteration time` or `<= 5 ms`. |
| `STOP_ITERATIONS` | `1` | Stops after `num_iterations` iterations. |
| `STOP_FMCALLS` | `2` | Stops based on the `num_fmcalls` rule above. |

**Heuristic**

| Field | Default | Used? | Description |
|-------|---------|-------|-------------|
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | Which `StateHeuristic` scores nodes. `getStateHeuristic()` rebuilds it at the start of every `act()`. |

| Constant | Value | Heuristic class |
|----------|-------|-----------------|
| `ENTROPY_HEURISTIC` | `0` | `TribesEntropyHeuristic` |
| `SIMPLE_HEURISTIC` | `1` | `TribesSimpleHeuristic` |
| `DIFF_HEURISTIC` | `2` | `TribesDiffHeuristic` |

**Inherited fields EMCTS does not use:** `epsilon` (`1e-6`), `FORCE_TURN_END` (`5`), `PRIORITIZE_ROOT` (`false`). `getPruneHeuristic()` returns `null`. The parameter-tuning hooks (`setParameterValue`, `getParameterValue`, `getParameters`, `getParameterValues`) are not overridden in `EMCTSParams`, so they do nothing.

---

## Node properties (`EMCTSTreeNode`)

| Field | Type | Meaning |
|-------|------|---------|
| `sequence` | `ArrayList<Action>` | The turn plan this node stands for. |
| `parent` | `EMCTSTreeNode` | The node this one was mutated from. `null` for the root. |
| `children` | `ArrayList<EMCTSTreeNode>` | Mutations of this node. |
| `value` | `double` | Heuristic value of the state reached after running `sequence`. |
| `score` | `double` | Exploration score used during selection. Set by `refreshScore`. |
| `times_visited` | `int` | Visit count. It goes up in `visited()` **and every time `getChildren()` is called**. |
| `gs` | `GameState` | Cached state after running `sequence`, so the node can be re-evaluated without replaying. |

Key methods: `refreshScore(bias)`, `returnNext()` (pops and returns the first action), `unlink()` (clears references so the node can be garbage-collected), `addChild`, `setGs` / `getGs`.

---

## Agent state (`EMCTSAgent`)

| Field | Meaning |
|-------|---------|
| `m_rnd` | `Random` seeded from the constructor `seed`. Used for every random choice. |
| `heuristic` | The `StateHeuristic` built from `params` at the start of each `act()`. |
| `params` | The `EMCTSParams` instance. |
| `root` | Root of the current search tree (a random initial sequence). |
| `bestNode` | The highest-`value` node seen so far. Its first action is returned. |
| `old_gs` | Copy of the state at the start of `act()`. Used as the heuristic's reference state. |
| `fmCallsCount` / `fmCallsRun` | Forward-model call counters (see above). |

---

## Usage

### From `Run.java`

Use the player type `"EMCTS"` (`Run.PlayerType.EMCTS`) in your run or tournament config. [`Run.getAgent`](../../Run.java) builds it with:

```java
EMCTSParams emctsParams = new EMCTSParams();
emctsParams.stop_type = emctsParams.STOP_FMCALLS;          // budget = num_fmcalls (2000)
emctsParams.heuristic_method = emctsParams.DIFF_HEURISTIC; // TribesDiffHeuristic
return new EMCTSAgent(agentSeed, emctsParams);
```

So when launched through `Run`, EMCTS uses a **2000 forward-model-call budget** and the **diff heuristic**. `bias`, `depth` and `NODE_SIZE` keep their defaults.

### Custom configuration

```java
EMCTSParams p = new EMCTSParams();
p.stop_type = p.STOP_ITERATIONS;
p.num_iterations = 500;
p.depth = 5;
p.bias = 0.5f;
p.NODE_SIZE = 30;
p.heuristic_method = p.DIFF_HEURISTIC;
Agent emcts = new EMCTSAgent(seed, p);
```

---

## Known quirks and limitations

These describe the current code. They are worth knowing when tuning or extending the agent.

- **`copy()` returns `null`.** Anything that relies on `Agent.copy()`, such as nesting EMCTS inside another agent, will not work.
- **Integer division in `refreshScore`.** `getParentVisited() / times_visited` divides two `int`s, so the ratio is floored before `Math.log`. If the parent has fewer visits than the child, the result is `ln(0) = -∞` and `sqrt` gives `NaN`. If the visits are equal, the bonus is `0`.
- **Selection loop.** In `nodeToExpand()`, the `if (!bigger)` check sits inside the `for` loop over children. Selection can therefore stop at the first child that doesn't beat the current score, without looking at the rest.
- **Backpropagation stops at a fixed depth.** It always updates exactly `depth` levels from the end of the chain, so nodes above where the chain started are not updated.
- **Side effect in `getChildren()`.** Calling it increments `times_visited`, so visit counts include selection traversals.
- **`returnNext()` changes `bestNode`.** This doesn't matter today because the tree is thrown away after each `act()`.
- **No reuse between turns.** The plan is searched again for every single action, even though each node stands for a whole turn.
- **`extend()` is unused.** The call is commented out in the search loop. It would try to make a node's sequence longer with extra random non-`END_TURN` actions.
