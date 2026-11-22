# MC: Flat Monte Carlo

`players.mc` contains a flat Monte Carlo agent for Tribes. There is no tree. For each decision it repeatedly picks a root action **uniformly at random**, plays a random rollout that starts with that action, scores the final state with a heuristic, and keeps a running mean per root action. Rollouts can run past `END_TURN` into opponents' turns, where opponents also play random actions. Before each rollout, the agent gives opponents a random set of technologies, because their tech trees are hidden in the observation.

| File | Purpose |
|------|---------|
| [MonteCarloAgent.java](MonteCarloAgent.java) | The `Agent`: root sampling, rollouts, forward-model counting and opponent tech injection |
| [MCParams.java](MCParams.java) | MC-specific parameters (extends `AlgParams`) |

---

## How it works

### `MonteCarloAgent.act(gs, ect)`

1. **Set up the heuristic.** `params.getStateHeuristic(playerID, allPlayerIDs)` is called again on every `act()`.
2. **Turn bookkeeping.** `lastTurn` and `actionTurnCounter` count how many actions were returned in the current tick. They are only used in commented-out debug output.
3. **Shortcut.** If `gs.getAllAvailableActions()` has only one action (normally `END_TURN`), return it without searching.
4. **Pick the root actions.**
   - If `PRIORITIZE_ROOT` is `true`, `Agent.determineActionGroup()` picks one group **at random** from *city*, *unit* and *tribe* actions, and only that group is sampled. The tribe group needs more than just `END_TURN`, and city `DESTROY` actions are left out. If no group is available, the agent returns `new EndTurn(gs.getActiveTribeID())`.
   - Otherwise every available action is a candidate.
5. **Set the iteration budget.** `params.num_iterations = rootActions.size() * N_ROLLOUT_MULT`. This **overwrites** the shared params object on every call.
6. **Main loop.** Until a stop condition fires:
   1. Pick `rootActionIndex` uniformly at random. While the picked action is `END_TURN`, pick again.
   2. Run `rollout(gs, act)` to get a score.
   3. Add the score to `scores[rootActionIndex]` (a `StatSummary`, created on first use).
   4. If this action's **new mean** is greater than `maxQ`, set `maxQ` to it and `bestAction = act`.
   5. Stop if `stop_type == STOP_FMCALLS && fmCalls >= num_fmcalls`, or if `stop_type == STOP_ITERATIONS && nRollouts >= num_iterations`.
7. Return `bestAction`.

Nothing is kept between `act()` calls except the turn counters.

### Rollout (`rollout(gs, act)`)

```
gsCopy = copyGameState(gs)                 // gs.copy() + opponent tech injection
step = 0
turnEndCountDown = FORCE_TURN_END
loop:
    if turnEndCountDown == 0 and EndTurn(activeTribe) is feasible:
        advance(gsCopy, EndTurn)           // forced turn end; `act` is skipped
        turnEndCountDown = FORCE_TURN_END
    else:
        advance(gsCopy, act)
        turnEndCountDown--
    step++
    end = gameOver or step == ROLLOUT_LENGTH or (STOP_FMCALLS and fmCalls >= num_fmcalls)
    if not end:
        actions = gsCopy.getAllAvailableActions()
        if |actions| == 1:
            act = actions[0]               // should be END_TURN
            if act is END_TURN: turnEndCountDown = FORCE_TURN_END + 1
            else: print a warning
        else:
            act = random action that is not END_TURN
return heuristic.evaluateState(gs, gsCopy)
```

- The first step applies the root action. Every later step is a random action for whichever tribe is active, so after an `END_TURN` the rollout plays the opponents' turns with random actions too.
- `END_TURN` only happens in a rollout when it is forced by `FORCE_TURN_END` or when it is the only available action. When it is the only action, the countdown is set to `FORCE_TURN_END + 1`, so it is back to `FORCE_TURN_END` after that `END_TURN` is applied.
- `ROLLOUT_LENGTH` counts **every** step, including forced `END_TURN`s and opponent actions.
- Random rollout actions are drawn from `getAllAvailableActions()`, so they can include `DESTROY` and `DISBAND`.
- Evaluation is `heuristic.evaluateState(gs, gsCopy)`: the end state is compared with the original state passed to `act()`, which suits `DIFF_HEURISTIC`.

### Opponent tech injection (`copyGameState`, `initTribesResearch`)

The observation the game gives an agent is `gs.copy(playerIdx)`. In it, every other tribe's tech tree is replaced with an empty `new TechnologyTree()` (see `Tribe.copy(hideInfo)`). To make up for this, every rollout copy researches `(int)(tick / 2.0)` random technologies for each opponent with `TechnologyTree.researchAtRandom(m_rnd)`. A new random set is drawn for each rollout. The copy passed to the heuristic as the reference state (`gs`) does **not** get this injection.

### Forward-model accounting

`fmCalls` goes up by one per `GameState.advance()`, through the private `advance()` wrapper. This covers the root action, every random rollout action and every forced `END_TURN`. `gs.copy()` and the tech injection are not counted.

With `STOP_FMCALLS`, the budget is checked after **every rollout step** as well as after every rollout. The search therefore never goes over `num_fmcalls`, but the last rollout can be cut short. Its score is still added to the root action's mean.

---

## Parameters

### MC-specific (`MCParams`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ROLLOUT_LENGTH` | `int` | `20` | Maximum rollout length in forward-model steps, counting the root action, forced `END_TURN`s and opponent actions. The rollout ends when `step == ROLLOUT_LENGTH`. |
| `N_ROLLOUT_MULT` | `int` | `3` | Rollouts per root action under `STOP_ITERATIONS`. `act()` sets `num_iterations = rootActions.size() * N_ROLLOUT_MULT`. It has no effect under `STOP_FMCALLS`. |

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

| Field | Default | Used by MC? | Description |
|-------|---------|-------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ⚠️ partly | Only `STOP_ITERATIONS` and `STOP_FMCALLS` are handled. `STOP_TIME` has no stop condition (see quirks). |
| `num_iterations` | `200` | ❌ (overwritten) | Recomputed from `N_ROLLOUT_MULT` on every `act()`, so any value you set is ignored. |
| `num_fmcalls` | `2000` | ✅ (`STOP_FMCALLS`) | Forward-model call budget per `act()`. Checked with `>=` after each rollout step and each rollout. |
| `num_time` | `40` | ❌ | Not read. The `ElapsedCpuTimer` passed to `act()` is not read either. |
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | `ENTROPY_HEURISTIC` (0), `SIMPLE_HEURISTIC` (1) or `DIFF_HEURISTIC` (2). |
| `epsilon` | `1e-6` | ❌ | Not read. There is no tie-break noise; the first action to reach a new `maxQ` wins. |
| `FORCE_TURN_END` | `5` | ✅ | In a rollout, every `FORCE_TURN_END` non-forced steps, apply `END_TURN` instead of the next action if it is feasible. Setting it above `ROLLOUT_LENGTH` turns this off. |
| `PRIORITIZE_ROOT` | `false` | ✅ | Limits root sampling to one randomly chosen action group (city, unit or tribe). See above. |

**Stop-type behaviour**

| Constant | Value | Stops when |
|----------|-------|------------|
| `STOP_TIME` | `0` | never (not implemented) |
| `STOP_ITERATIONS` | `1` | `nRollouts >= rootActions.size() * N_ROLLOUT_MULT` |
| `STOP_FMCALLS` | `2` | `fmCalls >= num_fmcalls` |

`MCParams` does not override the tuning hooks (`setParameterValue`, `getParameters`, ...). It inherits the `AlgParams` no-ops.

---

## Usage

### Constructor

There is only one constructor, `new MonteCarloAgent(seed, params)`. `m_rnd` is seeded with `seed`. Plain `new MCParams()` defaults are `STOP_TIME`, `SIMPLE_HEURISTIC`, `ROLLOUT_LENGTH = 20`, `N_ROLLOUT_MULT = 3`, `FORCE_TURN_END = 5`, `PRIORITIZE_ROOT = false`. Because `STOP_TIME` is not handled, **set `stop_type` before using it** (see quirks).

### From `Run.java`, `Tournament.java` and `Play.java`

Player type `"MC"`. [`Run.getAgent`](../../Run.java) builds it with:

```java
MCParams mcparams = new MCParams();
mcparams.stop_type        = mcparams.STOP_FMCALLS;     // 2000 FM calls per action
mcparams.heuristic_method = mcparams.DIFF_HEURISTIC;
mcparams.PRIORITIZE_ROOT  = true;
mcparams.ROLLOUT_LENGTH   = MAX_LENGTH;                // JSON "Search Depth"
mcparams.FORCE_TURN_END   = FORCE_TURN_END ? 5 : mcparams.ROLLOUT_LENGTH + 1; // JSON "Force End"
return new MonteCarloAgent(agentSeed, mcparams);
```

These JSON config keys affect MC:

| JSON key | Maps to | Effect |
|----------|---------|--------|
| `"Search Depth"` | `ROLLOUT_LENGTH` | Rollout length in steps |
| `"Force End"` | `FORCE_TURN_END` | `true` forces `END_TURN` every 5 steps. `false` sets it to `ROLLOUT_LENGTH + 1`, which turns forcing off. |

`"Rollouts"` does not affect MC. `num_fmcalls` and `N_ROLLOUT_MULT` cannot be set from JSON and keep their defaults (2000 and 3).

For example, the current `tournament.json` sets `"Search Depth": 20` and `"Force End": false`. That gives rollouts of up to 20 steps with no forced turn ends, scored by the diff heuristic. Since rollout actions avoid `END_TURN`, these rollouts only move into opponents' turns when `END_TURN` is the only action left.

If `Tournament.java` finds no config file (or an empty one), it plays MC vs MC without setting `Run.MAX_LENGTH`, so `ROLLOUT_LENGTH = 0` and `FORCE_TURN_END = 1` (see quirks).

### Custom configuration

```java
MCParams p = new MCParams();
p.stop_type = p.STOP_FMCALLS;
p.num_fmcalls = 5000;
p.ROLLOUT_LENGTH = 15;
p.FORCE_TURN_END = 16;          // > ROLLOUT_LENGTH: no forced turn ends
p.PRIORITIZE_ROOT = false;
p.heuristic_method = p.DIFF_HEURISTIC;
Agent mc = new MonteCarloAgent(seed, p);
```

---

## Known quirks and limitations

These describe the current code.

- **`STOP_TIME` never stops.** The main loop only ends on `STOP_ITERATIONS` or `STOP_FMCALLS`, and rollouts only stop early on `STOP_FMCALLS`. With the default `stop_type` (`STOP_TIME`), `act()` loops forever. The `ElapsedCpuTimer` is ignored.
- **`maxQ` is a high-water mark, not the current best mean.** `bestAction` only changes when some action's mean goes **above the highest mean seen so far**. If an action gets a lucky early score and its mean later drops, it is still returned unless another action's mean beats that early peak. The result is not the action with the highest final mean.
- **Root sampling is random, not round-robin.** Root actions are drawn uniformly with replacement, so some may get no rollouts and others only one before the budget runs out.
- **`num_iterations` is overwritten.** `act()` replaces it with `rootActions.size() * N_ROLLOUT_MULT` in the shared params object, so a configured value never takes effect.
- **`END_TURN` is never chosen voluntarily.** At the root, `END_TURN` is resampled away. The agent returns `END_TURN` only through the single-action shortcut, or `new EndTurn(...)` when `determineActionGroup` returns `null`. It keeps acting until nothing else is available.
- **`DESTROY` and `DISBAND` are not filtered consistently.** With `PRIORITIZE_ROOT`, the city group removes `DESTROY`, but the unit group keeps `DISBAND` (from `determineActionGroup`). Without `PRIORITIZE_ROOT`, and in every rollout step, both are possible.
- **Forced end can switch itself off.** If the countdown reaches 0 while `END_TURN` is not feasible (for example while a city is levelling up), the planned action runs and the countdown goes negative. It never reaches 0 again in that rollout unless a single-action `END_TURN` resets it.
- **Tech injection adds noise to the diff heuristic.** The injected opponent techs exist only in the rollout copy, not in the reference state `gs`. `TribesDiffHeuristic` counts opponents' tech gains (`4 ×` techs researched), so each rollout score includes a random penalty from the injection that has nothing to do with the root action.
- **Truncated rollouts are scored.** When the FM budget runs out mid-rollout, the partial rollout's score is still added to that root action's mean.
- **`ROLLOUT_LENGTH <= 0` means no length limit.** The check is `step == ROLLOUT_LENGTH` with `step` starting at 1. In the no-config `Tournament` fallback (`ROLLOUT_LENGTH = 0`, `FORCE_TURN_END = 1`), each rollout runs until the game ends or the FM budget is used up, alternating one action with a forced `END_TURN`.
- **`copy()` returns `null`.** MC can't be nested inside or cloned by other agents.
- **Unused state.** `lastTurn`, `actionTurnCounter` and several imports are only referenced by commented-out debug code.
