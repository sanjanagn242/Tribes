# OSLA: One-Step Look-Ahead

`players.osla` contains a greedy one-step look-ahead agent for Tribes. For each decision it tries every available action once on a copy of the state, scores the resulting state with a heuristic, and returns the action with the highest (slightly noised) score. There is no search beyond one action, no rollout and no opponent model.

| File | Purpose |
|------|---------|
| [OneStepLookAheadAgent.java](OneStepLookAheadAgent.java) | The `Agent`: evaluates each action one step ahead and returns the best |
| [OSLAParams.java](OSLAParams.java) | Parameters. An empty subclass of `AlgParams` with no fields of its own |

---

## How it works

### `OneStepLookAheadAgent.act(gs, ect)`

1. **Shortcut.** If `gs.getAllAvailableActions()` has only one action (normally `END_TURN`), return it without evaluating.
2. **Set up.** Reset `fmCalls` to 0, set `bestAction = new EndTurn()` and `maxQ = -∞`, and build the heuristic with `params.getStateHeuristic(playerID, allPlayerIDs)`.
3. **Evaluate actions in list order.** For `actionIdx = 0, 1, ...`:
   - Skip the action if its type is `END_TURN`, `DESTROY` or `DISBAND`.
   - Otherwise:
     ```
     gsCopy = gs.copy()
     advance(gsCopy, act, computeActions = false)    // one forward-model call
     Q = heuristic.evaluateState(gs, gsCopy)
     Q = (Q + ε) * (1 + ε * (rand - 0.5))            // random tie-break noise
     if Q > maxQ: maxQ = Q, bestAction = act
     ```
   - Stop when every action has been looked at, or when `stop_type == STOP_FMCALLS && fmCalls >= num_fmcalls`.
4. Return `bestAction`.

`getAllAvailableActions()` lists **tribe** actions first, then **city** actions, then **unit** actions. This order matters when the FM budget cuts the loop short.

Nothing is kept between `act()` calls.

### Forward-model accounting

`fmCalls` goes up by one per `GameState.advance()`, which is one per evaluated action. Skipped actions (`END_TURN`, `DESTROY`, `DISBAND`) and `gs.copy()` are not counted. The budget is checked after every action, so the agent never goes over `num_fmcalls`.

---

## Parameters

### OSLA-specific (`OSLAParams`)

`OSLAParams` adds no fields and does not override any `AlgParams` methods.

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

| Field | Default | Used by OSLA? | Description |
|-------|---------|---------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ⚠️ partly | Only `STOP_FMCALLS` is checked. With any other value, every action is evaluated. |
| `num_iterations` | `200` | ❌ | Not read. |
| `num_fmcalls` | `2000` | ✅ (`STOP_FMCALLS`) | Maximum number of actions evaluated per `act()`. |
| `num_time` | `40` | ❌ | Not read. The `ElapsedCpuTimer` passed to `act()` is not read either. |
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | `ENTROPY_HEURISTIC` (0), `SIMPLE_HEURISTIC` (1) or `DIFF_HEURISTIC` (2). |
| `epsilon` | `1e-6` | ✅ | Sets the size of the tie-break noise added to each score. |
| `FORCE_TURN_END` | `5` | ❌ | Not read. |
| `PRIORITIZE_ROOT` | `false` | ❌ | Not read. All available actions are always considered. |

**Stop-type behaviour**

| Constant | Value | Stops when |
|----------|-------|------------|
| `STOP_TIME` | `0` | all actions have been evaluated |
| `STOP_ITERATIONS` | `1` | all actions have been evaluated |
| `STOP_FMCALLS` | `2` | all actions have been evaluated, or `fmCalls >= num_fmcalls` |

---

## Usage

### Constructor

There is only one constructor, `new OneStepLookAheadAgent(seed, params)`. `m_rnd` (used for the noise) is seeded with `seed`. With plain `new OSLAParams()` the agent uses `SIMPLE_HEURISTIC` and, since `STOP_TIME` is not checked, evaluates every action.

### From `Run.java`, `Tournament.java` and `Play.java`

Player type `"OSLA"`. [`Run.getAgent`](../../Run.java) builds it with:

```java
OSLAParams oslaParams = new OSLAParams();
oslaParams.stop_type        = oslaParams.STOP_FMCALLS;   // upper bound: 2000 evaluated actions
oslaParams.heuristic_method = oslaParams.DIFF_HEURISTIC;
return new OneStepLookAheadAgent(agentSeed, oslaParams);
```

No JSON config keys affect OSLA. `"Search Depth"`, `"Force End"` and `"Rollouts"` are not used, and `num_fmcalls` keeps its default of 2000.

### Custom configuration

```java
OSLAParams p = new OSLAParams();
p.stop_type = p.STOP_FMCALLS;
p.num_fmcalls = 500;
p.heuristic_method = p.SIMPLE_HEURISTIC;
Agent osla = new OneStepLookAheadAgent(seed, p);
```

---

## Known quirks and limitations

These describe the current code.

- **`END_TURN` is never chosen voluntarily.** `END_TURN` is always skipped, so the agent keeps acting until `END_TURN` is the only available action (the shortcut). The only other way it returns `END_TURN` is the fallback below.
- **The fallback `EndTurn` has tribe ID 0.** `bestAction` starts as `new EndTurn()`, which leaves `tribeId` at its default of 0. It is returned when no action gets evaluated, for example when everything except `END_TURN` is `DESTROY` or `DISBAND`. `EndTurn.isFeasible` then checks `canEndTurn` for tribe 0, not for the active tribe.
- **The FM budget cuts off unit actions first.** Actions are evaluated in list order (tribe, city, unit). If there are more than `num_fmcalls` evaluable actions, the ones at the end of the list, which are unit actions, are never looked at.
- **One step only.** Each action is scored only by the state right after it. Later actions, including the rest of our own turn and opponents' replies, are not simulated.
- **`DESTROY` and `DISBAND` are never played.** They are filtered out in every state.
- **The heuristic is relative to the current state.** `evaluateState(gs, gsCopy)` compares the state after the action with the state before it, which suits `DIFF_HEURISTIC`.
- **`copy()` returns `null`.** OSLA can't be nested inside or cloned by other agents.
