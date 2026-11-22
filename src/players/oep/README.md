# OEP: Online Evolutionary Planning

`players.oep` contains an Online Evolutionary Planning agent for Tribes. Each individual is a **plan for the rest of the current turn**: a sequence of up to `NODE_SIZE` of our own actions, usually ending in `END_TURN`. Plans never extend into opponents' turns. A population of plans is evolved with truncation, tournament selection, uniform crossover, repair and mutation, and each plan is scored by the state heuristic at the state reached after its actions. The population is rebuilt on every `act()` call, and only the first action of the best plan is returned.

| File | Purpose |
|------|---------|
| [OEPAgent.java](OEPAgent.java) | The `Agent`: the evolutionary loop and all genetic operators |
| [OEPParams.java](OEPParams.java) | OEP-specific parameters (extends `AlgParams`) |
| [Individual.java](Individual.java) | A plan: action list, the game state it leads to, and its value |

---

## How it works

### `OEPAgent.act(gs, ect)`

1. **Reset.** Set `fmCallsCount = 0` and `numIters = 0`.
2. **Initial population.** Build `POP_SIZE` individuals with `randomActions(gs.copy())` (see below). These forward-model calls count towards the budget.
3. Create the state heuristic with `params.getStateHeuristic(playerID, allPlayerIDs)`, set `bestIndividual = null` and keep `root = gs.copy()` as the reference state for evaluation.
4. **Generation loop** (see *One generation*) until the stop condition is met.
5. Return `bestIndividual.returnNext()`, which removes and returns the first action of the best plan from the most recent evaluation.

There is **no shortcut** when only one action is available, and **nothing is reused between calls**: every `act()` call evolves a new population from scratch, even if it is the same turn.

### Random plans (`randomActions`)

While the game isn't over, it is still our turn, and the plan has fewer than `NODE_SIZE - 1` actions:
- pick a random action from `Agent.allGoodActions(gs)` (every city, unit and tribe action, minus city `DESTROY` and unit `DISBAND`);
- if it isn't `END_TURN`, apply it and add it to the plan;
- if it is `END_TURN` and it is the **only** action available, apply it and add it (the turn ends);
- otherwise (`END_TURN` picked while other actions exist) discard it and pick again.

If the plan reached `NODE_SIZE - 1` actions and its last action is not `END_TURN`, the `END_TURN` action from the current list is appended **without** being applied. The individual stores a copy of the final state.

So random plans only end their turn early when nothing else is possible.

### One generation

1. **Evaluate.** For every individual:
   ```
   value = heuristic.evaluateState(root.copy(), individual.gs.copy())
   ```
   No forward-model calls are used here: each individual already holds the state its plan leads to.
2. **Sort and pick the best.** `Collections.sort(population, Collections.reverseOrder())` sorts by value ascending (worst first), so `bestIndividual` is the **last** element.
3. **Cull.** Remove the first `(int)(size × KILL_RATE)` individuals, i.e. the worst ones. With the defaults, 25 of 50 survive.
4. **Budget check** (`STOP_FMCALLS` only, see *Stop conditions*). May `break` here.
5. **Crossover (`procreate`).**
   - If the number of survivors is odd, remove one at random (`person1`) first.
   - Repeatedly run `tournamentSelection` on the remaining survivors, put the two winners in `group1` and `group2`, and remove them from the pool, until the pool is empty.
   - Cross `group1[i]` with `group2[i]` for each pair. The children replace the population.
   - If `person1` was set aside, cross it with a random **child** and add the result.
6. **Budget check** again (`STOP_FMCALLS` only).
7. **Build the next population (`shiftPop`).**
   - Take the **last** individual of the child population, call `shift()` on it (see below) and keep it.
   - Add `POP_SIZE/2 - 1` mutated copies of that same individual (`mutateInd`).
   - Fill up to `POP_SIZE` with fresh `randomActions` plans.
8. **Stop check** (see *Stop conditions*).

### Tournament selection (`tournamentSelection`)

- If the pool has fewer than `TOURNAMENT_SIZE` individuals: return the same individual twice if there is only one, otherwise sort the pool by value (best first) and return the top two.
- Otherwise run two independent tournaments. Each samples `TOURNAMENT_SIZE` **distinct** individuals; the best of each tournament is a winner.

The two tournaments can pick the same individual. Then only one individual leaves the pool, so a pool of `n` produces between `n/2` and `n` children.

### Crossover (`crossover`)

Balanced uniform crossover on the two action lists, followed by repair:

```
in1 = shorter parent's actions, in2 = longer parent's actions
smallSize = len(in1) - 1
if lengths differ: drop the last action of in1              // modifies the parent's list in place
in1amount = len(in1) / 2 ;  in2amount = len(in1) - in1amount
for i in 0 .. len(in2)-1:
    if i >= smallSize:          child += in2[i]             // tail always from the longer parent
    elif in1amount == 0:        child += in2[i]
    elif in2amount == 0:        child += in1[i]
    else: 50/50 pick, decrementing the chosen parent's counter
child = repair(rootStateCopy, child)
```

The child has the longer parent's length before repair. Genes are aligned by **position**, not by which unit or city they belong to.

### Repair (`repair`)

The child is replayed from a copy of the current state, one gene at a time:

- If it is no longer our turn, stop and return what has been built so far.
- **Mutation.** With a small probability (see *Mutation probability*), and **at most once per child**, replace this gene with a random action from `allGoodActions`.
- Otherwise, if the gene passes `checkActionFeasibility`, apply and keep it. If it doesn't, apply and keep a random action from `allGoodActions` instead.
- If applying an action throws, the action is dropped and the state is restored from a copy taken before the gene.

After the last gene, if it is still our turn, the `END_TURN` action is appended **without** being applied.

A random replacement can itself be `END_TURN`, which ends the turn and cuts the plan short.

### Feasibility check (`checkActionFeasibility`)

Returns `false` if the game is over, if a unit action's actor is not a `Unit` of the active tribe, or if a city action's actor is not a `City` of the active tribe. Otherwise it returns `action.isFeasible(gs)`, treating exceptions as infeasible. It does not use the forward model.

### Mutation probability

Both `repair` and `mutateInd` decide per gene with:

```java
int chance = m_rnd.nextInt((int)(MUTATION_RATE * 100));   // uniform in [0, 100·MUTATION_RATE - 1]
if (m_rnd.nextInt(100) < chance) mutate();
```

The effective per-gene probability is `(100·MUTATION_RATE − 1) / 200`, which is **4.5 %** for the default `MUTATION_RATE = 0.1`, not 10 %.

### Shift (`shift`) and mutation (`mutateInd`)

- **`shift`** drops the plan's first action, replays the remaining actions from the current state while they are feasible, then **appends** random actions while it is still our turn and fewer than `NODE_SIZE` actions have been played. The final state is stored and scored. No `END_TURN` is appended, and random actions here may include `END_TURN`.
- **`mutateInd`** replays the plan from the current state. Each gene is replaced by a random action with the mutation probability above (no once-only limit); otherwise it is kept if feasible and replaced by a random action if not. It stops early if the turn ends and does not append `END_TURN`.

### Stop conditions

| `stop_type` | Checked | Stops when |
|-------------|---------|------------|
| `STOP_TIME` (0) | end of each generation | remaining time `<= 2 × average generation time` or `<= 5 ms`. The average leaves out the initial population. |
| `STOP_ITERATIONS` (1) | end of each generation | `numIters >= num_iterations` |
| `STOP_FMCALLS` (2) | after the cull, after crossover, and at the end of each generation | after the cull or crossover: `fmCallsCount > num_fmcalls`, or (first generation only) `fmCallsCount >= 0.9 × num_fmcalls`. At the end: `fmCallsCount > num_fmcalls`, or the generation just run (`fmCallsRun`) used more calls than are left. |

The last check in a generation always happens **after** `shiftPop`, and the `STOP_FMCALLS` breaks happen before it. Either way, the last population built is never evaluated. The returned action always comes from the evaluation at the start of the last generation.

### Forward-model accounting

`advance()` increments `fmCallsCount` and `fmCallsRun` once per `GameState.advance(action, true)`. That covers the initial population, repair, shift, `mutateInd` and random fill-ins. Feasibility checks and heuristic evaluations are not counted. The appended `END_TURN` actions are never applied, so they cost nothing.

With the `Run.java` settings (2000 calls, 50 plans of up to 20 actions), when plans are close to full length the initial population costs about 950 calls and one generation about 1,200–1,500 more. The loop then usually stops after the first generation, and the returned action is the first action of the best **initial random** plan.

---

## Parameters

### OEP-specific (`OEPParams`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `POP_SIZE` | `int` | `50` | Number of individuals in the initial population and in each population built by `shiftPop`. |
| `MUTATION_RATE` | `double` | `0.1` | Controls the per-gene mutation chance in `repair` and `mutateInd`. The effective probability is `(100·MUTATION_RATE − 1)/200` (4.5 % by default). Must be at least `0.01`, otherwise `nextInt(0)` throws. |
| `KILL_RATE` | `double` | `0.5` | Fraction of the population (the worst) removed each generation before crossover. |
| `NODE_SIZE` | `int` | `20` | Maximum plan length. Random plans get `NODE_SIZE - 1` actions plus an appended `END_TURN`, and `shift` refills up to `NODE_SIZE` played actions. |
| `TOURNAMENT_SIZE` | `int` | `5` | Number of distinct individuals sampled in each tournament. |

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

| Field | Default | Used by OEP? | Description |
|-------|---------|--------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ✅ | Which budget ends the search. All three types are supported. |
| `num_iterations` | `200` | ✅ (`STOP_ITERATIONS`) | Generations per `act()`. |
| `num_fmcalls` | `2000` | ✅ (`STOP_FMCALLS`) | Forward-model call budget per `act()`, including the initial population. |
| `num_time` | `40` | ❌ | Not read. `STOP_TIME` uses the `ElapsedCpuTimer` passed to `act()`. |
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | `ENTROPY_HEURISTIC` (0), `SIMPLE_HEURISTIC` (1) or `DIFF_HEURISTIC` (2). |
| `epsilon` | `1e-6` | ❌ | Not read. |
| `FORCE_TURN_END` | `5` | ❌ | Not read. Plans always stop at the end of our turn. |
| `PRIORITIZE_ROOT` | `false` | ❌ | Not read. |

### Tuning interface

`OEPParams` does not override `setParameterValue`, `getParameterValue`, `getParameters` or `getParameterValues`, so the `AlgParams` no-op versions apply and OEP can't be tuned through that interface.

---

## Individual properties (`Individual`)

| Field / method | Meaning |
|----------------|---------|
| `actions` | The plan: an ordered list of `Action`s for the current turn. |
| `gs` | The `GameState` reached after applying the plan (the appended `END_TURN` is not applied). Used by `eval()`. |
| `value` | Heuristic score of `gs` relative to the root state. |
| `returnNext()` | Removes and returns the first action, or `null` if the plan is empty. |
| `shift()` | Replaces `actions` with a new list without the first action. |
| `compareTo()` | Orders by value **descending** (higher value sorts first). `act()` sorts with `reverseOrder()`, so there the best ends up last. |

---

## Usage

### Default constructor

There is no seed-only constructor. `new OEPAgent(seed, new OEPParams())` uses the plain defaults: `STOP_TIME`, `SIMPLE_HEURISTIC`, `POP_SIZE = 50`, `MUTATION_RATE = 0.1`, `KILL_RATE = 0.5`, `NODE_SIZE = 20`, `TOURNAMENT_SIZE = 5`.

### From `Run.java`, `Tournament.java` and `Play.java`

Player type `"OEP"`. [`Run.getAgent`](../../Run.java) builds it with:

```java
OEPParams oepParams = new OEPParams();
oepParams.stop_type        = oepParams.STOP_FMCALLS;   // 2000 FM calls per action
oepParams.heuristic_method = oepParams.DIFF_HEURISTIC;
return new OEPAgent(agentSeed, oepParams);
```

**No JSON config key affects OEP.** `"Search Depth"`, `"Population Size"`, `"Force End"` and `"Rollouts"` are not passed to it, so it always runs with `POP_SIZE = 50`, `NODE_SIZE = 20` and `num_fmcalls = 2000`.

### Custom configuration

```java
OEPParams p = new OEPParams();
p.stop_type = p.STOP_FMCALLS;
p.num_fmcalls = 10000;
p.heuristic_method = p.DIFF_HEURISTIC;
p.POP_SIZE = 30;
p.NODE_SIZE = 15;
p.KILL_RATE = 0.5;
p.TOURNAMENT_SIZE = 4;
p.MUTATION_RATE = 0.2;
Agent oep = new OEPAgent(seed, p);
```

---

## Known quirks and limitations

These describe the current code.

- **`copy()` returns `null`.** OEP can't be nested inside or cloned by other agents.
- **Most crossover children are thrown away.** `shiftPop` keeps only the **last** child in the list, which was never evaluated and is not the best one. It also builds the mutants from that child. Every other child, and every survivor of the cull, is discarded. Each new population is therefore one child, its mutants, and about half fresh random plans.
- **No elitism.** `bestIndividual` is overwritten each generation and isn't carried into the next population. The returned plan is the best of the **last evaluated** population, not the best seen.
- **The last generation is never scored.** The loop always ends after `shiftPop` or breaks before it, so the final population built is never evaluated. With the `Run.java` budget this often means returning the best initial random plan (see *Forward-model accounting*).
- **`shift()` is applied within a single decision.** In `shiftPop` the kept child loses its first action every generation, even though no action has been played. Shifting a plan normally happens after its first action has been played (as in RHEA), but here it is done inside one `act()` call.
- **`shift()` can leave the plan out of sync with its state.** If a replayed action is infeasible, replay stops but the infeasible actions stay in the list, and random actions are **appended** after them. The stored state and score then reflect a different action sequence from `actions`. The list can also grow past `NODE_SIZE`. If that individual later ranks best, `act()` can return an action that isn't feasible in the current state.
- **The mutation rate is not what it says.** The two-step random draw gives `(100·MUTATION_RATE − 1)/200` per gene (4.5 % at the default 0.1). `repair` mutates at most one gene per child. `MUTATION_RATE < 0.01` throws.
- **Crossover modifies its parents.** When the parents' lengths differ, the shorter parent's action list loses its last action **in place**. That parent may be `bestIndividual`.
- **Population sizes can drift and crash.** When both tournaments pick the same winner, only one individual leaves the pool, so the number of children varies. If exactly one individual survives the cull (for example `POP_SIZE` 1 or 2 with `KILL_RATE = 0.5`), it is set aside as `person1`, no children are produced, and `population.get(m_rnd.nextInt(0))` throws.
- **Positional crossover.** Genes are combined by index, so a child's action at position *i* often refers to a unit or city in a different situation. Most of the child's quality then depends on `repair` swapping in random feasible actions.
- **Single-turn horizon.** Plans stop when our turn ends, and opponents are not modelled. The appended `END_TURN` is not simulated, so plans are scored on the state just before the turn ends.
- **No reuse between calls.** The population is rebuilt on every `act()` call, so a full-turn plan is re-evolved from scratch for each action in the turn.
