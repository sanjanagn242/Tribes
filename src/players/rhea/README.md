# RHEA: Rolling Horizon Evolutionary Algorithm

`players.rhea` contains a Rolling Horizon Evolutionary Algorithm agent for Tribes. Each genome is a fixed-length sequence of `INDIVIDUAL_LENGTH` actions, played from the current state. Unlike OEP, a genome is **not** limited to our own turn: once a genome's `END_TURN` is applied, the following genes are actions of whichever tribe is active, so opponents' moves are part of the genome and are evolved to maximise **our** heuristic. The population is evolved with tournament selection, uniform crossover, mutation and elitism. Within one game tick the population is kept between `act()` calls using the shift buffer.

| File | Purpose |
|------|---------|
| [RHEAAgent.java](RHEAAgent.java) | The `Agent`: population management, evolutionary loop and genetic operators |
| [RHEAParams.java](RHEAParams.java) | RHEA-specific parameters (extends `AlgParams`) |
| [Genome.java](Genome.java) | A genome: an action list and its value |

---

## How it works

### `RHEAAgent.act(gs, ect)`

1. Create the state heuristic with `params.getStateHeuristic(playerID, allPlayerIDs)` and reset `fmcalls = 0`.
2. **New-turn detection.** If `gs.getTick()` differs from the stored `currentTurn`, store it and set `newTurn = true`. A tick covers one turn of every tribe, so this happens on the first action of each of our turns. `newTurn` also starts as `true` for the first call.
3. **Population.**
   - On a new turn, `init(gs)` builds `POP_SIZE` random genomes.
   - Otherwise `shiftPopulation(gs)` builds the population from the previous best genome (see *Shift buffer*).
4. Run `rheaLoop(gs)` until the budget is used up, then set `newTurn = false`.
5. Return the **first action of `pop[0]`**, the best genome after the final sort.

There is **no shortcut** when only one action is available.

### Random genomes (`newRandomIndividual`)

From a copy of the state, while the game isn't over and the genome has fewer than `INDIVIDUAL_LENGTH` actions: pick a random action from `Agent.allGoodActions(gs)` (every city, unit and tribe action of the **active** tribe, minus city `DESTROY` and unit `DISBAND`, `END_TURN` included), apply it and add it. Then score the genome:

```
value = heuristic.evaluateState(gs, finalState)
```

`END_TURN` doesn't stop the genome. The remaining genes are random actions of the next tribe(s).

### Evolution loop (`rheaLoop`)

```
repeat:
    sort pop by value, best first
    pop = nextGeneration(gs)
    numIters++
    end = (STOP_FMCALLS    and fmcalls  >= num_fmcalls) or
          (STOP_ITERATIONS and numIters >= num_iterations)
until end
sort pop by value, best first
```

At least one generation always runs. There is **no `STOP_TIME` branch** (see *Known quirks*).

### Next generation (`nextGeneration`)

- **Elitism.** If `ELITISM` is on and `POP_SIZE > 1`, the current best genome is copied unchanged (not re-evaluated) into the new population.
- **`POP_SIZE > 1`.** Fill the rest of the population with `newIndividual(gs)` (selection, crossover and mutation).
- **`POP_SIZE == 1`.** Mutate the single genome and keep the mutant if its value is `>=` the parent's. This is a (1+1) hill climber.

### Selection (`selection`)

Two tournaments over **indices** `0 .. POP_SIZE-1` of the sorted population:

- **Parent 1.** Sample `TOURNAMENT_SIZE` distinct indices and keep the one with the highest noisy value:
  ```
  noisy = (value + ε) * (1 + ε * (rand - 0.5))        // tie-break noise, ε = epsilon
  ```
- **Parent 2.** Same, except the duplicate-index loop is `while (contains(ind) && ind != p1idx)`, so parent 1's index is always accepted, even if it is already in the tournament. Parent 2 can be the same genome as parent 1.

### Crossover and mutation (`newIndividual`)

The child is built gene by gene while it is applied to a copy of the current state:

```
for actIdx in 0 .. INDIVIDUAL_LENGTH-1 (while game not over):
    with probability MUTATION_RATE:
        candidate = random action from allGoodActions        // always accepted
    else:
        pick a parent at 50/50; take its gene at actIdx if it exists and is feasible
        otherwise try the other parent's gene at actIdx
    if a feasible candidate was found: apply it and add it to the child
    else: skip this position (actIdx still advances)
fill the child with random actions until it has INDIVIDUAL_LENGTH actions (or the game ends)
value = heuristic.evaluateState(gs, finalState)
```

Genes are aligned by **position**. A skipped position makes the child shorter, and the missing genes are added as random actions at the **end**, so later genes are no longer aligned with the state they were evolved in.

### Mutation-only operator (`mutate`)

Used for the `POP_SIZE == 1` hill climber and in `shiftPopulation`. It replays the genome from a copy of the state. For each gene:
- with probability `MUTATION_RATE`, replace it with a random action;
- otherwise keep it if feasible, or replace it with a random action if not.

The result has the same length as the input genome (shorter only if the game ends) and is re-evaluated.

### Feasibility check (`checkActionFeasibility`)

Returns `false` if the game is over, if a unit action's actor is not a `Unit` of the active tribe, or if a city action's actor is not a `City` of the active tribe. Otherwise it returns `action.isFeasible(gs)`, treating exceptions as infeasible. Tribe actions (`END_TURN`, `RESEARCH_TECH`, `BUILD_ROAD`, …) are **not** checked against the active tribe.

### Shift buffer (`shiftPopulation`)

On the second and later `act()` calls in the same tick:

1. **Shift the best genome (`shift`).** Drop its first action, which was the one just played. Replay the remaining actions from the current state while they are feasible. Then, starting from the number of actions replayed (`j`), **append** random actions until `INDIVIDUAL_LENGTH` actions have been played. Re-evaluate it.
2. Add `MUTATE_BEST` mutants (`mutate`) of the shifted genome (capped by `POP_SIZE`).
3. Fill up to `POP_SIZE` with new random genomes.

So within one turn, only the best genome is kept from one decision to the next. On a new tick the population is rebuilt from scratch.

### Forward-model accounting

`advance()` increments `fmcalls` once per `GameState.advance(action, true)`. That covers random genomes, shift, crossover, mutation and random fill-ins. Feasibility checks and heuristic evaluations are not counted. `fmcalls` is reset at the start of each `act()`, so the cost of `init` or `shiftPopulation` counts towards that call's budget.

The budget is only checked **after a full generation**. A generation costs up to about `(POP_SIZE − 1) × INDIVIDUAL_LENGTH` calls with elitism, so the search can go over budget by up to one generation. With the class defaults (`POP_SIZE = 100`, `INDIVIDUAL_LENGTH = 20`, 2000 calls), `init` alone uses about 2000 calls, and the mandatory first generation then roughly doubles the total.

---

## Parameters

### RHEA-specific (`RHEAParams`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `POP_SIZE` | `int` | `100` | Population size. `1` switches to the (1+1) hill-climber path. For other values it must be `>= TOURNAMENT_SIZE` (see *Known quirks*). |
| `MUTATION_RATE` | `double` | `0.1` | Per-gene probability of using a random action in `newIndividual` and `mutate`. |
| `INDIVIDUAL_LENGTH` | `int` | `20` | Genome length in actions, counted across all tribes' turns. This is the rolling horizon. |
| `TOURNAMENT_SIZE` | `int` | `3` | Number of indices sampled in each selection tournament. |
| `MUTATE_BEST` | `int` | `9` | Number of mutants of the shifted best genome added by `shiftPopulation`. |
| `ELITISM` | `boolean` | `true` | Copy the best genome unchanged into each new generation (only when `POP_SIZE > 1`). |

`RHEAParams.print()` writes these six values to stdout.

### Inherited from `AlgParams` ([../heuristics/AlgParams.java](../heuristics/AlgParams.java))

| Field | Default | Used by RHEA? | Description |
|-------|---------|---------------|-------------|
| `stop_type` | `STOP_TIME` (0) | ⚠️ | Only `STOP_ITERATIONS` and `STOP_FMCALLS` are handled. `STOP_TIME`, the default, never stops (see *Known quirks*). |
| `num_iterations` | `200` | ✅ (`STOP_ITERATIONS`) | Generations per `act()`. |
| `num_fmcalls` | `2000` | ✅ (`STOP_FMCALLS`) | Forward-model call budget per `act()`. The search stops once `fmcalls >= num_fmcalls` after a generation. |
| `num_time` | `40` | ❌ | Not read. |
| `heuristic_method` | `SIMPLE_HEURISTIC` (1) | ✅ | `ENTROPY_HEURISTIC` (0), `SIMPLE_HEURISTIC` (1) or `DIFF_HEURISTIC` (2). |
| `epsilon` | `1e-6` | ✅ | Size of the tie-break noise in tournament selection. |
| `FORCE_TURN_END` | `5` | ❌ | Not read. `Run.java` sets it, but it has no effect. |
| `PRIORITIZE_ROOT` | `false` | ❌ | Not read. |

### Tuning interface

`RHEAParams` does not override `setParameterValue`, `getParameterValue`, `getParameters` or `getParameterValues`, so the `AlgParams` no-op versions apply and RHEA can't be tuned through that interface.

---

## Genome properties (`Genome`)

| Field / method | Meaning |
|----------------|---------|
| `actions` | The action sequence. It may contain actions of several tribes. |
| `value` | Heuristic score of the state reached by the genome, relative to the state passed to `act()`. |
| `shift()` | Replaces `actions` with a new list without the first action. |
| `compareTo()` | Orders by value **descending**, so `Collections.sort(pop)` puts the best genome at index 0. |
| `toString()` | Prints the actions and the value. |

A `Genome` doesn't store a game state. Every operator replays actions from the current state.

`RHEAAgent` also keeps `pop`, `currentTurn` (last tick seen), `newTurn` and `actionInTurn` between calls. `actionInTurn` is incremented but never read.

---

## Usage

### Default constructor

There is no seed-only constructor. `new RHEAAgent(seed, new RHEAParams())` uses the plain defaults, including `stop_type = STOP_TIME`, which RHEA doesn't handle: **`act()` never returns** with these defaults. Always set `stop_type` to `STOP_FMCALLS` or `STOP_ITERATIONS`.

### From `Run.java`, `Tournament.java` and `Play.java`

Player type `"RHEA"`. [`Run.getAgent`](../../Run.java) builds it with:

```java
RHEAParams rheaParams = new RHEAParams();
rheaParams.stop_type         = rheaParams.STOP_FMCALLS;          // 2000 FM calls per action
rheaParams.heuristic_method  = rheaParams.DIFF_HEURISTIC;
rheaParams.INDIVIDUAL_LENGTH = MAX_LENGTH;                       // JSON "Search Depth"
rheaParams.FORCE_TURN_END    = rheaParams.INDIVIDUAL_LENGTH + 1; // not read by RHEA
rheaParams.POP_SIZE          = POP_SIZE;                         // JSON "Population Size"
return new RHEAAgent(agentSeed, rheaParams);
```

These JSON config keys affect RHEA:

| JSON key | Maps to | Effect |
|----------|---------|--------|
| `"Search Depth"` | `INDIVIDUAL_LENGTH` | Genome length (rolling horizon) |
| `"Population Size"` | `POP_SIZE` | Population size. `1` gives the (1+1) hill climber. |

`"Force End"` and `"Rollouts"` have no effect on RHEA. `MUTATION_RATE`, `TOURNAMENT_SIZE`, `MUTATE_BEST`, `ELITISM` and `num_fmcalls` can't be set from JSON and keep their defaults.

For example, the current `tournament.json` and `play.json` set `"Search Depth": 20` and `"Population Size": 1`. That gives a (1+1) hill climber over 20-action genomes: one random genome, then repeated mutate-and-keep-if-not-worse, about 100 mutations per action within the 2000-call budget.

### Custom configuration

```java
RHEAParams p = new RHEAParams();
p.stop_type = p.STOP_FMCALLS;          // required: STOP_TIME is not supported
p.num_fmcalls = 5000;
p.heuristic_method = p.DIFF_HEURISTIC;
p.POP_SIZE = 10;                       // must be 1 or >= TOURNAMENT_SIZE
p.INDIVIDUAL_LENGTH = 15;
p.TOURNAMENT_SIZE = 3;
p.MUTATION_RATE = 0.2;
p.MUTATE_BEST = 4;
p.ELITISM = true;
Agent rhea = new RHEAAgent(seed, p);
```

---

## Known quirks and limitations

These describe the current code.

- **`copy()` returns `null`.** RHEA can't be nested inside or cloned by other agents.
- **`STOP_TIME` hangs.** `rheaLoop` only sets its exit flag for `STOP_FMCALLS` and `STOP_ITERATIONS`. With the default `STOP_TIME`, `act()` loops forever, and the `ElapsedCpuTimer` is ignored.
- **Small populations hang.** For `1 < POP_SIZE < TOURNAMENT_SIZE`, parent 1's tournament can't find `TOURNAMENT_SIZE` distinct indices, and the `while (tournament.contains(ind))` loop never ends. With the default `TOURNAMENT_SIZE = 3`, `POP_SIZE = 2` hangs.
- **Parent 2 can equal parent 1.** The condition `contains(ind) && ind != p1idx` seems meant to exclude parent 1, but it lets parent 1's index into the second tournament, even more than once.
- **Opponents are evolved to help us.** After an `END_TURN`, the genome's genes are opponents' actions, chosen and kept by evolution to maximise our heuristic. That is an optimistic opponent model, not a paranoid or random one.
- **Tribe actions skip the active-tribe check.** `checkActionFeasibility` only checks ownership for unit and city actions. Crossover aligns genes by index across several tribes' turns, so a tribe action from one tribe's turn (for example another tribe's `END_TURN` or `RESEARCH_TECH`) can pass the check and be applied while a different tribe is active.
- **`shift()` can leave the genome out of sync with its score.** If a replayed action is infeasible, replay stops but the infeasible actions stay in the list, and the random actions are **appended** after them. The genome's value then reflects a different sequence from `actions`, and the list can grow past `INDIVIDUAL_LENGTH`. If the shifted genome stays best (elitism keeps it), `act()` can return an action that isn't feasible in the current state. The game then prints a `couldn't execute?` warning instead of applying it.
- **The budget is checked late.** It is checked only after each full generation, and `init`/`shiftPopulation` costs count too, so a call can use far more than `num_fmcalls` (about 2× with the class defaults).
- **Elites are not re-evaluated.** The elite and the shifted best keep the value they were given when created.
- **A wasted feasibility call.** In `newIndividual`, `checkActionFeasibility(candidate, crossState)` is called again after `advance` and its result is ignored.
- **Limited reuse.** Within a tick, only the best genome survives to the next `act()` call (plus its mutants). At each new tick the whole population is rebuilt randomly.
