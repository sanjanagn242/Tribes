# `players`: Tribes agents

`players` holds every AI (and the human interface) that can play Tribes. Each player is a subclass of the abstract [`Agent`](Agent.java). The game engine calls `act()` **once per action**, not once per turn, and keeps calling it until the agent returns `END_TURN`, no actions are left, or (for AI players, when `TURN_TIME_LIMITED`) the turn times out. Simple agents live as single files in this folder. The search-based agents each have their own subfolder, and they share the parameter and heuristic classes in [`heuristics/`](heuristics/).

| Location | Contents |
|----------|----------|
| [Agent.java](Agent.java) | Abstract base class for all players |
| [DoNothingAgent.java](DoNothingAgent.java), [EndTurnAgent.java](EndTurnAgent.java) | Always end the turn |
| [RandomAgent.java](RandomAgent.java) | Uniformly random legal action |
| [SimpleAgent.java](SimpleAgent.java) | Rule-based agent ("Rule Based" in JSON) |
| [HumanAgent.java](HumanAgent.java), [ActionController.java](ActionController.java), [KeyController.java](KeyController.java) | Human play through the GUI |
| [heuristics/](heuristics/) | `AlgParams`, state heuristics and the pruning heuristic shared by the search agents |
| `mc/`, `osla/`, `mcts/`, `emcts/`, `oep/`, `rhea/`, `portfolio/`, `portfolioMCTS/` | Search-based agents (see the [index](#agent-subfolders)) |

---

## The `Agent` base class

[`Agent.java`](Agent.java) defines the interface the game uses, plus a few helpers for subclasses.

### Fields

| Field | Type | Set by | Meaning |
|-------|------|--------|---------|
| `playerID` | `int` | `setPlayerIDs` | This agent's tribe ID |
| `allPlayerIDs` | `ArrayList<Integer>` | `setPlayerIDs` | IDs of all players in the game. `Play`/`Tournament` pass `0..n-1`. The heuristics rely on this. |
| `seed` | `long` | constructor / `reset` | Random seed. Subclasses build their own `Random` from it. |

### Methods

| Method | Kind | What it does |
|--------|------|--------------|
| `Agent(long seed)` | protected constructor | Calls `reset(seed)`. |
| `act(GameState gs, ElapsedCpuTimer ect)` | **abstract** | Return the next action to play. `ect` is the turn timer (`TURN_TIME_MILLIS`). |
| `copy()` | **abstract** | Return a copy of the agent. Many agents (DoNothing, EndTurn, Random, MCTS, pMCTS, ...) just return `null`. |
| `result(GameState gs, double reward)` | hook, empty by default | Called by `Game` at the end of the game with a copy of the final state and the tribe's score. |
| `getPlayerID()` / `getSeed()` | `final` getters | |
| `setPlayerIDs(int playerID, ArrayList<Integer> allIds)` | setter | Called by `Play`/`Tournament`/`RunElites` after `Run.getAgent()`. |
| `reset(long seed)` | | Stores the seed. It does **not** re-seed any `Random` a subclass has already built. |
| `determineActionGroup(gs, rnd)` | protected helper | Picks one action group at random and returns only those actions (see below). Used by MC and MCTS when `PRIORITIZE_ROOT` is on. |
| `allGoodActions(gs, rnd)` | protected helper | All available actions **except** city `DESTROY` and unit `DISBAND`; tribe actions (including `END_TURN`) are kept. `rnd` is unused. Used by EMCTS and OEP. |
| `actionsPerUnit(gs)` | public, overridable | One entry per city, per unit, plus one for the tribe: the number of actions each actor has. `Game` logs it for branching-factor stats. pMCTS overrides it to count portfolio assignments instead. |
| `actionsPerGameState(gs)` | public, overridable | `gs.getAllAvailableActions().size()`. Also logged by `Game`. |

### `determineActionGroup`

1. **City group:** all city actions except `DESTROY`. Available if non-empty.
2. **Unit group:** available if `getAllUnitActions()` is non-empty. A `DISBAND`-filtered list is built but **not used**: the unfiltered list is both tested and returned, so `DISBAND` stays in.
3. **Tribe group:** available only if there is more than one tribe action, i.e. something other than `END_TURN`.
4. If no group is available, return `null` (callers then return `END_TURN`). Otherwise pick one group **uniformly at random** and return its list.

The private enum `ACTION_TYPE { CITY, TRIBE, UNIT }` labels the groups.

---

## Single-file agents

### `DoNothingAgent` and `EndTurnAgent`

Both are identical: `act()` returns `new EndTurn(gs.getActiveTribeID())` every time, and `copy()` returns `null`. `DoNothingAgent` is reachable from JSON as `"Do Nothing"`. `EndTurnAgent` is not referenced anywhere else in the code.

### `RandomAgent`

Builds `Random(seed)` in the constructor. `act()` returns a uniformly random element of `gs.getAllAvailableActions()`, so `END_TURN`, `DESTROY` and `DISBAND` are as likely as any other action. `copy()` returns `null`. JSON: `"Random"`.

### `SimpleAgent` (rule-based)

JSON: `"Rule Based"`. `copy()` returns a new `SimpleAgent` with the same seed (a fresh `Random`).

#### Action selection (`act`)

1. Score **every** action in `gs.getAllAvailableActions()` with `evalAction()`. Scores are integers from `-2` to `5`.
2. Group actions by score in a `HashMap<Integer, ArrayList<Action>>`.
3. Starting from the best score and counting down to `-1`, take the first non-empty group and return a **uniformly random** action from it.

So it is greedy over hand-written priorities with random tie-breaking. There is no lookahead and no forward-model call. Because `END_TURN` scores `-1`, the agent keeps acting until every remaining action scores below `0` (`-2` actions are never chosen; `-1` ones only if nothing better exists).

#### Score table (`evalAction`)

Unlisted action types (`CLIMB_MOUNTAIN`, `DECLARE_WAR`, ...) score `0`.

| Group | Action | Score |
|-------|--------|-------|
| Unit | `CAPTURE`, `EXAMINE`, `MAKE_VETERAN` | `5` |
| Unit | `DISBAND` | `-2` |
| Unit | `MOVE` | `evalMove` |
| Unit | `ATTACK` | `evalAttack` |
| Unit | `CONVERT` | `evalConvert` |
| Unit | `HEAL_OTHERS` | `evalHeal` |
| Unit | `RECOVER` | `evalRecover` |
| Unit | `UPGRADE_BOAT`, `UPGRADE_SHIP` | `evalUpgrade` |
| City | `DESTROY` | `-2` |
| City | `BUILD` | `evalBuild` |
| City | `SPAWN` | `evalSpawn` |
| City | `LEVEL_UP` | `evalLevelUp` |
| City | `RESOURCE_GATHERING` | `evalResourceGathering` |
| City | `BURN_FOREST` / `CLEAR_FOREST` / `GROW_FOREST` | `evalBurn` / `evalClear` / `evalGrow` |
| Tribe | `END_TURN` | `-1` |
| Tribe | `RESEARCH_TECH` | `evalResearch` |
| Tribe | `BUILD_ROAD` | `evalRoad` |
| Tribe | `SEND_STARS` | `evalSendStars` |

`neededToLevelUp(c) = c.getLevel() + 1 - c.getProduction()` is used by several city rules.

#### Unit rules

**`evalMove`.** Checked in this order; the first rule that fires returns.

1. Scan every tile the tribe can see (row by row). For the **first** visible enemy unit found:
   - If the enemy is weaker (`enemy.DEF < unit.ATK` and `unit HP >= enemy HP`) and the move gets closer to it (Chebyshev distance): **3**.
   - Otherwise (enemy is stronger), if the move gets farther away and the unit is within the enemy's attack range: **4**.
   - If neither applies, the scan continues to the next enemy.
2. Scan the tiles within `unit.RANGE` (attack range) of the unit's current position. For each visible tile:
   - If it is inside an enemy city's borders and the move gets closer to that city's centre: **4**.
   - If it is a `VILLAGE` tile and the move gets closer to it: **5**.
3. If any tile next to the destination is visible: **3** (commented as "incentive to explore: next to fog", but the code tests for a *visible* tile, so this almost always fires).
4. Otherwise **0**.

**`evalAttack`.** `allegiance` = `diplomacy[attackerTribe][defenderTribe]`.

| Attacker | Condition | Score |
|----------|-----------|-------|
| Melee | HP ≥ defender HP and ATK > defender DEF | `5` if allegiance < 0; `2` if 0 ≤ allegiance < 15; else `0` |
| Melee | HP ≥ defender HP and ATK ≤ DEF | `1` if allegiance < 0, else `0` |
| Melee | HP < defender HP and ATK > DEF | `1` if allegiance < 0, else `0` |
| Melee | HP < defender HP and ATK ≤ DEF | `0` |
| Ranged | attacker is within the defender's range and defender ATK > attacker DEF | `0` (retaliation danger) |
| Ranged | defender is in the attacker's range but the attacker is not in the defender's range | `5` / `2` / `0` (same allegiance bands as the first row) |
| Ranged | otherwise, defender DEF < ATK and HP ≥ defender HP | `4` if allegiance < 0; `1` if < 15; else `0` |

"In range" is `chebyshev(attacker, defender) <= attacker.RANGE`.

**`evalConvert`.** By the target's unit type, using allegiance bands:

| Target type | allegiance < 0 | 0 ≤ allegiance < 15 | ≥ 15 |
|-------------|----------------|---------------------|------|
| `BATTLESHIP`, `SUPERUNIT`, `SWORDMAN`, `KNIGHT`, `CATAPULT` | `5` | `2` | falls through to the next row, which also gives `0` |
| `MIND_BENDER`, `BOAT`, `SHIP`, `WARRIOR` | `4` | `1` | `0` |
| any other | `0` | `0` | `0` |

(See *Known quirks*: the allegiance lookup uses unit IDs.)

**`evalHeal`.** For each target returned by `HealOthers.getTargets(gs)` that is below max HP, count the enemy units (of a tribe other than the target's) within `target.RANGE` of the **healer's** position whose HP is higher than the target's. Score = `min(count, 5)`.

**`evalRecover`.** Scan visible tiles for enemy units. For the **first** enemy that has this unit in its range: `3` if the enemy's HP is lower than ours, else `1`. If no visible enemy has us in range: `4`.

**`evalUpgrade`.**
- If the unit type is `BATTLESHIP`: `3` when the tribe's max production > 5 and stars > 8.
- Otherwise: `3` when max production > 3 and stars > 6.
- Else `0`.

The unit being upgraded is always a `BOAT` or `SHIP`, so only the second branch is ever used.

#### City rules

**`evalBuild`.** If the tribe has fewer than 8 stars: `0`. Otherwise `min(5, max(0, base + 5 - neededToLevelUp))`. The `switch` that sets `base` is written with tiered values (4 for farm/mine/forge/windmill/customs house, 3 for port/sawmill/lumber hut, 5 for wonders, 1 for temples), but the `break`s are commented out, so everything falls through to the temple case. The **effective** `base` is:

| Buildings | Effective base |
|-----------|----------------|
| `FARM`, `MINE`, `FORGE`, `WINDMILL`, `CUSTOMS_HOUSE`, `PORT`, `SAWMILL`, `LUMBER_HUT`, all wonders, `TEMPLE`, `WATER_TEMPLE`, `MOUNTAIN_TEMPLE` | `1` |
| `FOREST_TEMPLE` (not in the switch) | `0` |

So building choice depends almost entirely on how close the city is to levelling up.

**`evalSpawn`.** Count visible enemy units standing on this city's tiles.

| Unit | No enemies in city | Enemies in city |
|------|--------------------|-----------------|
| `DEFENDER` | `3` | `5` |
| `SWORDMAN`, `KNIGHT` | `3` | `4` |
| `WARRIOR` | `3` | `3` |
| `RIDER`, `ARCHER`, `CATAPULT` | `3` | `0` |
| `MIND_BENDER` | `2` | `0` |

**`evalLevelUp`.** `BORDER_GROWTH`, `WORKSHOP`, `RESOURCES`, `SUPERUNIT` → `5`. `EXPLORER`, `CITY_WALL`, `POP_GROWTH`, `PARK` → `0`.

**`evalResourceGathering`.** `min(5, max(0, 5 - neededToLevelUp))`.

**`evalBurn`.** If the tribe has more than 5 stars: `1 + (number of farms in the city, if fewer than 2)`, i.e. `1` or `2`. Otherwise `0`.

**`evalClear`.** `3` if the tribe has 0 stars, else `0`.

**`evalGrow`.** If the city has fewer than 2 forest tiles **and** fewer than 2 lumber huts: `4`. Otherwise `2`.

#### Tribe rules

**`evalResearch`.**

| Technologies | Score |
|--------------|-------|
| `ORGANIZATION`, `CLIMBING`, `MINING`, `FISHING`, `HUNTING`, `WHALING` | `5` |
| `FARMING`, `AQUATISM`, `RIDING`, `FORESTRY`, `ARCHERY`, `SAILING`, `SHIELDS`, `CHIVALRY` | `4` |
| every other tech | `3` |

**`evalRoad`.** Look at the 8 tiles around the road position in order; the first match returns:
- Neighbour is inside one of our cities' borders:
  - it is the city centre → `3` if max production > 5 and stars > 4 (otherwise keep looking);
  - it already has a road → `3`.
- Neighbour is outside our borders and already has a road → `4` if max production > 5 and stars > 4, else `3`.
- Nothing matched → `0`.

**`evalSendStars`.** `allegiance = diplomacy[us][target]`, `sent = stars being sent`.

| Condition | Score |
|-----------|-------|
| stars > 60 | `4` |
| 30 < stars ≤ 60, allegiance < 0 and `allegiance + sent > 0` | `3` |
| 30 < stars ≤ 60, otherwise | `2` |
| stars ≤ 30, `-5 ≤ allegiance < 0` and `sent ≤ 5` | `4` |
| otherwise | `-1` |

### `HumanAgent`, `ActionController`, `KeyController`

| Class | Role |
|-------|------|
| [`ActionController`](ActionController.java) | Extends `Agent` (seed 0). Holds a FIFO `Queue<Action>`. The GUI (`GUI`, `InfoView`) calls `addAction(action, gs)` when the human clicks something, including the End Turn button; `reset()` clears the queue. `getAction()` / `act()` poll the queue and return `null` when it is empty. `copy()` copies the queue. |
| [`HumanAgent`](HumanAgent.java) | Extends `Agent` (seed 0). Wraps an `ActionController`; `act()` returns `ac.getAction()`. `copy()` shares the same controller. `Game.processTurn` checks `instanceof HumanAgent` to skip the GUI action delay, branching-factor logging and the turn time-out / no-actions-left checks, so the turn only ends on `END_TURN`. |
| [`KeyController`](KeyController.java) | A Swing `KeyAdapter` added to the GUI window. Arrow keys + Space (`primaryKeys = true`, which is what `Play` uses) or WASD + Shift map to `Types.DIRECTIONS` and are queued. Keys `0`–`4` set `focusedPlayer = key - 1`. Neither the queue nor `focusedPlayer` has a getter, so nothing reads them. |

`Play` creates a `KeyController(true)` and an `ActionController` and passes both to `Run.runGame`, which hands them to the `GUI`. **No code constructs a `HumanAgent`** (see *Known quirks*).

---

## Shared heuristics (`heuristics/`)

### `AlgParams`

Base parameter class extended by every search agent's params (`MCParams`, `OSLAParams`, `MCTSParams`, `EMCTSParams`, `OEPParams`, `RHEAParams`, `PortfolioMCTSParams`).

**Constants** (instance `final int`s, so they are accessed as `params.STOP_FMCALLS`):

| Constant | Value |
|----------|-------|
| `STOP_TIME` | `0` |
| `STOP_ITERATIONS` | `1` |
| `STOP_FMCALLS` | `2` |
| `ENTROPY_HEURISTIC` | `0` |
| `SIMPLE_HEURISTIC` | `1` |
| `DIFF_HEURISTIC` | `2` |

**Fields:**

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `heuristic_method` | `int` | `SIMPLE_HEURISTIC` | Which `StateHeuristic` `getStateHeuristic()` builds |
| `epsilon` | `double` | `1e-6` | Small constant for noise / division by zero |
| `stop_type` | `int` | `STOP_TIME` | Which budget ends a search |
| `num_iterations` | `int` | `200` | Budget for `STOP_ITERATIONS` |
| `num_fmcalls` | `int` | `2000` | Budget for `STOP_FMCALLS` |
| `num_time` | `int` | `40` | Time budget in ms. No agent reads it; time-based stopping uses the `ElapsedCpuTimer` passed to `act()`. |
| `FORCE_TURN_END` | `int` | `5` | Agents that support it force `END_TURN` every `FORCE_TURN_END` plies |
| `PRIORITIZE_ROOT` | `boolean` | `false` | Restrict root actions to one random group via `determineActionGroup` |

How each stop type is checked (and whether it is supported at all) is up to each agent; see the agent READMEs.

**Methods:**
- `getStateHeuristic(playerID, allIDs)` → `TribesEntropyHeuristic`, `TribesSimpleHeuristic` or `TribesDiffHeuristic` depending on `heuristic_method` (`null` for any other value).
- `getPruneHeuristic()` → `null`. `PortfolioMCTSParams` overrides it to return its `PrunePortfolioHeuristic`.
- Tuning hooks `setParameterValue`, `getParameterValue`, `getParameters`, `getParameterValues` are no-ops / return `null`; subclasses override them.
- `constantNames()` maps `"heuristic_method"` to `{"SIMPLE_HEURISTIC", "ENTROPY_HEURISTIC"}`. Nothing calls it.

### `StateHeuristic`

An interface with two **default** methods, both returning `0.0`:

```java
double evaluateState(GameState gameState);                      // absolute
double evaluateState(GameState oldState, GameState newState);   // relative
```

Each implementation overrides only one of them. Every agent in this package calls the **two-argument** form.

All three heuristics share this outer shape: compute a per-tribe value `s_i` for each `i` in `0 .. allIds.size()-1`, then

```
value = s_me − mean(s_i for i ≠ me) + own_term
```

where `me = playerID`.

### `TribesSimpleHeuristic` (`SIMPLE_HEURISTIC`, 1)

Overrides only `evaluateState(gs)` (single state).

```
s_i = boost_i
    + numAvailableActions          // of the active tribe; same for every i
    + 100 · production_i
    + 10  · techsResearched_i
    +       score_i
    + 500 · numCities_i
    + 10  · numUnits_i

boost_i = 1e9 if game over and tribe i WON or LOST, else 0

own_term = 10 · (visible tiles / total tiles)     // active tribe's visibility
```

### `TribesEntropyHeuristic` (`ENTROPY_HEURISTIC`, 0)

Overrides only `evaluateState(old, new)`. Counts how many metrics **went up** (each indicator is 0 or 1):

```
s_i = [production↑] + [techs↑] + [score↑] + [cities↑] + [units↑] + [kills↑]

own_term = [my connected cities↑] + [visible tiles↑]
```

### `TribesDiffHeuristic` (`DIFF_HEURISTIC`, 2)

Overrides only `evaluateState(old, new)`. Weighted sum of metric **changes** (Δ = new − old):

```
s_i =   5   · Δproduction_i
      + 4   · ΔtechsResearched_i
      + 0.1 · Δscore_i
      + 4   · ΔnumCities_i
      + 2   · ΔnumUnits_i
      + 3   · Δkills_i
      + 2   · Δ(sum of city levels)_i

own_term = Δ(my connected cities) + Δ(number of visible tiles)
```

The visibility term is a raw tile count, so revealing 4 tiles is worth as much as one new city in `s_me`. `Run.getAgent` sets `DIFF_HEURISTIC` for every search agent.

### `PruneHeuristic`

Interface used only by portfolio MCTS. Three default methods: `evaluatePrune(state, aas)` → `0.0`, `prune(parent, actions, state, k)` → `null`, `unprune(parent, actions, state, pruned, rnd)` → `null`.

### `PrunePortfolioHeuristic`

The `PruneHeuristic` that `Run` gives to pMCTS (built from the `SimplePortfolio`).

- **Weights.** Every action type in the portfolio starts with weight `0.5`. `setWeights(double[])` (called when the JSON has `"pMCTS Weights"`) sets the `SPAWN` and `RESEARCH_TECH` type weights to `-1` and assigns one weight per script of those types, in portfolio order, stored by script class name. The calls for the other action types are commented out.
- **`evaluatePrune(state, aas)`** = `w × aas.getValue()`, where `w` is the action type's weight, or the script's own weight if the type weight is `-1`.
- **`prune(parent, actions, state, k)`** ranks the expanded children by `evaluatePrune` and leaves the top `k` unpruned; everything else, including unexpanded children, is marked pruned.
- **`unprune(...)`** un-prunes one expanded, currently pruned child with the highest `evaluatePrune`, breaking ties at random.
- pMCTS also uses `evaluatePrune / (child visits + ε)` as a progressive-bias term when `PROGBIAS` is on.

---

## Agent subfolders

| Folder | Main class | JSON name | Description | README |
|--------|------------|-----------|-------------|--------|
| `mc/` | `MonteCarloAgent` | `"MC"` | Flat Monte Carlo: repeatedly picks a random non-`END_TURN` root action, plays a random rollout of `ROLLOUT_LENGTH` actions, and returns the root action with the best mean heuristic value. | [mc/README.md](mc/README.md) |
| `osla/` | `OneStepLookAheadAgent` | `"OSLA"` | One-step lookahead: applies each action (skipping `END_TURN`, `DESTROY`, `DISBAND`) to a copy of the state and returns the one with the best heuristic value. | [osla/README.md](osla/README.md) |
| `mcts/` | `MCTSPlayer` | `"MCTS"` | Closed-loop UCT MCTS over single actions; opponents minimise our value; searches from scratch on every `act()`. | [mcts/README.md](mcts/README.md) |
| `emcts/` | `EMCTSAgent` | `"EMCTS"` | Evolutionary MCTS: tree nodes are whole-turn action sequences and children are mutations of their parent. | [emcts/README.md](emcts/README.md) |
| `oep/` | `OEPAgent` | `"OEP"` | Online Evolutionary Planning: evolves a population of turn-long action sequences (crossover, mutation, kill rate) and plays the next action of the chosen individual. | [oep/README.md](oep/README.md) |
| `rhea/` | `RHEAAgent` | `"RHEA"` | Rolling Horizon Evolutionary Algorithm: evolves fixed-length action sequences, shifts the population between calls within a turn, and plays the first action of the best genome. | [rhea/README.md](rhea/README.md) |
| `portfolio/` | `Portfolio`, `SimplePortfolio`, `RandomPortfolio` | – (not an agent) | Script portfolios: maps each action type to scripts (`scripts/*Scr.java`) that each pick one action per actor, producing the `ActionAssignment`s that pMCTS searches over. | [portfolio/README.md](portfolio/README.md) |
| `portfolioMCTS/` | `PortfolioMCTSPlayer` | `"pMCTS"` | MCTS whose branches are portfolio script assignments instead of raw actions, with optional progressive pruning/unpruning and progressive bias from `PrunePortfolioHeuristic`. | [portfolioMCTS/README.md](portfolioMCTS/README.md) |

---

## How agents are selected (`Run.java`)

[`Run.java`](../Run.java) holds the `PlayerType` enum, the JSON string parser, and the `getAgent` factory. `Play`, `Tournament` and `RunElites` all build their players through it.

1. `Play` / `Tournament` / `RunElites` read the config JSON and copy the shared settings into static fields on `Run`.
2. Each entry of `"Players"` goes through `Run.parsePlayerTypeStr()` to get a `PlayerType`. An unknown string throws `"Error: unrecognized Player Type"`.
3. `Run.getAgent(type, agentSeed)` builds the agent; the caller then calls `setPlayerIDs(i, [0..n-1])`. Every player in a game gets the **same** `agentSeed`.

### JSON settings copied into `Run`

| JSON key | `Run` field | Default in `Run` | Used by |
|----------|-------------|------------------|---------|
| `"Search Depth"` | `MAX_LENGTH` | – | MC, MCTS, pMCTS (`ROLLOUT_LENGTH`); RHEA (`INDIVIDUAL_LENGTH`) |
| `"Force End"` | `FORCE_TURN_END` | – | MC, MCTS |
| `"Rollouts"` | `MCTS_ROLLOUTS` | – | MCTS |
| `"Population Size"` | `POP_SIZE` | – | RHEA |
| `"Pruning"` | `PRUNING` | – | pMCTS |
| `"Progressive Bias"` | `PROGBIAS` | – | pMCTS |
| `"K init mult"` | `K_INIT_MULT` | `0.5` | pMCTS |
| `"T mult"` | `T_MULT` | `2.0` | pMCTS |
| `"A mult"` | `A_MULT` | `1.5` | pMCTS |
| `"B"` | `B` | `1.3` | pMCTS |
| `"pMCTS Weights"` (optional) | `pMCTSweights` | `null` | pMCTS (`PrunePortfolioHeuristic.setWeights`) |

### Player type mapping

| JSON string | `PlayerType` | Class | Parameters set by `getAgent` |
|-------------|--------------|-------|------------------------------|
| `"Do Nothing"` | `DONOTHING` | `DoNothingAgent` | – |
| `"Random"` | `RANDOM` | `RandomAgent` | – |
| `"Rule Based"` | `SIMPLE` | `SimpleAgent` | – |
| `"Human"` | `HUMAN` | – | No case in `getAgent`: returns `null` |
| `"OSLA"` | `OSLA` | `OneStepLookAheadAgent` | `STOP_FMCALLS`, `DIFF_HEURISTIC` |
| `"MC"` | `MC` | `MonteCarloAgent` | `STOP_FMCALLS`, `DIFF_HEURISTIC`, `PRIORITIZE_ROOT = true`, `ROLLOUT_LENGTH = MAX_LENGTH`, `FORCE_TURN_END = FORCE_TURN_END ? 5 : ROLLOUT_LENGTH + 1` |
| `"MCTS"` | `MCTS` | `MCTSPlayer` | `STOP_FMCALLS`, `DIFF_HEURISTIC`, `PRIORITIZE_ROOT = true`, `ROLLOUT_LENGTH = MAX_LENGTH`, `FORCE_TURN_END = FORCE_TURN_END ? 5 : ROLLOUT_LENGTH + 1`, `ROLOUTS_ENABLED = MCTS_ROLLOUTS` |
| `"pMCTS"` | `PORTFOLIO_MCTS` | `PortfolioMCTSPlayer` | `STOP_FMCALLS`, `DIFF_HEURISTIC`, `PRIORITIZE_ROOT = false`, `ROLLOUT_LENGTH = MAX_LENGTH`, `PRUNING`, `PROGBIAS`, `K_init_mult`, `A_mult`, `B`, `T_mult` from `Run`; portfolio = `new SimplePortfolio(agentSeed)`; `pruneHeuristic = new PrunePortfolioHeuristic(portfolio)`, with `setWeights(pMCTSweights)` if weights were given |
| `"OEP"` | `OEP` | `OEPAgent` | `STOP_FMCALLS`, `DIFF_HEURISTIC` |
| `"EMCTS"` | `EMCTS` | `EMCTSAgent` | `STOP_FMCALLS`, `DIFF_HEURISTIC` |
| `"RHEA"` | `RHEA` | `RHEAAgent` | `STOP_FMCALLS`, `DIFF_HEURISTIC`, `INDIVIDUAL_LENGTH = MAX_LENGTH`, `FORCE_TURN_END = INDIVIDUAL_LENGTH + 1` (forcing off), `POP_SIZE = POP_SIZE` |

`num_fmcalls` is never set by `Run`, so every search agent uses the `AlgParams` default of **2000 forward-model calls per `act()`**. `EndTurnAgent` has no `PlayerType`.

`RunElites` declares its own `PlayerType` enum (without `EMCTS`), but its config parsing and agent creation use `Run.parsePlayerTypeStr` and `Run.getAgent`.

---

## Known quirks and limitations

These describe the current code.

- **`"Human"` doesn't work from JSON.** `Run.getAgent(HUMAN, ...)` returns `null`. `Play.getPlayers` only `assert`s it is non-null (asserts are off by default) and then calls `setPlayerIDs` on it, which throws a `NullPointerException`. Nothing in the codebase constructs a `HumanAgent`.
- **`SIMPLE_HEURISTIC` evaluates to 0 for every agent.** `TribesSimpleHeuristic` overrides only the one-argument `evaluateState`, but all agents call the two-argument form, which falls back to the interface default `0.0`. Any agent left on the `AlgParams` default heuristic therefore sees every state as equal. `Run` avoids this by always setting `DIFF_HEURISTIC`.
- **`TribesSimpleHeuristic` boosts losses too.** `LOSE_BOOST` is `+1e9`, the same as `WIN_BOOST`.
- **Visibility terms use the active tribe.** `GameState.getVisibilityMap()` returns the *active* tribe's observation grid, so in states where an opponent is to move (deep in MCTS trees, for example) the heuristics' visibility term measures that opponent's vision.
- **Heuristics assume IDs `0..n-1`.** They loop `i = 0 .. allIds.size()-1` and compare `i` with `playerID` rather than iterating over `allIds`.
- **`determineActionGroup` keeps `DISBAND`.** The filtered unit list is built but the unfiltered one is returned. The city group does drop `DESTROY`.
- **`SimpleAgent.evalConvert` indexes diplomacy by unit IDs.** It reads `allegianceStatus[con.getUnitId()][con.getTargetId()]`, but the array is `numTribes × numTribes` and unit actor IDs come after all the city IDs, so this lookup goes out of bounds. Since `act()` scores every available action, any turn where a `CONVERT` action is available would throw `ArrayIndexOutOfBoundsException`.
- **`SimpleAgent.evalBuild` falls through.** The `break`s are commented out, so the tiered building priorities collapse to `1` (see the table above).
- **`SimpleAgent.evalUpgrade`'s battleship branch is dead.** It checks whether the unit *is* a `BATTLESHIP`, which an upgradable unit never is.
- **`SimpleAgent.evalMove` "explore" rule checks for visible tiles, not fog**, so almost any move that doesn't trigger an earlier rule scores `3`. The enemy rules use only the first matching enemy in scan order, and `maxCities` is computed but unused.
- **`copy()` often returns `null`** (DoNothing, EndTurn, Random, MCTS, pMCTS, ...), so those agents can't be cloned.
- **`ActionController.act()` returns `null`** when the queue is empty. `Game.processTurn` tolerates a `null` action (it just loops and asks again).
- **`KeyController` state is never read.** The direction queue and `focusedPlayer` have no getters.
- **`AlgParams.num_time` and `constantNames()` are unused.**
- **`PrunePortfolioHeuristic.setWeights` only covers `SPAWN` and `RESEARCH_TECH`.** The weight array must have exactly one entry per script of those two types (in portfolio order); a shorter array throws.
