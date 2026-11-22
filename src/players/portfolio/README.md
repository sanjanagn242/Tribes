# Portfolio: scripted action abstraction

`players.portfolio` is not an agent. It is an **action abstraction** used by search agents. Instead of searching over every raw action the game allows, the agent searches over *action assignments*: an actor (unit, city or tribe) paired with a hand-written **script** that picks one concrete action for it. Each script encodes a simple strategy, such as "attack the closest enemy" or "spawn the cheapest unit". Several scripts often pick the same action, and those duplicates are merged, so the agent sees far fewer options than the raw action list.

The only agent that uses it is Portfolio MCTS (`"pMCTS"`), in [../portfolioMCTS/](../portfolioMCTS/README.md). [PrunePortfolioHeuristic](../heuristics/PrunePortfolioHeuristic.java) also reads it, to rank assignments for pruning, and `RunElites.java` tunes that heuristic's weights.

| File | Purpose |
|------|---------|
| [Portfolio.java](Portfolio.java) | Abstract base: `initPortfolio()`, `produceActionAssignments(state)`, `getPortfolio()`, `scripts(actionType)` |
| [SimplePortfolio.java](SimplePortfolio.java) | The portfolio in use. Maps each `Types.ACTION` to one or more scripts |
| [RandomPortfolio.java](RandomPortfolio.java) | One `RandomScr` per actor. Not used anywhere, and doesn't work as written (see quirks) |
| [ActionAssignment.java](ActionAssignment.java) | An `(actor, script)` pair, plus the action and value the script returned |
| [scripts/BaseScript.java](scripts/BaseScript.java) | Script base class. The default picks the first candidate action with value `0.5` |
| [scripts/](scripts/) | 48 `BaseScript` subclasses (listed below) |
| [scripts/utils/MilitaryFunc.java](scripts/utils/MilitaryFunc.java) | Shared helpers: pick by unit attribute, pick research tech, move towards points of interest, move to the best-scoring tile |
| [scripts/utils/BuildingFunc.java](scripts/utils/BuildingFunc.java) | Shared helpers for building placement: support buildings, base buildings, "idle" tiles |
| [scripts/utils/InterestPoint.java](scripts/utils/InterestPoint.java), [ValuePoint.java](scripts/utils/ValuePoint.java) | Callbacks used by the move helpers: a tile is of interest (`boolean`), or has a score (`double`) |

---

## How it works

### Scripts

A script is a `BaseScript` with:

- `setActions(ArrayList<Action>)`: the candidate actions it may choose from. These are all the available actions of **one action type** for **one actor**.
- `process(GameState gs, Actor ac)`: returns `Pair<Action, Double>`, the chosen action and a value, or `null` if the script has nothing to suggest.

The value is a rough score for how good the choice is (usually, but not always, in `[0, 1]`). The portfolio itself ignores it. Portfolio MCTS uses it through `PrunePortfolioHeuristic`, for pruning and progressive bias.

### `SimplePortfolio.initPortfolio()`

This builds a `TreeMap<Types.ACTION, BaseScript[]>`. Each action type is handled by:

| Kind | Action types | Scripts |
|------|--------------|---------|
| Wrapper (plain `BaseScript`, takes the first action) | `CAPTURE`, `EXAMINE`, `HEAL_OTHERS`, `MAKE_VETERAN`, `RECOVER`, `UPGRADE_BOAT`, `UPGRADE_SHIP`, `END_TURN` | 1 each |
| Single script | `BURN_FOREST`, `GROW_FOREST`, `RESOURCE_GATHERING`, `BUILD_ROAD` | 1 each |
| Multiple scripts | `ATTACK` (4), `CLEAR_FOREST` (4), `CONVERT` (3), `SPAWN` (6), `BUILD` (10), `LEVEL_UP` (2), `RESEARCH_TECH` (5), `MOVE` (9) | 43 |
| Not in the portfolio | `DESTROY`, `DISBAND`, `DECLARE_WAR`, `SEND_STARS`, `CLIMB_MOUNTAIN` | none, so these are never produced |

Scripts that make random choices share one `Random`, seeded with the seed passed to the constructor.

### `SimplePortfolio.produceActionAssignments(state)`

1. Call `initPortfolio()` again, which creates fresh script objects.
2. For each unit of the active tribe with available actions, each city with available actions, and then the tribe itself, call `extract`:
   - For each action type in the portfolio (in `Types.ACTION` enum order), and for each script registered for that type:
     - collect the actor's available actions of that type. If there are none, skip;
     - `script.setActions(thoseActions)`, and create `ActionAssignment(actor, script)`;
     - `process(state)` runs the script. If it returns a non-null action, and no assignment with the **same actor and the same `Action` object** is already in the list, add it.
3. Return the list. Each entry is one option for the search: `getAction()` is what gets played.

So the number of options for an actor is at most the number of scripts for its action types, and is smaller when scripts agree. When two scripts pick the same action, the one reached first (enum order, then script order in the table below) keeps it.

### Action values from the helpers

- **`MilitaryFunc.getActionByActorAttr(feature, maximize)`** (attack, convert, spawn). Looks at the target unit (attack/convert) or a freshly created unit of the spawn type, and keeps the action whose `feature` (HP, COST, MOVEMENT, RANGE, DEFENCE, ATTACK, DISTANCE from the actor, DAMAGE) is highest or lowest. When maximising, the value is `normalise(best, 0, best)`, which is `1.0` whenever `best > 0`. When minimising, it is `1 − best / (value of the first candidate)`.
- **`MilitaryFunc.getPreferredResearchTech(techs)`**. Among available research actions whose tech is in the branch list, keep those with the lowest tier and pick one at random. The value is `(4 − tier) / 4`. Returns `null` if no tech in the branch is available.
- **`MilitaryFunc.moveTowards(InterestPoint)`**. Scans the whole board for tiles of interest, then keeps the move whose destination is closest (`custom_dist`) to any of them, breaking ties at random. The value is `1 − minDistance / maxDistance`. Returns `null` if no tile is of interest.
- **`MilitaryFunc.position(minValue, ValuePoint)`**. Scores each move's destination tile and keeps the best, breaking ties at random. It only returns the move if `bestValue >= minValue`. The value is the tile score.
- **`BuildingFunc.buildSupportBuilding(target)`** (customs house, sawmill, windmill, forge, and forest clearing). Scores each build or clear-forest tile by the fraction of its 8 neighbours that suit `target` (water for customs house, crops/farm for windmill, ore mountain/mine for forge, forest/lumber hut for sawmill). Keeps the best, with a random tie-break. The value is that fraction.
- **`BuildingFunc.buildBaseBuilding(toBuild, bonus)`** (farm, lumber hut, mine, port). Meant to keep the build tile next to the best spot for the matching support building. See the quirks: it drops the best tile.
- **`BuildingFunc.buildInIdle(target)`** (monuments, temples). Meant to prefer tiles that are poor for both base and support buildings. See the quirks: it always returns `null`.

---

## Script reference

Every file in [scripts/](scripts/), grouped by action type. "Value" is the second element of the returned pair.

| Script | Action type | Selects | Value |
|--------|-------------|---------|-------|
| [BaseScript](scripts/BaseScript.java) | wrappers | The first candidate action | `0.5` |
| [RandomScr](scripts/RandomScr.java) | any (`RandomPortfolio` only) | A uniformly random candidate action | `0.5` |
| [AttackClosestScr](scripts/AttackClosestScr.java) | `ATTACK` | Attack on the enemy unit nearest the attacker (min DISTANCE) | min-scaled |
| [AttackWeakestScr](scripts/AttackWeakestScr.java) | `ATTACK` | Attack on the enemy with the lowest DEF | min-scaled |
| [AttackMaxDamageScr](scripts/AttackMaxDamageScr.java) | `ATTACK` | Meant to pick the attack doing the most damage (bonus for a kill). In practice it picks the first attack (see quirks) | `0` |
| [AttackStrongestScr](scripts/AttackStrongestScr.java) | `ATTACK` | Attack on the enemy with the highest ATK | `1.0` |
| [ClearForestForCustomScr](scripts/ClearForestForCustomScr.java) | `CLEAR_FOREST` | Clears the forest tile with the most water neighbours (customs-house site) | neighbour fraction |
| [ClearForestForForgeScr](scripts/ClearForestForForgeScr.java) | `CLEAR_FOREST` | Clears the forest tile with the most ore/mine neighbours (forge site) | neighbour fraction |
| [ClearForestForSawmillScr](scripts/ClearForestForSawmillScr.java) | `CLEAR_FOREST` | Clears the forest tile with the most forest/lumber-hut neighbours (sawmill site) | neighbour fraction |
| [ClearForestForWindmillScr](scripts/ClearForestForWindmillScr.java) | `CLEAR_FOREST` | Clears the forest tile with the most crops/farm neighbours (windmill site) | neighbour fraction |
| [ConvertStrongestScr](scripts/ConvertStrongestScr.java) | `CONVERT` | Converts the enemy with the highest ATK | `1.0` |
| [ConvertHighestHPScr](scripts/ConvertHighestHPScr.java) | `CONVERT` | Converts the enemy with the highest current HP | `1.0` |
| [ConvertHighestDefenceScr](scripts/ConvertHighestDefenceScr.java) | `CONVERT` | Converts the enemy with the highest DEF | `1.0` |
| [SpawnStrongestScr](scripts/SpawnStrongestScr.java) | `SPAWN` | Spawns the available unit type with the highest ATK | `1.0` |
| [SpawnDefensiveScr](scripts/SpawnDefensiveScr.java) | `SPAWN` | Spawns the unit type with the highest DEF | `1.0` |
| [SpawnCheapestScr](scripts/SpawnCheapestScr.java) | `SPAWN` | Spawns the unit type with the lowest COST | min-scaled |
| [SpawnFastestScr](scripts/SpawnFastestScr.java) | `SPAWN` | Spawns the unit type with the highest MOV | `1.0` |
| [SpawnMaxHPScr](scripts/SpawnMaxHPScr.java) | `SPAWN` | Spawns the unit type with the highest starting HP | `1.0` |
| [SpawnRangeScr](scripts/SpawnRangeScr.java) | `SPAWN` | Spawns the unit type with the highest RANGE | `1.0` |
| [BuildCustomHouseScr](scripts/BuildCustomHouseScr.java) | `BUILD` | Customs house on the tile with the most water neighbours | neighbour fraction |
| [BuildSawmillScr](scripts/BuildSawmillScr.java) | `BUILD` | Sawmill on the tile with the most forest/lumber-hut neighbours | neighbour fraction |
| [BuildWindmillScr](scripts/BuildWindmillScr.java) | `BUILD` | Windmill on the tile with the most crops/farm neighbours | neighbour fraction |
| [BuildForgeScr](scripts/BuildForgeScr.java) | `BUILD` | Forge on the tile with the most ore/mine neighbours | neighbour fraction |
| [BuildFarmScr](scripts/BuildFarmScr.java) | `BUILD` | Farm next to the best windmill site (`buildBaseBuilding`, see quirks) | best neighbour score |
| [BuildMineScr](scripts/BuildMineScr.java) | `BUILD` | Mine next to the best forge site (`buildBaseBuilding`) | best neighbour score |
| [BuildLumberHutScr](scripts/BuildLumberHutScr.java) | `BUILD` | Lumber hut next to the best sawmill site (`buildBaseBuilding`) | best neighbour score |
| [BuildPortScr](scripts/BuildPortScr.java) | `BUILD` | Port next to the best customs-house site (`buildBaseBuilding`) | best neighbour score |
| [BuildMonumentScr](scripts/BuildMonumentScr.java) | `BUILD` | Meant to place a monument on an "idle" tile. Always returns `null` (see quirks) | n/a |
| [BuildTempleScr](scripts/BuildTempleScr.java) | `BUILD` | Meant to place a temple (any kind) on an "idle" tile. Always returns `null` | n/a |
| [LevelUpGrowthScr](scripts/LevelUpGrowthScr.java) | `LEVEL_UP` | The first of WORKSHOP / RESOURCES / POP_GROWTH. Otherwise a random PARK or SUPERUNIT. Never EXPLORER, CITY_WALL or BORDER_GROWTH | `0.5` |
| [LevelUpMilitaryScr](scripts/LevelUpMilitaryScr.java) | `LEVEL_UP` | The first of CITY_WALL / BORDER_GROWTH / SUPERUNIT / WORKSHOP (in the order the actions are listed) | `0.5` |
| [ResearchFarmsScr](scripts/ResearchFarmsScr.java) | `RESEARCH_TECH` | Lowest-tier tech among ORGANIZATION, FARMING, SHIELDS, CONSTRUCTION | `(4 − tier) / 4` |
| [ResearchNavalScr](scripts/ResearchNavalScr.java) | `RESEARCH_TECH` | Lowest-tier tech among FISHING, WHALING, SAILING, NAVIGATION, AQUATISM | `(4 − tier) / 4` |
| [ResearchMountainsScr](scripts/ResearchMountainsScr.java) | `RESEARCH_TECH` | Lowest-tier tech among CLIMBING, MINING, MEDITATION, SMITHERY, PHILOSOPHY | `(4 − tier) / 4` |
| [ResearchRangeScr](scripts/ResearchRangeScr.java) | `RESEARCH_TECH` | Lowest-tier tech among HUNTING, ARCHERY, FORESTRY, SPIRITUALISM, MATHEMATICS | `(4 − tier) / 4` |
| [ResearchRoadsScr](scripts/ResearchRoadsScr.java) | `RESEARCH_TECH` | Lowest-tier tech among RIDING, ROADS, FREE_SPIRIT, CHIVALRY, TRADE | `(4 − tier) / 4` |
| [MoveToCaptureScr](scripts/MoveToCaptureScr.java) | `MOVE` | Move closest to a ruin, village or city centre not owned by us | `1 − d/max` |
| [MoveToOwnCityCentreScr](scripts/MoveToOwnCityCentreScr.java) | `MOVE` | Move closest to one of our city centres | `1 − d/max` |
| [MoveDefensivelyScr](scripts/MoveDefensivelyScr.java) | `MOVE` | Move closest to any tile inside our cities' borders | `1 − d/max` |
| [MoveOffensivelyScr](scripts/MoveOffensivelyScr.java) | `MOVE` | Move closest to any tile inside another tribe's city borders | `1 − d/max` |
| [MoveToDisembarkScr](scripts/MoveToDisembarkScr.java) | `MOVE` | Only for a unit on water: move closest to any land tile | `1 − d/max` |
| [MoveToEmbarkScr](scripts/MoveToEmbarkScr.java) | `MOVE` | Only for a unit on land: move closest to one of our ports | `1 − d/max` |
| [MoveToAttackRangeScr](scripts/MoveToAttackRangeScr.java) | `MOVE` | Move to the tile where the most tiles within the unit's RANGE hold enemies. Needs a score of 1, so it almost never fires | fraction |
| [MoveToConvergeScr](scripts/MoveToConvergeScr.java) | `MOVE` | Move to the tile with the most allied neighbours. Needs all 8 neighbours allied (score 1) | fraction |
| [MoveToDivergeScr](scripts/MoveToDivergeScr.java) | `MOVE` | Move to the tile with the fewest allied neighbours. Needs a score of 6, which is impossible, so it never fires | n/a |
| [BurnForestScr](scripts/BurnForestScr.java) | `BURN_FOREST` | Burns the forest tile where (farm-site value − lumber-hut-site value) is highest. In practice, the forest that is worst for a lumber hut (see quirks) | `normalise(score, −1, 1)` |
| [GrowForestScr](scripts/GrowForestScr.java) | `GROW_FOREST` | Meant to grow forest where a lumber hut would sit best next to a sawmill site. In practice, a random tile (see quirks) | `0` |
| [ResourceGatheringScr](scripts/ResourceGatheringScr.java) | `RESOURCE_GATHERING` | WHALES if available. Otherwise a tile that is a good support-building site, or else a random gather | `1.0`, `0.5 + 0.5·score`, or `0.5` |
| [BuildRoadScr](scripts/BuildRoadScr.java) | `BUILD_ROAD` | Only if some city is not connected to the capital: a random road on a "hot" tile next to the capital's trade network | `0.5` |

"min-scaled" means `1 − best / first-candidate-value` (see *Action values from the helpers*). "Neighbour fraction" is the share of the 8 surrounding tiles that suit the building.

---

## Usage

The portfolio is set on Portfolio MCTS parameters. [`Run.getAgent`](../../Run.java) does this for `"pMCTS"`:

```java
Portfolio p = new SimplePortfolio(agentSeed);
portfolioMCTSParams.setPortfolio(p);
portfolioMCTSParams.pruneHeuristic = new PrunePortfolioHeuristic(p);
if (Run.pMCTSweights != null)                       // JSON "pMCTS Weights"
    portfolioMCTSParams.pruneHeuristic.setWeights(Run.pMCTSweights);
```

To use it in another agent:

```java
Portfolio p = new SimplePortfolio(seed);
for (ActionAssignment aas : p.produceActionAssignments(gs)) {
    Action a = aas.getAction();        // the concrete action to play
    double v = aas.getValue();         // the script's own score
    BaseScript s = aas.getScript();    // which strategy chose it
}
```

To add a script, subclass `BaseScript`, override `process`, and register it under its action type in `SimplePortfolio.initPortfolio()`. If you register it under `SPAWN` or `RESEARCH_TECH`, remember that `PrunePortfolioHeuristic.setWeights` expects one weight per script in those groups (11 in total now).

---

## Known quirks and limitations

These describe the current code.

- **`buildBaseBuilding` throws away its best tile.** When a tile beats the current best, the code adds it to the candidate list and then **replaces** the list with an empty one. So `BuildFarmScr`, `BuildMineScr`, `BuildLumberHutScr` and `BuildPortScr` return `null` when the best tile is unique, and otherwise return one of the *later* tiles tied with it.
- **`buildInIdle` never returns its result.** It builds `new Pair<>(chosen, value)` but doesn't `return` it, and then returns `null`. `BuildMonumentScr` and `BuildTempleScr` therefore never produce an assignment, so pMCTS can't build monuments or temples.
- **`MoveToDivergeScr` never fires, and two others almost never do.** `position()` requires `bestValue >= minValue`, but the tile scores are fractions in `[0, 1]`. Diverge uses `minValue = 6`, which can't be reached. Converge and AttackRange use `minValue = 1`, which needs every neighbouring tile (or every tile within range) to hold an ally or an enemy.
- **The forest scripts evaluate the tile before the change.** `valueForBaseBuilding` checks the building's terrain requirement against the tile as it is now. `GrowForestScr` targets plain tiles, where a lumber hut (forest only) is never valid, so every tile scores 0 and the choice is random with value 0. `BurnForestScr` targets forest tiles, where a farm (plain only) is never valid, so it just picks the forest that is worst for a lumber hut.
- **`AttackMaxDamageScr` doesn't measure damage.** `getActionByActorAttr` runs the attack on a copy of the state but reads HP from the target in the *original* state, so the difference is always 0. The script returns the first attack action with value `0`.
- **Script values aren't on one scale.** Maximising attribute scripts nearly always return `1.0`. Minimising ones return `1 − best/first`, which depends on the order of the action list (and is `0` when there is one candidate). Wrappers return `0.5`, research returns `(4 − tier)/4`, and so on. Any heuristic that compares values across action types (as `PrunePortfolioHeuristic` does) inherits this.
- **Deduplication compares object identity.** `ActionAssignment.equals` uses `action == other.action`. This works because all scripts pick from the same action objects, but it means the script reached first keeps the assignment, and later scripts that agree get no credit.
- **Some action types are missing.** `DESTROY`, `DISBAND`, `DECLARE_WAR`, `SEND_STARS` and `CLIMB_MOUNTAIN` are not in the portfolio, so agents using it can never play them.
- **Each call rebuilds everything.** `produceActionAssignments` calls `initPortfolio()` and creates about 50 new script objects every time. The move scripts scan the whole board for each unit and script, and `AttackMaxDamageScr` copies the game state for each attack action. None of this counts as forward-model calls, but it dominates CPU time in pMCTS rollouts.
- **`RandomPortfolio` doesn't work.** It never calls `setActions` or `process`, so its assignments have a `null` action. `getPortfolio()` returns `null`, so `PrunePortfolioHeuristic` can't be built from it. Its `Random` is unseeded. Nothing uses it.
- **Script comments are often copy-pasted.** For example, `BuildFarmScr` says "builds a Forge", `ResourceGatheringScr` describes level-up, and the `ClearForestFor*` scripts all mention the customs house. Trust the code and the table above.
