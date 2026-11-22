# Tribes Action Types

This file lists every action type in the game and where its code lives. The action types are defined in the `ACTION` enum in [src/core/Types.java](src/core/Types.java) (`public enum ACTION`).

## How many actions are there?

The original Tribes design has **21 action types**:
- 8 city actions
- 3 tribe actions (Build Road, Research Tech, End Turn)
- 10 unit actions. Upgrade counts as one action here, even though the enum splits it into `UPGRADE_BOAT` and `UPGRADE_SHIP`.

The diplomacy update added 2 more tribe actions, **Declare War** and **Send Stars**, so this codebase can generate **23** action types. The enum also defines `CLIMB_MOUNTAIN`. It has no action class, factory or command, so the game never generates or executes it.

## How actions are generated and executed

Each action has three parts:
1. **Action class**, e.g. `Spawn.java`. It holds the action's parameters and its `isFeasible(GameState)` check.
2. **Factory**, e.g. `factory/SpawnFactory.java`. Its `computeActionVariants(actor, gs)` lists every feasible version of the action for one actor.
3. **Command**, e.g. `command/SpawnCommand.java`. Its `execute(action, gs)` applies the action to the game state. `Types.ACTION.getCommand()` maps each action type to its command.

Each actor type has a builder that runs all of its factories:

| Builder | Called from |
|---|---|
| [CityActionBuilder.java](src/core/actions/cityactions/factory/CityActionBuilder.java) | `GameState.computePlayerActions` ([GameState.java:191](src/core/game/GameState.java#L191)) |
| [UnitActionBuilder.java](src/core/actions/unitactions/factory/UnitActionBuilder.java) | `GameState.computePlayerActions` ([GameState.java:236](src/core/game/GameState.java#L236)) |
| [TribeActionBuilder.java](src/core/actions/tribeactions/factory/TribeActionBuilder.java) | `GameState.computePlayerActions` ([GameState.java:246](src/core/game/GameState.java#L246)) |

Two special cases in the builders:
- **City:** when `LEVEL_UP` is possible, it is the only action the city gets that turn.
- **Unit:** `UPGRADE` is always generated. When the unit has already finished its turn (`isFinished()`), Upgrade is the only action it gets.

---

## City actions (8)
Base path: [src/core/actions/cityactions/](src/core/actions/cityactions/)

| # | Action (`Types.ACTION`) | Tech required | Action class | Factory (generation) | Command (execution) |
|---|---|---|---|---|---|
| 1 | `BUILD` | none (depends on the building) | [Build.java](src/core/actions/cityactions/Build.java) | [BuildFactory.java](src/core/actions/cityactions/factory/BuildFactory.java) | [BuildCommand.java](src/core/actions/cityactions/command/BuildCommand.java) |
| 2 | `BURN_FOREST` | Chivalry | [BurnForest.java](src/core/actions/cityactions/BurnForest.java) | [BurnForestFactory.java](src/core/actions/cityactions/factory/BurnForestFactory.java) | [BurnForestCommand.java](src/core/actions/cityactions/command/BurnForestCommand.java) |
| 3 | `CLEAR_FOREST` | Forestry | [ClearForest.java](src/core/actions/cityactions/ClearForest.java) | [ClearForestFactory.java](src/core/actions/cityactions/factory/ClearForestFactory.java) | [ClearForestCommand.java](src/core/actions/cityactions/command/ClearForestCommand.java) |
| 4 | `DESTROY` | Construction | [Destroy.java](src/core/actions/cityactions/Destroy.java) | [DestroyFactory.java](src/core/actions/cityactions/factory/DestroyFactory.java) | [DestroyCommand.java](src/core/actions/cityactions/command/DestroyCommand.java) |
| 5 | `GROW_FOREST` | Spiritualism | [GrowForest.java](src/core/actions/cityactions/GrowForest.java) | [GrowForestFactory.java](src/core/actions/cityactions/factory/GrowForestFactory.java) | [GrowForestCommand.java](src/core/actions/cityactions/command/GrowForestCommand.java) |
| 6 | `LEVEL_UP` (city bonus) | none | [LevelUp.java](src/core/actions/cityactions/LevelUp.java) | [LevelUpFactory.java](src/core/actions/cityactions/factory/LevelUpFactory.java) | [LevelUpCommand.java](src/core/actions/cityactions/command/LevelUpCommand.java) |
| 7 | `RESOURCE_GATHERING` | none (depends on the resource) | [ResourceGathering.java](src/core/actions/cityactions/ResourceGathering.java) | [ResourceGatheringFactory.java](src/core/actions/cityactions/factory/ResourceGatheringFactory.java) | [ResourceGatheringCommand.java](src/core/actions/cityactions/command/ResourceGatheringCommand.java) |
| 8 | `SPAWN` | none (depends on the unit type) | [Spawn.java](src/core/actions/cityactions/Spawn.java) | [SpawnFactory.java](src/core/actions/cityactions/factory/SpawnFactory.java) | [SpawnCommand.java](src/core/actions/cityactions/command/SpawnCommand.java) |

Notes:
- **Level-up bonuses:** each city level offers different bonuses. These are defined in the `Types.CITY_LEVEL_UP` enum (`getActions(level)`, `validType`). The bonuses are WORKSHOP, EXPLORER, CITY_WALL, RESOURCES, POP_GROWTH, BORDER_GROWTH, PARK and SUPERUNIT, and the `switch` in `LevelUpCommand` applies each one.
- **Spawn feasibility** (`Spawn.isFeasible`): all of these must hold.
  - The unit type is `spawnable()`. Naval units and giants are not.
  - The tribe has enough stars.
  - The city has unit capacity (`canAddUnit()`).
  - The city tile is empty.
  - The required tech has been researched.

## Tribe actions (5: 3 original + 2 diplomacy)
Base path: [src/core/actions/tribeactions/](src/core/actions/tribeactions/)

| # | Action (`Types.ACTION`) | Tech required | Action class | Factory (generation) | Command (execution) |
|---|---|---|---|---|---|
| 9 | `BUILD_ROAD` | Roads | [BuildRoad.java](src/core/actions/tribeactions/BuildRoad.java) | [BuildRoadFactory.java](src/core/actions/tribeactions/factory/BuildRoadFactory.java) | [BuildRoadCommand.java](src/core/actions/tribeactions/command/BuildRoadCommand.java) |
| 10 | `RESEARCH_TECH` | none | [ResearchTech.java](src/core/actions/tribeactions/ResearchTech.java) | [ResearchTechFactory.java](src/core/actions/tribeactions/factory/ResearchTechFactory.java) | [ResearchTechCommand.java](src/core/actions/tribeactions/command/ResearchTechCommand.java) |
| 11 | `END_TURN` | none | [EndTurn.java](src/core/actions/tribeactions/EndTurn.java) | [EndTurnFactory.java](src/core/actions/tribeactions/factory/EndTurnFactory.java) | [EndTurnCommand.java](src/core/actions/tribeactions/command/EndTurnCommand.java) |
| (+) | `DECLARE_WAR` *(diplomacy)* | none | [DeclareWar.java](src/core/actions/tribeactions/DeclareWar.java) | [DeclareWarFactory.java](src/core/actions/tribeactions/factory/DeclareWarFactory.java) | [DeclareWarCommand.java](src/core/actions/tribeactions/command/DeclareWarCommand.java) |
| (+) | `SEND_STARS` *(diplomacy)* | none | [SendStars.java](src/core/actions/tribeactions/SendStars.java) | [SendStarsFactory.java](src/core/actions/tribeactions/factory/SendStarsFactory.java) | [SendStarsCommand.java](src/core/actions/tribeactions/command/SendStarsCommand.java) |

## Unit actions (10)
Base path: [src/core/actions/unitactions/](src/core/actions/unitactions/)

| # | Action (`Types.ACTION`) | Tech required | Action class | Factory (generation) | Command (execution) |
|---|---|---|---|---|---|
| 12 | `ATTACK` | none | [Attack.java](src/core/actions/unitactions/Attack.java) | [AttackFactory.java](src/core/actions/unitactions/factory/AttackFactory.java) | [AttackCommand.java](src/core/actions/unitactions/command/AttackCommand.java) |
| 13 | `CAPTURE` | none | [Capture.java](src/core/actions/unitactions/Capture.java) | [CaptureFactory.java](src/core/actions/unitactions/factory/CaptureFactory.java) | [CaptureCommand.java](src/core/actions/unitactions/command/CaptureCommand.java) |
| 14 | `CONVERT` | none (Mind Bender only) | [Convert.java](src/core/actions/unitactions/Convert.java) | [ConvertFactory.java](src/core/actions/unitactions/factory/ConvertFactory.java) | [ConvertCommand.java](src/core/actions/unitactions/command/ConvertCommand.java) |
| 15 | `DISBAND` | Free Spirit | [Disband.java](src/core/actions/unitactions/Disband.java) | [DisbandFactory.java](src/core/actions/unitactions/factory/DisbandFactory.java) | [DisbandCommand.java](src/core/actions/unitactions/command/DisbandCommand.java) |
| 16 | `EXAMINE` (ruins) | none | [Examine.java](src/core/actions/unitactions/Examine.java) | [ExamineFactory.java](src/core/actions/unitactions/factory/ExamineFactory.java) | [ExamineCommand.java](src/core/actions/unitactions/command/ExamineCommand.java) |
| 17 | `HEAL_OTHERS` | none (Mind Bender only) | [HealOthers.java](src/core/actions/unitactions/HealOthers.java) | [HealOthersFactory.java](src/core/actions/unitactions/factory/HealOthersFactory.java) | [HealOthersCommand.java](src/core/actions/unitactions/command/HealOthersCommand.java) |
| 18 | `MAKE_VETERAN` | none | [MakeVeteran.java](src/core/actions/unitactions/MakeVeteran.java) | [MakeVeteranFactory.java](src/core/actions/unitactions/factory/MakeVeteranFactory.java) | [MakeVeteranCommand.java](src/core/actions/unitactions/command/MakeVeteranCommand.java) |
| 19 | `MOVE` | none (terrain may need a tech) | [Move.java](src/core/actions/unitactions/Move.java) (with the helper [StepMove.java](src/core/actions/unitactions/StepMove.java)) | [MoveFactory.java](src/core/actions/unitactions/factory/MoveFactory.java) | [MoveCommand.java](src/core/actions/unitactions/command/MoveCommand.java) |
| 20 | `RECOVER` | none | [Recover.java](src/core/actions/unitactions/Recover.java) | [RecoverFactory.java](src/core/actions/unitactions/factory/RecoverFactory.java) | [RecoverCommand.java](src/core/actions/unitactions/command/RecoverCommand.java) |
| 21 | `UPGRADE_BOAT` / `UPGRADE_SHIP` | Sailing / Navigation | [Upgrade.java](src/core/actions/unitactions/Upgrade.java) | [UpgradeFactory.java](src/core/actions/unitactions/factory/UpgradeFactory.java) | [UpgradeCommand.java](src/core/actions/unitactions/command/UpgradeCommand.java) |

## Declared but not implemented
| Action (`Types.ACTION`) | Tech | Status |
|---|---|---|
| `CLIMB_MOUNTAIN` | Climbing | No action class, factory or command. Instead, the Climbing tech check for mountain tiles is in [Board.java](src/core/game/Board.java) (around lines 264, 339 and 519). |

## Shared base classes
- [Action.java](src/core/actions/Action.java): abstract base class for all actions.
- [ActionFactory.java](src/core/actions/ActionFactory.java): interface for the factories (`computeActionVariants`).
- [ActionCommand.java](src/core/actions/ActionCommand.java): interface for the commands (`execute`).
- [CityAction.java](src/core/actions/cityactions/CityAction.java), [TribeAction.java](src/core/actions/tribeactions/TribeAction.java), [UnitAction.java](src/core/actions/unitactions/UnitAction.java): base classes for each actor type.
