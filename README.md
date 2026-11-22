# Tribes

![GitHub release (latest by date)](https://img.shields.io/github/v/release/GAIGResearch/Tribes?style=plastic) ![Tribes Java 93.5%](https://img.shields.io/github/languages/top/GAIGResearch/Tribes?style=plastic) ![GitHub All Releases](https://img.shields.io/github/downloads/GAIGResearch/Tribes/total?style=plastic) ![GitHub stars](https://img.shields.io/github/stars/GAIGResearch/Tribes?style=social)

# About

A turn-based strategy game framework. Tribes is a multi-player, multi-agent, stochastic and partially observable game that involves strategic and tactical combat decisions. A good playing strategy requires the management of a technology tree, build orders and economy. The framework provides a Forward Model, which can be used by Statistical Forward Planning methods.

# Requirements

**Java 8** or later is required to run the project. Download Java [here](https://java.com/en/download/).

# Getting started

In order to run the code, you must either download the repository, or clone it. If you are looking for a particular release, you can find all listed [here](https://github.com/GAIGResearch/Tribes/releases).

![Tribes Download](/img/download.png)

The simplest way to run the code is to create a new project in [IntelliJ IDEA](https://www.jetbrains.com/idea/) or a similar IDE. In IntelliJ, create a **new project from existing sources**, pointing to the code downloaded or cloned. This process should automatically set up the environment and add any project libraries as well (JSON library only required for version 1.0). 

Alternatively, open the code directly in your IDE of choice and add libraries included in the download from the `lib/` package. Make sure `src/` is marked as sources root.

# Running the framework

The main class for running the framework is `Play.java` found in the sources root directory. The `main` method in this class contains 3 ways of running the code. The parameters for all these execution modes can be found in the configuration file `play.json`.

1. **Play one game with visuals, using the level generator.** Call the `play` method with the following possible settings:
    * **tribes**: array of type `Types.TRIBE` containing the tribes that will be assigned to the players.
    * **level seed**: random seed for the level generator
    * **players**: array of type `PlayerType` containing the players taking part in the game; the tribe they receive depends on the order, mapping directly to the tribes array (thus the two arrays must also be of the same length).
    * **game mode**: variable of type `Types.GAME_MODE`, which can take either `Capitals` or `Score` values in version 1.0, to change the mode of the game played.

2. **Play one game with visuals, with level loaded from a file.** Call the alternative `play` method with the following possible settings:
    * **file name**: path to file containing the level to be loaded.
    * **players**: array of type `PlayerType` containing the players taking part in the game; the tribe they receive depends on the order, mapping directly to the tribes defined in the level; read order of tribes in level files is top-left to bottom-right (the player array must contain exactly enough players for the loaded level).
    * **game mode**: variable of type `Types.GAME_MODE`, which can take either `Capitals` or `Score` values in version 1.0, to change the mode of the game played.
    
3. **Play one game with visuals, with whole game loaded from a file.** Call the `load` method with the following possible settings:
    * **players**: array of type `PlayerType` containing the players taking part in the game; the tribe they receive depends on the order, mapping directly to the tribes array saved in the file (thus the two arrays must also be of the same length).
    * **file name**: path to file containing the JSON saved game to be loaded.

Additionally, other variables can be modified for an effect with all methods of running the framework in this file:
* **AGENT_SEED**: this variable defined at the top of the file can be given a new value, to set the random seed used by the AI agents.
* **GAME_SEED**: this variable defined at the top of the file can be given a new value, to set the random seed used by the game (e.g. for deciding random bonuses when exploring ruins).
* **RUN_VERBOSE**: this variable defined at the top of the file can be set to true (producing detailed output while running the game) or false.

The `_getAgent` method and the `PlayerType` enum can be used to set which AI players are available to run in the framework, and new ones can be defined by including both a type, and a corresponding constructor. Currently defined players have parameter options set as well, which can be modified for different behaviours.


## No visuals

To run **without visuals**, set the following variable in the `core.Constants.java` class to false:

```
    public static boolean VISUALS = true;
```

Other intersting variables in the `core.Constants.java` class for running games include:

```
    static final int MAX_TURNS = 30;  // Maximum number of turns when playing in Score mode
    static final int MAX_TURNS_CAPITALS = 50; // Maximum number of turns when playing in Capitals mode
    public static final boolean PLAY_WITH_FULL_OBS = true; // If false, agents receive game states with information hidden for areas covered by fog of war
    public static boolean GUI_FORCE_FULL_OBS = false; // If true, display still shows full observable game states even if players play with partial observability
    public static boolean WRITE_SAVEGAMES = false;  // If true, all games played are saved to files
    public static boolean DISABLE_NON_HUMAN_ACTION_HIGHLIGHT = true;  // If true, human observing/playing doesn't have access to actions of non-human players
    public static int FRAME_DELAY = 500; // The display delay between frames
    public static boolean TURN_LIMITED = false;  // Limits games to maximum number of turns
    public static long TURN_TIME_MILLIS = 10000000; // Limits a player's thinking time to this many milliseconds
    public static boolean GUI_PAN_TO_TRIBE = false;  // Pans to a tribe's capital when turn changes
    public static boolean GUI_DRAW_EFFECTS = false;  // Draws animations for unit actions
```

## Tournaments

To run round-robin tournaments between multiple AI players, run class `Tournament.java`. Running a tourament requires some parameters, which are indicated in the file `tournament.json`. Among these parameters, you may determine:

* **game mode**: Int, 0 for Capitals, 1 for Score
* **# repetitions**: Int, number of repetitions per match-up
* **max length**: Int, maximum number of turns per game
* **force turn end**: Int, 1 if true, 0 if false, forces players to end their turn after 5 moves if true.
* **MCTS rollouts**: Int, 1 if true, 0 if false, decides if MCTS is using rollouts or not
* **population size**: Int, for population-based algorithms, e.g. RHEA
* **players**: String x N, N player types taking part in the game
* **tribes**: String x N, N tribes for the game, same number as players
* **Level Seeds**: A series of random seeds for the game levels.

Tournament results indicate number of games played (N), number of wins (W) and win rate, score (S), number of technologies researched (T), number of cities (C) and star production (P) for each player. 

### Reading the results

At the end of a tournament, `Tournament.java` prints one line per player under `--------- RESULTS ---------`, for example:

```
--------- RESULTS ---------
[N:1];[%:100.00];[W:1];[S:13407.00];[T:24.00];[C:3.00];[P:28.00];[D:17.00];[R:202.00];[Player:1:MCTS]
[N:1];[%:0.00];[W:0];[S:7121.00];[T:24.00];[C:0.00];[P:0.00];[D:28.00];[R:171.00];[Player:0:EMCTS]
```

| Field | Meaning | Aggregated as |
|-------|---------|---------------|
| `N` | Games this player took part in | count |
| `%` | Win rate, `100 · W / N` | percentage |
| `W` | Games won | total |
| `S` | Final game score | mean over games |
| `T` | Technologies researched by the end of the game (24 is the whole tech tree) | mean |
| `C` | Cities owned at the end of the game | mean |
| `P` | Star production per turn at the end of the game (sum over the tribe's cities) | mean |
| `D` | Wars declared by this tribe | mean |
| `R` | Stars sent to other tribes | mean |
| `Player:id:TYPE` | Index of the player in the config's `"Players"` list, and its agent type | — |

All values except `N`, `%` and `W` are **averages per game**, measured on the final game state. Lines are sorted best first: by total wins, then mean `S`, `T`, `C`, `P`, `D`, `R`.

In the example, MCTS won the only game. It finished with 3 cities and 28 stars per turn, and EMCTS ended with no cities (and therefore 0 production) because it had been conquered.

**How a game is won**

* **Capitals mode:** a tribe wins as soon as it owns every capital. If nobody has done that by the turn limit (`Constants.MAX_TURNS_CAPITALS`, 50 by default), the game ends and the tribe ranked first wins.
* **Score mode:** the game runs to the turn limit (`MAX_TURNS`, 30), and the tribe ranked first wins.
* **Ranking**, used to pick the winner at the turn limit: highest score first, then techs, cities, production, wars declared and stars sent, in that order. A full tie is broken at random.

So a win in Capitals mode can mean either a conquest or a score win at the turn limit. A winner with all the cities (as above) is usually a conquest. Look at the per-game lines (`#1: Tribe ... WIN`) further up in the output to check individual games.

**Other lines in the output**

* `Playing with [0:MCTS(XIN_XI), 1:EMCTS(IMPERIUS)] (3/10)`: the game about to start, with each player's id, agent type and tribe. With `"Shift Tribes": true`, players rotate tribes and turn order from one game to the next.
* `#1: Tribe IMPERIUS (EMCTSAgent): WIN, 13986 points; #tech: 24, #cities: 3, production: 28` and the `Diplomacy - ...` line below it: final results of a single game.
* `FM: Action [...] couldn't execute?`: printed by the forward model when an agent simulates an action that is no longer legal during its search. This is harmless. Filter it out with `grep -v '^FM:'`.
* `Error running a game, trying again.`: the game crashed (the stack trace is on stderr) and is replayed with a new game seed. Crashed games are not counted in the results.

**Pairwise round robin**

`Tournament.java` puts *all* listed players into every game (an N-player free-for-all) and only rotates their seats. To get 1v1 results for every pair instead, use `round_robin.py`:

```
python3 round_robin.py --players MCTS EMCTS RHEA --jobs 3
```

It runs each pair as its own tournament with the other settings from `tournament.json`, and writes to `results/round_robin_<timestamp>/`:

* the `RESULTS` lines above, for each pair, in `<pair>.out`;
* `summary.txt`: a win-rate matrix (the row player's wins against the column player), overall wins, win rate, average score and average rounds per player, then games, rounds (avg/min/max/total) and time per pair, totals, and total wall time;
* `summary.csv`: one row per player per pair, with games, wins, win rate, average score, average rounds and time.

"Rounds" are the turns a game lasted (the number in the `44; Game Results:` line). In Capitals mode a game that hits the turn limit ends at round 51, so 51 means no conquest: the winner was decided by ranking. A lower number means the game ended by conquest.

## Game configuration

The game itself can be modified by changing its configuration (e.g. attack power of certain units) in the `core.TribesConfig.java` file.

## In-game randomness

Besides map generation (driven by the level seed), the game has its own random number generator, seeded with the **game seed**. `Game.init` creates it (`rnd = new Random(seed)` in `src/core/game/Game.java`) and passes it to `GameState`. Game code reads it through `GameState.getRandomGenerator()`. It decides the following:

| Event | Where |
|-------|-------|
| Bonus from examining ruins (picked at random) | `ExamineCommand.java` → `Types.EXAMINE_BONUS.random(...)` |
| Explorer's random walk (only reveals tiles with partial observability) | `Board.launchExplorer(...)` |
| Random technology received on meeting another tribe (only with partial observability) | `Tribe.meetTribe(...)` (called from `Tribe.clearView(...)`) |
| Random technology research | `TechnologyTree.researchAtRandom(...)` |
| Where a pushed unit ends up (e.g. after an attack) | `Board.pushUnit(...)` / `Board.tryPush(...)`, called from `AttackCommand.java` |
| Which cities units move to when a city is captured | `Board.moveOneToNewCity(...)` / `Board.moveAllFromCity(...)` (`Collections.shuffle`) |

Combat damage is deterministic.

Fixing the game seed (`"Game Seed"` in `play.json`; `-1` means use the current time) makes the real game's random events repeatable. There are two exceptions:

* `GameState.copy()` gives each copy a fresh, unseeded `new Random()`. Agents that simulate with the forward model (MCTS, RHEA, OEP, ...) therefore see different random outcomes in their simulations from one run to the next.
* `TribeResult.compareTo(...)` breaks a full tie in the final ranking with an unseeded `new Random()`.

# Associated research

* Diego Perez Liebana, Yu-Jhen Hsu, Stavros Emmanouilidis, Bobby Khaleque, Raluca Gaina, "**Tribes: A New Turn-Based Strategy Game for AI**", in _Sixteenth AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE)_, 2020.

# Acknowledgements

This work is supported by UK EPSRC research grants EP/T008962/1 and [IGGI CDT](http://iggi.org.uk) EP/L015846/1
