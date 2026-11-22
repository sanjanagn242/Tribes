package players.actor_critic.selfplay;

import core.Types;

/**
 * Settings for one data-collection run. Defaults are placeholders; tune them.
 */
public class SelfPlayConfig {

    public int numGames = 100;
    public int iteration = 0;                     // policy version that collected this data (for folder naming)
    public String outputDir = "data/ac";
    public String modelPath = null;               // the CURRENT policy (iteration 0: a randomly initialised export)

    public Types.TRIBE[] tribes = {Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS};
    public Types.GAME_MODE gameMode = Types.GAME_MODE.CAPITALS;
    public long levelSeed = -1;                   // -1 -> new random level each game
    public long baseSeed = 42;

    /** Read overrides from a JSON file (same style as play.json) or from command-line args. */
    public static SelfPlayConfig fromArgs(String[] args) {
        // TODO: parse args / a JSON config with org.json; fall back to the defaults above.
        return new SelfPlayConfig();
    }
}
