package players.actor_critic_mcts.selfplay;

import core.Types;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Iterator;

/**
 * Settings for one self-play run. The defaults are for a real run of one iteration on a single machine
 * (same values as the cluster pipeline in slurm/config.sh): 200 games of 2-player SCORE mode at 1000 FM calls per move,
 * which gives ~60k training samples.
 *
 * Override them from the command line, as a JSON file and/or key=value pairs (later ones win):
 *   java ... SelfPlay selfplay.json iteration=1 modelPath=models/iter_001/best.pt tribes=XIN_XI,IMPERIUS
 * The JSON file uses the same keys, e.g. {"numGames": 10, "tribes": ["XIN_XI", "IMPERIUS"], "gameMode": "SCORE"}.
 *
 * To use several CPU cores, run a few SelfPlay processes side by side with the same iteration, splitting the games
 * with numGames/gameOffset (e.g. 4 processes: numGames=50 with gameOffset=0, 50, 100, 150). They write separate files
 * into the same data/selfplay/iter_XXX folder, and the training script reads them all.
 * A quick test instead of a real run:  numGames=3 fmCallsPerMove=200
 */
public class SelfPlayConfig {

    public int numGames = 200;                    // games per iteration (~300 samples each)
    public int gameOffset = 0;                    // first game id; give each parallel process its own range
    public int iteration = 0;                     // self-play / training iteration (output folder + seeds)
    public String outputDir = "data/selfplay";    // data goes to <outputDir>/iter_XXX/
    public String modelPath = null;               // null -> UniformNetwork (iteration 0); then models/iter_XXX/best.pt

    public Types.TRIBE[] tribes = {Types.TRIBE.XIN_XI, Types.TRIBE.IMPERIUS};
    public Types.GAME_MODE gameMode = Types.GAME_MODE.SCORE;   // 30 turns and always a winner; CAPITALS runs up to 50 turns
    public long levelSeed = -1;                   // -1 -> new level each game (derived from baseSeed, iteration, game id)
    public long baseSeed = 42;                    // change it to get a different set of maps for a repeat run

    public int fmCallsPerMove = 1000;             // search budget per act() (params.STOP_FMCALLS); ~1-2 min per game with the network
    public int maxRolloutLength = 20;             // tree depth limit (params.ROLLOUT_LENGTH)
    public int forceTurnEnd = 5;                  // force EndTurn every this many plies in the tree

    public double temperature = 1.0;              // sample moves from visits^(1/T) ...
    public int tempMoves = 60;                    // ... for each player's first 60 actions (~6-8 turns), then argmax
    public boolean rootNoise = true;              // Dirichlet noise on root priors

    public boolean compress = true;               // write .jsonl.gz instead of .jsonl
    public boolean quiet = true;                  // silence the game's per-turn logging / stats

    /** Read overrides from a JSON file (any arg ending in .json) and/or key=value args; fall back to the defaults above. */
    public static SelfPlayConfig fromArgs(String[] args) {
        SelfPlayConfig cfg = new SelfPlayConfig();
        if (args == null) return cfg;
        for (String arg : args) {
            if (arg.endsWith(".json")) {
                JSONObject json = readJson(arg);
                Iterator<String> keys = json.keys();
                while (keys.hasNext()) {
                    String k = keys.next();
                    Object v = json.get(k);
                    cfg.set(k, v instanceof JSONArray ? joinArray((JSONArray) v) : String.valueOf(v));
                }
            } else {
                int eq = arg.indexOf('=');
                if (eq <= 0) throw new IllegalArgumentException("Expected key=value or a .json file, got: " + arg);
                cfg.set(arg.substring(0, eq).trim(), arg.substring(eq + 1).trim());
            }
        }
        return cfg;
    }

    private void set(String key, String v) {
        switch (key) {
            case "numGames": numGames = Integer.parseInt(v); break;
            case "gameOffset": gameOffset = Integer.parseInt(v); break;
            case "iteration": iteration = Integer.parseInt(v); break;
            case "outputDir": outputDir = v; break;
            case "modelPath": modelPath = (v.isEmpty() || v.equals("null")) ? null : v; break;
            case "tribes": {
                String[] names = v.split(",");
                tribes = new Types.TRIBE[names.length];
                for (int i = 0; i < names.length; i++) tribes[i] = Types.TRIBE.valueOf(names[i].trim().toUpperCase());
                break;
            }
            case "gameMode": gameMode = Types.GAME_MODE.valueOf(v.toUpperCase()); break;
            case "levelSeed": levelSeed = Long.parseLong(v); break;
            case "baseSeed": baseSeed = Long.parseLong(v); break;
            case "fmCallsPerMove": fmCallsPerMove = Integer.parseInt(v); break;
            case "maxRolloutLength": maxRolloutLength = Integer.parseInt(v); break;
            case "forceTurnEnd": forceTurnEnd = Integer.parseInt(v); break;
            case "temperature": temperature = Double.parseDouble(v); break;
            case "tempMoves": tempMoves = Integer.parseInt(v); break;
            case "rootNoise": rootNoise = Boolean.parseBoolean(v); break;
            case "compress": compress = Boolean.parseBoolean(v); break;
            case "quiet": quiet = Boolean.parseBoolean(v); break;
            default: throw new IllegalArgumentException("Unknown self-play setting: " + key);
        }
    }

    private static JSONObject readJson(String path) {
        try {
            return new JSONObject(new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new UncheckedIOException("Could not read config " + path, e);
        }
    }

    private static String joinArray(JSONArray a) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length(); i++) {
            if (i > 0) sb.append(',');
            sb.append(a.get(i));
        }
        return sb.toString();
    }

    /** All settings as JSON, saved next to the data so every run documents how it was made. */
    public JSONObject toJson() {
        JSONObject o = new JSONObject();
        o.put("numGames", numGames);
        o.put("gameOffset", gameOffset);
        o.put("iteration", iteration);
        o.put("outputDir", outputDir);
        o.put("modelPath", modelPath == null ? JSONObject.NULL : modelPath);
        JSONArray t = new JSONArray();
        for (Types.TRIBE tr : tribes) t.put(tr.name());
        o.put("tribes", t);
        o.put("gameMode", gameMode.name());
        o.put("levelSeed", levelSeed);
        o.put("baseSeed", baseSeed);
        o.put("fmCallsPerMove", fmCallsPerMove);
        o.put("maxRolloutLength", maxRolloutLength);
        o.put("forceTurnEnd", forceTurnEnd);
        o.put("temperature", temperature);
        o.put("tempMoves", tempMoves);
        o.put("rootNoise", rootNoise);
        o.put("compress", compress);
        return o;
    }
}
