package players.actor_critic_mcts.selfplay;

import core.Constants;
import core.game.Game;
import org.json.JSONArray;
import org.json.JSONObject;
import players.Agent;
import players.actor_critic_mcts.ACMCTSParams;
import players.actor_critic_mcts.ACMCTSPlayer;
import players.actor_critic_mcts.encoding.ActionEncoder;
import players.actor_critic_mcts.encoding.StateEncoder;
import players.actor_critic_mcts.network.ActorCriticNetwork;
import players.actor_critic_mcts.network.ExternalNetwork;
import players.actor_critic_mcts.network.UniformNetwork;
import players.actor_critic_mcts.records.GameRecord;
import players.actor_critic_mcts.records.RecordWriter;
import players.actor_critic_mcts.records.TrainingSample;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Self-play data generator: the same network plays every tribe, each player records its own samples,
 * and the finished records are written to disk for training.
 *
 * The full loop (outside this class) is:
 *     iteration 0: SelfPlay with UniformNetwork  ->  train in Python  ->  model_1
 *     iteration k: SelfPlay with model_k          ->  train            ->  model_{k+1}
 *     (optionally) Evaluate model_{k+1} vs model_k / vs plain MCTS before promoting it.
 *
 * Output, in outputFolder(cfg):
 *     games_0000-0099.jsonl.gz : one TrainingSample per line (see TrainingSample.toJson)
 *     schema.json              : the config used, feature sizes and the name of every feature column
 *
 * Run (after compiling):  java -cp <classes>:lib/json.jar players.actor_critic_mcts.selfplay.SelfPlay [config.json] [key=value ...]
 */
public class SelfPlay {

    public static void main(String[] args) {
        SelfPlayConfig cfg = SelfPlayConfig.fromArgs(args);
        if (cfg.quiet) {
            Constants.VERBOSE = false;
            Constants.LOG_STATS = false;
        }
        String folder = outputFolder(cfg);
        int first = cfg.gameOffset, last = cfg.gameOffset + cfg.numGames - 1;
        String fileName = String.format("games_%04d-%04d.jsonl%s", first, last, cfg.compress ? ".gz" : "");
        System.out.println("Self-play iteration " + cfg.iteration + ": games " + first + "-" + last + " -> " + folder + "/" + fileName
                + " (model: " + (cfg.modelPath == null ? "uniform" : cfg.modelPath) + ")");

        ActorCriticNetwork net = buildNetwork(cfg);
        long start = System.currentTimeMillis();
        try (RecordWriter writer = new RecordWriter(folder)) {
            writer.open(fileName);
            writeSchema(cfg, folder);
            for (int g = first; g <= last; g++) {
                long t0 = System.currentTimeMillis();
                List<GameRecord> records = playGame(g, cfg, net);
                writer.writeAll(records);
                logProgress(g, records, System.currentTimeMillis() - t0);
            }
            writer.commit();   // only now does the file appear under its final name
            System.out.printf("Done: %d samples in %.1f min -> %s%n", writer.getSamplesWritten(),
                    (System.currentTimeMillis() - start) / 60000.0, writer.getPath());
            if (net instanceof ExternalNetwork) {
                ExternalNetwork ext = (ExternalNetwork) net;
                System.out.printf("Network: %d evaluations, %.2f ms each on average%n",
                        ext.getEvaluations(), ext.getMeanEvalMillis());
            }
        } finally {
            net.close();
        }
    }

    /** UniformNetwork when cfg.modelPath == null, otherwise ExternalNetwork loaded from cfg.modelPath. */
    static ActorCriticNetwork buildNetwork(SelfPlayConfig cfg) {
        if (cfg.modelPath == null) return new UniformNetwork();
        ActorCriticNetwork net = new ExternalNetwork(new StateEncoder(), new ActionEncoder());
        net.load(cfg.modelPath);
        return net;
    }

    /** Self-play params: priors + value net on, root noise on, temperature sampling on, recording on. */
    static ACMCTSParams buildParams(SelfPlayConfig cfg) {
        ACMCTSParams p = new ACMCTSParams();
        // Same base settings as Run.getAgent() uses for AC_MCTS ...
        p.stop_type = p.STOP_FMCALLS;
        p.num_fmcalls = cfg.fmCallsPerMove;
        p.heuristic_method = p.DIFF_HEURISTIC;
        p.ROLLOUT_LENGTH = cfg.maxRolloutLength;
        p.FORCE_TURN_END = cfg.forceTurnEnd;
        // ... except PRIORITIZE_ROOT: the policy target must cover every legal action, not one action group.
        p.PRIORITIZE_ROOT = false;

        p.USE_PRIORS = true;
        // UniformNetwork's value is a constant 0, so only trust the critic once a trained model is loaded;
        // until then leaves are evaluated with the DIFF heuristic.
        p.USE_VALUE_NET = cfg.modelPath != null;
        p.ROOT_NOISE = cfg.rootNoise;
        p.TEMPERATURE = cfg.temperature;
        p.TEMP_MOVES = cfg.tempMoves;
        p.RECORD_SAMPLES = true;
        return p;
    }

    /** One ACMCTSPlayer per tribe, all sharing the same network, each with its own seed and GameRecord. */
    static ArrayList<Agent> createPlayers(int gameId, SelfPlayConfig cfg, ActorCriticNetwork net) {
        ArrayList<Agent> players = new ArrayList<>();
        Random seeds = new Random(gameSeedBase(cfg, gameId) ^ 0x5DEECE66DL);
        for (int i = 0; i < cfg.tribes.length; i++) {
            long seed = seeds.nextLong();
            ACMCTSPlayer player = new ACMCTSPlayer(seed, buildParams(cfg));
            player.setNetwork(net);
            players.add(player);
        }
        return players;
    }

    /**
     * Play one headless game and return each player's finished record.
     * Game.run(null, null) runs without GUI and calls Agent.result() on every player at the end,
     * which is where ACMCTSPlayer assigns value targets.
     */
    static List<GameRecord> playGame(int gameId, SelfPlayConfig cfg, ActorCriticNetwork net) {
        ArrayList<Agent> players = createPlayers(gameId, cfg, net);
        Random seeds = new Random(gameSeedBase(cfg, gameId));
        long levelSeed = cfg.levelSeed == -1 ? seeds.nextLong() : cfg.levelSeed;
        long gameSeed = seeds.nextLong();

        Game game = new Game();
        game.init(players, levelSeed, cfg.tribes, gameSeed, cfg.gameMode);
        for (Agent p : players) ((ACMCTSPlayer) p).startRecording(gameId);   // after init: playerID is set now
        game.run(null, null);

        List<GameRecord> records = new ArrayList<>();
        for (Agent p : players) records.add(((ACMCTSPlayer) p).getGameRecord());
        return records;
    }

    /**
     * Seed for everything random in one game (level, game, agents). It depends on baseSeed, the iteration and the
     * game id, so different nodes (different gameOffset) and different iterations never replay the same game.
     */
    static long gameSeedBase(SelfPlayConfig cfg, int gameId) {
        long h = cfg.baseSeed;
        h = h * 0x9E3779B97F4A7C15L + cfg.iteration;
        h = h * 0x9E3779B97F4A7C15L + gameId;
        return h ^ (h >>> 31);
    }

    static String outputFolder(SelfPlayConfig cfg) {
        return cfg.outputDir + "/iter_" + String.format("%03d", cfg.iteration);
    }

    /** Print game id, each player's outcome (value target), number of samples, time taken. */
    static void logProgress(int gameId, List<GameRecord> records, long millis) {
        StringBuilder sb = new StringBuilder();
        int total = 0;
        for (GameRecord r : records) {
            List<TrainingSample> s = r.getSamples();
            total += s.size();
            double z = s.isEmpty() ? Double.NaN : s.get(0).valueTarget;
            sb.append(String.format("  p%d: z=%+.2f n=%d", r.getPlayerID(), z, s.size()));
        }
        System.out.printf("game %4d | %5.1fs | %5d samples |%s%n", gameId, millis / 1000.0, total, sb);
    }

    /** Config + feature layout, so the training script knows how to split and name every column. */
    static void writeSchema(SelfPlayConfig cfg, String folder) {
        StateEncoder se = new StateEncoder();
        ActionEncoder ae = new ActionEncoder();
        JSONObject o = new JSONObject();
        o.put("config", cfg.toJson());
        o.put("maxPlayers", se.maxPlayers());
        o.put("valueRange", new JSONArray(new double[]{-1, 1}));
        o.put("nodeFeatureSize", se.nodeFeatureSize());
        o.put("edgeFeatureSize", se.edgeFeatureSize());
        o.put("globalFeatureSize", se.globalFeatureSize());
        o.put("actionFeatureSize", ae.actionFeatureSize());
        o.put("actionSrcNodeColumn", ae.srcNodeColumn());
        o.put("actionTgtNodeColumn", ae.tgtNodeColumn());
        o.put("edgeNeighbourOrder", "(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)");
        o.put("nodeFeatureNames", new JSONArray(se.nodeFeatureNames()));
        o.put("edgeFeatureNames", new JSONArray(se.edgeFeatureNames()));
        o.put("globalFeatureNames", new JSONArray(se.globalFeatureNames()));
        o.put("actionFeatureNames", new JSONArray(ae.featureNames()));
        // Several array tasks write the same schema into the same folder at once: write a private temp file and
        // rename it into place, so readers never see a half-written schema.json.
        try {
            java.nio.file.Path tmp = Files.createTempFile(Paths.get(folder), "schema", ".tmp");
            Files.write(tmp, o.toString(2).getBytes(StandardCharsets.UTF_8));
            Files.move(tmp, Paths.get(folder, "schema.json"), java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not write schema.json", e);
        }
    }
}
