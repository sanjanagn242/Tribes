package players.actor_critic.selfplay;

import core.game.Game;
import players.Agent;
import players.actor_critic.ActorCriticParams;
import players.actor_critic.ActorCriticPlayer;
import players.actor_critic.network.ActorCriticNetwork;
import players.actor_critic.records.RecordWriter;
import players.actor_critic.records.Trajectory;

import java.util.ArrayList;
import java.util.List;

/**
 * Data collection: the current policy plays every tribe, and each player records its own trajectory.
 *
 * The full on-policy loop (outside this class) is:
 *     export policy_k  ->  SelfPlay (collect with policy_k)  ->  Python: compute returns/advantages,
 *     update actor and critic  ->  export policy_{k+1}  ->  repeat.
 * Train only on data from the current policy (or use PPO-style ratios with Transition.actionProb).
 *
 * Run (after compiling):  java -cp <classes>:lib/json.jar players.actor_critic.selfplay.SelfPlay
 */
public class SelfPlay {

    public static void main(String[] args) {
        SelfPlayConfig cfg = SelfPlayConfig.fromArgs(args);
        // TODO: wire the steps together, roughly:
        //   ActorCriticNetwork net = buildNetwork(cfg);
        //   try (RecordWriter writer = new RecordWriter(outputFolder(cfg))) {
        //       writer.open(...);
        //       for (int g = 0; g < cfg.numGames; g++) {
        //           List<Trajectory> trajectories = playGame(g, cfg, net);
        //           writer.writeAll(trajectories);
        //           logProgress(g, trajectories);
        //       }
        //   }
        //   net.close();
    }

    /** ExternalNetwork (with your StateEncoder / ActionEncoder) loaded from cfg.modelPath. */
    static ActorCriticNetwork buildNetwork(SelfPlayConfig cfg) {
        // TODO
        throw new UnsupportedOperationException("TODO: SelfPlay.buildNetwork");
    }

    /** Collection params: SAMPLE_ACTIONS = true, TEMPERATURE = 1.0, RECORD_TRANSITIONS = true. */
    static ActorCriticParams buildParams(SelfPlayConfig cfg) {
        // TODO
        throw new UnsupportedOperationException("TODO: SelfPlay.buildParams");
    }

    /** One ActorCriticPlayer per tribe, all sharing the same network, each with its own seed. */
    static ArrayList<Agent> createPlayers(int gameId, SelfPlayConfig cfg, ActorCriticNetwork net) {
        // TODO: new ActorCriticPlayer(seed, buildParams(cfg)); player.setNetwork(net);
        throw new UnsupportedOperationException("TODO: SelfPlay.createPlayers");
    }

    /**
     * Play one headless game and return each player's finished trajectory.
     * Game.run(null, null) runs without GUI and calls Agent.result() on every player at the end,
     * which is where the terminal reward is assigned.
     */
    static List<Trajectory> playGame(int gameId, SelfPlayConfig cfg, ActorCriticNetwork net) {
        // TODO:
        //   ArrayList<Agent> players = createPlayers(gameId, cfg, net);
        //   Game game = new Game();
        //   game.init(players, levelSeed, cfg.tribes, gameSeed, cfg.gameMode);
        //   for each player: ((ActorCriticPlayer) p).startRecording(gameId);   // after init: playerID is set now
        //   game.run(null, null);
        //   collect ((ActorCriticPlayer) p).getTrajectory() from every player.
        throw new UnsupportedOperationException("TODO: SelfPlay.playGame");
    }

    /** e.g. cfg.outputDir + "/iter_" + String.format("%03d", cfg.iteration) */
    static String outputFolder(SelfPlayConfig cfg) {
        // TODO
        throw new UnsupportedOperationException("TODO: SelfPlay.outputFolder");
    }

    /** Print game id, winner, trajectory lengths, total reward per player, time taken. */
    static void logProgress(int gameId, List<Trajectory> trajectories) {
        // TODO
    }
}
