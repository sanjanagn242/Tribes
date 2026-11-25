package players.actor_critic_gae;

import core.Constants;
import core.Types;
import core.actions.Action;
import core.game.Game;
import core.game.GameState;
import players.Agent;
import players.RandomAgent;
import utils.ElapsedCpuTimer;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Tribes as an environment for the Python trainer (python/tribes_env.py starts this as a subprocess).
 * It plays the role of env_step in tic-tac-toe's rollout.py:
 *
 *   reset(seed, seat)  new game: the agent sits in `seat` (-1 = random) and a players.RandomAgent in the
 *                      other; returns the first state where the agent is to move
 *   step(action)       plays the agent's action (an index into the legal actions it was sent), then
 *                      everything up to the agent's next decision, including the opponent's whole turn
 *                      if the action ended the agent's turn; returns that state, or the end of the game
 *
 * Every state sent is one where the agent is to move: obs(s) and one feature row per legal action
 * (see Features). The reward is 0 until the game ends, then +1 win / -1 loss / 0 otherwise.
 *
 * Tribes' own Game loop runs the game (in a separate thread) through the forward model, just as Play
 * and Tournament do. The agent's seat is a BridgePlayer whose act() hands the state to Python and waits
 * for the chosen index. (GameState.advance() alone can't run a full game: it never increments the
 * tick, which only Game can do, so turn limits and per-turn income would break.)
 *
 * Protocol on stdin/stdout, big-endian (python/tribes_io.py):
 *   startup   -> int MAGIC, int obsDim, int actDim, int boardSize
 *   RESET     <- int 1, long seed, int seat      -> response
 *   STEP      <- int 2, int action               -> response
 *   QUIT      <- int 0
 *   response  =  int done (0/1), double reward, int seat, and if not done: Features.Encoded.write
 *
 * Arguments (KEY=VALUE): tribes=XIN_XI,IMPERIUS  game_mode=SCORE  level_seed=-1 (random level per game)
 */
public final class EnvServer {

    static final int MAGIC = 0x41434745;   // "ACGE"
    static final int OP_QUIT = 0, OP_RESET = 1, OP_STEP = 2;

    /** What the game thread hands to the main thread: a decision of the agent, or the end of the game. */
    private static final class Msg {
        final Features.Encoded decision;   // null when done
        final double reward;
        final Throwable error;

        Msg(Features.Encoded decision, double reward, Throwable error) {
            this.decision = decision; this.reward = reward; this.error = error;
        }
    }

    /** The agent's seat: sends each state to Python (through the main thread) and plays the index it gets back. */
    private static final class BridgePlayer extends Agent {
        private final Features features;
        private final LinkedBlockingQueue<Msg> toMain;
        private final LinkedBlockingQueue<Integer> toGame;

        BridgePlayer(long seed, Features features, LinkedBlockingQueue<Msg> toMain, LinkedBlockingQueue<Integer> toGame) {
            super(seed);
            this.features = features; this.toMain = toMain; this.toGame = toGame;
        }

        @Override
        public Action act(GameState gs, ElapsedCpuTimer ect) {
            ArrayList<Action> actions = gs.getAllAvailableActions();
            try {
                toMain.put(new Msg(features.encode(gs, actions, playerID), 0.0, null));
                int a = toGame.take();
                if (a < 0 || a >= actions.size())
                    throw new IllegalArgumentException("action " + a + " out of range: " + actions.size() + " legal actions");
                return actions.get(a);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Agent copy() { return null; }
    }

    public static void main(String[] args) throws Exception {
        ActorCriticGAEParams p = new ActorCriticGAEParams();
        for (String a : args) {
            int eq = a.indexOf('=');
            if (eq <= 0) throw new IllegalArgumentException("arguments are KEY=VALUE, got '" + a + "'");
            p.set(a.substring(0, eq).trim(), a.substring(eq + 1));
        }

        // stdout carries the protocol; the game's own prints (final ranking etc.) go nowhere.
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(FileDescriptor.out)));
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(FileDescriptor.in)));
        System.setOut(new PrintStream(new OutputStream() {
            @Override public void write(int b) {}
            @Override public void write(byte[] b, int off, int len) {}
        }));
        Constants.VISUALS = false;
        Constants.VERBOSE = false;
        Constants.LOG_STATS = false;

        Features features = Features.forTribes(p.TRIBES.length);
        out.writeInt(MAGIC);
        out.writeInt(features.obsDim);
        out.writeInt(features.actDim);
        out.writeInt(features.boardSize);
        out.flush();

        LinkedBlockingQueue<Msg> toMain = new LinkedBlockingQueue<>();
        LinkedBlockingQueue<Integer> toGame = new LinkedBlockingQueue<>();
        Thread gameThread = null;
        int seat = -1;

        while (true) {
            int op;
            try { op = in.readInt(); } catch (EOFException e) { break; }
            if (op == OP_QUIT) break;

            if (op == OP_RESET) {
                long seed = in.readLong();
                int wantSeat = in.readInt();
                if (gameThread != null && gameThread.isAlive())
                    throw new IllegalStateException("reset while a game is still running; play every game to the end");
                toMain.clear();
                toGame.clear();

                Random r = new Random(seed);
                seat = wantSeat >= 0 ? wantSeat : r.nextInt(p.TRIBES.length);
                long levelSeed = p.LEVEL_SEED >= 0 ? p.LEVEL_SEED : r.nextInt(Integer.MAX_VALUE);
                long engineSeed = r.nextLong();
                ArrayList<Agent> players = new ArrayList<>();
                BridgePlayer bridge = new BridgePlayer(r.nextLong(), features, toMain, toGame);
                for (int i = 0; i < p.TRIBES.length; i++)
                    players.add(i == seat ? bridge : new RandomAgent(r.nextLong()));

                gameThread = new Thread(() -> {
                    try {
                        Game game = new Game();
                        game.init(players, levelSeed, p.TRIBES, engineSeed, p.GAME_MODE);
                        game.run(null, null);
                        Types.RESULT res = game.getWinnerStatus()[bridge.getPlayerID()];
                        double outcome = res == Types.RESULT.WIN ? 1.0 : res == Types.RESULT.LOSS ? -1.0 : 0.0;
                        toMain.put(new Msg(null, outcome, null));
                    } catch (Throwable t) {
                        toMain.add(new Msg(null, 0.0, t));
                    }
                }, "tribes-game");
                gameThread.setDaemon(true);
                gameThread.start();

            } else if (op == OP_STEP) {
                toGame.put(in.readInt());
            } else {
                throw new IllegalArgumentException("unknown op " + op);
            }

            Msg m = toMain.take();
            if (m.error != null) {
                m.error.printStackTrace();
                System.exit(1);
            }
            if (m.decision == null) gameThread.join();   // game over: its thread is just returning
            out.writeInt(m.decision == null ? 1 : 0);
            out.writeDouble(m.reward);
            out.writeInt(seat);
            if (m.decision != null) m.decision.write(out);
            out.flush();
        }
        out.flush();
        System.exit(0);
    }
}
