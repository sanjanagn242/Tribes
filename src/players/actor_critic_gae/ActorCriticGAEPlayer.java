package players.actor_critic_gae;

import core.actions.Action;
import core.game.GameState;
import players.Agent;
import utils.ElapsedCpuTimer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * The trained GAE actor-critic as a Tribes player, for Run (play.json / tournament.json: "ACGAE").
 * Each act() is one decision: pi(.|s) from the Python actor (PythonPolicy, loading params.MODEL_PATH),
 * then argmax when params.GREEDY, else a sample.
 *
 * Training doesn't go through this class: python/train.py plays games through EnvServer.
 */
public class ActorCriticGAEPlayer extends Agent {

    private final Random rnd;
    private final ActorCriticGAEParams params;
    private PythonPolicy policy;
    private Features features;

    public ActorCriticGAEPlayer(long seed, ActorCriticGAEParams params) {
        super(seed);
        this.rnd = new Random(seed);
        this.params = params;
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        ArrayList<Action> actions = gs.getAllAvailableActions();
        try {
            if (policy == null) start(gs.getTribes().length);
            double[] probs = policy.policy(features.encode(gs, actions, playerID));
            return actions.get(params.GREEDY ? argmax(probs) : sample(probs));
        } catch (IOException e) {
            throw new RuntimeException("ACGAE: Python policy failed", e);
        }
    }

    private void start(int numTribes) throws IOException {
        if (params.MODEL_PATH == null)
            throw new IllegalStateException("ACGAE needs a checkpoint: -Dtribes.acgae.model=<run dir>/latest.pkl");
        policy = new PythonPolicy(params.MODEL_PATH);
        features = Features.forTribes(numTribes);
        if (features.obsDim != policy.obsDim || features.actDim != policy.actDim)
            throw new IllegalStateException("checkpoint was trained on a " + policy.boardSize + "x" + policy.boardSize
                    + " board; this game is " + features.boardSize + "x" + features.boardSize);
    }

    /** First maximum, like np.argmax. */
    static int argmax(double[] p) {
        int best = 0;
        for (int j = 1; j < p.length; j++) if (p[j] > p[best]) best = j;
        return best;
    }

    private int sample(double[] p) {
        double u = rnd.nextDouble(), c = 0.0;
        for (int j = 0; j < p.length; j++) {
            c += p[j];
            if (u < c) return j;
        }
        return p.length - 1;
    }

    @Override
    public Agent copy() {
        return null;
    }
}
