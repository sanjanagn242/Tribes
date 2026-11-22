package players.actor_critic_mcts;

import players.heuristics.AlgParams;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Parameters for MCTS guided by an actor-critic network.
 * The first block is copied unchanged from players.mcts.MCTSParams.
 * The second block is new: with every flag at its default (false), this agent behaves exactly like MCTS.
 */
@SuppressWarnings("WeakerAccess")
public class ACMCTSParams extends AlgParams {

    // ---------------- Plain MCTS parameters (copied) ----------------
    public double K = Math.sqrt(2);
    public int ROLLOUT_LENGTH = 10;//10;
    public boolean ROLOUTS_ENABLED = true;

    // ---------------- Actor-critic parameters (new) ----------------

    /** Use the actor (policy head) as priors P(s,a), and PUCT instead of UCT in selection. */
    public boolean USE_PRIORS = false;

    /** Use the critic (value head) to evaluate leaves instead of (or mixed with) the heuristic/rollout. */
    public boolean USE_VALUE_NET = false;

    /** Mixing weight for leaf evaluation: lambda * critic + (1 - lambda) * heuristic. 1.0 = critic only. */
    public double VALUE_MIX_LAMBDA = 1.0;

    /** Exploration constant in PUCT:  Q + C_PUCT * P(s,a) * sqrt(N(s)) / (1 + N(s,a)). */
    public double C_PUCT = 1.5;

    /** Dirichlet noise added to root priors during self-play (AlphaZero style). */
    public boolean ROOT_NOISE = false;
    public double DIRICHLET_ALPHA = 0.3;
    public double DIRICHLET_EPS = 0.25;

    /** Self-play action selection: sample from visit counts ^ (1/T) for the first TEMP_MOVES moves, then argmax. */
    public double TEMPERATURE = 1.0;
    public int TEMP_MOVES = 30;

    /** Record (state, search policy) samples on every act() call, for training. */
    public boolean RECORD_SAMPLES = false;

    public void setParameterValue(String param, Object value) {
        switch(param) {
            case "K": K = (double) value; break;
            case "ROLLOUT_LENGTH": ROLLOUT_LENGTH = (int) value; break;
            case "heuristic_method": heuristic_method = (int) value; break;
            case "C_PUCT": C_PUCT = (double) value; break;
        }
    }

    public Object getParameterValue(String param) {
        switch(param) {
            case "K": return K;
            case "ROLLOUT_LENGTH": return ROLLOUT_LENGTH;
            case "heuristic_method": return heuristic_method;
            case "C_PUCT": return C_PUCT;
        }
        return null;
    }


    public ArrayList<String> getParameters() {
        ArrayList<String> paramList = new ArrayList<>();
        paramList.add("K");
        paramList.add("rollout_depth");
        paramList.add("heuristic_method");
        paramList.add("C_PUCT");
        return paramList;
    }

    public Map<String, Object[]> getParameterValues() {
        HashMap<String, Object[]> parameterValues = new HashMap<>();
        parameterValues.put("K", new Double[]{1.0, Math.sqrt(2), 2.0});
        parameterValues.put("rollout_depth", new Integer[]{5, 8, 10, 12, 15});
        parameterValues.put("heuristic_method", new Integer[]{ENTROPY_HEURISTIC, SIMPLE_HEURISTIC, DIFF_HEURISTIC});
        parameterValues.put("C_PUCT", new Double[]{0.5, 1.0, 1.5, 2.5});
        return parameterValues;
    }

}
