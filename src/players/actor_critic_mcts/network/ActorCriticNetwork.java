package players.actor_critic_mcts.network;

import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;

/**
 * Anything that can give MCTS a policy (priors over the legal actions) and a value for a state.
 * Implementations:
 *   - UniformNetwork : no learning, uniform priors + 0 value. Use it to test that PUCT plumbing works.
 *   - ExternalNetwork: the real trained model (e.g. trained in Python, called through ONNX / a socket / DJL).
 */
public interface ActorCriticNetwork {

    /**
     * Run the actor and the critic on one state.
     * @param gs       state to evaluate.
     * @param actions  legal actions at gs, in the SAME order the tree node uses for its children.
     * @param playerID perspective for the value (the searching player, not necessarily the active tribe).
     * @return priors aligned with `actions`, and a value.
     */
    PolicyValue evaluate(GameState gs, ArrayList<Action> actions, int playerID);

    /** Load weights / open the connection to the model. Called once before the first evaluate(). */
    void load(String modelPath);

    /** Release resources (sockets, native sessions...). */
    void close();
}
