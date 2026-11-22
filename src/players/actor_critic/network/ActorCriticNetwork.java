package players.actor_critic.network;

import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;

/**
 * The actor-critic model as seen from Java: given a state and its legal actions, return
 *   - the actor's probability for each legal action  pi(a | s)
 *   - the critic's estimate of the state's value     V(s)
 *
 * Java only runs the model (inference). Building and training the model happens in Python on the
 * trajectories written by selfplay.SelfPlay. ExternalNetwork is the bridge.
 */
public interface ActorCriticNetwork {

    /**
     * Run the actor and the critic on one state.
     * @param gs       state to evaluate.
     * @param actions  legal actions at gs. The returned probabilities must be in the same order.
     * @param playerID whose point of view the value is from (the acting player).
     * @return action probabilities aligned with `actions`, and V(s).
     */
    PolicyValue evaluate(GameState gs, ArrayList<Action> actions, int playerID);

    /** Load weights / open the connection to the model. Called once before the first evaluate(). */
    void load(String modelPath);

    /** Release resources (sockets, native sessions...). */
    void close();
}
