package players.actor_critic.network;

import core.actions.Action;
import core.game.GameState;
import players.actor_critic.encoding.ActionEncoder;
import players.actor_critic.encoding.StateEncoder;

import java.util.ArrayList;

/**
 * The trained actor-critic. The model lives outside Java (a PyTorch model trained on the trajectories
 * produced by selfplay.SelfPlay). Pick ONE way to call it and implement it here:
 *   (a) export to ONNX and run it with onnxruntime-java,
 *   (b) run a small Python server and talk to it over a socket / HTTP,
 *   (c) re-implement the forward pass in Java from exported weights (fine for a small MLP).
 *
 * For the very first data collection, export a randomly initialised model from Python: on-policy
 * actor-critic needs data from the policy being trained, so don't collect it with a different agent.
 */
public class ExternalNetwork implements ActorCriticNetwork {

    private final StateEncoder stateEncoder;
    private final ActionEncoder actionEncoder;

    public ExternalNetwork(StateEncoder stateEncoder, ActionEncoder actionEncoder) {
        this.stateEncoder = stateEncoder;
        this.actionEncoder = actionEncoder;
    }

    @Override
    public void load(String modelPath) {
        // TODO: load the weights / start the session / open the connection.
    }

    @Override
    public PolicyValue evaluate(GameState gs, ArrayList<Action> actions, int playerID) {
        // TODO:
        //  1. x = stateEncoder.encode(gs, playerID)
        //  2. encode each legal action with actionEncoder (index into a fixed head, or a feature vector per action)
        //  3. run the model -> one logit per legal action + V(s)
        //  4. probs = PolicySampler.maskedSoftmax(logits) over the LEGAL actions only
        throw new UnsupportedOperationException("TODO: ExternalNetwork.evaluate");
    }

    @Override
    public void close() {
        // TODO: release the session / connection.
    }
}
