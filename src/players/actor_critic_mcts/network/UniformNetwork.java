package players.actor_critic_mcts.network;

import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;

/**
 * Baseline "network" with no learning: every action gets the same prior and every state is worth 0.
 * PUCT with uniform priors should play about as well as UCT; check that first before plugging in a real model.
 */
public class UniformNetwork implements ActorCriticNetwork {

    @Override
    public PolicyValue evaluate(GameState gs, ArrayList<Action> actions, int playerID) {
        // Uniform priors 1/|actions|, and a neutral value (0 for a [-1, 1] loss..win convention; use 0.5 for [0, 1]).
        int n = actions.size();
        double[] priors = new double[n];
        if (n > 0)
            java.util.Arrays.fill(priors, 1.0 / n);
        double value = 0.0;
        return new PolicyValue(priors, value);
    }

    @Override
    public void load(String modelPath) {
        // No weights: the uniform baseline has nothing to load.
    }

    @Override
    public void close() {
        // No resources held: nothing to release.
    }
}
