package players.actor_critic_mcts.network;

/**
 * Output of one forward pass of the actor-critic network for a single state.
 *   priors[i] : actor's probability for the i-th action of the action list that was passed in (should sum to 1).
 *   value     : critic's estimate of the outcome for the player whose perspective was requested.
 *               Decide your convention (e.g. [-1, 1] = loss..win, or [0, 1]) and keep it consistent
 *               with how you build value targets in records.GameRecord.
 */
public class PolicyValue {

    public final double[] priors;
    public final double value;

    public PolicyValue(double[] priors, double value) {
        this.priors = priors;
        this.value = value;
    }
}
