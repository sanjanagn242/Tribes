package players.actor_critic.network;

/**
 * Output of one forward pass of the actor-critic network for a single state.
 *   probs[i] : actor's probability pi(a_i | s) for the i-th legal action passed in (sums to 1 over legal actions).
 *   value    : critic's estimate V(s) of the expected (discounted) return from s for the acting player.
 *              Its scale must match the rewards produced by reward.RewardFunction.
 */
public class PolicyValue {

    public final double[] probs;
    public final double value;

    public PolicyValue(double[] probs, double value) {
        this.probs = probs;
        this.value = value;
    }
}
