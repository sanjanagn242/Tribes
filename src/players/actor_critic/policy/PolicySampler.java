package players.actor_critic.policy;

import java.util.Random;

/**
 * Stateless helpers to turn the actor's output into an action choice.
 * Training (collecting data): SAMPLE from pi, so the agent explores and the data matches the policy.
 * Evaluation (playing for real): usually take the ARGMAX.
 */
public final class PolicySampler {

    private PolicySampler() {}

    /** Numerically stable softmax over the logits of the legal actions (subtract the max before exp). */
    public static double[] maskedSoftmax(double[] legalLogits) {
        // TODO
        throw new UnsupportedOperationException("TODO: PolicySampler.maskedSoftmax");
    }

    /** Sharpen (T < 1) or flatten (T > 1) a distribution: p_i^(1/T) / sum_j p_j^(1/T). */
    public static double[] applyTemperature(double[] probs, double temperature) {
        // TODO: careful, recorded behaviour probabilities must be the ones you actually sampled from.
        throw new UnsupportedOperationException("TODO: PolicySampler.applyTemperature");
    }

    /** Draw an index i with probability probs[i]. */
    public static int sample(double[] probs, Random rnd) {
        // TODO
        throw new UnsupportedOperationException("TODO: PolicySampler.sample");
    }

    /** Index of the largest probability (break ties randomly). */
    public static int argmax(double[] probs, Random rnd) {
        // TODO
        throw new UnsupportedOperationException("TODO: PolicySampler.argmax");
    }

    /** Entropy -sum p log p. Useful to log: if it collapses to 0 early, the policy stopped exploring. */
    public static double entropy(double[] probs) {
        // TODO
        throw new UnsupportedOperationException("TODO: PolicySampler.entropy");
    }
}
