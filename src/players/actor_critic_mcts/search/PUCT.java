package players.actor_critic_mcts.search;

import java.util.Random;

/**
 * Stateless maths used by ACTreeNode / ACMCTSPlayer when the actor-critic is switched on.
 * Kept apart from the tree so each piece can be unit-tested on its own.
 */
public final class PUCT {

    private PUCT() {}

    /**
     * AlphaZero selection score:  Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
     * @param q            normalised mean value of the child (from the searching player's point of view).
     * @param prior        P(s,a) from the actor.
     * @param parentVisits N(s).
     * @param childVisits  N(s,a).
     * @param cPuct        exploration constant.
     */
    public static double score(double q, double prior, int parentVisits, int childVisits, double cPuct) {
        // TODO
        throw new UnsupportedOperationException("TODO: PUCT.score");
    }

    /**
     * Root exploration noise for self-play: p' = (1 - eps) * p + eps * Dir(alpha).
     * Returns a new array; does not modify `priors`.
     */
    public static double[] addDirichletNoise(double[] priors, double alpha, double eps, Random rnd) {
        // TODO: use sampleDirichlet.
        throw new UnsupportedOperationException("TODO: PUCT.addDirichletNoise");
    }

    /** Sample from a symmetric Dirichlet(alpha) of size n (hint: normalised Gamma(alpha, 1) samples). */
    public static double[] sampleDirichlet(int n, double alpha, Random rnd) {
        // TODO: java.util.Random has no Gamma sampler; implement Marsaglia-Tsang or use alpha<1 boosting trick.
        throw new UnsupportedOperationException("TODO: PUCT.sampleDirichlet");
    }

    /**
     * Turn root visit counts into the training target pi(a | s) = N(s,a)^(1/T) / sum_b N(s,b)^(1/T).
     * Unexpanded children (null) count as 0 visits.
     */
    public static double[] visitDistribution(int[] visitCounts, double temperature) {
        int n = visitCounts.length;
        double[] pi = new double[n];
        if (n == 0) return pi;

        int best = 0;
        long total = 0;
        for (int i = 0; i < n; i++) {
            total += visitCounts[i];
            if (visitCounts[i] > visitCounts[best]) best = i;
        }
        if (total == 0) {                         // no visits at all: uniform
            java.util.Arrays.fill(pi, 1.0 / n);
            return pi;
        }
        if (temperature < 1e-3) {                 // T -> 0: one-hot on the most visited action
            pi[best] = 1.0;
            return pi;
        }

        // N^(1/T), scaled by the max count first so large counts / small T don't overflow.
        double inv = 1.0 / temperature, max = visitCounts[best], sum = 0;
        for (int i = 0; i < n; i++) {
            pi[i] = visitCounts[i] > 0 ? Math.pow(visitCounts[i] / max, inv) : 0.0;
            sum += pi[i];
        }
        for (int i = 0; i < n; i++) pi[i] /= sum;
        return pi;
    }

    /** Sample an index from a probability distribution. */
    public static int sample(double[] probs, Random rnd) {
        if (probs.length == 0) throw new IllegalArgumentException("Empty distribution");
        double total = 0;
        for (double p : probs) total += p;
        double r = rnd.nextDouble() * total, acc = 0;
        int lastPositive = -1;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] <= 0) continue;
            acc += probs[i];
            lastPositive = i;
            if (r < acc) return i;
        }
        // Rounding left r just above the running sum: return the last action with positive probability.
        return lastPositive >= 0 ? lastPositive : rnd.nextInt(probs.length);
    }
}
