package players.actor_critic.records;

/**
 * One decision of one player: (s, legal actions, a, pi_old(a|s), V(s), r, done).
 * The reward is NOT known when the action is chosen. It's filled in at this player's next decision
 * (Trajectory.setLastReward) or at the end of the game (Trajectory.finish).
 */
public class Transition {

    public int tick;
    public float[] stateFeatures;      // StateEncoder output for s
    public float[][] actionFeatures;   // one row per legal action (or int[] actionIndices for the fixed-index scheme)
    public int actionIndex;            // which legal action was taken (index into actionFeatures)
    public double actionProb;          // pi_old(a|s) from the policy that actually chose it (needed for PPO-style ratios)
    public double value;               // V(s) from the critic at collection time (for logging / GAE)
    public double reward = Double.NaN; // r after this action. NaN until filled in
    public boolean done;               // true for the last transition of the game

    /** Serialise as one JSON object (one line in a .jsonl file). */
    public String toJson() {
        // TODO: use org.json (lib/json.jar) or build the string by hand.
        throw new UnsupportedOperationException("TODO: Transition.toJson");
    }
}
