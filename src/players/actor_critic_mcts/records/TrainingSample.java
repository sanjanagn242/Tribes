package players.actor_critic_mcts.records;

/**
 * One training example = one act() call of ACMCTSPlayer:
 *   stateFeatures  : StateEncoder.encode(gs, playerID)                     -> network input
 *   actionFeatures : one entry per root action (ActionEncoder output)      -> which actions pi refers to
 *   searchPolicy   : PUCT.visitDistribution(root visits, T)                -> actor (policy) target
 *   valueTarget    : final game result for playerID, filled in at the end  -> critic (value) target
 */
public class TrainingSample {

    public int gameId;
    public int playerID;
    public int tick;
    public int boardSize;              // needed to split stateFeatures back into nodes / edges / globals
    public float[] stateFeatures;
    public float[][] actionFeatures;   // or int[] actionIndices if you use the fixed-index scheme
    public double[] searchPolicy;
    public double rootValue;           // optional: MCTS root Q, useful for diagnostics / mixed value targets
    public double valueTarget = Double.NaN;   // NaN until GameRecord.assignOutcome() runs

    /**
     * Serialise this sample as one JSON object (one line in a .jsonl file). Keys:
     * gameId, playerID, tick, boardSize, rootValue, valueTarget, searchPolicy [A],
     * stateFeatures [S], actionFeatures [A][110].
     * Built by hand rather than with org.json: a sample has ~20k floats, mostly zeros, written as "0".
     */
    public String toJson() {
        int approx = 64 + (stateFeatures == null ? 0 : stateFeatures.length * 2)
                + (actionFeatures == null ? 0 : actionFeatures.length * 120 * 2);
        StringBuilder sb = new StringBuilder(approx);
        sb.append("{\"gameId\":").append(gameId)
          .append(",\"playerID\":").append(playerID)
          .append(",\"tick\":").append(tick)
          .append(",\"boardSize\":").append(boardSize)
          .append(",\"rootValue\":");
        appendDouble(sb, rootValue);
        sb.append(",\"valueTarget\":");
        appendDouble(sb, valueTarget);

        sb.append(",\"searchPolicy\":[");
        if (searchPolicy != null)
            for (int i = 0; i < searchPolicy.length; i++) {
                if (i > 0) sb.append(',');
                appendDouble(sb, searchPolicy[i]);
            }
        sb.append("],\"stateFeatures\":");
        appendFloats(sb, stateFeatures);

        sb.append(",\"actionFeatures\":[");
        if (actionFeatures != null)
            for (int i = 0; i < actionFeatures.length; i++) {
                if (i > 0) sb.append(',');
                appendFloats(sb, actionFeatures[i]);
            }
        sb.append("]}");
        return sb.toString();
    }

    private static void appendFloats(StringBuilder sb, float[] v) {
        sb.append('[');
        if (v != null)
            for (int i = 0; i < v.length; i++) {
                if (i > 0) sb.append(',');
                float x = v[i];
                if (x == (int) x) sb.append((int) x);          // 0, 1, node ids: no ".0"
                else sb.append(x);                              // Float.toString: shortest exact form
            }
        sb.append(']');
    }

    /** NaN / infinity aren't valid JSON, so they're written as null. */
    private static void appendDouble(StringBuilder sb, double x) {
        if (Double.isNaN(x) || Double.isInfinite(x)) sb.append("null");
        else if (x == (long) x) sb.append((long) x);
        else sb.append(x);
    }
}
