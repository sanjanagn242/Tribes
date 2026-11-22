package players.actor_critic_mcts.records;

import core.Types;
import core.actors.Tribe;
import core.game.GameState;

import java.util.ArrayList;

/**
 * All samples produced by ONE player during ONE game. The value target is only known when the game ends,
 * so samples are buffered here and completed in assignOutcome().
 */
public class GameRecord {

    private final int gameId;
    private final int playerID;
    private final ArrayList<TrainingSample> samples = new ArrayList<>();

    public GameRecord(int gameId, int playerID) {
        this.gameId = gameId;
        this.playerID = playerID;
    }

    /** Buffer one sample taken during search. */
    public void add(TrainingSample sample) {
        sample.gameId = gameId;
        sample.playerID = playerID;
        samples.add(sample);
    }

    /**
     * Called from ACMCTSPlayer.result() at game end: fill valueTarget of every buffered sample.
     * Use finalState.getTribeWinStatus(playerID) (Types.RESULT) and/or the final score.
     */
    public void assignOutcome(GameState finalState) {
        double z = outcomeToValue(finalState);
        for (TrainingSample s : samples)
            s.valueTarget = z;   // no discounting: every sample of the game gets the final outcome
    }

    /**
     * Map the final result to the critic's [-1, 1] range (same convention as ACTreeNode's terminal reward
     * and UniformNetwork's neutral 0):
     *   WIN -> 1, LOSS -> -1,
     *   INCOMPLETE -> score rank among the tribes, squeezed into [-0.5, 0.5] so it never looks like a real win/loss:
     *                 best score 0.5, worst -0.5, ties share the better rank.
     */
    double outcomeToValue(GameState finalState) {
        Types.RESULT res = finalState.getTribeWinStatus(playerID);
        if (res == Types.RESULT.WIN) return 1.0;
        if (res == Types.RESULT.LOSS) return -1.0;

        Tribe[] tribes = finalState.getTribes();
        if (tribes.length < 2) return 0.0;
        int myScore = tribes[playerID].getScore();
        int better = 0;
        for (Tribe t : tribes)
            if (t.getScore() > myScore) better++;
        double rankFrac = better / (double) (tribes.length - 1);   // 0 = best, 1 = worst
        return 0.5 * (1.0 - 2.0 * rankFrac);
    }

    public ArrayList<TrainingSample> getSamples() {
        return samples;
    }

    public int getGameId() { return gameId; }
    public int getPlayerID() { return playerID; }
}
