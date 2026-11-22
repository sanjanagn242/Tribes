package players.actor_critic.records;

import java.util.ArrayList;

/**
 * The ordered transitions of ONE player in ONE game (an episode from that player's point of view).
 * Returns and advantages are computed by the Python trainer from (reward, value, done), not here.
 */
public class Trajectory {

    private final int gameId;
    private final int playerID;
    private final ArrayList<Transition> transitions = new ArrayList<>();

    public Trajectory(int gameId, int playerID) {
        this.gameId = gameId;
        this.playerID = playerID;
    }

    /** Append the transition for the decision just made (reward still unknown). */
    public void add(Transition t) {
        // TODO
    }

    /** Fill in the reward of the most recent transition, once the next decision state is known. */
    public void setLastReward(double reward) {
        // TODO: no-op if the trajectory is empty (first decision of the game).
    }

    /** End of game: set the last transition's reward and mark it done. */
    public void finish(double terminalReward) {
        // TODO
    }

    /** Sanity check before writing: every transition has a reward, only the last one is done. */
    public boolean isComplete() {
        // TODO
        throw new UnsupportedOperationException("TODO: Trajectory.isComplete");
    }

    public ArrayList<Transition> getTransitions() { return transitions; }
    public int getGameId() { return gameId; }
    public int getPlayerID() { return playerID; }
}
