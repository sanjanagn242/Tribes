package players.actor_critic.reward;

import core.game.GameState;

/**
 * Defines what the agent is trained to maximise. Pure actor-critic learns only from these rewards
 * (there is no search to fall back on), so this choice matters a lot.
 *
 *  - Terminal reward: win / loss at the end of the game. Correct, but very sparse: a game is hundreds of
 *    actions long, so learning from it alone is slow.
 *  - Shaped step reward: change in a heuristic between two consecutive decisions of the same player,
 *    e.g. TribesDiffHeuristic / TribesSimpleHeuristic in players.heuristics, or the change in score.
 *    Potential-based shaping, r = gamma * phi(s') - phi(s), does not change which policy is optimal.
 *
 * "Step" here means: from this player's action at state s to this player's NEXT decision at s'.
 * The opponents' moves in between are part of the environment.
 */
public class RewardFunction {

    private final int playerID;
    private final double gamma;
    private final double shapingWeight;   // 0 -> terminal reward only

    public RewardFunction(int playerID, double gamma, double shapingWeight) {
        this.playerID = playerID;
        this.gamma = gamma;
        this.shapingWeight = shapingWeight;
    }

    /** Reward for the transition prevState -> currState (both are states where this player had to act). */
    public double stepReward(GameState prevState, GameState currState) {
        // TODO: shapingWeight * (gamma * potential(currState) - potential(prevState))
        throw new UnsupportedOperationException("TODO: RewardFunction.stepReward");
    }

    /**
     * Reward for the last transition, ending at the final state. Use finalState.getTribeWinStatus(playerID):
     * WIN / LOSS, and decide what INCOMPLETE (turn limit reached) is worth, e.g. based on rank or score.
     */
    public double terminalReward(GameState lastDecisionState, GameState finalState) {
        // TODO: outcome reward (+ the final shaping term, if you use shaping).
        throw new UnsupportedOperationException("TODO: RewardFunction.terminalReward");
    }

    /** phi(s) for shaping: a heuristic evaluation of s for playerID, scaled to a small range. */
    double potential(GameState state) {
        // TODO: e.g. a StateHeuristic from players.heuristics, or the tribe's score, normalised.
        throw new UnsupportedOperationException("TODO: RewardFunction.potential");
    }
}
