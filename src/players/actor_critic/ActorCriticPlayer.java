package players.actor_critic;

import core.actions.Action;
import core.game.GameState;
import players.Agent;
import players.actor_critic.network.ActorCriticNetwork;
import players.actor_critic.network.PolicyValue;
import players.actor_critic.records.Trajectory;
import players.actor_critic.records.Transition;
import players.actor_critic.reward.RewardFunction;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

/**
 * Pure actor-critic agent: no search and no forward-model simulation. Each act() is one network call:
 *     pi(.|s), V(s) = network(s, legal actions)  ->  a ~ pi  (or argmax)
 *
 * Java only ACTS and RECORDS. Learning (returns, advantages, policy and value losses, gradient steps)
 * happens in Python on the trajectories written by selfplay.SelfPlay.
 *
 * Reward bookkeeping: the reward for the previous decision is only known at this player's NEXT decision
 * (after the opponents have moved) or at the end of the game, so it's filled in one step late.
 */
public class ActorCriticPlayer extends Agent {

    private final Random m_rnd;
    private final ActorCriticParams params;

    private ActorCriticNetwork network;
    private RewardFunction rewardFunction;
    private Trajectory trajectory;          // null -> not recording
    private GameState prevDecisionState;    // state at this player's previous decision, for the step reward

    public ActorCriticPlayer(long seed) {
        this(seed, new ActorCriticParams());
    }

    public ActorCriticPlayer(long seed, ActorCriticParams params) {
        super(seed);
        this.m_rnd = new Random(seed);
        this.params = params;
    }

    public void setNetwork(ActorCriticNetwork network) {
        this.network = network;
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        // TODO:
        //  1. ArrayList<Action> actions = gs.getAllAvailableActions();   (if only one, you may still record it or skip it; decide)
        //  2. if recording: close the previous transition -> trajectory.setLastReward(rewardFunction.stepReward(prevDecisionState, gs))
        //  3. PolicyValue pv = network.evaluate(gs, actions, playerID);
        //  4. int idx = chooseAction(pv.probs);
        //  5. if recording: trajectory.add(buildTransition(gs, actions, idx, pv)); prevDecisionState = gs.copy();
        //  6. return actions.get(idx);
        throw new UnsupportedOperationException("TODO: ActorCriticPlayer.act");
    }

    /** Sample from pi when params.SAMPLE_ACTIONS (training), argmax otherwise (evaluation). Use policy.PolicySampler. */
    private int chooseAction(double[] probs) {
        // TODO
        throw new UnsupportedOperationException("TODO: ActorCriticPlayer.chooseAction");
    }

    /** Fill a Transition: encoded state and legal actions, chosen index, its probability, V(s), tick. Reward stays NaN. */
    private Transition buildTransition(GameState gs, ArrayList<Action> actions, int actionIdx, PolicyValue pv) {
        // TODO: use encoding.StateEncoder / encoding.ActionEncoder.
        throw new UnsupportedOperationException("TODO: ActorCriticPlayer.buildTransition");
    }

    /**
     * Start recording a new game. Called by SelfPlay AFTER Game.init, because playerID is only
     * assigned there (Agent.setPlayerIDs).
     */
    public void startRecording(int gameId) {
        // TODO: trajectory = new Trajectory(gameId, playerID);
        //       rewardFunction = new RewardFunction(playerID, params.GAMMA, params.SHAPING_WEIGHT);
        //       prevDecisionState = null;
    }

    public Trajectory getTrajectory() {
        return trajectory;
    }

    /** End of game (called by Game.terminate): give the last transition its terminal reward and mark it done. */
    @Override
    public void result(GameState gs, double reward) {
        // TODO: if recording: trajectory.finish(rewardFunction.terminalReward(prevDecisionState, gs));
    }

    @Override
    public Agent copy() {
        return null;
    }
}
