package players.actor_critic_mcts;

import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.game.GameState;
import players.Agent;
import players.actor_critic_mcts.encoding.ActionEncoder;
import players.actor_critic_mcts.encoding.StateEncoder;
import players.actor_critic_mcts.network.ActorCriticNetwork;
import players.actor_critic_mcts.network.UniformNetwork;
import players.actor_critic_mcts.records.GameRecord;
import players.actor_critic_mcts.records.TrainingSample;
import players.actor_critic_mcts.search.PUCT;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

/**
 * Copy of players.mcts.MCTSPlayer with hooks for an actor-critic network:
 *   - the network is handed to the tree (priors in selection, critic at leaves),
 *   - each act() can be recorded as a TrainingSample for self-play,
 *   - result() assigns the final outcome to all recorded samples.
 * With all actor-critic flags in ACMCTSParams left false, this behaves like plain MCTS.
 */
public class ACMCTSPlayer extends Agent {

    private Random m_rnd;
    private ACMCTSParams params;

    private ActorCriticNetwork network;   // null -> no priors / no critic available
    private GameRecord gameRecord;        // null -> not recording
    private int movesPlayed = 0;          // for the temperature schedule

    // Encoders for recorded samples. Stateless after construction, so copies share them.
    // Their maxPlayers must match whatever ExternalNetwork uses at inference time.
    private StateEncoder stateEncoder = new StateEncoder();
    private ActionEncoder actionEncoder = new ActionEncoder();

    public ACMCTSPlayer(long seed)
    {
        super(seed);
        m_rnd = new Random(seed);
        this.params = new ACMCTSParams();
        this.network = new UniformNetwork();   // default until a trained model is set via setNetwork()
    }

    public ACMCTSPlayer(long seed, ACMCTSParams params) {
        this(seed);
        this.params = params;
    }

    public Action act(GameState gs, ElapsedCpuTimer ect) {
        //Gather all available actions:
        ArrayList<Action> allActions = gs.getAllAvailableActions();

        if(allActions.size() == 1)
            return allActions.get(0); //EndTurn, it's possible.

        ArrayList<Action> rootActions = params.PRIORITIZE_ROOT ? determineActionGroup(gs, m_rnd) : allActions;
        if(rootActions == null)
            return new EndTurn();

        ACTreeNode m_root = new ACTreeNode(params, m_rnd, rootActions.size(), rootActions, this.playerID, 1.0);
        m_root.setRootGameState(m_root, gs, allPlayerIDs);
        m_root.setNetwork(network);

        // if(params.USE_PRIORS && network != null)
        //     m_root.initRootPriors(params.ROOT_NOISE);

        m_root.mctsSearch(ect);

        if(params.RECORD_SAMPLES && gameRecord != null)
            recordSample(gs, rootActions, m_root);

        int chosen = params.TEMPERATURE > 0 && movesPlayed < params.TEMP_MOVES && params.RECORD_SAMPLES
                ? sampleActionByVisits(m_root)
                : m_root.mostVisitedAction();
        movesPlayed++;
        return rootActions.get(chosen);
    }

    // ---------------------------------------------------------------- actor-critic / self-play hooks

    public void setNetwork(ActorCriticNetwork network) {
        this.network = network;
    }

    /**
     * Start a fresh record for a new game. Called by SelfPlay AFTER Game.init, because playerID is only
     * assigned there (Agent.setPlayerIDs).
     */
    public void startRecording(int gameId) {
        gameRecord = new GameRecord(gameId, playerID);
        movesPlayed = 0;
    }

    public GameRecord getGameRecord() {
        return gameRecord;
    }

    /**
     * Build one TrainingSample from the finished search:
     * state features, action features for rootActions, visit distribution at the root, root value.
     */
    private void recordSample(GameState gs, ArrayList<Action> rootActions, ACTreeNode root) {
        // Note: with PRIORITIZE_ROOT=true the root only sees one action group, so pi only covers that group.
        // Probably set PRIORITIZE_ROOT=false in self-play.
        int[] visits = root.getRootVisitCounts();
        if (visits.length != rootActions.size())
            throw new IllegalStateException("Root has " + visits.length + " children but " + rootActions.size() + " actions");

        TrainingSample sample = new TrainingSample();
        sample.tick = gs.getTick();
        sample.boardSize = gs.getBoard().getSize();
        // Same playerID for both encoders, so tribe slots in the state and action features line up.
        sample.stateFeatures = stateEncoder.encode(gs, playerID);
        sample.actionFeatures = actionEncoder.toFeatures(rootActions, gs, playerID);   // row i <-> rootActions.get(i)
        sample.searchPolicy = PUCT.visitDistribution(visits, 1.0);                     // T = 1: raw visit proportions
        sample.rootValue = root.getRootValue();
        gameRecord.add(sample);
    }

    /** Self-play move choice: sample from visit counts ^ (1/T) instead of always taking the argmax. */
    private int sampleActionByVisits(ACTreeNode root) {
        double[] pi = PUCT.visitDistribution(root.getRootVisitCounts(), params.TEMPERATURE);
        return PUCT.sample(pi, m_rnd);
    }

    /** End of game (called by Game.terminate): give every recorded sample its value target. */
    @Override
    public void result(GameState gs, double reward) {
        if (gameRecord != null) gameRecord.assignOutcome(gs);
    }

    @Override
    public Agent copy() {
        ACMCTSPlayer c = new ACMCTSPlayer(seed, params);
        c.network = network;
        c.stateEncoder = stateEncoder;
        c.actionEncoder = actionEncoder;
        return c;
    }

}
