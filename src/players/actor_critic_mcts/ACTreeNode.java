package players.actor_critic_mcts;

import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.game.GameState;
import core.Types;
import players.actor_critic_mcts.network.ActorCriticNetwork;
import players.actor_critic_mcts.network.PolicyValue;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;
import static core.Types.ACTION.*;

/**
 * Copy of players.mcts.SingleTreeNode, plus actor-critic hooks:
 *   priors[]          : P(s,a) for each child, from the actor (null when priors are off).
 *   selection         : UCT (copied) or PUCT (search.PUCT.score) when params.USE_PRIORS.
 *   leaf evaluation   : heuristic/rollout (copied) or the critic when params.USE_VALUE_NET.
 */
class ACTreeNode
{
    private ACMCTSParams params;

    private ACTreeNode root;
    private ACTreeNode parent;
    private ACTreeNode[] children;
    private double totValue;
    private int nVisits;
    private Random m_rnd;
    private int m_depth;
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int fmCallsCount;
    private int playerID;

    private ArrayList<Action> actions;
    private GameState state;

    private GameState rootState;
    private StateHeuristic rootStateHeuristic;

    // ---- actor-critic additions ----
    private ActorCriticNetwork network;   // only set on the root; children use root.network
    private double prior;              // P(s,a) for the edge from parent; null until computed
    private double reward;                // this node's own evaluation: critic V(s), or terminal reward. Set on expansion.

    //From ACMCTSPlayer
    ACTreeNode(ACMCTSParams p, Random rnd, int num_actions, ArrayList<Action> actions, int playerID, double prior) {
        this(p, null, rnd, num_actions, actions, null, playerID, null, null, prior);
    }

    private ACTreeNode(ACMCTSParams p, ACTreeNode parent, Random rnd, int num_actions,
                           ArrayList<Action> actions, StateHeuristic sh, int playerID, ACTreeNode root, GameState state, double prior) {
        this.params = p;
        this.fmCallsCount = 0;
        this.parent = parent;
        this.m_rnd = rnd;
        this.actions = actions;
        this.root = root;
        children = new ACTreeNode[num_actions];
        totValue = 0.0;
        this.prior = prior;
        this.playerID = playerID;
        this.state = state;
        if(parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        }
        else {
            m_depth = 0;
        }

    }

    void setRootGameState(ACTreeNode root, GameState gs, ArrayList<Integer> allIDs)
    {
        this.state = gs;
        this.root = root;
        this.rootState = gs;
        this.rootStateHeuristic = params.getStateHeuristic(playerID, allIDs);
    }

    void setNetwork(ActorCriticNetwork network) {
        this.network = network;
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;
        boolean stop = false;

        // Root: expand once to get the actor's priors for rootActions, then (self-play only) mix in Dirichlet noise:
        //   P'(s,a) = (1 - eps) * P(s,a) + eps * Dir(alpha)_a
        if (this.notFullyExpanded()) {
            double rootValue = this.expand_and_evaluate();
            // backUp(this, rootValue);

            if (params.ROOT_NOISE && children.length > 0) {
                // Dir(alpha) sample = Gamma(alpha, 1) samples normalised to sum 1.
                // Gamma via Marsaglia-Tsang; for alpha < 1 sample Gamma(alpha + 1) and scale by U^(1/alpha).
                double alpha = params.DIRICHLET_ALPHA;
                double a = alpha < 1 ? alpha + 1 : alpha;
                double d = a - 1.0 / 3.0, c = 1.0 / Math.sqrt(9.0 * d);
                double[] g = new double[children.length];
                double sum = 0;
                for (int i = 0; i < g.length; i++) {
                    double z, v;
                    while (true) {
                        z = m_rnd.nextGaussian();
                        double t = 1 + c * z;
                        if (t <= 0) continue;
                        v = t * t * t;
                        if (Math.log(m_rnd.nextDouble()) < 0.5 * z * z + d - d * v + d * Math.log(v)) break;
                    }
                    g[i] = d * v;
                    if (alpha < 1) g[i] *= Math.pow(m_rnd.nextDouble(), 1.0 / alpha);
                    sum += g[i];
                }
                double eps = params.DIRICHLET_EPS;
                for (int i = 0; i < children.length; i++)
                    children[i].prior = (1 - eps) * children[i].prior + eps * g[i] / sum;
            }
        }

        while(!stop){
//            System.out.println("------- " + root.actions.size() + " -------");
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            ACTreeNode selected = treePolicy();
            double delta;
            if (selected.notFullyExpanded() || selected.state.isGameOver() || selected.nVisits == 0)
                delta = selected.expand_and_evaluate();
            else
                // already-expanded depth-limit node: reuse its stored evaluation instead of re-expanding
                delta = selected.reward;
            backUp(selected, delta);
            numIters++;

            //Stopping condition
            if(params.stop_type == params.STOP_TIME) {
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
                avgTimeTaken  = acumTimeTaken/numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            }else if(params.stop_type == params.STOP_ITERATIONS) {
                stop = numIters >= params.num_iterations;
            }else if(params.stop_type == params.STOP_FMCALLS)
            {
                stop = fmCallsCount > params.num_fmcalls;
            }
        }
    }

    private ACTreeNode treePolicy() {
        ACTreeNode cur = this;
        while (!cur.state.isGameOver() && cur.m_depth < params.ROLLOUT_LENGTH)
        {
            if (cur.notFullyExpanded()) {
                return cur;   // unexpanded leaf: mctsSearch expands + evaluates it
            } else {
                cur = cur.p_uct();
            }
        }
        return cur;   // terminal or depth-limit node
    }

    private int tryForceEnd(GameState state, EndTurn endTurn, int depth)
    {
        boolean willForceEnd = (depth > 0 && (depth % params.FORCE_TURN_END) == 0) && endTurn.isFeasible(state);
        if(!willForceEnd)
            return -1; //Not the time, or not available.

        ArrayList<Action> availableActions = state.getAllAvailableActions();
        int actionIdx = 0;
        while(actionIdx < availableActions.size())
        {
            Action act = availableActions.get(actionIdx);
            if(act.getActionType() == END_TURN)
            {
                //Here's the end turn, return it's index.
                return actionIdx;
            }else actionIdx++;
        }

        //This should not happen, but EndTurn is not available here.
        return -1;
    }

    private double expand_and_evaluate() {

        // Terminal: with the critic, use the true outcome on the critic's [-1, 1] scale;
        // otherwise use the heuristic, same as plain MCTS.
        if (this.state.isGameOver()) {
            if (params.USE_VALUE_NET) {
                Types.RESULT res = state.getTribeWinStatus(playerID);
                this.reward = res == Types.RESULT.WIN ? 1.0 : (res == Types.RESULT.LOSS ? -1.0 : 0.0);
            } else {
                this.reward = rootStateHeuristic.evaluateState(root.rootState, state);
            }
            return this.reward;
        }

        // Use this node's own action list so children[i] <-> actions.get(i)
        // (at the root this is rootActions, which may be a PRIORITIZE_ROOT subset).
        PolicyValue out = root.network.evaluate(state, actions, playerID);
        int n = actions.size();

        // for every action, create a child node
        for (int i = 0; i < n; i++) {
            double p = params.USE_PRIORS ? out.priors[i] : 1.0 / n;

            GameState nextState = state.copy();
            ArrayList<Action> nextActions = advance(nextState, actions.get(i), true);
            children[i] = new ACTreeNode(params, this, this.m_rnd, nextActions.size(),
                    nextActions, rootStateHeuristic, this.playerID, this.root, nextState, p);
        }

        // Leaf value: critic, or the heuristic (as plain MCTS without rollouts).
        this.reward = params.USE_VALUE_NET
                ? out.value
                : rootStateHeuristic.evaluateState(root.rootState, state);
        return this.reward;
    }



    private ArrayList<Action> advance(GameState gs, Action act, boolean computeActions)
    {
        gs.advance(act, computeActions);
        root.fmCallsCount++;
        return gs.getAllAvailableActions();
    }


    private ACTreeNode p_uct() {

        ACTreeNode selected;
        boolean IamMoving = (state.getActiveTribeID() == this.playerID);
        int bestAction = tryForceEnd(state, new EndTurn(state.getActiveTribeID()), this.m_depth);
        if(bestAction == -1)
        {
            //No end turn, use uct.
            double[] vals = new double[this.children.length];
            for(int i = 0; i < this.children.length; ++i)
            {
                ACTreeNode child = children[i];

                // Q in [0, 1] from the mover's point of view. Unvisited children get Q = 1 (optimistic),
                // so every child is tried once, as in UCT; avoids 0/0 = NaN.
                double q;
                if (child.nVisits == 0) {
                    q = 1.0;
                } else {
                    q = normalise(child.totValue / child.nVisits, bounds[0], bounds[1]);
                    if (!IamMoving) q = 1.0 - q;   // opponent maximises its own value, not minimises Q+U
                }

                double p_uctValue = q +
                        params.C_PUCT * child.prior * Math.sqrt(this.nVisits) / (1 + child.nVisits);

                vals[i] = noise(p_uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
            }

            int which = -1;
            double bestValue = -Double.MAX_VALUE;
            for(int i = 0; i < vals.length; ++i) {
                if (vals[i] > bestValue){
                    which = i;
                    bestValue = vals[i];
                }
            }
            

            if (which == -1)
            {
                //if(this.children.length == 0)
                System.out.println("Warning! couldn't find the best UCT value " + which + " : " + this.children.length + " " +
                //throw new RuntimeException("Warning! couldn't find the best UCT value " + which + " : " + this.children.length + " " +
                        + bounds[0] + " " + bounds[1]);
                System.out.print(this.m_depth + ", AmIMoving? " + IamMoving + ";");
                for(int i = 0; i < this.children.length; ++i)
                    System.out.printf(" %f2", vals[i]);
                System.out.println("; selected: " + which);

                which = m_rnd.nextInt(children.length);
            }

            selected = children[which];

//            System.out.print(this.m_depth + ", AmIMoving? " + IamMoving + ";");
//            for(int i = 0; i < this.children.length; ++i)
//                System.out.printf(" %f2", vals[i]);
//            System.out.println("; selected: " + which);

        }else
        {
            selected = children[bestAction];
        }

        //Roll the state. This is closed loop, we don't advance the state. We can't do open loop here because the
        // number of actions available on a state depend on the state itself, and random events triggered by multiple
        // runs over the same tree node would have different outcomes (i.e Examine ruins).
        //advance(state, actions.get(selected.childIdx), true);

        root.fmCallsCount++;

        return selected;
    }


    private void backUp(ACTreeNode node, double result)
    {
        ACTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.totValue += result;
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }

    int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null)
            {
                if(first == -1)
                    first = children[i].nVisits;
                else if(first != children[i].nVisits)
                {
                    allEqual = false;
                }

                double childValue = children[i].nVisits;
                childValue = noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }

        return selected;
    }

    /** N(root, a) for every root action, aligned with the root's action list. Unexpanded children count as 0. */
    int[] getRootVisitCounts() {
        int[] visits = new int[children.length];
        for (int i = 0; i < children.length; i++)
            visits[i] = children[i] == null ? 0 : children[i].nVisits;
        return visits;
    }

    /**
     * Mean backed-up value at the root, from playerID's point of view. It's on the critic's scale when
     * USE_VALUE_NET is on, and on the heuristic's scale otherwise. Falls back to the root's own evaluation if
     * no iteration has been backed up yet.
     */
    double getRootValue() {
        return nVisits == 0 ? reward : totValue / nVisits;
    }

    private int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null) {
                double childValue = children[i].totValue / (children[i].nVisits + params.epsilon);
                childValue = noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (ACTreeNode tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }

    private double normalise(double a_value, double a_min, double a_max)
    {
        if(a_min < a_max)
            return (a_value - a_min)/(a_max - a_min);
        else    // if bounds are invalid, then return same value
            return a_value;
    }

    private double noise(double input, double epsilon, double random)
    {
        return (input + epsilon) * (1.0 + epsilon * (random - 0.5));
    }

}
