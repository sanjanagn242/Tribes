package players.actor_critic_gae;

import core.TribesConfig;
import core.actions.Action;
import core.game.GameState;
import players.actor_critic_mcts.encoding.ActionEncoder;
import players.actor_critic_mcts.encoding.GraphState;
import players.actor_critic_mcts.encoding.StateEncoder;

import java.util.List;

/**
 * Turns a Tribes state and its legal actions into the inputs of the two MLPs (python/networks.py), reusing the encoders of
 * players.actor_critic_mcts.encoding (see its README for every column).
 *
 *   obs(s)      = [ node features of every tile (N x 104, row by row) ; global features (80) ]
 *                 from playerID's point of view. This is the flat MLP input, like game.observation() in
 *                 tic-tac-toe. (The graph's edge features are left out: they are functions of the nodes.)
 *
 *   act(s, a)   = [ ActionEncoder features without the two raw node-id columns ;
 *                   node features of a's source tile ; node features of a's target tile ]
 *                 (zeros where a has no source / target tile).
 *
 * Tribes has a variable, state-dependent list of legal actions, so the actor scores each one instead:
 *     logit(s, a) = actor([obs(s) ; act(s, a)])
 * and pi(.|s) is the softmax over the legal actions, which is the same as masking all others to -1e9.
 *
 * Both are sent to Python (EnvServer / PythonPolicy) as sparse vectors: obs indices in [0, obsDim),
 * action indices in [0, actDim). Python turns them back into dense float32 arrays.
 */
public class Features {

    public final int boardSize;
    public final int numTiles;
    public final int nodeDim, globalDim, obsDim;
    public final int rawActDim, actDim;

    private final StateEncoder stateEncoder;
    private final ActionEncoder actionEncoder;
    private final int srcCol, tgtCol, hasSrcCol, hasTgtCol;

    /** One encoded decision: the observation and one feature row per legal action. */
    public static final class Encoded {
        public final SparseVec obs;
        public final SparseVec[] actions;

        Encoded(SparseVec obs, SparseVec[] actions) {
            this.obs = obs;
            this.actions = actions;
        }

        /** obs, then the number of legal actions, then each action row (read by python/tribes_io.read_decision). */
        public void write(java.io.DataOutputStream out) throws java.io.IOException {
            obs.write(out);
            out.writeInt(actions.length);
            for (SparseVec a : actions) a.write(out);
        }
    }

    /** Features for games with `numTribes` tribes on the default map size. */
    public static Features forTribes(int numTribes) {
        return new Features(TribesConfig.DEFAULT_MAP_SIZE[numTribes - 1]);   // indexed by #tribes - 1
    }

    public Features(int boardSize) {
        this.boardSize = boardSize;
        this.numTiles = boardSize * boardSize;
        this.stateEncoder = new StateEncoder();
        this.actionEncoder = new ActionEncoder(stateEncoder.maxPlayers());
        this.nodeDim = stateEncoder.nodeFeatureSize();
        this.globalDim = stateEncoder.globalFeatureSize();
        this.obsDim = numTiles * nodeDim + globalDim;
        this.rawActDim = actionEncoder.actionFeatureSize();
        this.srcCol = actionEncoder.srcNodeColumn();
        this.tgtCol = actionEncoder.tgtNodeColumn();
        this.hasSrcCol = srcCol - 1;
        this.hasTgtCol = tgtCol - 1;
        this.actDim = (rawActDim - 2) + 2 * nodeDim;
    }

    public Encoded encode(GameState gs, List<Action> actions, int playerID) {
        int size = gs.getBoard().getSize();
        if (size != boardSize)
            throw new IllegalStateException("board is " + size + "x" + size + " but the network was built for "
                    + boardSize + "x" + boardSize + " (keep the number of tribes fixed)");
        GraphState g = stateEncoder.encodeGraph(gs, playerID);
        SparseVec obs = observation(g);

        float[][] raw = actionEncoder.toFeatures(actions, gs, playerID);
        SparseVec[] acts = new SparseVec[raw.length];
        for (int j = 0; j < raw.length; j++) acts[j] = actionRow(raw[j], g);
        return new Encoded(obs, acts);
    }

    private SparseVec observation(GraphState g) {
        float[] dense = new float[obsDim];
        for (int n = 0; n < numTiles; n++)
            System.arraycopy(g.nodeFeatures[n], 0, dense, n * nodeDim, nodeDim);
        System.arraycopy(g.globalFeatures, 0, dense, numTiles * nodeDim, globalDim);
        return SparseVec.fromDense(dense);
    }

    private SparseVec actionRow(float[] raw, GraphState g) {
        float[] dense = new float[actDim];
        int k = 0;
        for (int c = 0; c < rawActDim; c++) {
            if (c == srcCol || c == tgtCol) continue;      // raw node ids are pointers, not features
            dense[k++] = raw[c];
        }
        if (raw[hasSrcCol] > 0f) System.arraycopy(g.nodeFeatures[(int) raw[srcCol]], 0, dense, k, nodeDim);
        k += nodeDim;
        if (raw[hasTgtCol] > 0f) System.arraycopy(g.nodeFeatures[(int) raw[tgtCol]], 0, dense, k, nodeDim);
        return SparseVec.fromDense(dense);
    }
}
