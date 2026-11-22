package players.actor_critic_mcts.encoding;

/**
 * A game state encoded as a graph, ready for a graph neural network (e.g. PyTorch Geometric).
 *
 *   nodes : one per board tile, id = x * boardSize + y (same [x][y] indexing as core.game.Board).
 *   edges : directed, between each tile and its 8 neighbours (Tribes movement and vision use the
 *           3x3 neighbourhood). Both directions are listed, so (u,v) and (v,u) are separate edges.
 *           The topology depends only on boardSize; the edge FEATURES change with the state.
 *   global: tribe-level scalars (stars, techs, turn, opponents...) that don't belong to any tile.
 *
 * In PyG terms:  x = nodeFeatures [N, F],  edge_index = edgeIndex [2, E],  edge_attr = edgeFeatures [E, Fe],
 *                u = globalFeatures [G].
 */
public class GraphState {

    public final int boardSize;
    public final int numNodes;
    public final int numEdges;

    /** [numNodes][nodeFeatureSize]. Layout: StateEncoder.nodeFeatureNames(). */
    public final float[][] nodeFeatures;

    /** [2][numEdges]: row 0 = source node, row 1 = target node. Shared between states of the same size: don't modify. */
    public final int[][] edgeIndex;

    /** [numEdges][edgeFeatureSize]. Layout: StateEncoder.edgeFeatureNames(). */
    public final float[][] edgeFeatures;

    /** [globalFeatureSize]. Layout: StateEncoder.globalFeatureNames(). */
    public final float[] globalFeatures;

    public GraphState(int boardSize, float[][] nodeFeatures, int[][] edgeIndex,
                      float[][] edgeFeatures, float[] globalFeatures) {
        this.boardSize = boardSize;
        this.numNodes = nodeFeatures.length;
        this.numEdges = edgeIndex[0].length;
        this.nodeFeatures = nodeFeatures;
        this.edgeIndex = edgeIndex;
        this.edgeFeatures = edgeFeatures;
        this.globalFeatures = globalFeatures;
    }

    public static int nodeId(int x, int y, int boardSize) {
        return x * boardSize + y;
    }

    /**
     * Flat vector = concat(nodeFeatures row by row, edgeFeatures row by row, globalFeatures).
     * edgeIndex is NOT included: rebuild it from boardSize (StateEncoder.edgeIndex) on the Python side.
     */
    public float[] flatten() {
        int nf = numNodes == 0 ? 0 : nodeFeatures[0].length;
        int ef = numEdges == 0 ? 0 : edgeFeatures[0].length;
        float[] out = new float[numNodes * nf + numEdges * ef + globalFeatures.length];
        int k = 0;
        for (float[] row : nodeFeatures) { System.arraycopy(row, 0, out, k, nf); k += nf; }
        for (float[] row : edgeFeatures) { System.arraycopy(row, 0, out, k, ef); k += ef; }
        System.arraycopy(globalFeatures, 0, out, k, globalFeatures.length);
        return out;
    }
}
