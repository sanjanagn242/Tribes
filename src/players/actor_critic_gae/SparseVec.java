package players.actor_critic_gae;

/**
 * A sparse float vector: only the non-zero entries (idx[i], val[i]) are stored.
 * The Tribes observations are mostly zeros (one-hots, empty tiles: ~650 of 12,664 values are non-zero),
 * so sending only the non-zero entries to Python keeps the pipe traffic small.
 */
public final class SparseVec {

    public static final SparseVec EMPTY = new SparseVec(new int[0], new float[0]);

    public final int[] idx;
    public final float[] val;

    public SparseVec(int[] idx, float[] val) {
        this.idx = idx;
        this.val = val;
    }

    public int nnz() { return idx.length; }

    /** Keeps the non-zero entries of dense[from .. from+len), re-indexed to start at `offset`. */
    public static SparseVec fromDense(float[] dense, int from, int len, int offset) {
        int n = 0;
        for (int i = from; i < from + len; i++) if (dense[i] != 0f) n++;
        int[] idx = new int[n];
        float[] val = new float[n];
        int k = 0;
        for (int i = from; i < from + len; i++) {
            if (dense[i] != 0f) { idx[k] = offset + i - from; val[k] = dense[i]; k++; }
        }
        return new SparseVec(idx, val);
    }

    public static SparseVec fromDense(float[] dense) {
        return fromDense(dense, 0, dense.length, 0);
    }

    /** Writes nnz, then nnz int32 indices, then nnz float32 values (big-endian; read by python/tribes_io.py). */
    public void write(java.io.DataOutputStream out) throws java.io.IOException {
        out.writeInt(idx.length);
        for (int i : idx) out.writeInt(i);
        for (float v : val) out.writeFloat(v);
    }
}
