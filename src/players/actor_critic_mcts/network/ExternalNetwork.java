package players.actor_critic_mcts.network;

import core.actions.Action;
import core.game.GameState;
import players.actor_critic_mcts.encoding.ActionEncoder;
import players.actor_critic_mcts.encoding.StateEncoder;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * The trained actor-critic GNN (py/actor_critic/gnn_model.py), run by a Python subprocess
 * (py/actor_critic/serve_gnn.py) that this class starts in load() and talks to over stdin/stdout.
 *
 * Why a subprocess: it runs the exact PyTorch model used in training (no export step, no second copy of the
 * model), needs no extra Java libraries, and a forward pass takes ~3-9 ms on CPU. The protocol is documented in
 * serve_gnn.py; numbers are big-endian, which is what DataInput/DataOutputStream use.
 *
 * Perspective: every training sample was recorded on the acting player's own turn, so the network is only
 * reliable from the acting tribe's point of view. evaluate() therefore encodes the state and actions from
 * gs.getActiveTribeID() (the priors are for that tribe's actions anyway), and converts the value to
 * `playerID`'s point of view: unchanged when playerID is acting, negated otherwise. Negating is exact for
 * two-player win/loss values and an approximation with more players.
 *
 * Settings (Java system properties, e.g. java -Dtribes.gnn.threads=8 ...):
 *   tribes.python      Python executable          (default: $TRIBES_PYTHON, else "python3")
 *   tribes.gnn.server  path to serve_gnn.py       (default: py/actor_critic/serve_gnn.py, relative to the working dir)
 *   tribes.gnn.threads torch CPU threads           (default: 4)
 *   tribes.gnn.device  torch device: auto | cpu | cuda | cuda:N   (default: auto = GPU if the node has one)
 */
public class ExternalNetwork implements ActorCriticNetwork {

    private static final int MAGIC = 0x54474E4E;   // "TGNN"
    private static final int VERSION = 1;
    private static final int OP_QUIT = 0, OP_EVALUATE = 1;

    private final StateEncoder stateEncoder;
    private final ActionEncoder actionEncoder;

    private Process process;
    private DataOutputStream toServer;
    private DataInputStream fromServer;
    private Thread shutdownHook;
    private String modelPath;

    private long evaluations = 0;
    private long evalNanos = 0;

    public ExternalNetwork() {
        this(new StateEncoder(), new ActionEncoder());
    }

    public ExternalNetwork(StateEncoder stateEncoder, ActionEncoder actionEncoder) {
        this.stateEncoder = stateEncoder;
        this.actionEncoder = actionEncoder;
    }

    /** Start the Python server with the checkpoint at `modelPath` (a .pt file from train_gnn.py) and check it matches the encoders. */
    @Override
    public synchronized void load(String modelPath) {
        if (process != null) close();
        if (!new File(modelPath).isFile())
            throw new IllegalArgumentException("Model checkpoint not found: " + modelPath);
        String python = System.getProperty("tribes.python",
                System.getenv("TRIBES_PYTHON") != null ? System.getenv("TRIBES_PYTHON") : "python3");
        String server = System.getProperty("tribes.gnn.server", "py/actor_critic/serve_gnn.py");
        if (!new File(server).isFile())
            throw new IllegalArgumentException("Inference server not found: " + server
                    + " (run from the repository root or set -Dtribes.gnn.server=...)");

        List<String> cmd = Arrays.asList(python, server, "--checkpoint", modelPath,
                "--threads", System.getProperty("tribes.gnn.threads", "4"),
                "--device", System.getProperty("tribes.gnn.device", "auto"));
        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);   // server logs and Python errors show up in our stderr
            process = pb.start();
        } catch (IOException e) {
            throw new UncheckedIOException("Could not start " + String.join(" ", cmd), e);
        }
        toServer = new DataOutputStream(new BufferedOutputStream(process.getOutputStream(), 1 << 17));
        fromServer = new DataInputStream(new BufferedInputStream(process.getInputStream(), 1 << 16));
        shutdownHook = new Thread(this::destroyProcess);
        Runtime.getRuntime().addShutdownHook(shutdownHook);
        this.modelPath = modelPath;

        // Handshake: the server reports the feature layout the model was trained on.
        try {
            int magic = fromServer.readInt(), version = fromServer.readInt();
            if (magic != MAGIC || version != VERSION)
                throw new IllegalStateException("Unexpected handshake from " + server + " (magic " + magic + ", version " + version + ")");
            int[] got = new int[7];
            for (int i = 0; i < got.length; i++) got[i] = fromServer.readInt();
            int[] want = {stateEncoder.nodeFeatureSize(), stateEncoder.edgeFeatureSize(), stateEncoder.globalFeatureSize(),
                    actionEncoder.actionFeatureSize(), actionEncoder.srcNodeColumn(), actionEncoder.tgtNodeColumn(),
                    stateEncoder.maxPlayers()};
            if (!Arrays.equals(got, want))
                throw new IllegalStateException("Model " + modelPath + " was trained with feature layout " + Arrays.toString(got)
                        + " but the encoders produce " + Arrays.toString(want)
                        + " [node, edge, global, action, srcCol, tgtCol, maxPlayers]. Retrain on data from the current encoders.");
        } catch (IOException e) {
            abortStartup();
            throw new IllegalStateException("Inference server exited during start-up (see its error above)", e);
        } catch (RuntimeException e) {
            abortStartup();
            throw e;
        }
    }

    private void abortStartup() {
        destroyProcess();
        removeShutdownHook();
        process = null;
    }

    @Override
    public synchronized PolicyValue evaluate(GameState gs, ArrayList<Action> actions, int playerID) {
        if (process == null) throw new IllegalStateException("ExternalNetwork.load() was not called");
        long t0 = System.nanoTime();

        int active = gs.getActiveTribeID();
        int perspective = active >= 0 ? active : playerID;
        float[] state = stateEncoder.encode(gs, perspective);
        float[][] feats = actionEncoder.toFeatures(actions, gs, perspective);
        int n = feats.length, af = actionEncoder.actionFeatureSize();

        double value;
        double[] logits = new double[n];
        try {
            toServer.writeInt(OP_EVALUATE);
            toServer.writeInt(gs.getBoard().getSize());
            toServer.writeInt(state.length);
            toServer.writeInt(n);
            toServer.writeInt(af);
            for (float v : state) toServer.writeFloat(v);
            for (float[] row : feats) for (float v : row) toServer.writeFloat(v);
            toServer.flush();

            int status = fromServer.readInt();
            if (status != 0) {
                byte[] msg = new byte[fromServer.readInt()];
                fromServer.readFully(msg);
                throw new IllegalStateException("Inference server error: " + new String(msg, StandardCharsets.UTF_8));
            }
            value = fromServer.readFloat();
            for (int i = 0; i < n; i++) logits[i] = fromServer.readFloat();
        } catch (IOException e) {
            throw new IllegalStateException("Lost connection to the inference server for " + modelPath
                    + " (see its error above)", e);
        }

        // The network's value is for the acting tribe; the tree wants it for playerID.
        if (perspective != playerID) value = -value;

        evaluations++;
        evalNanos += System.nanoTime() - t0;
        return new PolicyValue(softmax(logits), value);
    }

    /** Numerically stable softmax over the legal-action logits. */
    static double[] softmax(double[] logits) {
        int n = logits.length;
        double[] p = new double[n];
        if (n == 0) return p;
        double max = Double.NEGATIVE_INFINITY;
        for (double l : logits) if (l > max) max = l;
        if (Double.isInfinite(max) || Double.isNaN(max)) {   // all -inf (or broken output): fall back to uniform
            Arrays.fill(p, 1.0 / n);
            return p;
        }
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp(logits[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < n; i++) p[i] /= sum;
        return p;
    }

    /** Number of evaluate() calls so far, and their mean wall time in milliseconds (encoding + server round trip). */
    public long getEvaluations() { return evaluations; }

    public double getMeanEvalMillis() { return evaluations == 0 ? 0 : evalNanos / 1e6 / evaluations; }

    @Override
    public synchronized void close() {
        if (process == null) return;
        try {
            toServer.writeInt(OP_QUIT);
            toServer.flush();
            toServer.close();
            if (!process.waitFor(5, TimeUnit.SECONDS)) process.destroyForcibly();
        } catch (IOException e) {
            process.destroyForcibly();   // already gone
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
        } finally {
            removeShutdownHook();
            process = null;
        }
    }

    private void destroyProcess() {
        if (process != null) process.destroyForcibly();
    }

    private void removeShutdownHook() {
        if (shutdownHook == null) return;
        try {
            Runtime.getRuntime().removeShutdownHook(shutdownHook);
        } catch (IllegalStateException ignored) {
            // JVM already shutting down
        }
        shutdownHook = null;
    }
}
