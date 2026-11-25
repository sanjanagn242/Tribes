package players.actor_critic_gae;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * The trained actor, running in Python (python/serve.py) as a subprocess; used by ActorCriticGAEPlayer
 * when it plays from Run (play.json / tournament.json). Training doesn't use this: there Python drives
 * the game through EnvServer.
 *
 * Protocol on the subprocess's stdin/stdout, big-endian (python/tribes_io.py):
 *   startup   -> int MAGIC, int obsDim, int actDim, int boardSize      (of the checkpoint)
 *   POLICY    <- int 1, Features.Encoded.write  -> int n, n x float64 pi(a|s)
 *   QUIT      <- int 0
 *
 * System properties: tribes.python (default $TRIBES_PYTHON or python3),
 *                    tribes.acgae.server (default src/players/actor_critic_gae/python/serve.py).
 */
public final class PythonPolicy implements AutoCloseable {

    private final Process process;
    private final DataOutputStream toServer;
    private final DataInputStream fromServer;
    public final int obsDim, actDim, boardSize;

    public PythonPolicy(String checkpoint) throws IOException {
        if (!new File(checkpoint).isFile()) throw new FileNotFoundException("checkpoint not found: " + checkpoint);
        String python = System.getProperty("tribes.python",
                System.getenv("TRIBES_PYTHON") != null ? System.getenv("TRIBES_PYTHON") : "python3");
        String server = System.getProperty("tribes.acgae.server", "src/players/actor_critic_gae/python/serve.py");
        if (!new File(server).isFile())
            throw new FileNotFoundException(server + " not found (run from the repo root or set -Dtribes.acgae.server=...)");

        List<String> cmd = new ArrayList<>();
        cmd.add(python); cmd.add(server); cmd.add(checkpoint);
        process = new ProcessBuilder(cmd).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        toServer = new DataOutputStream(new BufferedOutputStream(process.getOutputStream()));
        fromServer = new DataInputStream(new BufferedInputStream(process.getInputStream()));
        if (fromServer.readInt() != EnvServer.MAGIC) throw new IOException("unexpected reply from " + server);
        obsDim = fromServer.readInt();
        actDim = fromServer.readInt();
        boardSize = fromServer.readInt();
        Runtime.getRuntime().addShutdownHook(new Thread(this::close));
    }

    /** pi(.|s) over the legal actions, in the order they were encoded. */
    public synchronized double[] policy(Features.Encoded enc) throws IOException {
        toServer.writeInt(1);
        enc.write(toServer);
        toServer.flush();
        int n = fromServer.readInt();
        double[] probs = new double[n];
        for (int i = 0; i < n; i++) probs[i] = fromServer.readDouble();
        return probs;
    }

    @Override
    public synchronized void close() {
        try {
            toServer.writeInt(0);
            toServer.flush();
        } catch (IOException ignored) {
        }
        process.destroy();
    }
}
