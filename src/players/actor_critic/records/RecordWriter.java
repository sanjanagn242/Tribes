package players.actor_critic.records;

import java.util.List;

/**
 * Writes finished Trajectories to disk for the Python trainer.
 * Suggested format: JSON Lines, one Transition per line, tagged with gameId and playerID so the trainer
 * can rebuild each trajectory in order (it needs the order to compute returns / advantages).
 * One folder per policy version, e.g. data/ac/iter_003/, since on-policy data goes stale after an update.
 */
public class RecordWriter implements AutoCloseable {

    private final String outputDir;

    public RecordWriter(String outputDir) {
        this.outputDir = outputDir;
    }

    /** Create the output directory and open the output file. */
    public void open(String fileName) {
        // TODO: java.nio.file.Files.createDirectories + a BufferedWriter.
    }

    /** Append every transition of a finished trajectory. Fail loudly if !trajectory.isComplete(). */
    public void write(Trajectory trajectory) {
        // TODO
    }

    public void writeAll(List<Trajectory> trajectories) {
        // TODO: loop over write(); flush at the end.
    }

    @Override
    public void close() {
        // TODO: flush and close the writer.
    }
}
