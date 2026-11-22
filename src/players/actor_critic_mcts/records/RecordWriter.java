package players.actor_critic_mcts.records;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.zip.GZIPOutputStream;

/**
 * Writes finished GameRecords to disk for the training script to read.
 * Format: JSON Lines, one TrainingSample per line (see TrainingSample.toJson()), one file per self-play run,
 * e.g. data/selfplay/iter_003/games_0000-0099.jsonl.gz.
 *
 * A file name ending in ".gz" is gzip-compressed. The state features are mostly zeros, so this shrinks the
 * files by roughly 20x. Python: gzip.open(path, "rt") then json.loads(line) per line.
 */
public class RecordWriter implements AutoCloseable {

    private final String outputDir;
    private BufferedWriter writer;
    private Path path;
    private Path partPath;
    private long samplesWritten = 0;

    public RecordWriter(String outputDir) {
        this.outputDir = outputDir;
    }

    /**
     * Create the output directory and open the output file. Samples go to "<fileName>.part" until commit(), so a
     * run that crashes or is killed (e.g. at the SLURM time limit) never leaves a truncated file that the training
     * script would pick up.
     */
    public void open(String fileName) {
        try {
            Path dir = Paths.get(outputDir);
            Files.createDirectories(dir);
            path = dir.resolve(fileName);
            partPath = dir.resolve(fileName + ".part");
            OutputStream out = Files.newOutputStream(partPath);
            if (fileName.endsWith(".gz")) out = new GZIPOutputStream(out, 1 << 16);
            writer = new BufferedWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8), 1 << 16);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not open " + outputDir + "/" + fileName, e);
        }
    }

    /** Append every sample of a finished game. Fails loudly if any valueTarget is still NaN. */
    public void write(GameRecord record) {
        if (writer == null) throw new IllegalStateException("RecordWriter.open() was not called");
        try {
            for (TrainingSample s : record.getSamples()) {
                if (Double.isNaN(s.valueTarget))
                    throw new IllegalStateException("Game " + record.getGameId() + ", player " + record.getPlayerID()
                            + ": valueTarget is NaN at tick " + s.tick + " (was assignOutcome called?)");
                writer.write(s.toJson());
                writer.newLine();
                samplesWritten++;
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Could not write to " + path, e);
        }
    }

    public void writeAll(List<GameRecord> records) {
        for (GameRecord r : records) write(r);
        try {
            writer.flush();
        } catch (IOException e) {
            throw new UncheckedIOException("Could not flush " + path, e);
        }
    }

    public long getSamplesWritten() { return samplesWritten; }

    public Path getPath() { return path; }

    /** Finish the file and move it to its final name. Call once, after the last write. */
    public void commit() {
        close();
        try {
            Files.move(partPath, path, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                    java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not rename " + partPath + " to " + path, e);
        }
    }

    /** Close the stream. Without commit() the data stays in the ".part" file and is ignored by training. */
    @Override
    public void close() {
        if (writer == null) return;
        try {
            writer.close();   // also finishes the gzip stream
        } catch (IOException e) {
            throw new UncheckedIOException("Could not close " + partPath, e);
        } finally {
            writer = null;
        }
    }
}
