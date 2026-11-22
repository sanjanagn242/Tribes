package players.actor_critic.encoding;

import core.game.GameState;

/**
 * Turns a GameState into the fixed-size input the network expects.
 * Encode from the perspective of `playerID` (own vs. enemy), so one network can play any tribe.
 * Only use information that player can see (GameState.getVisibilityMap()), or the network learns to cheat.
 *
 * A common layout: C planes of size boardSize x boardSize (terrain, resources, own/enemy units by type,
 * own/enemy cities, visibility...) flattened, followed by a vector of global scalars (stars, techs, turn...).
 */
public class StateEncoder {

    /** Total length of the vector returned by encode(). Must stay constant across a training run. */
    public int featureSize(int boardSize) {
        // TODO: numPlanes * boardSize * boardSize + numScalars
        throw new UnsupportedOperationException("TODO: StateEncoder.featureSize");
    }

    /** Full encoding = concat(spatial planes, scalar features). */
    public float[] encode(GameState gs, int playerID) {
        // TODO: call encodeBoardPlanes and encodeTribeScalars and concatenate.
        throw new UnsupportedOperationException("TODO: StateEncoder.encode");
    }

    /** Per-tile features: terrain, resources, buildings, units (own/enemy, by type), cities, fog of war. */
    float[][][] encodeBoardPlanes(GameState gs, int playerID) {
        // TODO: one [boardSize][boardSize] plane per feature. Look at gs.getBoard() and gs.getVisibilityMap().
        throw new UnsupportedOperationException("TODO: StateEncoder.encodeBoardPlanes");
    }

    /** Global features: stars, score, researched technologies, number of cities/units, turn number... */
    float[] encodeTribeScalars(GameState gs, int playerID) {
        // TODO: use gs.getTribe(playerID), gs.getTribeTechTree(playerID), gs.getTick(); normalise to ~[0, 1].
        throw new UnsupportedOperationException("TODO: StateEncoder.encodeTribeScalars");
    }
}
