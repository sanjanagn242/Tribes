package players.actor_critic.encoding;

import core.actions.Action;
import core.game.GameState;

/**
 * Tribes has a variable, state-dependent action list, so the actor can't just output "action #17".
 * Pick one scheme and implement the matching method(s):
 *
 *  (A) Fixed index space: map every action to an index in [0, actionSpaceSize()) built from
 *      (Types.ACTION type, source tile / actor, target tile, extra id like unit type or tech).
 *      The policy head outputs actionSpaceSize() logits; mask the illegal ones.
 *
 *  (B) Action features: describe each legal action with a small feature vector; the network scores each
 *      (state embedding, action features) pair -> one logit per legal action. Simpler for Tribes.
 */
public class ActionEncoder {

    // ---- Scheme (A) ----

    /** Size of the fixed policy head. */
    public int actionSpaceSize(int boardSize) {
        // TODO (scheme A)
        throw new UnsupportedOperationException("TODO: ActionEncoder.actionSpaceSize");
    }

    /** Index of `a` in the fixed policy head. */
    public int toIndex(Action a, GameState gs) {
        // TODO (scheme A): use a.getActionType() and the action's specific fields (look in core/actions/*).
        throw new UnsupportedOperationException("TODO: ActionEncoder.toIndex");
    }

    // ---- Scheme (B) ----

    /** Length of the vector returned by toFeatures(). */
    public int actionFeatureSize() {
        // TODO (scheme B)
        throw new UnsupportedOperationException("TODO: ActionEncoder.actionFeatureSize");
    }

    /** Feature vector describing `a` (one-hot action type, normalised source/target coords, unit type...). */
    public float[] toFeatures(Action a, GameState gs) {
        // TODO (scheme B)
        throw new UnsupportedOperationException("TODO: ActionEncoder.toFeatures");
    }
}
