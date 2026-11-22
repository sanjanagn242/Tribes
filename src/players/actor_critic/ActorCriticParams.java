package players.actor_critic;

/**
 * Settings for the pure actor-critic agent. There's no search, so none of the MCTS budget settings apply.
 * Learning-rate, entropy bonus and similar training settings belong to the Python trainer, not here.
 */
@SuppressWarnings("WeakerAccess")
public class ActorCriticParams {

    /** true: sample from pi (training / data collection). false: argmax (evaluation). */
    public boolean SAMPLE_ACTIONS = true;

    /** Temperature applied to pi before sampling. Keep 1.0 when collecting on-policy data. */
    public double TEMPERATURE = 1.0;

    /** Record a Transition on every act() call. */
    public boolean RECORD_TRANSITIONS = false;

    /** Discount factor. Also used by RewardFunction for potential-based shaping, and must match the trainer. */
    public double GAMMA = 0.99;

    /** Weight of the shaped (heuristic-change) step reward. 0 -> learn from the win/loss reward only. */
    public double SHAPING_WEIGHT = 0.1;

    /** Path of the trained model; read by whoever builds the ExternalNetwork. */
    public String MODEL_PATH = null;
}
