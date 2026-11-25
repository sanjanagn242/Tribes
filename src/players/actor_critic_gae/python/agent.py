"""GAE actor-critic: tic-tac-toe's gae-actor-critic-with-random/agent.py, adapted to Tribes.

The only changes from tic-tac-toe come from the action space:
  - the actor is called as actor.apply(theta, obs, acts) and returns one logit per action slot
    (networks.ActionScorer), instead of actor.apply(theta, obs) -> 9 logits;
  - legal_mask marks which of the (zero-padded) action slots hold a real legal action;
  - act() takes the state's action features instead of a 9-cell mask.
Losses, GAE, gamma^k weighting, entropy bonus and the update are unchanged.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

MIN_BATCH = 64              # batches are zero-padded to a power of two >= this
MIN_ACTIONS = 16            # action lists are zero-padded to a power of two >= this


def padded_size(n, minimum=MIN_BATCH):
    """Smallest power of two >= max(n, minimum). jax.jit compiles once per input shape, so padding
    the variable-length batches (and action lists) to a few fixed sizes avoids recompiling."""
    size = minimum
    while size < n:
        size *= 2
    return size


def pad_batch(batch):
    """Zero-pads every array in `batch` along axis 0 to padded_size(n) and adds `mask`
    (1 = real transition, 0 = padding). Padded rows get done = 1 so nothing (e.g. a GAE sum)
    carries across them; the losses weight every row by `mask`."""
    n = len(batch["action"])
    size = padded_size(n)
    out = {k: np.concatenate([v, np.zeros((size - n,) + v.shape[1:], v.dtype)])
           for k, v in batch.items()}
    out["done"][n:] = 1.0
    out["mask"] = np.concatenate([np.ones(n, np.float32), np.zeros(size - n, np.float32)])
    return out


def pad_actions(acts, size=None):
    """[A, D] action features -> ([size, D] zero-padded, [size] legal mask)."""
    n = len(acts)
    size = size or padded_size(n, MIN_ACTIONS)
    out = np.zeros((size, acts.shape[1]), np.float32)
    out[:n] = acts
    mask = np.zeros(size, np.float32)
    mask[:n] = 1.0
    return out, mask


def masked_mean(x, mask):
    """Mean of x over the real (mask = 1) rows only."""
    return jnp.sum(mask * x) / jnp.sum(mask)


def gae(delta, done, gamma, lam):
    """Generalised advantage estimates for a batch of whole games laid end to end.

        A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
            = sum_l (gamma * lam)^l delta_{t+l}      (summed up to the end of t's game)

    `delta` and `done` are in play order, games concatenated (as rollout.to_batch builds them);
    done_t = 1 on each game's last move, which stops the sum from running into the next game.
    lam = 0 gives the one-step TD error delta_t; lam = 1 gives the Monte Carlo advantage G_t - U(s_t).
    """
    def step(a_next, xs):
        d, dn = xs
        a = d + gamma * lam * (1.0 - dn) * a_next
        return a, a

    _, adv = jax.lax.scan(step, jnp.zeros((), delta.dtype), (delta, done), reverse=True)
    return adv


class ActorCriticAgent:
    def __init__(self, actor, critic, obs_size, act_size, gamma=0.99, alpha=1e-3, alpha_phi=1e-3,
                 beta=0.0, lam=0.95, seed=0):
        self.actor = actor
        self.critic = critic
        self.obs_size = obs_size
        self.act_size = act_size
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_phi = alpha_phi
        self.beta = beta            # entropy bonus weight; 0 = no entropy term
        self.lam = lam              # GAE lambda; 0 = one-step TD, 1 = Monte Carlo

        # seed s -> keys (2s, 2s + 1); seed 0 gives the old fixed keys (0, 1)
        dummy_obs = jnp.zeros((1, obs_size), jnp.float32)
        dummy_acts = jnp.zeros((1, 1, act_size), jnp.float32)
        self.params = {
            "theta": actor.init(jax.random.PRNGKey(2 * seed), dummy_obs, dummy_acts),
            "phi": critic.init(jax.random.PRNGKey(2 * seed + 1), dummy_obs),
        }

        self.actor_opt = optax.adam(alpha)
        self.critic_opt = optax.adam(alpha_phi)
        self.a_state = self.actor_opt.init(self.params["theta"])
        self.c_state = self.critic_opt.init(self.params["phi"])

        self._actor_forward = jax.jit(actor.apply)
        self._critic_forward = jax.jit(critic.apply)
        self._update_jit = jax.jit(self._update_impl)

    def policy(self, obs, acts):
        """pi(.|s) over the legal actions; acts is [A, D], one row per legal action."""
        padded, legal = pad_actions(acts)
        logits = self._actor_forward(self.params["theta"], jnp.asarray(obs, jnp.float32)[None],
                                     jnp.asarray(padded)[None])[0]
        masked = jnp.where(jnp.asarray(legal) > 0, logits, -1e9)
        return np.asarray(jax.nn.softmax(masked), dtype=np.float64)[:len(acts)]

    def act(self, obs, acts, rng, greedy=False):
        probs = self.policy(obs, acts)
        if greedy:
            return int(np.argmax(probs))
        return int(rng.choice(len(probs), p=probs / probs.sum()))

    def critic_loss(self, phi, batch):
        v = self.critic.apply(phi, batch["obs"]).squeeze(-1)
        return 0.5 * masked_mean((v - batch["rtg"]) ** 2, batch["mask"])

    def actor_loss(self, theta, phi, batch):
        logits = self.actor.apply(theta, batch["obs"], batch["acts"])
        logp = jax.nn.log_softmax(jnp.where(batch["legal_mask"] > 0, logits, -1e9))
        logp_a = jnp.take_along_axis(logp, batch["action"][:, None], 1)[:, 0]

        v = self.critic.apply(phi, batch["obs"]).squeeze(-1)
        v_next = self.critic.apply(phi, batch["next_obs"]).squeeze(-1)
        delta = batch["reward"] + self.gamma * (1.0 - batch["done"]) * v_next - v
        adv = jax.lax.stop_gradient(gae(delta, batch["done"], self.gamma, self.lam))

        weight = self.gamma ** batch["k"]
        pg = masked_mean(weight * adv * logp_a, batch["mask"])

        # Mean policy entropy over legal moves, H(pi(.|s)) = -sum_a pi log pi.
        # Illegal (padding) slots have pi = 0 exactly, so they contribute 0 (masked to be safe).
        p = jnp.exp(logp)
        entropy = -masked_mean(jnp.sum(jnp.where(batch["legal_mask"] > 0, p * logp, 0.0), axis=-1),
                               batch["mask"])

        # Gradient ascent on  U(theta) + beta * H(theta)  ==  descent on its negative.
        return -(pg + self.beta * entropy)

    def _update_impl(self, theta, phi, a_state, c_state, batch):
        a_loss, a_g = jax.value_and_grad(self.actor_loss)(theta, phi, batch)
        c_loss, c_g = jax.value_and_grad(self.critic_loss)(phi, batch)

        a_upd, a_state = self.actor_opt.update(a_g, a_state, theta)
        c_upd, c_state = self.critic_opt.update(c_g, c_state, phi)
        return (optax.apply_updates(theta, a_upd),
                optax.apply_updates(phi, c_upd),
                a_state, c_state, a_loss, c_loss)

    def update(self, batch):
        batch = {k: jnp.asarray(v) for k, v in pad_batch({k: np.asarray(v) for k, v in batch.items()}).items()}
        for key in ("obs", "acts", "next_obs", "reward", "done", "rtg", "k", "legal_mask", "mask"):
            batch[key] = batch[key].astype(jnp.float32)
        batch["action"] = batch["action"].astype(jnp.int32)

        (self.params["theta"], self.params["phi"], self.a_state, self.c_state,
         a_loss, c_loss) = self._update_jit(self.params["theta"], self.params["phi"],
                                            self.a_state, self.c_state, batch)
        return {"actor_loss": float(a_loss), "critic_loss": float(c_loss)}
