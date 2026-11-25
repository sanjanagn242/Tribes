from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """The critic: obs -> scalar (same as tic-tac-toe's networks.MLP)."""
    out_dim: int
    hidden: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.relu(nn.Dense(h)(x))
        return nn.Dense(self.out_dim)(x)


class ActionScorer(nn.Module):
    """The actor: one logit per (state, legal action) pair.

    Tic-tac-toe has 9 fixed actions, so its actor maps obs -> 9 logits. Tribes has a different list of
    legal actions in every state, so the actor scores each one instead:

        logit(s, a_j) = MLP([obs(s) ; act(s, a_j)])          obs [B, O], acts [B, A, D] -> logits [B, A]

    The first layer is one Dense on the concatenation [obs ; act], written as two matrix products
    (kernel[:O] for obs, kernel[O:] for act) so obs @ kernel[:O] is computed once per state rather than
    once per action. Same parameters, same init and same result as nn.Dense(hidden[0]) on the concat.
    """
    hidden: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, obs, acts):
        # Shapes:  obs  [B, O]     B states, O = 12,664 state features each
        #          acts [B, A, D]  for each state, A action slots (legal actions + zero padding),
        #                          D = 316 features per action
        O, D = obs.shape[-1], acts.shape[-1]

        # ---- Layer 1: nn.Dense(hidden[0]) applied to the concatenation [obs ; act] ----
        # Conceptually, for every (state, action) pair:
        #     x = relu([obs ; act] @ kernel + bias)          kernel: [O + D, h0]
        # Splitting kernel into its top O rows (state part) and bottom D rows (action part):
        #     [obs ; act] @ kernel = obs @ kernel[:O] + act @ kernel[O:]
        # so we never build the big concatenated array. obs @ kernel[:O] is computed once per
        # state ([B, h0]) and broadcast over the A actions ([..., None, :] -> [B, 1, h0]);
        # act @ kernel[O:] is computed per action ([B, A, h0]). The sum is the same as the Dense
        # on the concatenation, with the same parameters and the same lecun_normal init.
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (O + D, self.hidden[0]))
        bias = self.param("bias", nn.initializers.zeros_init(), (self.hidden[0],))
        x = (obs @ kernel[:O])[..., None, :] + acts @ kernel[O:] + bias      # [B, A, h0]
        x = nn.relu(x)

        # ---- Remaining hidden layers: exactly the loop in MLP above ----
        # nn.Dense acts on the last axis only, so each (state, action) pair goes through
        # the same layers independently: x stays [B, A, h] with h = 64 by default.
        for h in self.hidden[1:]:
            x = nn.relu(nn.Dense(h)(x))

        # ---- Output: one number per (state, action) pair = that action's logit ----
        # Dense(1) gives [B, A, 1]; [..., 0] drops the last axis -> [B, A].
        # In tic-tac-toe the actor's last layer was Dense(9) on one hidden vector (9 logits);
        # here it is Dense(1) on each action's own hidden vector (A logits). agent.py then masks
        # the padding slots with legal_mask and takes the softmax over each state's A logits,
        # exactly as it masked illegal cells before.
        return nn.Dense(1)(x)[..., 0]                                         # [B, A]
