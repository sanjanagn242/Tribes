"""Collecting experience: games against a random opponent -> transitions -> batches.

Same as tic-tac-toe's rollout.py, with the Tribes environment (tribes_env.TribesEnv) in place of
game.py + env_step: the agent takes one of the two seats at random each game, and after each of its
decisions the environment plays on until the agent is to move again.
"""
import numpy as np

from agent import padded_size, MIN_ACTIONS


def reward_to_go(rewards, gamma):
    """Discounted return from each step to the end: G_t = r_t + gamma * G_{t+1}."""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


def play_vs_random(agent, env, rng, greedy=False, seat=-1, record=True):
    """One game of the agent against a random opponent.

    Each transition is one agent decision (one Tribes action), all from the agent's point of view:
        obs, acts       : s with the agent to move, and its legal actions' features
        action, reward  : the agent's action (index into acts) and its reward (+1/-1/0 at the end, else 0)
        next_obs        : s', at the agent's next decision
        done            : 1.0 if the game ended before the agent's next decision
        k               : index of the agent's decision in the game (0, 1, 2, ...)
        rtg             : discounted return from this decision to the end
    Returns (transitions, final reward); transitions is None when record=False (evaluation).
    """
    obs, acts, reward, done = env.reset(int(rng.integers(2 ** 62)), seat)
    transitions, k = [], 0
    while not done:
        action = agent.act(obs, acts, rng, greedy=greedy)
        next_obs, next_acts, reward, done = env.step(action)
        if record:
            # Nobody moves at a terminal s'; U(s') is multiplied by 0, so zeros are fine.
            transitions.append({
                "obs": obs, "acts": acts, "action": action, "reward": reward,
                "next_obs": np.zeros_like(obs) if done else next_obs, "done": float(done), "k": k,
            })
        obs, acts = next_obs, next_acts
        k += 1

    if not record:
        return None, reward
    # Returns need the whole game, so they are filled in after it ends.
    rtg = reward_to_go([t["reward"] for t in transitions], agent.gamma)
    for t, g in zip(transitions, rtg):
        t["rtg"] = g
    return transitions, reward


def to_batch(transitions):
    """Stacks a list of transitions into a dict of arrays for agent.update().
    Action lists are zero-padded to the same length A; legal_mask marks the real ones."""
    A = padded_size(max(len(t["acts"]) for t in transitions), MIN_ACTIONS)
    act_size = transitions[0]["acts"].shape[1]
    acts = np.zeros((len(transitions), A, act_size), np.float32)
    legal = np.zeros((len(transitions), A), np.float32)
    for i, t in enumerate(transitions):
        acts[i, :len(t["acts"])] = t["acts"]
        legal[i, :len(t["acts"])] = 1.0
    return {
        "obs": np.stack([t["obs"] for t in transitions]).astype(np.float32),
        "acts": acts,
        "legal_mask": legal,
        "action": np.array([t["action"] for t in transitions], dtype=np.int32),
        "reward": np.array([t["reward"] for t in transitions], dtype=np.float32),
        "next_obs": np.stack([t["next_obs"] for t in transitions]).astype(np.float32),
        "done": np.array([t["done"] for t in transitions], dtype=np.float32),
        "k": np.array([t["k"] for t in transitions], dtype=np.float32),
        "rtg": np.array([t["rtg"] for t in transitions], dtype=np.float32),
    }
