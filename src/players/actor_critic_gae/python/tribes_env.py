"""Tribes against a random opponent, as an environment for the agent (the Java side is EnvServer.java).

    env = TribesEnv(classpath, cfg)
    obs, acts, reward, done = env.reset(seed)       # first state where the agent is to move
    obs, acts, reward, done = env.step(action)      # action = index into acts (the legal actions)

Like env_step in tic-tac-toe's rollout.py: step() plays the agent's action and then everything up to
the agent's next decision (the opponent's whole turn, if the action ended the agent's turn), so every
state returned has the agent to move. reward is 0 until done, then +1 win / -1 loss / 0 otherwise.

    obs   [obs_size]           the state from the agent's point of view (121 tiles x 104 + 80 globals)
    acts  [A, act_size]        one feature row per legal action (A changes from state to state)
"""
import os
import struct
import subprocess

from tribes_io import MAGIC, read_decision, read_double, read_int, write_ints

OP_QUIT, OP_RESET, OP_STEP = 0, 1, 2
# Repo root, when this file is at src/players/actor_critic_gae/python/ (run copies pass repo_dir instead).
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


class TribesEnv:
    def __init__(self, classpath, cfg, java="java", heap="2g", repo_dir=REPO_DIR):
        args = [f"tribes={','.join(cfg['tribes'])}", f"game_mode={cfg['game_mode']}",
                f"level_seed={cfg['level_seed']}"]
        # Tribes reads terrainProbs.json etc. relative to the working directory: run from the repo root.
        self.proc = subprocess.Popen([java, f"-Xmx{heap}", "-cp", classpath,
                                      "players.actor_critic_gae.EnvServer", *args],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=repo_dir)
        self.r, self.w = self.proc.stdout, self.proc.stdin
        if read_int(self.r) != MAGIC:
            raise RuntimeError("EnvServer didn't answer with the expected header")
        self.obs_size, self.act_size, self.board_size = read_int(self.r), read_int(self.r), read_int(self.r)
        self.seat = None

    def _response(self):
        done, reward, self.seat = read_int(self.r), read_double(self.r), read_int(self.r)
        if done:
            return None, None, reward, True
        obs, acts = read_decision(self.r, self.obs_size, self.act_size)
        return obs, acts, reward, False

    def reset(self, seed, seat=-1):
        """New game; seat = -1 picks the agent's seat at random (from `seed`)."""
        write_ints(self.w, OP_RESET)
        self.w.write(struct.pack(">q", seed))
        write_ints(self.w, seat)
        self.w.flush()
        return self._response()

    def step(self, action):
        write_ints(self.w, OP_STEP, int(action))
        self.w.flush()
        return self._response()

    def close(self):
        try:
            write_ints(self.w, OP_QUIT)
            self.w.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
