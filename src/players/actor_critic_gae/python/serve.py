"""Serves a trained actor to Java (PythonPolicy.java), so ActorCriticGAEPlayer can play from Run
(play.json / tournament.json as "ACGAE"). Java starts it as:

    python serve.py <checkpoint.pkl>

and sends one encoded state at a time; the reply is pi(.|s) over that state's legal actions.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tribes_io import MAGIC, read_decision, read_int, write_ints  # noqa: E402
from train import build_agent, load_checkpoint                    # noqa: E402


def main():
    ckpt = load_checkpoint(sys.argv[1])
    agent = build_agent(ckpt["config"], ckpt["obs_size"], ckpt["act_size"])
    agent.params = ckpt["params"]

    r, w = sys.stdin.buffer, sys.stdout.buffer
    write_ints(w, MAGIC, ckpt["obs_size"], ckpt["act_size"], ckpt["board_size"])
    w.flush()
    while True:
        try:
            op = read_int(r)
        except EOFError:
            break
        if op == 0:
            break
        obs, acts = read_decision(r, ckpt["obs_size"], ckpt["act_size"])
        probs = agent.policy(obs, acts)
        write_ints(w, len(probs))
        w.write(struct.pack(f">{len(probs)}d", *probs))
        w.flush()


if __name__ == "__main__":
    main()
