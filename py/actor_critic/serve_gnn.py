"""
Inference server for the Tribes GNN, started by the Java ExternalNetwork as a subprocess.

    python py/actor_critic/serve_gnn.py --checkpoint models/iter_001/best.pt [--threads 4] [--device cpu]

It talks binary over stdin/stdout. All numbers are big-endian (Java's DataInput/DataOutputStream);
stdout carries nothing else, and logs go to stderr.

Handshake (server -> client, once the model is loaded):
    int32 MAGIC (0x54474E4E, "TGNN"), int32 VERSION,
    int32 nodeFeatures, edgeFeatures, globalFeatures, actionFeatures, srcCol, tgtCol, maxPlayers

Request (client -> server):
    int32 op            1 = evaluate, 0 = quit
    int32 boardSize, int32 stateLength, int32 numActions, int32 actionFeatures
    float32[stateLength]                  StateEncoder.encode(...)
    float32[numActions * actionFeatures]  ActionEncoder.toFeatures(...), row by row

Response (server -> client):
    int32 status        0 = ok, 1 = error
    ok:    float32 value, float32[numActions] logits   (softmax is done in Java)
    error: int32 length, utf-8 bytes message
"""

import argparse
import os
import struct
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import TribesGNN  # noqa: E402
from selfplay_data import Schema  # noqa: E402

MAGIC = 0x54474E4E
VERSION = 1
OP_QUIT, OP_EVALUATE = 0, 1


def log(*args):
    print("[serve_gnn]", *args, file=sys.stderr, flush=True)


def read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("client closed the connection")
        buf += chunk
    return bytes(buf)


def read_ints(stream, k):
    return struct.unpack(f">{k}i", read_exact(stream, 4 * k))


def read_floats(stream, k):
    return np.frombuffer(read_exact(stream, 4 * k), dtype=">f4").astype(np.float32) if k else np.zeros(0, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto", help="auto (GPU if available) | cpu | cuda | cuda:0 ...")
    args = p.parse_args()

    torch.set_num_threads(max(1, args.threads))
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log(f"GPU: {torch.cuda.get_device_name(device)}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = Schema.from_dict(ckpt["schema"])
    model = TribesGNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    log(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch')}), device={device}, threads={args.threads}")

    inp, out = sys.stdin.buffer, sys.stdout.buffer
    out.write(struct.pack(">9i", MAGIC, VERSION, schema.node_features, schema.edge_features, schema.global_features,
                          schema.action_features, schema.src_col, schema.tgt_col, schema.max_players))
    out.flush()

    n_eval = 0
    while True:
        try:
            (op,) = read_ints(inp, 1)
        except EOFError:
            break
        if op == OP_QUIT:
            break
        if op != OP_EVALUATE:
            log(f"unknown op {op}, exiting")
            break

        size, state_len, n_actions, action_f = read_ints(inp, 4)
        state = read_floats(inp, state_len)
        actions = read_floats(inp, n_actions * action_f)   # read the whole request before any error reply
        try:
            if state_len != schema.state_length(size):
                raise ValueError(f"state length {state_len} != {schema.state_length(size)} expected for board {size}")
            if action_f != schema.action_features:
                raise ValueError(f"action feature size {action_f} != {schema.action_features}")

            flat = torch.from_numpy(state).to(device).unsqueeze(0)
            x, edge_attr, u = schema.split_state(flat, size)
            # The model needs at least one action slot; with no actions, pass one masked-out dummy.
            a = torch.from_numpy(actions.reshape(n_actions, action_f).copy()) if n_actions else torch.zeros(1, action_f)
            mask = torch.ones(1, a.shape[0], dtype=torch.bool) if n_actions else torch.zeros(1, 1, dtype=torch.bool)
            with torch.inference_mode():
                logits, value = model(x, edge_attr, u, a.unsqueeze(0).to(device), mask.to(device))
            logits = logits[0, :n_actions].cpu().numpy().astype(">f4")
            out.write(struct.pack(">if", 0, float(value.item())))
            out.write(logits.tobytes())
        except Exception as e:  # report to Java instead of dying, so the error message isn't lost
            msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}".encode("utf-8")
            out.write(struct.pack(">ii", 1, len(msg)))
            out.write(msg)
        out.flush()
        n_eval += 1

    log(f"exiting after {n_eval} evaluations")


if __name__ == "__main__":
    main()
