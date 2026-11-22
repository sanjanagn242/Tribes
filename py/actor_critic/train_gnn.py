"""
Train the Tribes actor-critic GNN on self-play data.

    python py/actor_critic/train_gnn.py --data data/selfplay/iter_000 --out models/iter_001

Loss = cross-entropy(search policy pi, softmax(logits)) + value_weight * MSE(z, value)

Outputs in --out:
    best.pt      checkpoint with the lowest validation loss (model weights + config + schema + args)
    last.pt      checkpoint after the final epoch
    metrics.csv  one row per epoch

Continue from an earlier model (next self-play iteration) with --init-from models/iter_001/best.pt.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_model import TribesGNN  # noqa: E402
from selfplay_data import SelfPlayDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True,
                   help="self-play folders (searched recursively) or .jsonl[.gz] files; mix iterations freely")
    p.add_argument("--out", required=True, help="output folder for checkpoints and metrics")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4, help="message-passing rounds")
    p.add_argument("--value-weight", type=float, default=1.0, help="weight of the value loss")
    p.add_argument("--val-frac", type=float, default=0.1, help="fraction of GAMES held out for validation")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--init-from", default=None, help="checkpoint to start from (keeps its architecture)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = torch default)")
    p.add_argument("--no-cache", action="store_true", help="don't read/write the .npz caches next to the data")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def losses(model, batch, value_weight):
    logits, value = model(batch["x"], batch["edge_attr"], batch["u"], batch["actions"], batch["action_mask"])
    mask, pi, z = batch["action_mask"], batch["pi"], batch["z"]

    logp = torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)    # padding: 0 * -inf would be NaN
    policy_loss = -(pi * logp).sum(-1).mean()
    value_loss = F.mse_loss(value, z)
    loss = policy_loss + value_weight * value_loss

    with torch.no_grad():
        # Entropy of pi: the policy loss can't go below it, so KL = CE - H(pi) shows how far there is to go.
        pi_entropy = -(pi * torch.log(pi.clamp(min=1e-12))).sum(-1).mean()
        top1 = (logits.argmax(-1) == pi.argmax(-1)).float().mean()          # agrees with the most-visited action
        sign = ((value > 0) == (z > 0)).float().mean()                        # predicts the winner
    stats = dict(loss=loss.item(), policy=policy_loss.item(), value=value_loss.item(),
                 kl=(policy_loss - pi_entropy).item(), top1=top1.item(), value_sign=sign.item())
    return loss, stats


def run_epoch(model, data, split, args, device, optimizer=None, scheduler=None, rng=None):
    train = optimizer is not None
    model.train(train)
    totals, n = {}, 0
    with torch.set_grad_enabled(train):
        for batch in data.batches(split, args.batch_size, shuffle=train, rng=rng):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, stats = losses(model, batch, args.value_weight)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
            bs = batch["z"].shape[0]
            for k, v in stats.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def save(path, model, schema, args, epoch, metrics):
    torch.save(dict(model_state=model.state_dict(), model_config=model.config, schema=schema.raw,
                    args=vars(args), epoch=epoch, metrics=metrics), path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    t0 = time.time()
    data = SelfPlayDataset(args.data, use_cache=not args.no_cache)
    train_split, val_split = data.split(args.val_frac, seed=args.seed)
    n_train = sum(len(i) for _, i in train_split)
    n_val = sum(len(i) for _, i in val_split)
    sizes = ", ".join(f"{s}x{s}: {len(b)}" for s, b in data.buckets.items())
    print(f"{len(data)} samples ({sizes}) loaded in {time.time() - t0:.1f}s -> {n_train} train / {n_val} val")
    if n_train == 0:
        raise SystemExit("No training samples (too few games for the validation split?)")

    s = data.schema
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        cfg = ckpt["model_config"]
        expected = (s.node_features, s.edge_features, s.global_features, s.action_features, s.src_col, s.tgt_col)
        got = tuple(cfg[k] for k in ("node_features", "edge_features", "global_features", "action_features",
                                     "src_col", "tgt_col"))
        if expected != got:
            raise SystemExit(f"--init-from model was trained on a different feature layout: {got} vs data {expected}")
        model = TribesGNN(**cfg)
        model.load_state_dict(ckpt["model_state"])
        print(f"Initialised from {args.init_from} (epoch {ckpt['epoch']})")
    else:
        model = TribesGNN(s.node_features, s.edge_features, s.global_features, s.action_features,
                          s.src_col, s.tgt_col, hidden=args.hidden, layers=args.layers)
    model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"hidden={model.config['hidden']}, layers={model.config['layers']}, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = sum((len(i) + args.batch_size - 1) // args.batch_size for _, i in train_split)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, pct_start=0.1,
                                                    total_steps=max(1, args.epochs * steps_per_epoch))

    metrics_path = os.path.join(args.out, "metrics.csv")
    fields = ["epoch", "seconds"] + [f"{p}_{k}" for p in ("train", "val")
                                     for k in ("loss", "policy", "value", "kl", "top1", "value_sign")]
    best = float("inf")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, data, train_split, args, device, optimizer, scheduler, rng)
            va = run_epoch(model, data, val_split, args, device) if n_val else {}
            row = {"epoch": epoch, "seconds": round(time.time() - t0, 1)}
            row.update({f"train_{k}": round(v, 5) for k, v in tr.items()})
            row.update({f"val_{k}": round(v, 5) for k, v in va.items()})
            writer.writerow(row)
            f.flush()

            msg = (f"epoch {epoch:3d} | {row['seconds']:6.1f}s | train loss {tr['loss']:.4f} "
                   f"(pol {tr['policy']:.4f}, val {tr['value']:.4f}, top1 {tr['top1']:.3f})")
            if va:
                msg += (f" | val loss {va['loss']:.4f} (pol {va['policy']:.4f}, kl {va['kl']:.4f}, "
                        f"val {va['value']:.4f}, top1 {va['top1']:.3f}, sign {va['value_sign']:.3f})")
            score = va["loss"] if va else tr["loss"]
            if score < best:
                best = score
                save(os.path.join(args.out, "best.pt"), model, s, args, epoch, row)
                msg += "  *"
            print(msg, flush=True)

    save(os.path.join(args.out, "last.pt"), model, s, args, args.epochs, row)
    print(f"Done. Best {'val' if n_val else 'train'} loss {best:.4f} -> {args.out}/best.pt; metrics in {metrics_path}")


if __name__ == "__main__":
    main()
