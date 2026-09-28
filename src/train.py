"""
FoveaNet training loop.

Points are kept on disk in compact int16 form and rasterised *on the fly*,
which (a) keeps memory flat and (b) lets every sample carry a fresh random
augmentation (yaw, mirror, scale, ego jitter, intensity jitter, range-
dependent return dropout).
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .grids import rasterize_bev
from .model import FoveaNet, MaskedCE, estimate_class_weights, count_params


# ==========================================================================
#  shard access
# ==========================================================================
class Shard:
    def __init__(self, path):
        self.d = np.load(path)
        self.ptr = self.d["ptr"]
        self.n = int(self.ptr.shape[0]) - 1
        self.path = path

    def frame(self, i):
        s, e = int(self.ptr[i]), int(self.ptr[i + 1])
        a = self.d["xyz"][s:e]
        return dict(
            xy=a[:, :2].astype(np.float32) * 0.01,
            z=a[:, 2].astype(np.float32) * 0.01,
            label=self.d["label"][s:e],
            inten=self.d["inten"][s:e],
            dyn=self.d["dyn"][s:e],
        )


class FrameSet:
    def __init__(self, pattern):
        self.shards = [Shard(p) for p in sorted(glob.glob(pattern))]
        self.index = []
        for si, s in enumerate(self.shards):
            for f in range(s.n):
                self.index.append((si, f))
        print(f"  {pattern}: {len(self.shards)} shard(s), {len(self.index)} frames")

    def __len__(self):
        return len(self.index)

    def get(self, i):
        si, f = self.index[i]
        return self.shards[si].frame(f)


# ==========================================================================
#  augmentation
# ==========================================================================
def augment(fr, rng):
    xy = fr["xy"].copy()
    z = fr["z"].copy()
    lab = fr["label"]
    itn = fr["inten"].astype(np.float32)

    th = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(th), np.sin(th)
    xy = np.stack([xy[:, 0] * c - xy[:, 1] * s, xy[:, 0] * s + xy[:, 1] * c], 1)

    if rng.random() < 0.5:
        xy[:, 1] = -xy[:, 1]

    sc = float(rng.uniform(0.88, 1.12))
    xy *= sc
    z *= sc

    xy += rng.uniform(-1.6, 1.6, 2).astype(np.float32)

    itn = np.clip(itn * float(rng.uniform(0.75, 1.25)) + rng.normal(0, 7.0), 0, 255)

    r = np.hypot(xy[:, 0], xy[:, 1])
    pdrop = float(rng.uniform(0.0, 0.25)) * np.clip((r - 12.0) / 55.0, 0.0, 1.0)
    keep = rng.random(len(xy)) >= pdrop
    if keep.sum() < 0.5 * len(xy):
        keep = np.ones(len(xy), bool)

    return xy[keep], z[keep], lab[keep], itn[keep]


def make_sample(fr, rng, aug=True):
    if aug:
        xy, z, lab, itn = augment(fr, rng)
    else:
        xy, z, lab, itn = fr["xy"], fr["z"], fr["label"], fr["inten"].astype(np.float32)
    feas, labs, vals = [], [], []
    for k, (extent, px, _) in enumerate(C.PYRAMID):
        f, l, v = rasterize_bev(xy, z, itn, lab, extent, px)
        f[7] = k / max(C.PYR_LEVELS - 1, 1)
        feas.append(f)
        labs.append(l)
        vals.append(v)
    return feas, labs, vals


CHANNELS_LAST = True


def collate(batch, device="cpu"):
    """batch: list of (feas, labs, vals) -> tensors stacked per level."""
    out = []
    for k in range(C.PYR_LEVELS):
        f = torch.from_numpy(np.stack([b[0][k] for b in batch])).to(
            memory_format=torch.channels_last if CHANNELS_LAST else torch.contiguous_format)
        l = torch.from_numpy(np.stack([b[1][k] for b in batch]).astype(np.int64))
        v = torch.from_numpy(np.stack([b[2][k] for b in batch]))
        l = torch.where(v, l, torch.zeros_like(l))
        out.append((f, l, v))
    return out


# ==========================================================================
#  evaluation
# ==========================================================================
@torch.no_grad()
def evaluate(model, frames, n=None, aug=False, rng=None):
    model.eval()
    inter = np.zeros(C.NUM_CLASSES)
    union = np.zeros(C.NUM_CLASSES)
    tot = 0
    corr = 0
    n = len(frames) if n is None else min(n, len(frames))
    for i in range(n):
        fr = frames.get(i)
        feas, labs, vals = make_sample(fr, rng, aug=aug)
        logits = model.forward_pyramid(feas)
        for k in range(C.PYR_LEVELS):
            pred = logits[k].argmax(0)
            gt = labs[k]
            v = vals[k]
            if v.sum() == 0:
                continue
            p = pred[v]
            g = gt[v].astype(np.int64)
            for c in range(C.NUM_CLASSES):
                inter[c] += int(((p == c) & (g == c)).sum())
                union[c] += int(((p == c) | (g == c)).sum())
            tot += g.size
            corr += int((p == g).sum())
    iou = inter / np.maximum(union, 1)
    miou = float(np.nanmean(iou[union > 0]))
    return dict(iou=iou, miou=miou, acc=corr / max(tot, 1), n=n)


# ==========================================================================
#  main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.TRAIN["epochs"])
    ap.add_argument("--iters", type=int, default=C.TRAIN["iters_per_epoch"])
    ap.add_argument("--batch", type=int, default=C.TRAIN["batch_size"])
    ap.add_argument("--lr", type=float, default=C.TRAIN["lr"])
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wpow", type=float, default=0.5,
                    help="class-weight exponent: 0.5=damped, 1.0=inverse freq")
    ap.add_argument("--wclip", type=float, default=6.0)
    ap.add_argument("--wmin", type=float, default=0.40)
    ap.add_argument("--init", type=str, default="",
                    help="checkpoint to start from (fine-tune)")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--tag", type=str, default="foveanet")
    args = ap.parse_args()

    torch.manual_seed(C.TRAIN["seed"])
    torch.set_num_threads(C.TRAIN["num_threads"])
    np.random.seed(C.TRAIN["seed"])
    rng = np.random.default_rng(C.TRAIN["seed"])

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    print("loading dataset ...")
    train = FrameSet(f"{C.DATA_DIR}/train_w*.npz")
    val = FrameSet(f"{C.DATA_DIR}/val_w*.npz")
    print(f"train frames {len(train)}   val frames {len(val)}")

    print("estimating class weights ...")
    sample = [train.get(i) for i in np.linspace(0, len(train) - 1, 18).astype(int)]
    w = estimate_class_weights(sample, wpow=args.wpow,
                              clip=(args.wmin, args.wclip))
    print("  weights:", np.round(w, 2).tolist())
    del sample

    model = FoveaNet(ch=(args.ch, args.ch * 2, args.ch * 4, min(args.ch * 6, 160)))
    model.use_channels_last(CHANNELS_LAST)
    print(f"model params: {count_params(model):,}  channels_last={CHANNELS_LAST}")
    crit = MaskedCE(w)

    init = args.init or args.resume
    if init and os.path.exists(init):
        model.load_state_dict(torch.load(init, map_location="cpu"))
        model.use_channels_last(CHANNELS_LAST)
        print("initialised from", init)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=C.TRAIN["weight_decay"])
    total_steps = args.epochs * args.iters
    if args.wpow != 0.5:
        print(f"  class weighting: exponent {args.wpow}, clip [{args.wmin},{args.wclip}]")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.06, anneal_strategy="cos")

    if args.smoke:
        print("--- smoke timing ---")
        batch = [make_sample(train.get(i), rng) for i in range(args.batch)]
        bt = collate(batch)
        t0 = time.time()
        for _ in range(3):
            for k in range(C.PYR_LEVELS):
                f, l, v = bt[k]
                out = model(f, k)
                loss = crit(out, l, v)
                loss.backward()
        torch.set_num_threads(C.TRAIN["num_threads"])
        t1 = time.time()
        print(f"  {(t1-t0)/3*1000:.1f} ms / step (batch={args.batch}, fwd+bwd)")
        print(f"  projected {(t1-t0)/3*total_steps/60:.1f} min for {total_steps} steps")
        t0 = time.time()
        with torch.no_grad():
            for k in range(C.PYR_LEVELS):
                f, l, v = bt[k]
                model(f, k)
        print(f"  {(time.time()-t0)*1000*args.batch:.1f} ms / frame inference (batch {args.batch})")
        return

    hist = dict(loss=[], miou=[], step=[], val_miou=[], val_iou=[], best=0.0)
    best = 0.0
    step = 0
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        for it in range(args.iters):
            idx = rng.integers(0, len(train), args.batch)
            batch = [make_sample(train.get(int(i)), rng) for i in idx]
            bt = collate(batch)
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for k in range(C.PYR_LEVELS):
                f, l, v = bt[k]
                out = model(f, k)
                loss = loss + crit(out, l, v)
            loss = loss / C.PYR_LEVELS
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ep_loss += float(loss)
            step += 1

            if step % 50 == 0:
                hist["loss"].append(ep_loss / 50)
                hist["step"].append(step)
                el = time.time() - t_start
                print(f"ep {ep+1:3d}/{args.epochs} it {it+1:4d} "
                      f"loss {ep_loss/50:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el:.0f}s  eta {el/step*(total_steps-step)/60:.1f}m", flush=True)
                ep_loss = 0.0

            if step % 400 == 0 or step == total_steps:
                r = evaluate(model, val, n=min(16, len(val)), rng=rng)
                hist["val_miou"].append(float(r["miou"]))
                hist["val_iou"].append([float(x) for x in r["iou"]])
                hist["miou"].append(step)
                print(f"   >> val mIoU {r['miou']:.4f}  acc {r['acc']:.4f}", flush=True)
                if r["miou"] > best:
                    best = r["miou"]
                    hist["best"] = float(best)
                    torch.save(model.state_dict(), f"{C.MODEL_DIR}/{args.tag}_best.pt")
        torch.save(model.state_dict(), f"{C.MODEL_DIR}/{args.tag}_last.pt")

    torch.save(model.state_dict(), f"{C.MODEL_DIR}/{args.tag}_final.pt")
    r = evaluate(model, val, n=len(val), rng=rng)
    hist["final_val"] = dict(miou=float(r["miou"]), acc=float(r["acc"]),
                             iou=[float(x) for x in r["iou"]])
    hist["args"] = vars(args)
    hist["class_weights"] = [float(x) for x in w]
    with open(f"{C.MODEL_DIR}/{args.tag}_hist.json", "w", encoding="utf-8") as fh:
        json.dump(hist, fh, indent=1)
    print(f"\nDONE  best val mIoU {best:.4f}   final val mIoU {r['miou']:.4f}")
    print("per-class IoU:", {C.CLASS_NAMES[i]: round(float(r['iou'][i]), 3)
                             for i in range(C.NUM_CLASSES)})


if __name__ == "__main__":
    main()
