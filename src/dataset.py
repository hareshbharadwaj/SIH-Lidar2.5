"""
Dataset builder: turns procedural worlds into labelled LiDAR sweeps and
stores them in a compact, shard-wise .npz format.

Layout of one shard  (data/<split>_w<seed>.npz)
    xyz        (N,3) int16   centimetres, ego frame (x fwd, y left, z up)
    label      (N,)  uint8   semantic class
    dyn        (N,)  uint8   1 = the return came from a moving agent
    inten      (N,)  uint8   intensity 0..255
    ptr        (F+1,) int64  point index bounds of every frame
    pose       (F,4) float64 x, y, z, yaw  (world)
    t          (F,)  float32 simulation time
"""

from __future__ import annotations
import os
import time
import numpy as np

from . import config as C
from .worldgen import generate_world, _sample
from .lidar import Simulator


def render_world(seed, n_frames, dt, tag="", sensor=None, gcs=8.0):
    t0 = time.time()
    w = generate_world(seed, n_frames, dt=dt)
    gres = w["gres"]
    ego_z = _sample(w["H"], w["poses"][:, 0], w["poses"][:, 1], gres).astype(np.float64)
    # little vertical ride motion from suspension
    rng = np.random.default_rng(seed + 31)
    ego_z = ego_z + 0.012 * np.sin(np.arange(n_frames) * 1.7) + rng.normal(0, 0.004, n_frames)
    sim = Simulator(w, sensor=sensor or C.SENSOR, gcs=gcs, seed=seed)

    xyzs, labs, dyns, ints, poses, ts = [], [], [], [], [], []
    for f in range(n_frames):
        s = sim.sweep(f, float(ego_z[f]))
        xyzs.append(s["xyz"])
        labs.append(s["label"])
        dyns.append(s["dyn"])
        ints.append(s["intensity"])
        poses.append(s["pose"])
        ts.append(s["t"])
        if f == 0 or (f + 1) % 10 == 0:
            print(f"  [{tag}] frame {f+1}/{n_frames}  pts={len(s['xyz'])}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)

    xyz = np.concatenate(xyzs).astype(np.float32)
    lab = np.concatenate(labs)
    dyn = np.concatenate(dyns)
    itn = np.concatenate(ints)
    ptr = np.concatenate([[0], np.cumsum([len(a) for a in xyzs])]).astype(np.int64)
    return dict(
        xyz=(xyz * 100.0).astype(np.int16),      # metres -> centimetres
        label=lab.astype(np.uint8),
        dyn=dyn.astype(np.uint8),
        inten=itn.astype(np.uint8),
        ptr=ptr,
        pose=np.array(poses, dtype=np.float64),
        t=np.array(ts, dtype=np.float32),
        seed=np.int64(seed),
        hmax=np.float32(w["H"].max()),
    )


def build_all(force=False, splits=None):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    cfg = C.DATASET
    plan = []
    for sp, nw, nf in (("train", cfg["n_train_worlds"], cfg["frames_per_world"]),
                       ("val", cfg["n_val_worlds"], cfg["frames_per_world"]),
                       ("test", cfg["n_test_worlds"], cfg["frames_per_world"])):
        base = {"train": 0, "val": 1000, "test": 2000}[sp]
        for k in range(nw):
            plan.append((sp, cfg["seed0"] + base + k * 7, nf, cfg["frame_dt"]))
    plan.append(("dash", cfg["seed0"] + 9137, cfg["dash_frames"], 0.15))

    made = []
    for sp, seed, nf, dt in plan:
        path = f"{C.DATA_DIR}/{sp}_w{seed}.npz"
        if os.path.exists(path) and not force:
            print(f"skip {path}")
            made.append(path)
            continue
        print(f"== rendering {sp} world seed={seed} frames={nf} dt={dt}")
        d = render_world(seed, nf, dt, tag=f"{sp}{seed}")
        np.savez_compressed(path, **d)
        print(f"   saved {path}  ({os.path.getsize(path)/1e6:.1f} MB, "
              f"{d['xyz'].shape[0]/max(nf,1):.0f} pts/frame)")
        made.append(path)
        del d
    return made


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    build_all(force=force)
