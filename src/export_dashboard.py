"""
Build the self-contained interactive dashboard.

For every frame of the demonstration sequence we run the full pipeline and
pack the resulting variable-resolution 2.5D map into PNG tiles (3 RGB images
per ring: class/probability/density, elevation, ground-truth + motion).
PNG is used because the browser decodes it natively and its filters compress
the very smooth elevation layers extremely well.

Outputs
-------
    outputs/dash_data.js   payload (data URIs + metrics)
    dashboard.html         single self-contained file, no network needed
"""

from __future__ import annotations
import argparse
import base64
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from . import config as C
from .grids import FoveatedGrid
from .model import FoveaNet
from .pipeline import infer_points, project_uniform, motion_score
from .train import FrameSet

UNIFORM_CELL = 0.50          # reference uniform grid shown in the UI


# ==========================================================================
#  tile packing
# ==========================================================================
def _elev_pair(v):
    """Split an elevation float into (fine @1cm, coarse @25cm) bytes."""
    v = np.asarray(v, dtype=np.float32)
    fine = np.clip(np.round((v + 1.0) * 100.0), 0, 254).astype(np.uint8)
    hi = np.zeros_like(fine)
    over = v > 1.54
    if over.any():
        hi[over] = np.clip(np.round((v[over] + 1.0) / 0.25), 0, 255).astype(np.uint8)
        fine[over] = 255
    return fine, hi


def _pack(a, b, c, level=9):
    arr = np.stack([a.astype(np.uint8), b.astype(np.uint8), c.astype(np.uint8)], -1)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=level)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_ring(rg, cls_gt, key="cls_ref"):
    """One ring (or one uniform tile) -> 3 RGB-PNG data URIs.

    Channel layout (9 planes over 3 images):
        A : class | confidence (32 levels) | return density (32 levels)
        B : surface height fine @1cm | surface height coarse @25cm | floor height fine
        C : floor height coarse | motion evidence | ground truth (delta coded)

    Confidence/density are quantised to 32 levels: they are diagnostics, not
    measurements, and quantisation roughly halves their compressed size.  The
    ground-truth plane is stored as a *residual* (0 where prediction == truth,
    truth+1 otherwise) which is almost everywhere zero and therefore nearly
    free; the browser reconstructs it as  gt = delta ? delta-1 : cls.
    """
    cls = np.asarray(rg[key], np.uint8).copy()
    valid = np.asarray(rg["valid"], bool)
    cls = np.where(valid, cls, 255).astype(np.uint8)

    prob = np.asarray(rg["prob"], np.float32)
    prob_q = (np.clip(np.round(prob / 255.0 * 31.0), 0, 31) * 8).astype(np.uint8)
    cnt = np.asarray(rg["cnt"], np.float32)
    cnt_q = (np.clip(np.round(np.log1p(cnt) / np.log1p(48.0) * 31.0), 0, 31) * 8).astype(np.uint8)

    fmax, hmax = _elev_pair(np.where(valid, rg["zmax"], -1.0))
    fmin, hmin = _elev_pair(np.where(valid, rg["zmin"], -1.0))
    aux = np.asarray(rg["dyn"], np.uint8)

    gt = np.where(valid, np.asarray(cls_gt, np.uint8), 255).astype(np.uint8)
    same = (gt == cls) | (cls == 255)
    gt_delta = np.where(same, 0, (gt + 1)).astype(np.uint8)

    return dict(
        a=_pack(cls, prob_q, cnt_q),
        b=_pack(fmax, hmax, fmin),
        c=_pack(hmin, aux, gt_delta),
    )


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{C.MODEL_DIR}/foveanet_best.pt")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", default=f"{C.ROOT}/dashboard.html")
    ap.add_argument("--shell", default=f"{C.ROOT}/src/dashboard_shell.html")
    ap.add_argument("--metrics", default=f"{C.OUT_DIR}/metrics.json")
    args = ap.parse_args()

    torch.set_num_threads(C.TRAIN["num_threads"])
    os.makedirs(C.OUT_DIR, exist_ok=True)

    model = FoveaNet(ch=(16, 32, 64, 96))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.use_channels_last(True)
    model.eval()
    print("model loaded")

    fs = FrameSet(f"{C.DATA_DIR}/dash_w*.npz")
    n = len(fs) if args.frames <= 0 else min(args.frames, len(fs))
    grid = FoveatedGrid()

    # warm-up: the first sweep pays allocator / thread-pool / page-fault
    # costs that are not representative of steady-state operation
    _ = infer_points(model, fs.get(0), grid=grid)
    if hasattr(model, "forward_pyramid"):
        _ = infer_points(model, fs.get(0), grid=grid)

    frames = []
    tot_ms = []
    t_all = time.time()
    for i in range(n):
        fr = fs.get(i)
        si, fi = fs.index[i]
        cur_pose = fs.shards[si].d["pose"][fi]
        mot = None
        if i > 0:
            psj, pfj = fs.index[i - 1]
            mot = motion_score(fs.get(i - 1), fr,
                               prev_pose=fs.shards[psj].d["pose"][pfj],
                               cur_pose=cur_pose)
        res = infer_points(model, fr, grid=grid, motion=mot)
        gt_rings = grid.project(fr["xy"], fr["z"], fr["label"].astype(np.int64),
                                np.ones(len(fr["z"]), np.float32),
                                np.zeros(len(fr["z"]), np.float32) if mot is None else mot)
        pred_pts = np.maximum(res["pred"], 0)
        uni = project_uniform(fr["xy"], fr["z"], pred_pts, res["prob"],
                              np.zeros(len(fr["z"]), np.float32) if mot is None else mot,
                              UNIFORM_CELL)
        # project_uniform returns per-point arrays at the top level and the
        # dense lattice under "tile"; the encoder wants the lattice.  It is
        # 50 cm, i.e. coarser than the 20 cm ceiling for geometric refinement,
        # so it gets none - which is exactly the point of the comparison.
        uni_gt = project_uniform(fr["xy"], fr["z"], fr["label"].astype(np.int64),
                                 np.ones(len(fr["z"]), np.float32),
                                 np.zeros(len(fr["z"]), np.float32), UNIFORM_CELL)

        entry = dict(
            i=i, t=float(fr.get("t", i * 0.15) if isinstance(fr, dict) else i * 0.15),
            pose=[float(v) for v in cur_pose],
            npts=int(len(fr["z"])),
            ms={k: round(float(v), 2) for k, v in res["times"].items()},
            rings=[encode_ring(rg, gt_rings[k]["cls"]) for k, rg in enumerate(res["rings"])],
            unif=encode_ring(uni["tile"], uni_gt["tile"]["cls"]),
        )
        frames.append(entry)
        tot_ms.append(res["times"]["total_ms"])
        size = sum(len(x) for e in entry["rings"] for x in e.values()) + \
            sum(len(x) for x in entry["unif"].values())
        print(f"  frame {i+1}/{n}  {res['times']['total_ms']:.1f} ms   "
              f"payload {size/1024:.0f} KB", flush=True)

    payload = dict(
        frames=frames,
        classes=[dict(name=n, color=c, group=g, drivable=d) for (_, n, g, d, c, _) in C.CLASS_DEF],
        rings=[dict(r_in=r[0], r_out=r[1], cell=r[2]) for r in C.GRID_RINGS],
        tile=C.RING_TILE,
        uniform_cell=UNIFORM_CELL,
        extent=C.MAP_EXTENT,
        pyramid=[dict(extent=p[0], px=p[1], res=p[2]) for p in C.PYRAMID],
        mean_ms=float(np.mean(tot_ms)), fps=float(1000.0 / np.mean(tot_ms)),
    )
    metrics = {}
    if os.path.exists(args.metrics):
        with open(args.metrics, encoding="utf-8") as fh:
            metrics = json.load(fh)
    payload["metrics"] = metrics

    js = "const DASH = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    with open(f"{C.OUT_DIR}/dash_data.js", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(js)
    print(f"payload {len(js)/1e6:.2f} MB")

    with open(args.shell, encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("/*__DASH_DATA__*/", js)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)  in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
