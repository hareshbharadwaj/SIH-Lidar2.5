"""
Full evaluation of the FoveaMap pipeline on the held-out test sequence.

Produces ``outputs/metrics.json`` with:

  * per-class / per-group / per-range-bin IoU and accuracy (point level)
  * drivability and object-detection task metrics
  * row-normalised confusion matrix
  * measured latency breakdown + FPS, and the cost/latency scaling of the
    network with input resolution (used to price a uniform 5 cm baseline)
  * memory footprint of the adaptive grid vs uniform grids and vs a 3D voxel
    map
  * the resolution ablation: map-level mIoU and curb recall for uniform cell
    sizes 5/10/20/50 cm against the adaptive (foveated) grid
"""

from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import torch

from . import config as C
from .grids import FoveatedGrid
from .model import FoveaNet
from .pipeline import infer_points, project_uniform, motion_score
from .train import FrameSet


# ==========================================================================
def iou_from_conf(pred, gt, K=C.NUM_CLASSES):
    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)
    m = (pred >= 0) & (gt >= 0) & (gt < K) & (pred < K)
    cm = np.bincount(gt[m] * K + pred[m], minlength=K * K).reshape(K, K)
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    return iou, cm, float((pred[m] == gt[m]).mean())


def group_iou(iou):
    out = {}
    for g, name in ((C.TERRAIN_CLASSES, "terrain"),
                    (C.STATIC_CLASSES, "static"),
                    (C.DYNAMIC_CLASSES, "dynamic")):
        out[name] = float(np.mean([iou[i] for i in g]))
    return out


def task_metrics(pred, gt, K=C.NUM_CLASSES):
    """Terrain drivability + object detection, in the terms of the brief."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    out = {}

    # ---- TASK A: drivability over ground points -------------------------
    terr = np.isin(gt, C.TERRAIN_CLASSES)
    if terr.sum():
        d_gt = C.DRIVABLE_ID[gt[terr]]
        d_pr = C.DRIVABLE_ID[np.clip(pred[terr], 0, K - 1)]
        tp = int(((d_gt == 0) & (d_pr == 0)).sum())
        fp = int(((d_gt == 1) & (d_pr == 0)).sum())
        fn = int(((d_gt == 0) & (d_pr == 1)).sum())
        iou = tp / max(tp + fp + fn, 1)
        out["drivable_iou"] = float(iou)
        out["drivable_acc"] = float((d_gt == d_pr).mean())
        out["false_drivable_rate"] = float(fp / max((d_gt == 1).sum(), 1))
        out["n_terrain"] = int(terr.sum())

    # ---- TASK B: object detection ---------------------------------------
    obj = np.isin(gt, C.STATIC_CLASSES + C.DYNAMIC_CLASSES)
    if obj.sum():
        gp = C.GROUP_ID[gt[obj]]
        pp = C.GROUP_ID[np.clip(pred[obj], 0, K - 1)]
        out["object_recall"] = float((pp != 0).mean())            # not called ground
        out["object_group_acc"] = float((pp == gp).mean())
        dyn = np.isin(gt, C.DYNAMIC_CLASSES)
        sta = np.isin(gt, C.STATIC_CLASSES)
        if dyn.sum():
            out["dynamic_recall"] = float((C.GROUP_ID[np.clip(pred[dyn], 0, K - 1)] != 0).mean())
            out["dynamic_group_acc"] = float((C.GROUP_ID[np.clip(pred[dyn], 0, K - 1)] == 2).mean())
            out["n_dynamic"] = int(dyn.sum())
        if sta.sum():
            out["static_group_acc"] = float((C.GROUP_ID[np.clip(pred[sta], 0, K - 1)] == 1).mean())
            out["n_static"] = int(sta.sum())
        out["n_object"] = int(obj.sum())
    return out


# ==========================================================================
def net_scaling(model, sizes=(64, 128, 176, 256, 352, 512), reps=4):
    """Measure ms/pixel of the network - lets us price any raster size."""
    model.eval()
    rows = []
    cl = torch.channels_last
    with torch.no_grad():
        for s in sizes:
            x = torch.randn(1, C.BEV_CHANNELS, s, s).to(memory_format=cl)
            for _ in range(2):
                model(x, 0)
            t = time.perf_counter()
            for _ in range(reps):
                model(x, 0)
            ms = (time.perf_counter() - t) / reps * 1e3
            rows.append(dict(px=int(s), ms=float(ms), px_count=int(s * s),
                             ms_per_mpx=float(ms / (s * s / 1e6))))
    return rows


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{C.MODEL_DIR}/foveanet_best.pt")
    ap.add_argument("--n", type=int, default=0, help="frames to evaluate (0=all)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=f"{C.OUT_DIR}/metrics.json")
    ap.add_argument("--ablation", action="store_true", default=True)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("FM_THREADS", C.TRAIN["num_threads"])))
    os.makedirs(C.OUT_DIR, exist_ok=True)

    model = FoveaNet(ch=(16, 32, 64, 96))
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.use_channels_last(True)
    model.eval()
    print(f"loaded {args.ckpt}  params={sum(p.numel() for p in model.parameters()):,}")

    fs = FrameSet(f"{C.DATA_DIR}/{args.split}_w*.npz")
    n = len(fs) if args.n <= 0 else min(args.n, len(fs))
    print(f"evaluating {n} frames from {args.split}")

    K = C.NUM_CLASSES
    cm_tot = np.zeros((K, K), np.int64)
    preds_all, gts_all, rad_all, prob_all, refs_all = [], [], [], [], []
    lat = {k: [] for k in ("raster_ms", "net_ms", "sample_ms", "project_ms", "total_ms")}
    grid = FoveatedGrid()

    # per-frame occupancy for the memory report
    occ_adapt, occ_uniform = [], {k: [] for k in C.BASELINES}

    # warm-up: the first sweep pays allocator / thread-pool / page-fault
    # costs that are not representative of steady-state operation
    _ = infer_points(model, fs.get(0), grid=grid)
    if hasattr(model, "forward_pyramid"):
        _ = infer_points(model, fs.get(0), grid=grid)

    t_wall = time.perf_counter()
    for i in range(n):
        fr = fs.get(i)
        prev = fs.get(max(i - 1, 0)) if i > 0 else None
        mot = None
        if prev is not None:
            mot = motion_score(prev, fr,
                               prev_pose=fs.shards[fs.index[max(i - 1, 0)][0]].d["pose"][fs.index[max(i - 1, 0)][1]],
                               cur_pose=fs.shards[fs.index[i][0]].d["pose"][fs.index[i][1]])
        res = infer_points(model, fr, grid=grid, motion=mot)
        for k in lat:
            lat[k].append(res["times"][k])

        r = np.hypot(fr["xy"][:, 0], fr["xy"][:, 1])
        keep = r < 100.0
        preds_all.append(res["pred"][keep])
        gts_all.append(fr["label"][keep].astype(np.int64))
        rad_all.append(r[keep])
        prob_all.append(res["prob"][keep])
        # labels after projection into the foveated lattice + geometric refine
        refs_all.append(FoveatedGrid.scatter(res["rings"], len(fr["z"]),
                                             key="cls_ref")[keep])

        occ_adapt.append(sum(int(rg["valid"].sum()) for rg in res["rings"]))
        for name, cell in C.BASELINES.items():
            ix = np.floor((fr["xy"][:, 0] + C.MAP_EXTENT / 2) / cell).astype(np.int64)
            iy = np.floor((fr["xy"][:, 1] + C.MAP_EXTENT / 2) / cell).astype(np.int64)
            nn = int(C.MAP_EXTENT / cell)
            m = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn) & keep
            occ_uniform[name].append(int(np.unique(iy[m] * nn + ix[m]).size))
        print(f"  frame {i+1}/{n}  total {res['times']['total_ms']:.1f} ms", flush=True)

    wall = time.perf_counter() - t_wall
    pred = np.concatenate(preds_all)
    gt = np.concatenate(gts_all)
    rad = np.concatenate(rad_all)
    prob = np.concatenate(prob_all)
    ref = np.concatenate(refs_all)

    iou, cm, acc = iou_from_conf(pred, gt)
    tm = task_metrics(pred, gt)
    iou_ref, _, acc_ref = iou_from_conf(ref, gt)
    tm_ref = task_metrics(ref, gt)

    # ---- accuracy vs range -------------------------------------------
    per_range = []
    for (lo, hi) in C.RANGE_BINS:
        m = (rad >= lo) & (rad < hi)
        if m.sum() < 50:
            per_range.append(dict(rng=f"{lo}-{hi}", n=int(m.sum()), miou=None, acc=None))
            continue
        ii, _, aa = iou_from_conf(pred[m], gt[m])
        iir, _, aar = iou_from_conf(ref[m], gt[m])
        per_range.append(dict(rng=f"{lo}-{hi}", n=int(m.sum()),
                              miou=float(np.nanmean(ii)),
                              miou_mapped=float(np.nanmean(iir)),
                              acc=float(aa), acc_mapped=float(aar),
                              iou=[float(x) for x in ii],
                              cell=C.cell_size_of(np.array([(lo + hi) / 2]))[0]))

    # ---- latency ------------------------------------------------------
    latency = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)),
                       p95=float(np.percentile(v, 95))) for k, v in lat.items()}
    fps = 1000.0 / latency["total_ms"]["mean"]

    # ---- network scaling ----------------------------------------------
    scaling = net_scaling(model)
    big = [s for s in scaling if s["px"] >= 176]
    ms_per_mpx = float(np.mean([s["ms_per_mpx"] for s in big]))
    unif5_px = int(C.MAP_EXTENT / 0.05) ** 2
    proj_unif5_ms = ms_per_mpx * unif5_px / 1e6

    # ---- memory -------------------------------------------------------
    bytes_cell = 8
    adapt_alloc = sum(C.RING_TILE ** 2 for _ in C.GRID_RINGS)
    mem = dict(bytes_per_cell=bytes_cell,
               adaptive_allocated=adapt_alloc,
               adaptive_occupied_mean=float(np.mean(occ_adapt)),
               adaptive_bytes=adapt_alloc * bytes_cell,
               uniform=[])
    for name, cell in C.BASELINES.items():
        nn = int(C.MAP_EXTENT / cell) ** 2
        mem["uniform"].append(dict(
            name=name, cell=cell, allocated=nn, bytes=nn * bytes_cell,
            occupied_mean=float(np.mean(occ_uniform[name])),
            ratio_vs_adaptive=nn / adapt_alloc))
    # 3D voxel reference: 200 x 200 x 12 m at 5 cm
    vox = int(C.MAP_EXTENT / 0.05) ** 2 * int(12 / 0.05)
    mem["voxel3d"] = dict(allocated=vox, bytes=vox * 4, cell=0.05, height=12.0)

    report = dict(
        split=args.split, n_frames=n, wall_s=float(wall),
        num_classes=K, class_names=C.CLASS_NAMES,
        per_class_iou=[float(x) for x in iou],
        miou=float(np.nanmean(iou)),
        accuracy=float(acc),
        freq=[float(v) for v in (np.bincount(gt, minlength=K) / len(gt))],
        per_class_iou_mapped=[float(x) for x in iou_ref],
        miou_mapped=float(np.nanmean(iou_ref)),
        accuracy_mapped=float(acc_ref),
        group_iou_mapped=group_iou(iou_ref),
        task_mapped=tm_ref,
        group_iou=group_iou(iou),
        task=tm,
        per_range=per_range,
        confusion=(cm / np.maximum(cm.sum(1, keepdims=True), 1)).tolist(),
        confusion_counts=cm.tolist(),
        latency=latency,
        fps=float(fps),
        net_scaling=scaling,
        ms_per_megapixel=ms_per_mpx,
        projected_uniform5cm_net_ms=float(proj_unif5_ms),
        projected_uniform5cm_fps=float(1000.0 / proj_unif5_ms),
        memory=mem,
        pyramid=[dict(extent=p[0], px=p[1], res=p[2]) for p in C.PYRAMID],
        rings=[dict(r_in=r[0], r_out=r[1], cell=r[2]) for r in C.GRID_RINGS],
        params=int(sum(p.numel() for p in model.parameters())),
    )

    # ---- resolution ablation -----------------------------------------
    if args.ablation:
        print("running resolution ablation ...")
        n_ab = min(n, 8)
        cache = []
        for i in range(n_ab):
            fr = fs.get(i)
            res = infer_points(model, fr, grid=grid, refine=False)
            r = np.hypot(fr["xy"][:, 0], fr["xy"][:, 1])
            m = r < 100.0
            cache.append((fr["xy"][m], fr["z"][m],
                          np.maximum(res["pred"][m], 0), res["prob"][m],
                          fr["label"][m].astype(np.int64), res))

        def occ_count(xy, cell):
            nn = int(C.MAP_EXTENT / cell)
            ix = np.floor((xy[:, 0] + C.MAP_EXTENT / 2) / cell).astype(np.int64)
            iy = np.floor((xy[:, 1] + C.MAP_EXTENT / 2) / cell).astype(np.int64)
            m = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn)
            return int(np.unique(iy[m] * nn + ix[m]).size)

        def metrics_of(pt, g):
            ii, _, accu = iou_from_conf(pt, g)
            return dict(miou=float(np.nanmean(ii)), acc=float(accu),
                        curb_iou=float(ii[names.index("curb")]),
                        pothole_iou=float(ii[names.index("pothole")]),
                        pedestrian_iou=float(ii[names.index("pedestrian")]),
                        vehicle_iou=float(ii[names.index("vehicle")]),
                        iou=[float(x) for x in ii])

        names = C.CLASS_NAMES
        abl = []
        for cell in (0.05, 0.10, 0.20, 0.35, 0.50):
            pi, gi, occ = [], [], 0
            for (xy, z, pr, pb, gl, _) in cache:
                rg = project_uniform(xy, z, pr, pb, np.zeros(len(xy), np.float32),
                                     cell, refine=True, crop_m=30.0)
                pi.append(rg["pt_ref"]); gi.append(gl)
                nn = int(C.MAP_EXTENT / cell)
                ix = np.floor((xy[:, 0] + C.MAP_EXTENT / 2) / cell).astype(np.int64)
                iy = np.floor((xy[:, 1] + C.MAP_EXTENT / 2) / cell).astype(np.int64)
                ok = (ix >= 0) & (ix < nn) & (iy >= 0) & (iy < nn)
                occ += int(np.unique(iy[ok] * nn + ix[ok]).size)
            d = metrics_of(np.concatenate(pi), np.concatenate(gi))
            d.update(mode="uniform", cell=cell,
                     allocated=int((C.MAP_EXTENT / cell) ** 2), occupied=occ)
            abl.append(d)
        # adaptive (foveated) grid, geometric refinement enabled on the fine rings
        pi, gi = [], []
        for i, (xy, z, pr, pb, gl, _) in enumerate(cache):
            res2 = infer_points(model, fs.get(i), grid=grid, refine=True)
            s_all = FoveatedGrid.scatter(res2["rings"], len(fs.get(i)["xy"]), key="cls_ref")
            r = np.hypot(fs.get(i)["xy"][:, 0], fs.get(i)["xy"][:, 1])
            pi.append(s_all[r < 100.0]); gi.append(gl)
        d = metrics_of(np.concatenate(pi), np.concatenate(gi))
        d.update(mode="adaptive", cell=None, allocated=adapt_alloc,
                 occupied=int(np.mean(occ_adapt) * n_ab))
        abl.append(d)
        report["ablation"] = abl

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    print(f"\n== {args.split} ==")
    print(f"mIoU {report['miou']:.4f}   acc {report['accuracy']:.4f}   "
          f"FPS {report['fps']:.1f}")
    for i, nm in enumerate(C.CLASS_NAMES):
        print(f"   {nm:12s} IoU {iou[i]:.3f}   freq {report['freq'][i]*100:6.2f}%")
    print("groups:", report["group_iou"])
    print(f"after grid mapping + geometric refine: mIoU {report['miou_mapped']:.4f}  "
          f"acc {report['accuracy_mapped']:.4f}")
    for i, nm in enumerate(C.CLASS_NAMES):
        print(f"   {nm:12s} IoU {iou[i]:.3f} -> {iou_ref[i]:.3f}")
    print("task:", {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in report["task"].items()})
    print(f"latency: {latency['total_ms']['mean']:.1f} ms  "
          f"(raster {latency['raster_ms']['mean']:.1f}, net {latency['net_ms']['mean']:.1f})")
    print("saved", args.out)


if __name__ == "__main__":
    main()
