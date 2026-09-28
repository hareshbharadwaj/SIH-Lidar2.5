"""
End-to-end inference pipeline:  points -> foveated BEV pyramid -> FoveaNet
-> per-point semantics -> variable-resolution 2.5D grid -> geometric refine.

Also implements:
  * ego-motion-compensated temporal differencing (a *predicted* motion layer,
    no ground truth involved), and
  * uniform-resolution reference grids used for the accuracy/memory ablation.
"""

from __future__ import annotations
import time

import numpy as np
import torch

from . import config as C
from .grids import (FoveatedGrid, rasterize_bev, level_for_radius, sample_logits)


# ==========================================================================
def infer_points(model, fr, refine=True, grid=None, motion=None, timing=False, want_margin=False):
    """Full pipeline for one sweep.

    Returns a dict with the per-point prediction and the projected ring maps.
    """
    tm = {}
    t0 = time.perf_counter()
    xy, z, inten = fr["xy"], fr["z"], fr["inten"].astype(np.float32)

    feas = []
    for k, (extent, px, _) in enumerate(C.PYRAMID):
        f, lab, val = rasterize_bev(xy, z, inten, fr["label"], extent, px)
        f[7] = k / max(C.PYR_LEVELS - 1, 1)
        feas.append(f)
    tm["raster_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    logits = model.forward_pyramid(feas)
    tm["net_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    r = np.hypot(xy[:, 0], xy[:, 1])
    lvl = level_for_radius(r)
    K = C.NUM_CLASSES
    pred = np.zeros(len(xy), np.int64)
    prob = np.zeros(len(xy), np.float32)
    marg = np.zeros(len(xy), np.float32)
    for k in range(C.PYR_LEVELS):
        m = np.flatnonzero(lvl == k)
        if m.size == 0:
            continue
        extent, px, _ = C.PYRAMID[k]
        lg = sample_logits(logits[k], xy[m], extent, px)      # (n, K)
        top = lg.argmax(1)
        pred[m] = top
        e = np.exp(lg - lg.max(1, keepdims=True))
        sm = e / e.sum(1, keepdims=True)
        prob[m] = sm[np.arange(m.size), top]
        if want_margin:
            # the full softmax sort costs ~8 ms/frame and nothing uses it
            srt = np.sort(sm, 1)
            marg[m] = srt[:, -1] - srt[:, -2]
    # returns beyond the outermost crop are unobserved
    far = r > C.PYRAMID[-1][0] * 0.5
    pred[far] = -1
    prob[far] = 0.0
    tm["sample_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    grid = grid or FoveatedGrid()
    aux = np.zeros(len(xy), np.float32) if motion is None else motion
    rings = grid.project(xy, z, np.maximum(pred, 0), prob, aux)
    if refine:
        for k, rg in enumerate(rings):
            if C.GRID_RINGS[k][2] <= 0.20:
                rg["cls_ref"] = FoveatedGrid.refine(rg, C.GRID_RINGS[k][2])
            else:
                rg["cls_ref"] = rg["cls"]
    tm["project_ms"] = (time.perf_counter() - t0) * 1e3
    tm["total_ms"] = sum(tm.values())

    return dict(pred=pred, prob=prob, margin=(marg if want_margin else None),
                rings=rings,
                logits=logits, level=lvl, times=tm)


# ==========================================================================
#  ego-motion-compensated temporal differencing
# ==========================================================================
def _ego_to_world(p, pose):
    x, y, z, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + p[:, 0] * c - p[:, 1] * s,
                     y + p[:, 0] * s + p[:, 1] * c,
                     z + p[:, 2]], 1)


def _world_to_ego(pw, pose):
    x, y, z, yaw = pose
    ex = pw[:, 0] - x
    ey = pw[:, 1] - y
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([ex * c + ey * s, -ex * s + ey * c, pw[:, 2] - z], 1)


def motion_score(prev_fr, cur_fr, prev_pose=None, cur_pose=None,
                 cell=0.5, extent=C.MAP_EXTENT):
    """Per-point motion evidence in [0, 1].

    The previous sweep is warped into the current ego frame using the true
    carrier pose (as any real stack would, from wheel/IMU odometry), then the
    two sweeps are compared on a coarse lattice.  A cell that changed height
    or appeared/vanished is flagged.  This uses *no* labels - it is the cue an
    autonomous vehicle would actually have.
    """
    xy = cur_fr["xy"]
    z = cur_fr["z"]
    n = int(extent / cell)
    ix = np.clip(((xy[:, 0] + extent / 2) / cell).astype(np.int64), 0, n - 1)
    iy = np.clip(((xy[:, 1] + extent / 2) / cell).astype(np.int64), 0, n - 1)
    cid = iy * n + ix
    N = n * n

    def stats(x, zz):
        c = np.bincount(x, minlength=N).astype(np.float32)
        s = np.bincount(x, weights=zz, minlength=N).astype(np.float32)
        mean = s / np.maximum(c, 1)
        zmin = np.full(N, 1e30, np.float32); np.minimum.at(zmin, x, zz)
        zmax = np.full(N, -1e30, np.float32); np.maximum.at(zmax, x, zz)
        return c, mean, zmin, zmax

    if prev_pose is not None and cur_pose is not None:
        pw = _ego_to_world(np.stack([prev_fr["xy"][:, 0], prev_fr["xy"][:, 1],
                                     prev_fr["z"]], 1), prev_pose)
        pe = _world_to_ego(pw, cur_pose)
        pxy, pz = pe[:, :2].astype(np.float32), pe[:, 2].astype(np.float32)
    else:
        pxy, pz = prev_fr["xy"], prev_fr["z"]
    pix = np.clip(((pxy[:, 0] + extent / 2) / cell).astype(np.int64), 0, n - 1)
    piy = np.clip(((pxy[:, 1] + extent / 2) / cell).astype(np.int64), 0, n - 1)
    pcid = piy * n + pix

    c0, m0, lo0, hi0 = stats(pcid, pz)
    c1, m1, lo1, hi1 = stats(cid, z)

    dz = np.abs(m1 - m0)
    dlo = np.abs(np.where(c0 > 0, lo0, m1) - lo1)
    appear = ((c0 == 0) & (c1 > 0)).astype(np.float32)
    vanish = ((c0 > 0) & (c1 == 0)).astype(np.float32)
    raw = (dz + 0.7 * dlo) * (c0 > 0) + 0.6 * appear + 0.4 * vanish
    score = np.clip(raw / 0.35, 0.0, 1.0)
    return score[cid]


# ==========================================================================
#  uniform-resolution reference grids (for the ablation)
# ==========================================================================
def project_uniform(xy, z, cls, prob, aux, cell, extent=C.MAP_EXTENT,
                    refine=True, crop_m=None):
    """Same accumulation as the foveated rings, but one global cell size.

    All statistics are accumulated over the *occupied* cells only, so a
    5 cm uniform grid over 200 m (16 M cells) costs no more memory than the
    points themselves.  The optional geometric refinement needs a dense
    lattice, so it is evaluated on a square crop (default: the nearest 60 m)
    and scattered back onto the points that fall inside it.
    """
    n = int(round(extent / cell))
    ix = np.clip(((xy[:, 0] + extent / 2) / cell).astype(np.int64), 0, n - 1)
    iy = np.clip(((xy[:, 1] + extent / 2) / cell).astype(np.int64), 0, n - 1)
    cid = iy * n + ix

    u, inv = np.unique(cid, return_inverse=True)
    m = u.size
    cnt = np.bincount(inv, minlength=m).astype(np.float32)
    zsum = np.bincount(inv, weights=z, minlength=m).astype(np.float32)
    psum = np.bincount(inv, weights=prob, minlength=m).astype(np.float32)
    asum = np.bincount(inv, weights=aux, minlength=m).astype(np.float32)
    zmax = np.full(m, -1e30, np.float32); np.maximum.at(zmax, inv, z)
    zmin = np.full(m, 1e30, np.float32); np.minimum.at(zmin, inv, z)

    ccls = np.clip(np.asarray(cls), 0, C.NUM_CLASSES - 1).astype(np.int64)
    cnts = np.zeros((m, C.NUM_CLASSES), np.int64)
    np.add.at(cnts, (inv, ccls), 1)
    lab = cnts.argmax(1).astype(np.uint8)
    pt_cls = lab[inv]
    pt_ref = pt_cls.copy()

    ring = dict(cell=cell, n=n, extent=extent, pt_cid=cid, pt_cls=pt_cls,
                pt_ref=pt_ref, u=u, inv=inv, cnt_u=cnt,
                zmean_u=zsum / np.maximum(cnt, 1),
                prob_u=np.clip(psum / np.maximum(cnt, 1), 0, 1),
                dyn_u=np.clip(asum / np.maximum(cnt, 1), 0, 1),
                zmax_u=zmax, zmin_u=zmin)

    # ---- dense lattice, on a crop if the full grid would be huge ---------
    if crop_m is None:
        crop_m = extent / 2 if (n * n) <= 6e6 else 30.0
    half = min(crop_m, extent / 2)
    T = int(round(2 * half / cell))
    if refine or T <= 4096:
        sel = (np.abs(xy[:, 0]) <= half) & (np.abs(xy[:, 1]) <= half)
        cx_ = np.clip(((xy[sel, 0] + half) / cell).astype(np.int64), 0, T - 1)
        cy_ = np.clip(((xy[sel, 1] + half) / cell).astype(np.int64), 0, T - 1)
        ccid = cy_ * T + cx_
        uc, invc = np.unique(ccid, return_inverse=True)
        cntc = np.bincount(invc, minlength=uc.size).astype(np.float32)
        valid = np.zeros(T * T, bool); valid[uc] = True
        zsumc = np.bincount(invc, weights=z[sel], minlength=uc.size)
        zmaxc = np.full(uc.size, -1e30, np.float32); np.maximum.at(zmaxc, invc, z[sel])
        zminc = np.full(uc.size, 1e30, np.float32); np.minimum.at(zminc, invc, z[sel])
        psumc = np.bincount(invc, weights=prob[sel], minlength=uc.size)
        asumc = np.bincount(invc, weights=aux[sel], minlength=uc.size)
        cntsc = np.zeros((uc.size, C.NUM_CLASSES), np.int64)
        np.add.at(cntsc, (invc, ccls[sel]), 1)
        labc = cntsc.argmax(1).astype(np.uint8)

        shp = (T, T)
        def scat(src, fill, dtype):
            a = np.full(T * T, fill, dtype); a[uc] = src; return a.reshape(shp)
        tile = dict(
            cls=scat(labc, 255, np.uint8),
            prob=scat(np.clip(psumc / np.maximum(cntc, 1), 0, 1) * 255, 0, np.uint8),
            zmax=scat(zmaxc, 0, np.float32),
            zmin=scat(zminc, 0, np.float32),
            zmean=scat((zsumc / np.maximum(cntc, 1)).astype(np.float32), 0, np.float32),
            cnt=scat(cntc, 0, np.float32),
            dyn=scat(np.clip(asumc / np.maximum(cntc, 1), 0, 1) * 255, 0, np.uint8),
            valid=valid.reshape(shp),
        )
        tile["cls_ref"] = FoveatedGrid.refine(tile, cell) if refine else tile["cls"]
        idx = np.flatnonzero(sel)
        ring["tile"] = tile
        ring["crop_half"] = half
        ring["pt_idx"] = idx
        ring["pt_ccid"] = ccid
        ref = tile["cls_ref"].reshape(-1)[ccid]
        keep = tile["cls_ref"].reshape(-1)[ccid] != 255
        pt_ref[idx[keep]] = ref[keep]
        ring["pt_ref"] = pt_ref
    return ring
