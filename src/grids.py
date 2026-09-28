"""
Grid engine.

Two tightly-coupled data structures:

1. ``rasterize_bev``  - projects a point cloud onto one square BEV crop of the
   foveated pyramid, producing an 8-channel feature tensor plus a per-cell
   majority semantic label and a validity mask.

2. ``FoveatedGrid``   - the deliverable 2.5D representation.  Points are binned
   into concentric rings whose cell edge grows with distance from the sensor
   (5 cm -> 50 cm).  Every ring is a fixed TILE x TILE lattice so the whole
   map is a small set of dense arrays - no hashes, no trees, O(1) lookup, and
   no alignment error because ring boundaries are computed from the same
   continuous radius function used to route each point.

   Per cell we keep the semantic class, the classifier confidence, min/max/
   mean elevation (=> curb height, pothole depth and overhang clearance),
   the number of returns and the fraction of returns coming from a moving
   agent.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter, zoom

from . import config as C


# ==========================================================================
#  1.  BEV pyramid rasterisation
# ==========================================================================
def rasterize_bev(xy, z, inten, label, extent, px, num_classes=C.NUM_CLASSES,
                  z_range=C.Z_RANGE, span_clip=C.Z_SPAN_CLIP):
    """Rasterise one pyramid level.

    Returns
    -------
    feat  (8, px, px) float32   normalised network input
    lab   (px, px) int64        majority semantic class (majority = -1)
    valid (px, px) bool         cell contains at least one return
    """
    res = extent / px
    n = px * px
    ix = np.clip(((xy[:, 0] + extent * 0.5) / res).astype(np.int64), 0, px - 1)
    iy = np.clip(((xy[:, 1] + extent * 0.5) / res).astype(np.int64), 0, px - 1)
    cid = iy * px + ix

    cnt = np.bincount(cid, minlength=n).astype(np.float32)
    valid = cnt > 0
    zsum = np.bincount(cid, weights=z, minlength=n).astype(np.float32)
    isum = np.bincount(cid, weights=inten.astype(np.float32), minlength=n)

    NEG, POS = -1e30, 1e30
    zmax = np.full(n, NEG, np.float32)
    zmin = np.full(n, POS, np.float32)
    np.maximum.at(zmax, cid, z)
    np.minimum.at(zmin, cid, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        zmean = np.where(valid, zsum / np.maximum(cnt, 1), 0.0)

    # "groundness": fraction of returns that sit close to the cell floor
    near = (z <= (zmin[cid] + 0.25)).astype(np.float32)
    gnd = np.bincount(cid, weights=near, minlength=n) / np.maximum(cnt, 1)

    # majority label -------------------------------------------------------
    cc = np.bincount(cid.astype(np.int64) * num_classes + label.astype(np.int64),
                     minlength=n * num_classes)
    cc = cc.reshape(n, num_classes)
    lab = np.argmax(cc, axis=1).astype(np.int64)
    lab[~valid] = -1

    lo, hi = z_range
    inv = 1.0 / (hi - lo)
    feat = np.empty((C.BEV_CHANNELS, px, px), dtype=np.float32)
    feat[0] = np.clip((zmax - lo) * inv, 0, 1).reshape(px, px)
    feat[1] = np.clip((zmean - lo) * inv, 0, 1).reshape(px, px)
    feat[2] = np.clip((zmin - lo) * inv, 0, 1).reshape(px, px)
    feat[3] = np.clip((zmax - zmin) / span_clip, 0, 1).reshape(px, px)
    feat[4] = (np.log1p(cnt) / np.log1p(256.0)).reshape(px, px)
    feat[5] = np.clip(isum / np.maximum(cnt, 1) / 255.0, 0, 1).reshape(px, px)
    feat[6] = gnd.reshape(px, px)
    feat[7] = 0.0                      # level code, filled by the caller
    return feat, lab.reshape(px, px), valid.reshape(px, px)


def pyramid_from_frame(xy, z, inten, label, augment=None):
    """Rasterise all three pyramid levels for one sweep (optionally augmented)."""
    if augment is not None:
        xy, z = augment(xy, z)
    outs = []
    for k, (extent, px, res) in enumerate(C.PYRAMID):
        f, lab, val = rasterize_bev(xy, z, inten, label, extent, px)
        f[7] = k / max(C.PYR_LEVELS - 1, 1)
        outs.append((f, lab, val))
    return outs


# ==========================================================================
#  2.  Foveated 2.5D grid
# ==========================================================================
class FoveatedGrid:
    """Variable-resolution elevation + semantics map centred on the sensor."""

    def __init__(self, rings=C.GRID_RINGS, tile=C.RING_TILE):
        self.rings = rings
        self.tile = tile
        self.n_ring = len(rings)

    # ------------------------------------------------------------------
    def project(self, xy, z, cls, prob=None, dyn=None, num_classes=C.NUM_CLASSES,
                need_pt_cls=False):
        """Bin labelled points into the ring lattice.

        cls/prob/dyn are *per point* (already transferred from the network).
        """
        tile = self.tile
        r = np.hypot(xy[:, 0], xy[:, 1])
        prob = np.full(len(z), 1.0, np.float32) if prob is None else prob
        dyn = np.zeros(len(z), np.uint8) if dyn is None else dyn
        out = []
        for k, (r_in, r_out, cell) in enumerate(self.rings):
            m = (r >= r_in) & (r < r_out)
            if not m.any():
                out.append(self._empty_ring(num_classes))
                continue
            x = xy[m, 0]
            y = xy[m, 1]
            zz = z[m]
            cc = cls[m]
            pp = prob[m]
            dd = dyn[m]
            ix = np.clip(((x + r_out) / cell).astype(np.int64), 0, tile - 1)
            iy = np.clip(((y + r_out) / cell).astype(np.int64), 0, tile - 1)
            cid = (iy * tile + ix).astype(np.int64)
            n = tile * tile

            # Aggregate over *occupied* cells only.  The naive version
            # allocates minlength=n*K counters for the class histogram on
            # every ring (5 x 1.76 M int64 = 70 MB of allocation + zeroing
            # per frame) and np.maximum.at over the full lattice; sorting the
            # cell ids once and using reduceat does the same job on ~40 k
            # cells with a single pass.
            order = np.argsort(cid, kind="stable")
            cs = cid[order]
            if cs.size:
                bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
                u = cs[bnd]
                cnts = np.diff(np.r_[bnd, cs.size])
            else:
                bnd = np.zeros(0, np.int64)
                u = np.zeros(0, np.int64)
                cnts = np.zeros(0, np.int64)

            # Dense arrays start at the "empty" sentinel (0 for every
            # continuous plane, 255 for the class), so occupied cells can be
            # scattered straight in and the six full-lattice np.where() calls
            # that used to follow are unnecessary.
            cnt = np.zeros(n, np.float32)
            zmax = np.zeros(n, np.float32)
            zmin = np.zeros(n, np.float32)
            zmean = np.zeros(n, np.float32)
            prob_d = np.zeros(n, np.uint8)
            dyn_d = np.zeros(n, np.uint8)
            lab_dense = np.full(n, 255, np.uint8)
            pt_cls = np.zeros(cc.size, np.uint8) if need_pt_cls else None

            if u.size:
                c_ = cnts.astype(np.float32)
                cnt[u] = c_
                zo = zz[order]
                zmax[u] = np.maximum.reduceat(zo, bnd)
                zmin[u] = np.minimum.reduceat(zo, bnd)
                zmean[u] = np.add.reduceat(zo, bnd) / c_
                prob_d[u] = np.clip(np.add.reduceat(pp[order], bnd) / c_, 0, 1
                                    ).astype(np.float32) * 255
                dyn_d[u] = np.clip(np.add.reduceat(dd[order].astype(np.float32), bnd) / c_,
                                   0, 1) * 255
                inv = np.repeat(np.arange(u.size), cnts)
                ccw = np.bincount(inv * num_classes + cc[order],
                                  minlength=u.size * num_classes).reshape(u.size, num_classes)
                lab = np.argmax(ccw, axis=1).astype(np.uint8)
                lab_dense[u] = lab
                if need_pt_cls:
                    pt_cls[order] = lab[inv]

            shp = (tile, tile)
            cell_d = dict(
                cls=lab_dense.reshape(shp),
                prob=prob_d.reshape(shp),
                zmax=zmax.reshape(shp),
                zmin=zmin.reshape(shp),
                zmean=zmean.reshape(shp),
                cnt=cnt.reshape(shp),
                dyn=dyn_d.reshape(shp),
                valid=(cnt > 0).reshape(shp),
            )
            cell_d["pt_idx"] = np.flatnonzero(m)
            cell_d["pt_cid"] = cid
            if need_pt_cls:
                cell_d["pt_cls"] = pt_cls
            out.append(cell_d)
        return out

    @staticmethod
    def scatter(rings, n_points, key="cls_ref"):
        """Scatter a per-cell array back onto the points that filled the cell."""
        out = np.full(n_points, -1, np.int64)
        for rg in rings:
            idx = rg.get("pt_idx")
            if idx is None or len(idx) == 0:
                continue
            flat = np.asarray(rg[key]).reshape(-1)
            out[idx] = flat[rg["pt_cid"]]
        return out

    def _empty_ring(self, num_classes):
        shp = (self.tile, self.tile)
        return dict(cls=np.full(shp, 255, np.uint8), prob=np.zeros(shp, np.uint8),
                    zmax=np.zeros(shp, np.float32), zmin=np.zeros(shp, np.float32),
                    zmean=np.zeros(shp, np.float32), cnt=np.zeros(shp, np.float32),
                    dyn=np.zeros(shp, np.uint8), valid=np.zeros(shp, bool))

    # ------------------------------------------------------------------
    @staticmethod
    @staticmethod
    def _curb_mask(post, z, valid, base, depth, pot_i, cell, ground_cls):
        """Cells that sit on the lip of a kerb-sized elevation step."""
        k = int(round(post["curb_max_width"] / cell))
        k = max(3, k | 1)
        zf = np.where(valid, z, 1e30)
        zfloor = minimum_filter(zf, size=k, mode="constant", cval=1e30)
        rise = z - zfloor
        is_terr = np.isin(base, np.asarray(ground_cls, dtype=np.uint8)) & (base != pot_i)
        # a cell that is itself below the road surface is part of a
        # depression, not the lip of a kerb
        return (valid & is_terr & (rise >= post["curb_min_step"])
                & (rise <= post["curb_max_height"])
                & (depth < 0.5 * post["curb_min_step"]))

    def refine(ring, cell, ground_cls=C.TERRAIN_CLASSES, post=None):
        """Geometric refinement that only the fine rings can afford.

        * curb    - the upper cell of a kerb-sized elevation step
                    (off by default, see C.POST["curb_enable"])
        * pothole - a cell sitting >`pothole_min_depth` below the *local road
                    surface*, where that surface is estimated from road-class
                    cells only (using a plain mean over the whole window would
                    be dragged up by the raised sidewalk next to the road and
                    would flag every kerbside cell as a pothole).

        Runs on the 2.5D elevation layer, so it is unavailable to any
        representation coarser than the feature it is trying to detect.
        """
        post = post or C.POST
        curb_i = C.CLASS_NAMES.index("curb")
        road_i = C.CLASS_NAMES.index("road")
        pot_i = C.CLASS_NAMES.index("pothole")
        valid = ring["valid"]
        z = np.where(valid, ring["zmean"], np.nan).astype(np.float32)
        base = ring["cls"]
        out = base.copy()

        # ---- local road surface (robust: road-class cells only) --------
        step = max(1, int(round(0.5 / cell)))
        small = z[::step, ::step]
        rmask = (base[::step, ::step] == road_i) | (base[::step, ::step] == pot_i)
        good = np.isfinite(small) & rmask
        num = uniform_filter(np.where(good, small, 0.0), size=9, mode="nearest")
        den = uniform_filter(good.astype(np.float32), size=9, mode="nearest")
        med = num / np.maximum(den, 1e-3)
        ref = np.kron(med, np.ones((step, step), np.float32))[:z.shape[0], :z.shape[1]]
        depth = ref - z

        # ---- potholes: coherent depression below the road surface ------
        thr = float(post["pothole_min_depth"])
        pot = np.zeros(z.shape, bool)
        if thr < 1.0:
            is_road = (out == road_i) | (out == pot_i)
            pot = valid & is_road & (depth >= thr)
            if post.get("pothole_min_cells", 1) > 1:
                # kill isolated noise spikes: keep only cells whose whole
                # neighbourhood is also depressed
                pot &= minimum_filter(pot.astype(np.uint8), size=3,
                                      mode="nearest").astype(bool)

        # ---- curbs: elevation rise above the local floor ---------------
        # disabled by default (see C.POST["curb_enable"])
        if post.get("curb_enable", 0):
            out = np.where(FoveatedGrid._curb_mask(post, z, valid, base, depth,
                                                   pot_i, cell, ground_cls),
                           curb_i, out)
        out = np.where(pot, pot_i, out)
        return out


# ==========================================================================
#  3.  baselines + memory accounting
# ==========================================================================
def uniform_occupancy(xy, cell, extent=C.MAP_EXTENT):
    """Number of distinct cells a *uniform* grid of `cell` metres would need."""
    ix = np.floor((xy[:, 0] + extent * 0.5) / cell).astype(np.int64)
    iy = np.floor((xy[:, 1] + extent * 0.5) / cell).astype(np.int64)
    n = int(extent / cell)
    m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    if not m.any():
        return 0
    return int(np.unique(iy[m] * n + ix[m]).size)


def memory_report(frame_stats, bytes_per_cell=8):
    """Adaptive vs uniform occupancy / allocation cost for one sweep."""
    rep = {}
    adapt_occ = sum(s["occupied"] for s in frame_stats)
    adapt_alloc = sum(s["allocated"] for s in frame_stats)
    rep["adaptive_occupied"] = adapt_occ
    rep["adaptive_allocated"] = adapt_alloc
    rep["adaptive_bytes"] = adapt_alloc * bytes_per_cell
    for name, cell in C.BASELINES.items():
        n = int(C.MAP_EXTENT / cell) ** 2
        rep[f"{name}_allocated"] = n
        rep[f"{name}_bytes"] = n * bytes_per_cell
        rep[f"{name}_occupied"] = None        # filled by the caller
    rep["bytes_per_cell"] = bytes_per_cell
    return rep


def ring_stats(rings):
    out = []
    for k, r in enumerate(rings):
        occ = int(r["valid"].sum())
        out.append(dict(ring=k, r_in=C.GRID_RINGS[k][0], r_out=C.GRID_RINGS[k][1],
                        cell=C.GRID_RINGS[k][2], occupied=occ,
                        allocated=r["valid"].size))
    return out


# ==========================================================================
#  4.  point <-> grid transfer helpers
# ==========================================================================
def level_for_radius(r):
    """Which pyramid level is authoritative at radial distance r?

    The *finest* level whose crop still covers the point.  Pyramid levels are
    ordered by increasing extent, so iterate from the outside in - otherwise
    every near point would satisfy the coarse tests too and be handed to the
    periphery, throwing away the fovea entirely.
    """
    r = np.asarray(r)
    lvl = np.full(r.shape, C.PYR_LEVELS - 1, dtype=np.int64)
    cov = [p[0] * 0.5 for p in C.PYRAMID]
    for k in range(C.PYR_LEVELS - 2, -1, -1):
        lvl = np.where(r < cov[k], k, lvl)
    return lvl


def sample_logits(logits, xy, extent, px):
    """Fetch the network logit vector for every point from one level's map."""
    res = extent / px
    ix = np.clip(((xy[:, 0] + extent * 0.5) / res).astype(np.int64), 0, px - 1)
    iy = np.clip(((xy[:, 1] + extent * 0.5) / res).astype(np.int64), 0, px - 1)
    return logits[:, iy, ix].T                      # (N, K)
