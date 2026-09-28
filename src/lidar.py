"""
Physically-motivated spinning-LiDAR simulator (Numba JIT).

Per sweep we cast one ray per (beam, azimuth) pair through a synthetic city:

  * the ground is a bilinearly-interpolated height field -> marched with a
    geometrically growing step and 14-step bisection,
  * objects (oriented boxes / z-cylinders / spheres) are intersected exactly,
    accelerated by a 2D uniform-grid DDA over the X/Y plane so that only the
    primitives in the columns the ray actually traverses are tested,
  * the nearest of {ground, object} wins  -> correct occlusion,
  * a rolling shutter is modelled: the ego carrier keeps moving while the
    mirror turns, so the ray origin is a function of azimuth,
  * returns carry range noise, an intensity that depends on material
    reflectivity, incidence angle and range, and a *range-dependent dropout*
    probability (this is what makes far field sparse and motivates foveation).
"""

from __future__ import annotations
import math
import numpy as np
from numba import njit, prange

from .config import SENSOR


# ==========================================================================
#   geometry kernels
# ==========================================================================
@njit(inline="always")
def _h_at(H, gres, x, y):
    n0 = H.shape[0]
    n1 = H.shape[1]
    fx = x / gres - 0.5
    fy = y / gres - 0.5
    i0 = int(math.floor(fx))
    j0 = int(math.floor(fy))
    tx = fx - i0
    ty = fy - j0
    if i0 < 0:
        i0 = 0
        tx = 0.0
    if j0 < 0:
        j0 = 0
        ty = 0.0
    if i0 > n1 - 2:
        i0 = n1 - 2
        tx = 1.0
    if j0 > n0 - 2:
        j0 = n0 - 2
        ty = 1.0
    h00 = H[j0, i0]
    h10 = H[j0, i0 + 1]
    h01 = H[j0 + 1, i0]
    h11 = H[j0 + 1, i0 + 1]
    return ((h00 * (1.0 - tx) + h10 * tx) * (1.0 - ty) +
            (h01 * (1.0 - tx) + h11 * tx) * ty)


@njit(inline="always")
def _lab_at(L, gres, x, y):
    n0 = L.shape[0]
    n1 = L.shape[1]
    i = int(x / gres)
    j = int(y / gres)
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    if i > n1 - 1:
        i = n1 - 1
    if j > n0 - 1:
        j = n0 - 1
    return L[j, i]


# ==========================================================================
#   main kernel
# ==========================================================================
@njit(parallel=True, fastmath=True, cache=False)
def _trace(ro, rd, H, L, M, gres, hmax,
           ptype, pa, pb, pyaw, plabel, pmat, pdyn, pmin, pmax,
           gcs, gnx, gny, col_start, col_items,
           max_range, min_range, rnd,
           ot, olab, odyn, oi, onx, ony, onz):
    N = ro.shape[0]
    W = H.shape[1] * gres
    nprim = ptype.shape[0]
    INF = 1.0e30

    for i in prange(N):
        ox = ro[i, 0]
        oy = ro[i, 1]
        oz = ro[i, 2]
        dx = rd[i, 0]
        dy = rd[i, 1]
        dz = rd[i, 2]

        best_t = INF
        best_lab = 255
        best_mat = 0.0
        best_dyn = 0
        bnx = 0.0
        bny = 0.0
        bnz = 1.0

        # ---------------- ground : height-field march --------------------
        if (dz < 0.0) or (oz < hmax):
            t = 0.0
            prev_t = 0.0
            z = oz
            h = _h_at(H, gres, ox, oy)
            prev_diff = z - h
            while t < max_range:
                dt = 0.16 + 0.018 * t
                if dt > 0.55:
                    dt = 0.55
                t += dt
                if t > max_range:
                    break
                px = ox + dx * t
                py = oy + dy * t
                if px < 0.0 or py < 0.0 or px >= W or py >= W:
                    break
                z = oz + dz * t
                h = _h_at(H, gres, px, py)
                diff = z - h
                if diff <= 0.0:
                    lo = prev_t
                    hi = t
                    lost = prev_diff
                    for _b in range(14):
                        mid = 0.5 * (lo + hi)
                        mz = oz + dz * mid
                        mh = _h_at(H, gres, ox + dx * mid, oy + dy * mid)
                        if mz - mh <= 0.0:
                            hi = mid
                        else:
                            lo = mid
                            lost = mz - mh
                    best_t = hi
                    gx = ox + dx * hi
                    gy = oy + dy * hi
                    eps = gres
                    hxp = _h_at(H, gres, gx + eps, gy)
                    hxm = _h_at(H, gres, gx - eps, gy)
                    hyp = _h_at(H, gres, gx, gy + eps)
                    hym = _h_at(H, gres, gx, gy - eps)
                    nx_ = -(hxp - hxm) / (2.0 * eps)
                    ny_ = -(hyp - hym) / (2.0 * eps)
                    inv = 1.0 / math.sqrt(nx_ * nx_ + ny_ * ny_ + 1.0)
                    bnx = nx_ * inv
                    bny = ny_ * inv
                    bnz = inv
                    gi = int(gx / gres)
                    gj = int(gy / gres)
                    if gi < 0:
                        gi = 0
                    if gj < 0:
                        gj = 0
                    if gi > L.shape[1] - 1:
                        gi = L.shape[1] - 1
                    if gj > L.shape[0] - 1:
                        gj = L.shape[0] - 1
                    best_lab = L[gj, gi]
                    mk = M[gj, gi]
                    best_mat = _MAT_OF(best_lab)
                    if mk == 1 or mk == 2:
                        best_mat = 0.88          # thermoplastic road paint
                    elif mk == 3:
                        best_mat = 0.82
                    best_dyn = 0
                    break
                prev_t = t
                prev_diff = diff
                if dz > 0.0 and z > hmax + 1.5:
                    break

        # ---------------- objects : 2D-grid DDA --------------------------
        adx = dx if dx >= 0.0 else -dx
        ady = dy if dy >= 0.0 else -dy
        ix = int((ox - 0.0) / gcs)
        iy = int((oy - 0.0) / gcs)
        if ix < 0:
            ix = 0
        elif ix > gnx - 1:
            ix = gnx - 1
        if iy < 0:
            iy = 0
        elif iy > gny - 1:
            iy = gny - 1

        if dx > 1e-12:
            stepx = 1
            tmx = ((ix + 1) * gcs - ox) / dx
            tdx = gcs / dx
        elif dx < -1e-12:
            stepx = -1
            tmx = (ix * gcs - ox) / dx
            tdx = -gcs / dx
        else:
            stepx = 0
            tmx = INF
            tdx = INF
        if dy > 1e-12:
            stepy = 1
            tmy = ((iy + 1) * gcs - oy) / dy
            tdy = gcs / dy
        elif dy < -1e-12:
            stepy = -1
            tmy = (iy * gcs - oy) / dy
            tdy = -gcs / dy
        else:
            stepy = 0
            tmy = INF
            tdy = INF

        tcur = 0.0
        guard = 0
        while guard < 4096:
            guard += 1
            if tcur > best_t or tcur > max_range:
                break
            cid = iy * gnx + ix
            for k in range(col_start[cid], col_start[cid + 1]):
                p = col_items[k]
                pt = ptype[p]
                hit = -1.0
                hnx = 0.0
                hny = 0.0
                hnz = 1.0
                if pt == 0:                                  # oriented box
                    yaw = pyaw[p]
                    c = math.cos(yaw)
                    s = math.sin(yaw)
                    ex = ox - pa[p, 0]
                    ey = oy - pa[p, 1]
                    ez = oz - pa[p, 2]
                    lx = c * ex + s * ey
                    ly = -s * ex + c * ey
                    lz = ez
                    ldx = c * dx + s * dy
                    ldy = -s * dx + c * dy
                    ldz = dz
                    hx = pb[p, 0]
                    hy = pb[p, 1]
                    hz = pb[p, 2]
                    t0 = 0.0
                    t1 = best_t
                    ok = True
                    axis = -1
                    sgn = 1.0
                    # x slab
                    if ldx > -1e-12 and ldx < 1e-12:
                        if lx < -hx or lx > hx:
                            ok = False
                    else:
                        inv = 1.0 / ldx
                        ta = (-hx - lx) * inv
                        tb = (hx - lx) * inv
                        sg = -1.0
                        if ta > tb:
                            tmp = ta
                            ta = tb
                            tb = tmp
                            sg = 1.0
                        if ta > t0:
                            t0 = ta
                            axis = 0
                            sgn = sg
                        if tb < t1:
                            t1 = tb
                        if t0 > t1:
                            ok = False
                    if ok:
                        if ldy > -1e-12 and ldy < 1e-12:
                            if ly < -hy or ly > hy:
                                ok = False
                        else:
                            inv = 1.0 / ldy
                            ta = (-hy - ly) * inv
                            tb = (hy - ly) * inv
                            sg = -1.0
                            if ta > tb:
                                tmp = ta
                                ta = tb
                                tb = tmp
                                sg = 1.0
                            if ta > t0:
                                t0 = ta
                                axis = 1
                                sgn = sg
                            if tb < t1:
                                t1 = tb
                            if t0 > t1:
                                ok = False
                    if ok:
                        if ldz > -1e-12 and ldz < 1e-12:
                            if lz < -hz or lz > hz:
                                ok = False
                        else:
                            inv = 1.0 / ldz
                            ta = (-hz - lz) * inv
                            tb = (hz - lz) * inv
                            sg = -1.0
                            if ta > tb:
                                tmp = ta
                                ta = tb
                                tb = tmp
                                sg = 1.0
                            if ta > t0:
                                t0 = ta
                                axis = 2
                                sgn = sg
                            if tb < t1:
                                t1 = tb
                            if t0 > t1:
                                ok = False
                    if ok and t0 > 1e-4 and t0 < best_t:
                        hit = t0
                        if axis == 0:
                            hnx = sgn
                            hny = 0.0
                            hnz = 0.0
                        elif axis == 1:
                            hnx = 0.0
                            hny = sgn
                            hnz = 0.0
                        else:
                            hnx = 0.0
                            hny = 0.0
                            hnz = sgn
                        # rotate normal back to world
                        rx = c * hnx - s * hny
                        ry = s * hnx + c * hny
                        hnx = rx
                        hny = ry
                elif pt == 1:                                # z-cylinder
                    ax_ = pa[p, 0]
                    ay_ = pa[p, 1]
                    z0 = pa[p, 2]
                    z1 = pb[p, 2]
                    r = pb[p, 0]
                    a = dx * dx + dy * dy
                    ex = ox - ax_
                    ey = oy - ay_
                    if a > 1e-14:
                        b = ex * dx + ey * dy
                        cc = ex * ex + ey * ey - r * r
                        disc = b * b - a * cc
                        if disc >= 0.0:
                            sq = math.sqrt(disc)
                            t1r = (-b - sq) / a
                            t2r = (-b + sq) / a
                            for _it in range(2):
                                tt = t1r if _it == 0 else t2r
                                if tt > 1e-4 and tt < best_t:
                                    zz = oz + dz * tt
                                    if zz >= z0 and zz <= z1:
                                        if hit < 0.0 or tt < hit:
                                            hit = tt
                                            px = ox + dx * tt
                                            py = oy + dy * tt
                                            hnx = (px - ax_) / r
                                            hny = (py - ay_) / r
                                            hnz = 0.0
                            if dz > 1e-12 or dz < -1e-12:
                                for _ic in range(2):
                                    zc = z0 if _ic == 0 else z1
                                    tc = (zc - oz) / dz
                                    if tc > 1e-4 and tc < best_t:
                                        px = ox + dx * tc - ax_
                                        py = oy + dy * tc - ay_
                                        if px * px + py * py <= r * r:
                                            if hit < 0.0 or tc < hit:
                                                hit = tc
                                                hnx = 0.0
                                                hny = 0.0
                                                hnz = 1.0 if dz < 0 else -1.0
                else:                                        # sphere
                    cx_ = pa[p, 0]
                    cy_ = pa[p, 1]
                    cz_ = pa[p, 2]
                    r = pb[p, 0]
                    ex = ox - cx_
                    ey = oy - cy_
                    ez = oz - cz_
                    b = ex * dx + ey * dy + ez * dz
                    cc = ex * ex + ey * ey + ez * ez - r * r
                    disc = b * b - cc
                    if disc >= 0.0:
                        sq = math.sqrt(disc)
                        tt = -b - sq
                        if tt <= 1e-4:
                            tt = -b + sq
                        if tt > 1e-4 and tt < best_t:
                            hit = tt
                            px = ox + dx * tt
                            py = oy + dy * tt
                            pz = oz + dz * tt
                            hnx = (px - cx_) / r
                            hny = (py - cy_) / r
                            hnz = (pz - cz_) / r

                if hit >= 0.0 and hit < best_t:
                    best_t = hit
                    best_lab = plabel[p]
                    best_mat = pmat[p]
                    best_dyn = pdyn[p]
                    bnx = hnx
                    bny = hny
                    bnz = hnz

            if stepx == 0 and stepy == 0:
                break
            if tmx < tmy:
                tcur = tmx
                ix += stepx
                if ix < 0 or ix > gnx - 1:
                    break
                tmx += tdx
            else:
                tcur = tmy
                iy += stepy
                if iy < 0 or iy > gny - 1:
                    break
                tmy += tdy

        # ---------------- measurement model ------------------------------
        if best_t >= max_range or best_lab == 255:
            ot[i] = -1.0
            continue

        ci = bnx * (-dx) + bny * (-dy) + bnz * (-dz)
        if ci < 0.0:
            ci = -ci
        if ci > 1.0:
            ci = 1.0

        # range-dependent detection probability
        base = 1.0
        if best_t > 30.0:
            base = math.exp(-(best_t - 30.0) / 95.0)
        mat = 0.88 + 0.35 * best_mat
        if mat > 1.0:
            mat = 1.0
        if mat < 0.62:
            mat = 0.62
        geom = 0.80 + 0.20 * ci
        pkeep = base * mat * geom
        if pkeep > 0.97:
            pkeep = 0.97
        if rnd[i, 0] > pkeep:
            ot[i] = -1.0
            continue

        sigma = 0.012 + 0.0009 * best_t
        tmeas = best_t + sigma * rnd[i, 1]
        if tmeas < min_range:
            ot[i] = -1.0
            continue

        ot[i] = tmeas
        olab[i] = best_lab
        odyn[i] = best_dyn
        inten = best_mat * (0.35 + 0.65 * ci) * (30.0 / (30.0 + best_t)) * 2.4
        inten += 0.02 * rnd[i, 2]
        if inten > 1.0:
            inten = 1.0
        if inten < 0.0:
            inten = 0.0
        oi[i] = inten
        onx[i] = bnx
        ony[i] = bny
        onz[i] = bnz


@njit(inline="always")
def _MAT_OF(lab):
    # ground-material reflectivity table (mirrors CLASS_REFL in config.py)
    if lab == 0:
        return 0.20     # road / asphalt
    if lab == 1:
        return 0.13     # pothole (loose, dark)
    if lab == 2:
        return 0.45     # sidewalk / concrete
    if lab == 3:
        return 0.42     # curb
    if lab == 4:
        return 0.30     # grass / dirt
    if lab == 5:
        return 0.55     # structure
    if lab == 6:
        return 0.62     # pole
    if lab == 7:
        return 0.35     # vegetation
    if lab == 8:
        return 0.68     # vehicle body
    if lab == 9:
        return 0.34     # pedestrian clothing
    return 0.46         # cyclist


# ==========================================================================
#   python-side driver
# ==========================================================================
def build_ray_batch(pose, sensor=SENSOR):
    """Ray origins/directions for one sweep, including the rolling shutter."""
    x, y, yaw = pose
    nb = sensor["n_beams"]
    nc = int(round(360.0 / sensor["az_res"]))
    u = np.linspace(0.0, 1.0, nb)
    elev = (sensor["fov_down"] +
            (sensor["fov_up"] - sensor["fov_down"]) * (1.0 - (1.0 - u) ** 1.5))
    elev = np.deg2rad(elev).astype(np.float32)
    az = np.deg2rad(sensor["az_start"] + sensor["az_res"] * np.arange(nc)).astype(np.float32)

    E, A = np.meshgrid(elev, az, indexing="ij")
    E = E.ravel()
    A = A.ravel()
    N = E.size

    ce, se = np.cos(E), np.sin(E)
    ca, sa = np.cos(A), np.sin(A)
    dxe = (ce * ca).astype(np.float32)
    dye = (ce * sa).astype(np.float32)
    dze = se.astype(np.float32)

    cy, sy = math.cos(yaw), math.sin(yaw)
    rd = np.empty((N, 3), dtype=np.float32)
    rd[:, 0] = dxe * cy - dye * sy
    rd[:, 1] = dxe * sy + dye * cy
    rd[:, 2] = dze

    # rolling shutter: the carrier advances while the mirror turns
    tj = (np.repeat(np.arange(nc), nb) / nc - 0.5) * sensor["sweep_time"]
    ro = np.empty((N, 3), dtype=np.float32)
    ro[:, 2] = sensor["height"]
    return ro, rd, tj, AzimuthHelper(tj.shape[0])


class AzimuthHelper:  # trivial container kept for readability
    def __init__(self, n):
        self.n = n


def build_col_grid(pmin, pmax, gcs, size):
    """CSR bucket list: for every X/Y column, the primitives that overlap it."""
    gnx = int(math.ceil(size / gcs))
    gny = gnx
    ncols = gnx * gny
    buckets = [[] for _ in range(ncols)]
    for p in range(pmin.shape[0]):
        i0 = int(max(0, np.floor(pmin[p, 0] / gcs)))
        i1 = int(min(gnx - 1, np.floor(pmax[p, 0] / gcs)))
        j0 = int(max(0, np.floor(pmin[p, 1] / gcs)))
        j1 = int(min(gny - 1, np.floor(pmax[p, 1] / gcs)))
        if i1 < 0 or j1 < 0 or i0 > gnx - 1 or j0 > gny - 1:
            continue
        for j in range(j0, j1 + 1):
            base = j * gnx
            for i in range(i0, i1 + 1):
                buckets[base + i].append(p)
    col_start = np.zeros(ncols + 1, dtype=np.int32)
    for c in range(ncols):
        col_start[c + 1] = col_start[c] + len(buckets[c])
    col_items = np.empty(col_start[ncols], dtype=np.int32)
    k = 0
    for c in range(ncols):
        for p in buckets[c]:
            col_items[k] = p
            k += 1
    return col_start, col_items, gnx, gny


class Simulator:
    """Owns one world and renders sweeps from it."""

    def __init__(self, world, sensor=SENSOR, gcs=8.0, seed=0):
        self.w = world
        self.S = sensor
        self.gcs = gcs
        self.rng = np.random.default_rng(seed + 977)
        st = world["static"]
        self.static = st
        self.col_start, self.col_items, self.gnx, self.gny = build_col_grid(
            st["pmin"], st["pmax"], gcs, world["size"])
        self.hmax = float(world["H"].max())
        self._warmup()

    def _warmup(self):
        """Trigger JIT compilation with a tiny synthetic call."""
        n = 64
        ro = np.zeros((n, 3), dtype=np.float32)
        rd = np.zeros((n, 3), dtype=np.float32)
        rd[:, 0] = 1.0
        H = np.zeros((8, 8), dtype=np.float32)
        L = np.zeros((8, 8), dtype=np.uint8)
        M = np.zeros((8, 8), dtype=np.uint8)
        st = self.static
        out = [np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.uint8),
               np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.float32),
               np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32),
               np.zeros(n, dtype=np.float32)]
        _trace(ro, rd, H, L, M, np.float32(1.0), np.float32(1.0),
               st["type"], st["a"], st["b"], st["yaw"], st["label"], st["refl"],
               st["dyn"], st["pmin"], st["pmax"],
               np.float32(self.gcs), self.gnx, self.gny,
               self.col_start, self.col_items, np.float32(5.0), np.float32(0.0),
               np.random.random((n, 3)).astype(np.float32), *out)

    def sweep(self, frame_idx, ego_z, with_agents=True):
        w = self.w
        S = self.S
        gres = w["gres"]
        pose = w["poses"][frame_idx]
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        t = frame_idx * w["dt"]

        ro, rd, tj, _ = build_ray_batch((x, y, yaw), S)
        # ego velocity -> rolling shutter displacement
        if frame_idx + 1 < len(w["poses"]):
            nxp = w["poses"][frame_idx + 1]
        else:
            nxp = w["poses"][max(frame_idx - 1, 0)]
        vx = (nxp[0] - x) / w["dt"]
        vy = (nxp[1] - y) / w["dt"]
        ro[:, 0] = x + vx * tj
        ro[:, 1] = y + vy * tj
        ro[:, 2] = ego_z + S["height"]

        # refresh moving agents into primitives for this instant
        from .worldgen import PrimSet, agents_at_time
        P = PrimSet()
        if with_agents:
            agents_at_time(w["agents"], t + S["sweep_time"] * 0.5, w["H"], gres, P)
        dyn = P.arrays()
        st = self.static
        if dyn["type"].size:
            ptype = np.concatenate([st["type"], dyn["type"]])
            pa = np.concatenate([st["a"], dyn["a"]]).astype(np.float32)
            pb = np.concatenate([st["b"], dyn["b"]]).astype(np.float32)
            pyaw = np.concatenate([st["yaw"], dyn["yaw"]]).astype(np.float32)
            plabel = np.concatenate([st["label"], dyn["label"]])
            pmat = np.concatenate([st["refl"], dyn["refl"]]).astype(np.float32)
            pdyn = np.concatenate([st["dyn"], dyn["dyn"]])
            pmin = np.concatenate([st["pmin"], dyn["pmin"]]).astype(np.float32)
            pmax = np.concatenate([st["pmax"], dyn["pmax"]]).astype(np.float32)
        else:
            (ptype, pa, pb, pyaw, plabel, pmat, pdyn, pmin, pmax) = (
                st["type"], st["a"], st["b"], st["yaw"], st["label"],
                st["refl"], st["dyn"], st["pmin"], st["pmax"])

        col_start, col_items, gnx, gny = build_col_grid(pmin, pmax, self.gcs, w["size"])

        N = ro.shape[0]
        rnd = self.rng.standard_normal((N, 3)).astype(np.float32)
        # rnd[:,1] is used as a gaussian for range noise -> convert col 0 to uniform
        rnd[:, 0] = self.rng.random(N).astype(np.float32)

        ot = np.full(N, -1.0, dtype=np.float32)
        olab = np.zeros(N, dtype=np.uint8)
        odyn = np.zeros(N, dtype=np.uint8)
        oi = np.zeros(N, dtype=np.float32)
        onx = np.zeros(N, dtype=np.float32)
        ony = np.zeros(N, dtype=np.float32)
        onz = np.zeros(N, dtype=np.float32)

        _trace(ro, rd, w["H"], w["L"], w["M"], np.float32(gres), np.float32(self.hmax),
               ptype, pa, pb, pyaw, plabel, pmat, pdyn, pmin, pmax,
               np.float32(self.gcs), gnx, gny, col_start, col_items,
               np.float32(S["max_range"]), np.float32(S["min_range"]), rnd,
               ot, olab, odyn, oi, onx, ony, onz)

        good = ot > 0.0
        pw = ro[good] + rd[good] * ot[good][:, None]
        # to ego frame (x forward, y left, z up) at the mid-sweep pose
        cy, sy = math.cos(yaw), math.sin(yaw)
        ex = pw[:, 0] - x
        ey = pw[:, 1] - y
        ez = pw[:, 2] - ego_z
        pts = np.empty_like(pw)
        pts[:, 0] = ex * cy + ey * sy
        pts[:, 1] = -ex * sy + ey * cy
        pts[:, 2] = ez

        out = dict(
            xyz=pts.astype(np.float32),
            label=olab[good],
            dyn=odyn[good],
            intensity=(oi[good] * 255.0).astype(np.uint8),
            normal=np.stack([onx[good], ony[good], onz[good]], 1).astype(np.float32),
            pose=np.array([x, y, ego_z, yaw], dtype=np.float64),
            t=t,
        )
        return out
