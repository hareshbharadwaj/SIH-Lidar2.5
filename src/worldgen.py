"""
Procedural synthetic-city generator.

Produces, for a given random seed:

  * a ground height-field  H   (float32, metres)
  * a ground label map     L   (uint8, semantic class of the surface)
  * a road-marking map     M   (uint8, 0 none / 1 lane / 2 crosswalk / 3 stop line)
  * a set of geometric primitives (oriented boxes, z-cylinders, spheres)
    with per-primitive semantic label, reflectivity and a "dynamic" flag
  * a set of moving agents (vehicles / pedestrians / cyclists) that are
    re-instantiated into primitives at every time step
  * an ego trajectory (polyline + headings) for the sensor carrier

The generator is deliberately physical: the road network follows a smooth
terrain profile, sidewalks are raised 15 cm, curbs appear as a one-cell
(10 cm) elevation step, potholes are elliptical depressions and speed
bumps are raised gaussian ridges.  All of it is what the 2.5D elevation
layer has to recover.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from .config import DATASET, CLASS_NAMES

BOX, CYL, SPH = 0, 1, 2

LBL = {n: i for i, n in enumerate(CLASS_NAMES)}

LANE_OFF = 2.25     # lane centre offset from the avenue centre-line
CY_OFF = 3.55       # cycle lane (kerbside, same side as the ego lane)


# ==========================================================================
# small helpers
# ==========================================================================
def _value_noise(rng, n, base, octaves=4, persistence=0.55):
    """Smooth fractal value-noise on an n x n lattice, zero-mean, ~unit var."""
    out = np.zeros((n, n), dtype=np.float32)
    amp = 1.0
    tot = 0.0
    for o in range(octaves):
        b = max(3, int(base * (2 ** o)))
        if b > n:
            break
        f = rng.standard_normal((b, b)).astype(np.float32)
        z = zoom(f, n / b, order=1)
        out += amp * z
        tot += amp
        amp *= persistence
    out /= max(tot, 1e-6)
    return out / (out.std() + 1e-6)


def _sample(H, x, y, gres):
    """Bilinear-ish (nearest) sample of the height field at world (x, y)."""
    ix = np.clip((np.asarray(x) / gres).astype(np.int64), 0, H.shape[1] - 1)
    iy = np.clip((np.asarray(y) / gres).astype(np.int64), 0, H.shape[0] - 1)
    return H[iy, ix]


# ==========================================================================
# primitive accumulator
# ==========================================================================
class PrimSet:
    def __init__(self):
        self.t = []      # type
        self.a = []      # vec3
        self.b = []      # vec3
        self.yaw = []
        self.label = []
        self.dyn = []
        self.refl = []

    def box(self, cx, cy, cz, hx, hy, hz, yaw, label, refl, dyn=0):
        self.t.append(BOX)
        self.a.append((cx, cy, cz))
        self.b.append((hx, hy, hz))
        self.yaw.append(yaw)
        self.label.append(label)
        self.refl.append(refl)
        self.dyn.append(dyn)

    def cyl(self, cx, cy, z0, z1, r, label, refl, dyn=0):
        self.t.append(CYL)
        self.a.append((cx, cy, z0))
        self.b.append((r, r, z1))
        self.yaw.append(0.0)
        self.label.append(label)
        self.refl.append(refl)
        self.dyn.append(dyn)

    def sph(self, cx, cy, cz, r, label, refl, dyn=0):
        self.t.append(SPH)
        self.a.append((cx, cy, cz))
        self.b.append((r, r, r))
        self.yaw.append(0.0)
        self.label.append(label)
        self.refl.append(refl)
        self.dyn.append(dyn)

    def arrays(self):
        t = np.asarray(self.t, dtype=np.uint8)
        a = np.asarray(self.a, dtype=np.float32)
        b = np.asarray(self.b, dtype=np.float32)
        yaw = np.asarray(self.yaw, dtype=np.float32)
        lab = np.asarray(self.label, dtype=np.uint8)
        dyn = np.asarray(self.dyn, dtype=np.uint8)
        refl = np.asarray(self.refl, dtype=np.float32)
        pmin = np.empty_like(a)
        pmax = np.empty_like(a)
        m = t == BOX
        if m.any():
            hy_ = np.abs(b[m, 0] * np.cos(yaw[m])) + np.abs(b[m, 1] * np.sin(yaw[m]))
            hx_ = np.abs(b[m, 0] * np.sin(yaw[m])) + np.abs(b[m, 1] * np.cos(yaw[m]))
            pmin[m, 0] = a[m, 0] - hx_
            pmax[m, 0] = a[m, 0] + hx_
            pmin[m, 1] = a[m, 1] - hy_
            pmax[m, 1] = a[m, 1] + hy_
            pmin[m, 2] = a[m, 2] - b[m, 2]
            pmax[m, 2] = a[m, 2] + b[m, 2]
        m = t == CYL
        if m.any():
            pmin[m, 0] = a[m, 0] - b[m, 0]
            pmax[m, 0] = a[m, 0] + b[m, 0]
            pmin[m, 1] = a[m, 1] - b[m, 1]
            pmax[m, 1] = a[m, 1] + b[m, 1]
            pmin[m, 2] = a[m, 2]
            pmax[m, 2] = b[m, 2]
        m = t == SPH
        if m.any():
            pmin[m] = a[m] - b[m]
            pmax[m] = a[m] + b[m]
        return dict(type=t, a=a, b=b, yaw=yaw, label=lab, dyn=dyn,
                    refl=refl, pmin=pmin, pmax=pmax)


# ==========================================================================
# ground : height field + label map + marking map
# ==========================================================================
def layout_roads(rng, size, lo=35.0, mn=52.0, mx=78.0):
    """Irregular grid of avenues -> centre-line coordinates."""
    def lines():
        out = []
        v = lo + rng.uniform(0, mn)
        while v < size - lo - mn:
            out.append(float(v))
            v += rng.uniform(mn, mx)
        return np.array(out, dtype=np.float32)
    return lines(), lines()


def build_ground(rng, size, gres, xl, yl, focus=None):
    n = int(round(size / gres))
    xs = (np.arange(n, dtype=np.float32) + 0.5) * gres
    ys = xs.copy()

    # --- road network : irregular grid of avenues -------------------------
    HW = 4.5                       # half road width  -> 9 m carriageway
    SW = 2.4                       # sidewalk width

    dx = np.min(np.abs(xs[:, None] - xl[None, :]), axis=1)   # (nx,)
    dy = np.min(np.abs(ys[:, None] - yl[None, :]), axis=1)   # (ny,)
    DX = dx[None, :]
    DY = dy[:, None]

    # --- terrain ----------------------------------------------------------
    p1, p2, p3, p4 = rng.uniform(0, 6.28, 4)
    Ax = (1.05 * np.sin(xs * 0.0075 + p1) + 0.45 * np.sin(xs * 0.019 + p3)).astype(np.float32)
    Ay = (0.90 * np.cos(ys * 0.0062 + p2) + 0.40 * np.sin(ys * 0.017 + p4)).astype(np.float32)
    base = (Ax[None, :] + Ay[:, None])                       # (ny,nx) float32

    H = base + _value_noise(rng, n, 6, octaves=4) * 0.80     # natural terrain
    L = np.full((n, n), LBL["grass"], dtype=np.uint8)
    M = np.zeros((n, n), dtype=np.uint8)

    # --- carriageway ------------------------------------------------------
    road = (DX <= HW) | (DY <= HW)
    side = (~road) & ((DX <= HW + SW) | (DY <= HW + SW))
    curb = (np.abs(DX - HW) <= gres) | (np.abs(DY - HW) <= gres)
    curb &= road | side

    L[road] = LBL["road"]
    L[side] = LBL["sidewalk"]
    L[curb] = LBL["curb"]

    # road follows the smooth terrain (no micro-noise), with asphalt texture
    asphalt = (gaussian_filter(rng.standard_normal((n // 4, n // 4)).astype(np.float32), 1.0))
    asphalt = zoom(asphalt, 4.0, order=1)[:n, :n] * 0.012
    coarse = gaussian_filter(rng.standard_normal((n // 8, n // 8)).astype(np.float32), 1.0)
    coarse = zoom(coarse, 8.0, order=1)[:n, :n] * 0.025
    H[road] = base[road] + asphalt[road] + coarse[road]
    H[side] = base[side] + 0.15 + asphalt[side] * 0.6
    H[curb] = base[curb] + 0.15
    del asphalt, coarse

    # --- block land-use ---------------------------------------------------
    blocks = []
    for i in range(len(xl) - 1):
        for j in range(len(yl) - 1):
            x0, x1 = xl[i] + HW + SW, xl[i + 1] - HW - SW
            y0, y1 = yl[j] + HW + SW, yl[j + 1] - HW - SW
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            r = rng.random()
            use = "park" if r < 0.18 else ("plaza" if r < 0.34 else "built")
            blocks.append(dict(i=i, j=j, x0=x0, y0=y0, x1=x1, y1=y1, use=use))

    for b in blocks:
        i0 = int(b["x0"] / gres); i1 = int(b["x1"] / gres)
        j0 = int(b["y0"] / gres); j1 = int(b["y1"] / gres)
        i0, j0 = max(i0, 0), max(j0, 0)
        i1, j1 = min(i1, n), min(j1, n)
        if i1 <= i0 or j1 <= j0:
            continue
        if b["use"] == "park":
            L[j0:j1, i0:i1] = LBL["grass"]
            rough = zoom(gaussian_filter(
                rng.standard_normal((max(4, (j1 - j0) // 8), max(4, (i1 - i0) // 8))).astype(np.float32), 1.0
            ), (float(j1 - j0) / max(4, (j1 - j0) // 8), float(i1 - i0) / max(4, (i1 - i0) // 8)), order=1)
            H[j0:j1, i0:i1] += rough[:j1 - j0, :i1 - i0] * 0.10
        elif b["use"] == "plaza":
            L[j0:j1, i0:i1] = LBL["road"]
            H[j0:j1, i0:i1] = base[j0:j1, i0:i1] + 0.012 * rng.standard_normal((j1 - j0, i1 - i0)).astype(np.float32)

    # --- lane markings, crosswalks, stop lines ----------------------------
    for lx in xl:
        ci = int(lx / gres)
        w = max(1, int(0.09 / gres))
        sl = slice(max(0, ci - w), min(n, ci + w + 1))
        dash = ((np.arange(n) * gres) % 9.0) < 4.0           # dashed centre line
        M[:, sl][dash, :] = 1
        M[:, sl][:, :] = np.where(dash[:, None], 1, M[:, sl])
    for ly in yl:
        cj = int(ly / gres)
        w = max(1, int(0.09 / gres))
        sl = slice(max(0, cj - w), min(n, cj + w + 1))
        dash = ((np.arange(n) * gres) % 9.0) < 4.0
        M[sl, :][:, dash] = 1
    # crosswalks + stop lines around every intersection
    for lx in xl:
        for ly in yl:
            for axis, sgn in ((0, -1), (0, 1), (1, -1), (1, 1)):
                if axis == 0:      # approach along x, stripes run along y
                    d0, d1 = HW + SW + 0.4, HW + SW + 5.4
                    ci0 = int((lx + sgn * d0) / gres); ci1 = int((lx + sgn * d1) / gres)
                    cj0 = int((ly - HW - 0.4) / gres); cj1 = int((ly + HW + 0.4) / gres)
                    a0, a1 = sorted((ci0, ci1)); b0, b1 = sorted((cj0, cj1))
                    a0, b0 = max(a0, 0), max(b0, 0)
                    a1, b1 = min(a1, n), min(b1, n)
                    if a1 <= a0 or b1 <= b0:
                        continue
                    xx = np.arange(a0, a1) * gres
                    stripe = (((xx - lx) * sgn) % 1.25) < 0.62
                    M[b0:b1, a0:a1] = np.where(stripe[None, :], 2, M[b0:b1, a0:a1])
                    # stop line 1.2 m before the crosswalk
                    s0 = int((lx + sgn * (d0 - 1.5)) / gres); s1 = int((lx + sgn * (d0 - 0.3)) / gres)
                    s0, s1 = sorted((max(s0, 0), min(s1, n)))
                    if s1 > s0:
                        M[b0:b1, s0:s1] = 3
                else:
                    d0, d1 = HW + SW + 0.4, HW + SW + 5.4
                    cj0 = int((ly + sgn * d0) / gres); cj1 = int((ly + sgn * d1) / gres)
                    ci0 = int((lx - HW - 0.4) / gres); ci1 = int((lx + HW + 0.4) / gres)
                    a0, a1 = sorted((ci0, ci1)); b0, b1 = sorted((cj0, cj1))
                    a0, b0 = max(a0, 0), max(b0, 0)
                    a1, b1 = min(a1, n), min(b1, n)
                    if a1 <= a0 or b1 <= b0:
                        continue
                    yy = np.arange(b0, b1) * gres
                    stripe = (((yy - ly) * sgn) % 1.25) < 0.62
                    M[b0:b1, a0:a1] = np.where(stripe[:, None], 2, M[b0:b1, a0:a1])
                    s0 = int((ly + sgn * (d0 - 1.5)) / gres); s1 = int((ly + sgn * (d0 - 0.3)) / gres)
                    s0, s1 = sorted((max(s0, 0), min(s1, n)))
                    if s1 > s0:
                        M[s0:s1, a0:a1] = 3

    # --- potholes & speed bumps on the carriageway ------------------------
    def _pt(in_focus=True):
        if focus is not None and in_focus:
            return (float(rng.uniform(focus[0], focus[1])),
                    float(rng.uniform(focus[2], focus[3])))
        return float(rng.uniform(20, size - 20)), float(rng.uniform(20, size - 20))

    n_pot = int(rng.integers(70, 110))
    for _p in range(n_pot):
        for _try in range(40):
            px, py = _pt(rng.random() < 0.8)
            ix, iy = int(px / gres), int(py / gres)
            if ix < 2 or iy < 2 or ix >= n - 2 or iy >= n - 2:
                continue
            if L[iy, ix] != LBL["road"]:
                continue
            r = float(rng.uniform(0.45, 1.25))
            depth = float(rng.uniform(0.06, 0.26))
            R = int(np.ceil(r / gres)) + 1
            i0, i1 = max(0, ix - R), min(n, ix + R + 1)
            j0, j1 = max(0, iy - R), min(n, iy + R + 1)
            if i1 <= i0 or j1 <= j0:
                continue
            gx = (np.arange(i0, i1) * gres + gres / 2 - px) / r
            gy = (np.arange(j0, j1) * gres + gres / 2 - py) / r
            d2 = gx[None, :] ** 2 + gy[:, None] ** 2
            prof = np.clip(1.0 - d2, 0, None) ** 1.3
            sub = H[j0:j1, i0:i1] - depth * prof
            H[j0:j1, i0:i1] = np.where(prof > 0.25, np.minimum(H[j0:j1, i0:i1], sub), H[j0:j1, i0:i1])
            L[j0:j1, i0:i1] = np.where(prof > 0.45, LBL["pothole"], L[j0:j1, i0:i1])
            break

    n_bump = int(rng.integers(8, 18))
    for _ in range(n_bump):
        for _try in range(40):
            px, py = _pt(rng.random() < 0.75)
            ix, iy = int(px / gres), int(py / gres)
            if ix < 2 or iy < 2 or ix >= n - 2 or iy >= n - 2:
                continue
            if L[iy, ix] != LBL["road"]:
                continue
            # ridge perpendicular to the road direction
            along_x = dx[ix] < dy[iy]        # road runs along y
            hgt = float(rng.uniform(0.07, 0.13))
            sig = 0.42
            R = int(np.ceil(3.0 * sig / gres)) + 1
            i0, i1 = max(0, ix - R), min(n, ix + R + 1)
            j0, j1 = max(0, iy - R), min(n, iy + R + 1)
            gx = np.arange(i0, i1) * gres + gres / 2 - px
            gy = np.arange(j0, j1) * gres + gres / 2 - py
            if along_x:
                prof = np.exp(-(gx ** 2) / (2 * sig ** 2))[None, :]
                prof = np.repeat(prof, j1 - j0, axis=0)
            else:
                prof = np.exp(-(gy ** 2) / (2 * sig ** 2))[:, None]
                prof = np.repeat(prof, i1 - i0, axis=1)
            H[j0:j1, i0:i1] += (hgt * prof).astype(np.float32)
            break

    return H.astype(np.float32), L, M, xl, yl, HW, SW, blocks


# ==========================================================================
# static scene content
# ==========================================================================
def build_statics(rng, H, L, M, xl, yl, HW, SW, blocks, gres, size, focus=None,
                  poses=None):
    P = PrimSet()
    if focus is None:
        focus = (14.0, size - 14.0, 14.0, size - 14.0)
    fx0, fx1, fy0, fy1 = focus

    def h_at(x, y):
        return float(_sample(H, x, y, gres))

    # ---------------- buildings -----------------------------------------
    for b in blocks:
        if b["use"] != "built":
            continue
        w = b["x1"] - b["x0"]; h = b["y1"] - b["y0"]
        nx = int(max(1, w // 22)); ny = int(max(1, h // 22))
        for ii in range(nx):
            for jj in range(ny):
                if rng.random() > 0.82:
                    continue
                cx0 = b["x0"] + (ii + 0.5) * w / nx
                cy0 = b["y0"] + (jj + 0.5) * h / ny
                bw = rng.uniform(0.55, 0.92) * w / nx
                bh = rng.uniform(0.55, 0.92) * h / ny
                cx = cx0 + rng.uniform(-0.15, 0.15) * w / nx
                cy = cy0 + rng.uniform(-0.15, 0.15) * h / ny
                tall = rng.random()
                bhz = rng.uniform(5.5, 14.0) if tall < 0.62 else rng.uniform(14.0, 34.0)
                yaw = rng.uniform(-0.10, 0.10)
                z0 = h_at(cx, cy)
                P.box(cx, cy, z0 + bhz / 2, bw / 2, bh / 2, bhz / 2, yaw,
                      LBL["structure"], float(rng.uniform(0.35, 0.72)))
                # roof-top clutter makes the elevation layer interesting
                if rng.random() < 0.45 and bhz < 20:
                    rw = rng.uniform(0.8, 2.5); rh = rng.uniform(0.8, 2.0)
                    P.box(cx + rng.uniform(-0.2, 0.2) * bw, cy + rng.uniform(-0.2, 0.2) * bh,
                          z0 + bhz + 0.7, rw, rh, 0.7, yaw, LBL["structure"], 0.55)

    # ---------------- garden walls / fences ------------------------------
    for b in blocks:
        if b["use"] == "built" or rng.random() > 0.42:
            continue
        sides = rng.choice([0, 1, 2, 3], size=int(rng.integers(1, 3)), replace=False)
        wh = float(rng.uniform(0.9, 2.1))
        for s in sides:
            if s == 0:
                P.box((b["x0"] + b["x1"]) / 2, b["y0"] + 0.15, h_at(b["x0"], b["y0"]) + wh / 2,
                      (b["x1"] - b["x0"]) / 2, 0.14, wh / 2, 0.0, LBL["structure"], 0.48)
            elif s == 1:
                P.box((b["x0"] + b["x1"]) / 2, b["y1"] - 0.15, h_at(b["x0"], b["y1"]) + wh / 2,
                      (b["x1"] - b["x0"]) / 2, 0.14, wh / 2, 0.0, LBL["structure"], 0.48)
            elif s == 2:
                P.box(b["x0"] + 0.15, (b["y0"] + b["y1"]) / 2, h_at(b["x0"], b["y0"]) + wh / 2,
                      0.14, (b["y1"] - b["y0"]) / 2, wh / 2, 0.0, LBL["structure"], 0.48)
            else:
                P.box(b["x1"] - 0.15, (b["y0"] + b["y1"]) / 2, h_at(b["x1"], b["y0"]) + wh / 2,
                      0.14, (b["y1"] - b["y0"]) / 2, wh / 2, 0.0, LBL["structure"], 0.48)

    # ---------------- street furniture along the avenues ------------------
    def sidewalk_pt(lv, other, axis, side):
        off = HW + SW * float(rng.uniform(0.25, 0.8))
        if axis == 0:
            return lv + side * off, other
        return other, lv + side * off

    n_pole = 70
    for _ in range(n_pole):
        axis = int(rng.integers(0, 2))
        side = 1 if rng.random() < 0.5 else -1
        lvs = xl if axis == 0 else yl
        lv = float(rng.choice(lvs))
        oth = float(rng.uniform(18, size - 18))
        px, py = sidewalk_pt(lv, oth, axis, side)
        if not (0 < px < size and 0 < py < size):
            continue
        z0 = h_at(px, py)
        ph = float(rng.uniform(5.0, 8.5))
        P.cyl(px, py, z0, z0 + ph, 0.085, LBL["pole"], 0.62)
        if rng.random() < 0.34:      # mast arm with a signal head
            alen = float(rng.uniform(1.6, 3.4))
            ax = px + (alen * (1 if axis == 0 else 0)) * (-side)
            ay = py + (alen * (1 if axis == 1 else 0)) * (-side)
            P.box((px + ax) / 2, (py + ay) / 2, z0 + ph - 0.18,
                  (alen / 2 if axis == 0 else 0.06), (0.06 if axis == 0 else alen / 2), 0.09,
                  0.0, LBL["pole"], 0.58)
            P.box(ax, ay, z0 + ph - 0.75, 0.16, 0.16, 0.42, 0.0, LBL["pole"], 0.55)

    # ---------------- trees ------------------------------------------------
    n_tree = 72
    for _ in range(n_tree):
        r = rng.random()
        if r < 0.55:
            axis = int(rng.integers(0, 2)); side = 1 if rng.random() < 0.5 else -1
            lvs = xl if axis == 0 else yl
            px, py = sidewalk_pt(float(rng.choice(lvs)), float(rng.uniform(max(18, fy0 - 40), min(size - 18, fy1 + 40))), axis, side)
        else:
            b = blocks[int(rng.integers(0, len(blocks)))]
            px = float(rng.uniform(b["x0"], b["x1"])); py = float(rng.uniform(b["y0"], b["y1"]))
        if not (3 < px < size - 3 and 3 < py < size - 3):
            continue
        z0 = h_at(px, py)
        th = float(rng.uniform(2.8, 5.2))
        P.cyl(px, py, z0, z0 + th, float(rng.uniform(0.13, 0.22)), LBL["pole"], 0.30)
        ns = int(rng.integers(3, 6))
        cr = float(rng.uniform(1.3, 2.5))
        for s in range(ns):
            P.sph(px + rng.uniform(-1.1, 1.1), py + rng.uniform(-1.1, 1.1),
                  z0 + th + rng.uniform(-0.4, 1.5), cr * float(rng.uniform(0.55, 1.0)),
                  LBL["vegetation"], float(rng.uniform(0.26, 0.44)))

    n_bush = 72
    for _ in range(n_bush):
        b = blocks[int(rng.integers(0, len(blocks)))]
        px = float(rng.uniform(b["x0"] - 1, b["x1"] + 1)); py = float(rng.uniform(b["y0"] - 1, b["y1"] + 1))
        if not (2 < px < size - 2 and 2 < py < size - 2):
            continue
        z0 = h_at(px, py)
        P.sph(px, py, z0 + 0.35, float(rng.uniform(0.45, 1.05)), LBL["vegetation"], 0.34)

    # ---------------- bollards --------------------------------------------
    for lx in xl:
        for ly in yl:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    if rng.random() > 0.45:
                        continue
                    for k in range(2):
                        px = lx + sx * (HW + 0.55) + rng.uniform(-0.2, 0.2)
                        py = ly + sy * (HW + 1.4 + k * 0.85)
                        if not (0 < px < size and 0 < py < size):
                            continue
                        z0 = h_at(px, py)
                        P.cyl(px, py, z0, z0 + 0.92, 0.06, LBL["pole"], 0.55)

    # ---------------- parked (static) vehicles ----------------------------
    n_park = 40
    pose_xy = poses[:, :2] if poses is not None else None
    for _ in range(n_park):
        axis = int(rng.integers(0, 2))
        side = 1 if rng.random() < 0.5 else -1
        lvs = xl if axis == 0 else yl
        lv = float(rng.choice(lvs))
        oth = float(rng.uniform(max(18, fy0 - 40), min(size - 18, fy1 + 40)))
        off = float(rng.uniform(0.9, 2.1))
        if axis == 0:
            px, py = lv + side * (HW - off), oth
            yaw = np.pi / 2 + rng.uniform(-0.03, 0.03)
        else:
            px, py = oth, lv + side * (HW - off)
            yaw = rng.uniform(-0.03, 0.03)
        z0 = h_at(px, py)
        if pose_xy is not None:
            dmin = float(np.linalg.norm(pose_xy - np.array([px, py]), axis=1).min())
            if dmin < 4.2:
                continue
        bl, bw, bh = 4.35, 1.82, 1.42
        P.box(px, py, z0 + bh / 2 + 0.10, bl / 2, bw / 2, bh / 2, yaw, LBL["vehicle"],
              float(rng.uniform(0.42, 0.86)))
        P.box(px + 0.15 * np.cos(yaw), py + 0.15 * np.sin(yaw), z0 + bh + 0.34,
              bl * 0.27, bw * 0.46, 0.30, yaw, LBL["vehicle"], 0.62)

    return P


# ==========================================================================
# moving agents
# ==========================================================================
def build_agents(rng, H, xl, yl, HW, SW, blocks, gres, size, focus=None,
                 poses=None, dt=0.4, n_frames=40):
    agents = []
    if focus is None:
        focus = (14.0, size - 14.0, 14.0, size - 14.0)
    fx0, fx1, fy0, fy1 = focus

    def _rand(a, b):
        return float(rng.uniform(a, b))

    # blocks / avenues that actually intersect the driven area
    near_blocks = [b for b in blocks
                   if b["x1"] > fx0 - 20 and b["x0"] < fx1 + 20
                   and b["y1"] > fy0 - 20 and b["y0"] < fy1 + 20]
    if not near_blocks:
        near_blocks = blocks
    nx_ = np.array([v for v in xl if fx0 - 40 < v < fx1 + 40], dtype=np.float32)
    ny_ = np.array([v for v in yl if fy0 - 40 < v < fy1 + 40], dtype=np.float32)
    xl_n = nx_ if len(nx_) else xl
    yl_n = ny_ if len(ny_) else yl

    # ---- vehicles : straight runs along a randomly chosen lane -----------
    pose_xy = poses[:, :2] if poses is not None else None

    def lane_pos(axis, lv, s0):
        """Right-hand-traffic lane centre for a vehicle at arc-length s0
        travelling along `axis` and offset LANE_OFF from the centre-line."""
        if axis == 0:                       # avenue runs along y
            return lv + LANE_OFF, s0
        return s0, lv - LANE_OFF            # avenue runs along x

    def clear(traj):
        if pose_xy is None:
            return True
        d = np.linalg.norm(traj[:, None, :] - pose_xy[None, :, :], axis=2)
        return d.min() > 6.5

    nveh = int(rng.integers(14, 22))
    for _ in range(nveh):
        axis = int(rng.integers(0, 2))
        lvs = xl_n if axis == 0 else yl_n
        lv = float(rng.choice(lvs))
        s0 = _rand(fx0 - 40, fx1 + 40)
        v = float(rng.uniform(4.0, 13.5)) * (1 if rng.random() < 0.5 else -1)
        kind = rng.random()
        if kind < 0.72:
            L_, W_, H_ = 4.45, 1.84, 1.45
        elif kind < 0.88:
            L_, W_, H_ = 5.9, 2.10, 1.95     # SUV / van
        else:
            L_, W_, H_ = 10.5, 2.50, 3.10    # bus / truck
        traj = []
        for f in range(n_frames):
            p = lane_pos(axis, lv, s0 + v * f * dt)
            traj.append(p)
        traj = np.array(traj, dtype=np.float64)
        if not clear(traj):
            continue
        x, y = lane_pos(axis, lv, s0)
        agents.append(dict(kind="vehicle", axis=axis, cx=x, cy=y, v=v,
                           L=L_, W=W_, H=H_, refl=float(rng.uniform(0.42, 0.88)),
                           phase=float(rng.uniform(0, 6.28)), wheels=kind < 0.72))

    # ---- pedestrians : walk the sidewalk ring of a block ----------------
    nped = int(rng.integers(46, 66))
    for _ in range(nped):
        b = near_blocks[int(rng.integers(0, len(near_blocks)))]
        pad = 1.15
        x0, x1 = b["x0"] - pad, b["x1"] + pad
        y0, y1 = b["y0"] - pad, b["y1"] + pad
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1]
        traj = []
        for f in range(n_frames):
            ss = (float(rng.uniform(0, 1)) * 0)  # placeholder, replaced below
        # sample the whole loop to test clearance cheaply
        _tt = np.linspace(0, total, 60, endpoint=False)
        _d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        _cum = cum
        _tp = []
        for tv in _tt:
            i = int(np.clip(np.searchsorted(_cum, tv) - 1, 0, len(_d) - 1))
            u = (tv - _cum[i]) / max(_d[i], 1e-6)
            _tp.append(pts[i] + u * (pts[i + 1] - pts[i]))
        _tp = np.array(_tp)
        if pose_xy is not None:
            dd = np.linalg.norm(_tp[:, None, :] - pose_xy[None, :, :], axis=2)
            if dd.min() < 3.0:
                continue
        agents.append(dict(kind="pedestrian", pts=pts, cum=cum, total=total,
                           s=float(rng.uniform(0, total)),
                           v=float(rng.uniform(0.85, 1.75)) * (1 if rng.random() < 0.5 else -1),
                           phase=float(rng.uniform(0, 6.28)),
                           refl=float(rng.uniform(0.22, 0.52)),
                           scale=float(rng.uniform(0.92, 1.08))))

    # ---- cyclists : ride the kerbside lane ------------------------------
    ncyc = int(rng.integers(7, 13))
    for _ in range(ncyc):
        axis = int(rng.integers(0, 2))
        lvs = xl_n if axis == 0 else yl_n
        lv = float(rng.choice(lvs))
        s0 = _rand(fx0 - 40, fx1 + 40)
        v = float(rng.uniform(3.5, 7.0)) * (1 if rng.random() < 0.5 else -1)
        if axis == 0:
            traj = np.stack([np.full(n_frames, lv + CY_OFF), s0 + v * np.arange(n_frames) * dt], 1)
            x, y = lv + CY_OFF, s0
        else:
            traj = np.stack([s0 + v * np.arange(n_frames) * dt, np.full(n_frames, lv - CY_OFF)], 1)
            x, y = s0, lv - CY_OFF
        if pose_xy is not None:
            dd = np.linalg.norm(traj[:, None, :] - pose_xy[None, :, :], axis=2)
            if dd.min() < 4.5:
                continue
        agents.append(dict(kind="cyclist", axis=axis, cx=x, cy=y, v=v,
                           refl=float(rng.uniform(0.35, 0.6)),
                           phase=float(rng.uniform(0, 6.28))))

    return agents


def agents_at_time(agents, t, H, gres, P: PrimSet):
    """Instantiate all moving agents into primitives at simulation time t."""
    for ag in agents:
        k = ag["kind"]
        if k == "vehicle":
            s = ag["cx"] if ag["axis"] == 1 else ag["cy"]
            s = s + ag["v"] * t
            if ag["axis"] == 0:
                px, py, yaw = ag["cx"], s, 0.0
            else:
                px, py, yaw = s, ag["cy"], np.pi / 2
            z0 = float(_sample(H, px, py, gres))
            L_, W_, Hv = ag["L"], ag["W"], ag["H"]
            bob = 0.012 * np.sin(2.2 * t + ag["phase"])
            P.box(px, py, z0 + 0.16 + Hv / 2 + bob, L_ / 2, W_ / 2, Hv / 2, yaw,
                  LBL["vehicle"], ag["refl"], dyn=1)
            P.box(px - 0.20 * np.cos(yaw), py - 0.20 * np.sin(yaw), z0 + 0.16 + Hv + 0.32 + bob,
                  L_ * 0.28, W_ * 0.45, 0.30, yaw, LBL["vehicle"], ag["refl"] * 0.9, dyn=1)
            if ag["wheels"]:
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        wx = px + sx * L_ * 0.32 * np.cos(yaw) - sy * W_ * 0.48 * np.sin(yaw)
                        wy = py + sx * L_ * 0.32 * np.sin(yaw) + sy * W_ * 0.48 * np.cos(yaw)
                        P.cyl(wx, wy, z0, z0 + 0.66, 0.33, LBL["vehicle"], 0.18, dyn=1)

        elif k == "pedestrian":
            s = (ag["s"] + ag["v"] * t) % ag["total"]
            d = np.diff(ag["pts"], axis=0)
            i = int(np.searchsorted(ag["cum"], s) - 1)
            i = int(np.clip(i, 0, len(d) - 1))
            u = (s - ag["cum"][i]) / (ag["cum"][i + 1] - ag["cum"][i])
            px = float(ag["pts"][i, 0] + u * d[i, 0])
            py = float(ag["pts"][i, 1] + u * d[i, 1])
            yaw = float(np.arctan2(d[i, 1], d[i, 0]))
            sc = ag["scale"]
            z0 = float(_sample(H, px, py, gres))
            gait = np.sin(2 * np.pi * 1.05 * t + ag["phase"])
            bob = 0.022 * np.cos(4 * np.pi * 1.05 * t + ag["phase"])
            nx_, ny_ = -np.sin(yaw), np.cos(yaw)
            for side in (-1, 1):
                lx = px + nx_ * side * 0.10 + np.cos(yaw) * (0.16 * gait * side * np.sign(ag["v"]))
                ly = py + ny_ * side * 0.10 + np.sin(yaw) * (0.16 * gait * side * np.sign(ag["v"]))
                P.cyl(lx, ly, z0 + 0.02, z0 + 0.86 * sc, 0.075 * sc, LBL["pedestrian"], ag["refl"], dyn=1)
            P.cyl(px, py, z0 + 0.80 * sc + bob, z0 + 1.44 * sc + bob, 0.165 * sc,
                  LBL["pedestrian"], ag["refl"], dyn=1)
            P.sph(px, py, z0 + 1.60 * sc + bob, 0.115 * sc, LBL["pedestrian"], ag["refl"], dyn=1)

        else:  # cyclist
            s = (ag["cx"] if ag["axis"] == 1 else ag["cy"]) + ag["v"] * t
            if ag["axis"] == 0:
                px, py, yaw = ag["cx"], s, 0.0
            else:
                px, py, yaw = s, ag["cy"], np.pi / 2
            z0 = float(_sample(H, px, py, gres))
            P.box(px, py, z0 + 0.52, 0.85, 0.09, 0.42, yaw, LBL["cyclist"], 0.30, dyn=1)
            P.cyl(px - 0.12 * np.cos(yaw), py - 0.12 * np.sin(yaw), z0 + 0.95, z0 + 1.55,
                  0.17, LBL["cyclist"], ag["refl"], dyn=1)
            P.sph(px - 0.16 * np.cos(yaw), py - 0.16 * np.sin(yaw), z0 + 1.78, 0.115,
                  LBL["cyclist"], ag["refl"], dyn=1)


# ==========================================================================
# ego trajectory
# ==========================================================================
def build_ego_route(rng, xl, yl, HW, size, n_frames, dt, speed):
    """Random walk on the avenue graph, confined to the central region."""
    lo, hi = 100.0, size - 100.0
    cand_x = [v for v in xl if lo < v < hi]
    cand_y = [v for v in yl if lo < v < hi]
    if len(cand_x) < 2:
        cand_x = list(xl)
    if len(cand_y) < 2:
        cand_y = list(yl)
    node = (int(rng.integers(0, len(cand_x))), int(rng.integers(0, len(cand_y))))
    nodes = [node]
    for _ in range(int(rng.integers(4, 9))):
        i, j = nodes[-1]
        opts = []
        if i + 1 < len(cand_x): opts.append((i + 1, j))
        if i - 1 >= 0: opts.append((i - 1, j))
        if j + 1 < len(cand_y): opts.append((i, j + 1))
        if j - 1 >= 0: opts.append((i, j - 1))
        opts = [o for o in opts if o != nodes[-1]] or opts
        nxt = opts[int(rng.integers(0, len(opts)))]
        if len(nodes) > 1 and nxt == nodes[-2]:
            nxt = opts[int(rng.integers(0, len(opts)))]
        nodes.append(nxt)
    wp = np.array([[cand_x[i], cand_y[j]] for i, j in nodes], dtype=np.float64)
    # rounded corners so the heading changes smoothly
    dense = []
    for k in range(len(wp) - 1):
        a, b = wp[k], wp[k + 1]
        n = max(2, int(np.linalg.norm(b - a) / 1.5))
        for u in np.linspace(0, 1, n, endpoint=False):
            dense.append(a + u * (b - a))
    dense.append(wp[-1])
    path = np.array(dense, dtype=np.float64)
    path = gaussian_filter(path, (2.2, 0), mode="nearest")   # smooth the corners

    # offset the centre-line polyline into the correct (right-hand) lane
    sm = gaussian_filter(path, (3.0, 0), mode="nearest")
    tan = np.zeros_like(sm)
    tan[1:-1] = sm[2:] - sm[:-2]
    tan[0] = sm[1] - sm[0]
    tan[-1] = sm[-1] - sm[-2]
    nrm = np.linalg.norm(tan, axis=1, keepdims=True) + 1e-6
    tan = tan / nrm
    right = np.stack([tan[:, 1], -tan[:, 0]], axis=1)      # right-hand side vector
    path = sm + right * LANE_OFF
    path = gaussian_filter(path, (2.4, 0), mode="nearest")

    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    step = speed * dt
    poses = []
    for f in range(n_frames):
        s = min(f * step, total - 0.01)
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(d) - 1))
        u = (s - cum[i]) / max(d[i], 1e-6)
        x = path[i, 0] + u * (path[i + 1, 0] - path[i, 0])
        y = path[i, 1] + u * (path[i + 1, 1] - path[i, 1])
        yaw = float(np.arctan2(path[i + 1, 1] - path[i, 1], path[i + 1, 0] - path[i, 0]))
        poses.append((float(x), float(y), yaw))
    return np.array(poses, dtype=np.float64), path


# ==========================================================================
# top level
# ==========================================================================
def generate_world(seed, n_frames, dt=None, cfg=DATASET):
    """Build one complete synthetic city + ego trajectory."""
    rng = np.random.default_rng(seed)
    size = cfg["world_size"]; gres = cfg["ground_res"]
    dt = cfg["frame_dt"] if dt is None else dt

    # 1) avenue layout, 2) the ego route, 3) everything else packed around it
    xl, yl = layout_roads(rng, size)
    HW, SW = 4.5, 2.4
    poses, path = build_ego_route(rng, xl, yl, HW, size, n_frames, dt, cfg["ego_speed"])
    pad = 65.0
    focus = (float(np.clip(path[:, 0].min() - pad, 12, size - 12)),
             float(np.clip(path[:, 0].max() + pad, 12, size - 12)),
             float(np.clip(path[:, 1].min() - pad, 12, size - 12)),
             float(np.clip(path[:, 1].max() + pad, 12, size - 12)))

    H, L, M, xl, yl, HW, SW, blocks = build_ground(rng, size, gres, xl, yl, focus)
    Ps = build_statics(rng, H, L, M, xl, yl, HW, SW, blocks, gres, size, focus, poses)
    static = Ps.arrays()
    agents = build_agents(rng, H, xl, yl, HW, SW, blocks, gres, size, focus,
                         poses=poses, dt=dt, n_frames=n_frames)
    return dict(
        seed=seed, size=size, gres=gres,
        H=H, L=L, M=M, blocks=blocks, xl=xl, yl=yl, HW=HW, SW=SW,
        static=static, agents=agents, poses=poses, path=path,
        n_frames=n_frames, dt=dt,
    )
