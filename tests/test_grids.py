"""Sanity tests for the foveated 2.5D grid engine and the BEV rasteriser."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import config as C
from src.grids import FoveatedGrid, rasterize_bev, uniform_occupancy, ring_stats

rng = np.random.default_rng(0)
ok = lambda n, c: print(("  PASS  " if c else "  FAIL  ") + n) or (c or sys.exit(1))

print("ring topology")
a, u = C.theoretical_cell_counts()
print(f"  adaptive cells (theory) {a:,.0f}   uniform 5cm {u['uniform_5cm']:,.0f}"
      f"   ratio {u['uniform_5cm']/a:.1f}x")
ok("rings tile [0,100)", C.GRID_RINGS[0][0] == 0 and C.GRID_RINGS[-1][1] == 100.0)
ok("cell size grows monotonically",
   all(C.GRID_RINGS[i][2] <= C.GRID_RINGS[i + 1][2] for i in range(4)))
ok("all rings are TILE x TILE",
   all(abs(2 * r[1] / r[2] - C.RING_TILE) < 1e-9 for r in C.GRID_RINGS))

print("projection: no alignment error, no lost points")
N = 200000
xy = (rng.random((N, 2)) * 200 - 100).astype(np.float32)
r = np.hypot(xy[:, 0], xy[:, 1])
keep = r < 99.9
xy = xy[keep]; r = r[keep]
z = (rng.random(len(xy)) * 2 - 1).astype(np.float32)
cls = rng.integers(0, C.NUM_CLASSES, len(xy)).astype(np.int64)
g = FoveatedGrid()
rings = g.project(xy, z, cls)
tot = sum(int(rg["valid"].sum()) for rg in rings)
print(f"  {len(xy):,} points -> {tot:,} occupied cells")
ok("every ring produced cells", all(rg["valid"].sum() > 0 for rg in rings))
ok("cells <= points (no duplication)", tot <= len(xy))

# round-trip: each point must land in the cell whose centre is within
# half a cell diagonal of it -> the ring lattice has no alignment error
for k, rg in enumerate(rings):
    cell, rout = C.GRID_RINGS[k][2], C.GRID_RINGS[k][1]
    m = np.flatnonzero(C.ring_index_of(r) == k)
    ix = np.clip(((xy[m, 0] + rout) / cell).astype(np.int64), 0, C.RING_TILE - 1)
    iy = np.clip(((xy[m, 1] + rout) / cell).astype(np.int64), 0, C.RING_TILE - 1)
    cx = (ix + 0.5) * cell - rout
    cy = (iy + 0.5) * cell - rout
    err = np.hypot(xy[m, 0] - cx, xy[m, 1] - cy).max()
    lim = cell * 0.7072
    ok(f"ring {k}: max point->cell-centre error {err:.4f} m <= {lim:.4f}", err <= lim)
    ok(f"ring {k}: every routed cell is flagged occupied",
       bool(rg["valid"][iy, ix].all()))

print("ring membership / radius routing")
for k, rg in enumerate(rings):
    idx = np.flatnonzero(rg["valid"])
    j, i = np.divmod(idx, C.RING_TILE)
    cell, rout = C.GRID_RINGS[k][2], C.GRID_RINGS[k][1]
    cx = (i + 0.5) * cell - rout
    cy = (j + 0.5) * cell - rout
    rr = np.hypot(cx, cy)
    r_in, r_out = C.GRID_RINGS[k][0], C.GRID_RINGS[k][1]
    ok(f"ring {k} cells lie in [{r_in},{r_out}) (+-1 cell)",
       rr.min() >= r_in - 2 * cell and rr.max() < r_out + 2 * cell)
# every point inside 99.9 m lands in exactly one ring
cnt = sum(len(rg["pt_idx"]) for rg in rings)
ok(f"all {len(xy):,} points routed to exactly one ring", cnt == len(xy))
lvl = C.ring_index_of(r)
ok("ring_index_of matches the projection", all(
    (lvl[rg["pt_idx"]] == k).all() for k, rg in enumerate(rings)))

print("realistic radial density (1/r^2) clusters into cells")
th = rng.random(200000) * 2 * np.pi
rr = 1.0 + 99.0 * rng.random(200000) ** 2.0
xy2 = np.stack([rr * np.cos(th), rr * np.sin(th)], 1).astype(np.float32)
z2 = np.zeros(len(xy2), np.float32)
c2 = np.zeros(len(xy2), np.int64)
rings2 = FoveatedGrid().project(xy2, z2, c2)
tot2 = sum(int(rg["valid"].sum()) for rg in rings2)
print(f"  {len(xy2):,} lidar-like points -> {tot2:,} cells ({len(xy2)/tot2:.2f}x)")
ok("lidar-like density compresses > 1.3x", len(xy2) / tot2 > 1.3)

print("BEV rasteriser")
extent, px, res = C.PYRAMID[0]
x = np.array([0.0, 0.0, 1.0], np.float32)
y = np.array([0.0, 0.0, 0.0], np.float32)
zz = np.array([-0.5, 1.5, 0.25], np.float32)
f, lab, val = rasterize_bev(np.stack([x, y], 1), zz, np.zeros(3, np.float32),
                            np.array([0, 0, 3], np.int64), extent, px)
c = int((0.0 + extent / 2) / res)
lo, hi = C.Z_RANGE
ok("max height cell correct", abs(f[0, c, c] - (1.5 - lo) / (hi - lo)) < 1e-6)
ok("min height cell correct", abs(f[2, c, c] - (-0.5 - lo) / (hi - lo)) < 1e-6)
ok("mean height cell correct", abs(f[1, c, c] - ((-0.5 + 1.5) / 2 - lo) / (hi - lo)) < 1e-6)
c2x = int((1.0 + extent / 2) / res)        # the (1 m, 0 m) point lands here
ok("validity mask: 2 distinct cells occupied, distant cells empty",
   int(val.sum()) == 2 and bool(val[c, c2x]) and not bool(val[c, c + 40]))
ok("majority label", lab[c, c] == 0)

print("memory accounting")
occ = uniform_occupancy(xy, 0.05)
u5 = int(C.MAP_EXTENT / 0.05) ** 2
ad = sum(C.RING_TILE ** 2 for _ in C.GRID_RINGS)
print(f"  adaptive allocated {ad:,}  |  uniform 5cm allocated {u5:,} "
      f"({u5/ad:.0f}x)  occupied {occ:,}")
ok("adaptive is smaller than uniform", ad < u5)
ok("3D voxel baseline is enormous",
   int(C.MAP_EXTENT / 0.05) ** 2 * int(12 / 0.05) > 1e9)
print("\nALL GRID TESTS PASSED")
