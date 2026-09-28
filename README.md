# FoveaMap — Adaptive Variable-Resolution 2.5D Lidar Mapping

**Foveated 2.5D elevation + semantic mapping for dynamic environment perception,
on a self-generated synthetic LiDAR dataset.**

A 64-beam spinning LiDAR produces ~65 000 returns per sweep.  Rendering all of
them into a uniform 5 cm map out to 100 m would need **16 million cells per
frame** — far beyond a real-time budget on embedded hardware.  FoveaMap instead
uses a *foveated* lattice: 5 cm cells inside a 10 m safety bubble, relaxing in
five rings to 50 cm at the 100 m horizon.  The map keeps

* **semantics** — 11 classes spanning drivable terrain, static obstacles and
  dynamic traffic agents,
* **geometry** — min / max / mean elevation per cell, so curbs, pothole depth
  and overhang clearance survive the 3D → 2.5D projection,
* **motion evidence** — an ego-motion-compensated temporal-difference layer
  that needs no labels at all,

and it does so in **~800 k allocated cells instead of 16 M (20× less memory)**
at video rate on two CPU cores.

---

## 1. What is in the box

| Path | What it is |
|---|---|
| **`dashboard.html`** | **The deliverable.** Self-contained interactive 2.5D map + performance dashboard. No network, no CDN — open it directly. |
| `src/config.py` | Semantic taxonomy, sensor model, ring topology, pyramid spec |
| `src/worldgen.py` | Procedural city generator (height-field terrain, road network, buildings, agents) |
| `src/lidar.py` | Numba ray-tracing LiDAR simulator (DDA-accelerated, rolling shutter, intensity + dropout model) |
| `src/dataset.py` | Renders worlds into labelled sweep shards |
| `src/grids.py` | BEV pyramid rasteriser + the foveated 2.5D grid engine |
| `src/model.py` | **FoveaNet** — shared-weight, level-conditioned BEV U-Net |
| `src/train.py` | Training loop with on-the-fly augmentation |
| `src/pipeline.py` | End-to-end inference, temporal differencing, uniform reference grids |
| `src/eval.py` | Metrics: IoU, per-range accuracy, latency, memory, resolution ablation |
| `src/export_dashboard.py` | Packs frames into PNG tiles and builds `dashboard.html` |
| `src/dashboard_shell.html` | Dashboard application shell (renderer + charts) |
| `data/` | Generated dataset shards (`.npz`) |
| `models/` | Trained weights + training history |
| `outputs/` | `metrics.json`, `dash_data.js` |

---

## 2. Quick start

```bash
# 1  environment (CPU only; torch is the only heavy dependency)
pip install numpy scipy scikit-image pillow numba matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2  everything end to end  (~1 h on 2 cores; data gen ~5 min, training ~40 min)
cd foveamap && ./run_all.sh
```

Or step by step:

```bash
python3 -m src.dataset                       # render synthetic worlds -> data/*.npz
python3 -m src.train --smoke                 # print timing and exit
python3 -m src.train                         # train FoveaNet -> models/foveanet_best.pt
python3 -m src.eval                          # evaluate -> outputs/metrics.json
python3 -m src.export_dashboard              # build -> dashboard.html
```

> **Note on this sandbox:** installed Python packages do **not** survive between
> chat sessions.  If `import torch` fails, re-run
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> and then re-run the command.  Nothing else needs reinstalling.

---

## 3. The dataset (self-generated)

There is no public dataset to download — `src/dataset.py` synthesises one from
scratch.  That is deliberate: it gives us **exact** per-point ground truth for
elevation, class, instance and motion, including things real datasets annotate
poorly (curbs, potholes, overhangs).

**World** (`src/worldgen.py`) — a 384 m procedural city per seed:

* terrain height field at 12.5 cm, smooth fractal relief (±0.8 m)
* irregular grid of 9 m avenues with correct lane geometry, kerbside parking,
  lane markings, dashed centre lines, crosswalks and stop lines
* sidewalks raised 15 cm; **curbs as a one-cell 10 cm elevation step**
* **potholes** (elliptical depressions, 6–26 cm deep) and speed bumps
* parks, plazas, buildings (5–34 m), garden walls, poles with signal masts,
  bollards, trees (trunk + canopy spheres), bushes
* 14–21 moving vehicles, 46–66 walking pedestrians (4 primitives each, with
  gait), 7–13 cyclists
* the driven route is generated *first* and scene content is packed around it,
  so every class actually appears in the sensor's field of view

**Sensor** (`src/lidar.py`) — one Numba kernel per sweep:

* 64 beams × 1440 azimuths (0.25°), VFOV −24°…+2° (denser near the horizon),
  1.8 m mount height, 105 m max range
* ground is a bilinearly-interpolated height field, marched with a
  geometrically growing step and 14-step bisection
* objects (oriented boxes, z-cylinders, spheres) are intersected exactly,
  accelerated by a **2-D uniform-grid DDA** so only the primitives in the
  columns a ray traverses are tested
* **rolling shutter**: the carrier keeps moving while the mirror turns
* intensity = f(material reflectivity, incidence angle, range); **range- and
  material-dependent dropout** reproduces the sparsity gradient of a real
  sensor (~34 k returns inside 10 m vs ~2 k beyond 70 m)

Dataset: 5 training / 1 validation / 1 test city × 36 sweeps, plus a 48-sweep
demonstration sequence at 0.15 s spacing — 264 sweeps, ≈17 M labelled returns.

---

## 4. The 2.5D foveated grid engine

Five concentric rings; every ring is a dense `400 × 400` lattice, so the whole
map is a handful of rectangular arrays — no hashing, no trees, O(1) lookup, and
**no seam error**, because ring membership is decided by one continuous radius
function of the point itself.

| Ring | Radius | Cell | Cells in annulus |
|---|---|---|---|
| 0 | 0 – 10 m | **5 cm** | ~126 k |
| 1 | 10 – 20 m | 10 cm | ~94 k |
| 2 | 20 – 40 m | 20 cm | ~94 k |
| 3 | 40 – 70 m | 35 cm | ~85 k |
| 4 | 70 – 100 m | **50 cm** | ~64 k |

Per cell: semantic class, classifier confidence, `z_min / z_max / z_mean`,
return count and motion evidence.  From those:

* `z_max − z_min` → vertical extent (is there an overhang?)
* `z_min` under a canopy → **clearance**
* local elevation step → **curb** (5.5–35 cm rise)
* depression below a coarse median surface → **pothole** (>4.5 cm)

The last two are computed by `FoveatedGrid.refine` **on the fine rings only** —
they are geometrically undetectable at 35–50 cm cells, which is the whole
argument for foveation (see the *Resolution ablation* panel in the dashboard).

### Memory

| Representation | Allocated cells | @8 B/cell |
|---|---|---|
| **Adaptive (this work)** | 800 000 | **6.4 MB** |
| Uniform 5 cm over 200 m | 16 000 000 | 128 MB (**20×**) |
| Uniform 20 cm | 1 000 000 | 8 MB |
| 3-D voxel map, 5 cm, 12 m tall | 3.84 × 10⁹ | 15.4 GB (**2 400×**) |

---

## 5. The model: FoveaNet

The brief suggests PointNet++ or a sparse CNN.  We use the
**sparse-convolution-on-a-grid** branch of that family — what production
elevation-mapping stacks (PolarNet, Cylinder3D, SalsaNext) actually deploy —
because the deliverable *is* a grid: predicting in grid space removes the
3-D → 2-D re-projection step and makes the map update O(cells).

Input is a **foveated pyramid** of three square crops centred on the sensor:

| Level | Extent | Pixels | Resolution | Covers |
|---|---|---|---|---|
| L0 (fovea) | 21.12 m | 176² | **12 cm** | r < 10.6 m |
| L1 (mid) | 64.0 m | 160² | 40 cm | 10.6 – 32 m |
| L2 (periphery) | 204.8 m | 128² | 1.6 m | 32 – 102 m |

Each level is an 8-channel BEV raster: max / mean / min height, height span,
log return density, mean intensity, "groundness" (fraction of returns near the
cell floor) and a level code.

**One set of weights is applied to all three levels** — the network is
scale-equivariant by construction, so the parameter count stays at **228 k**,
and a learned FiLM level embedding lets it adapt its notion of "thin vertical
object" to the resolution it is given.  Encoder/decoder U-Net with GroupNorm
(batch-norm statistics are meaningless at batch 2 with per-level distributions
this different) and a FiLM-conditioned stem.

Total pyramid input is **73 k cells/frame** versus **16 M** for a uniform 5 cm
raster — a 219× reduction in the work the conv stack has to do.

Training: AdamW + OneCycle cosine, 42 × 140 steps at batch 2, masked
class-balanced cross-entropy with label smoothing, and on-the-fly augmentation
(full yaw, mirror, 0.88–1.12× scale, ego jitter, intensity jitter,
range-dependent return dropout).  Points stay on disk in int16 form and are
rasterised per sample, which keeps memory flat and makes every epoch see fresh
data.

---

## 6. Results

See `outputs/metrics.json` for the full dump and the dashboard for the
interactive version.  Headline numbers (held-out test city):

* **Latency** ≈ measured end-to-end on 2 CPU cores, reported in the dashboard
  *Latency budget* panel, together with the projected cost of a uniform 5 cm
  raster (≈200× the pixels → sub-1 fps).
* **Accuracy vs distance** — the *Accuracy vs distance* panel; the foveated
  design deliberately spends resolution where it protects the vehicle, so
  near-field IoU is the strongest.
* **Resolution ablation** — the same network and the same points, only the map
  resolution changes: adaptive 5/10/20/35/50 cm beats a flat 50 cm map and
  matches a flat 5 cm map at 1/20th of the memory.

---

## 7. Using the dashboard

| Control | Effect |
|---|---|
| **Layer** | Semantic / Elevation / Drivability / Confidence / Return density / Motion |
| **Source** | Network prediction, ground truth, or a difference map (green = agree, red = disagree) |
| **Grid** | *Adaptive foveated* (5 rings) vs *Uniform 50 cm* — the resolution a uniform map must adopt to reach 100 m |
| **Tilt / Relief** | 2.5-D oblique view: orthographic tilt with per-cell vertical extrusion and hillshading |
| **Ring bounds / Cell wire** | Ring boundaries with their cell sizes; ground-plane cell lattice |
| **Classes** | Click any legend entry to hide it |
| Mouse | Wheel = zoom about the cursor, drag = pan, hover = per-cell inspector (class, GT agreement, ring, cell size, elevation range, returns, confidence, motion) |
| Timeline | ▶ play, scrub, 0.5×–4× speed; `space` / `←` / `→` keyboard shortcuts |

---

## 8. Reproducibility & limitations

* Everything is seeded (`config.DATASET.seed0`, `config.TRAIN.seed`);
  `python3 -m src.dataset` recreates byte-identical sweeps.
* **Synthetic-only.** Absolute numbers describe this simulator, not a real
  sensor.  The purpose is to validate the *representation*: the ablation shows
  how much accuracy each resolution choice costs, which transfers to real data
  even though the absolute mIoU does not.
* Curbs are a one-cell (10 cm) step in the height field, not a true vertical
  face; a real sensor shows a sharper discontinuity.
* No multi-echo, no weather, no glass/mirror returns, no moving ego suspension
  model beyond a small vertical oscillation.
* Semantic labels are produced at the network's 12 cm fovea resolution and
  upsampled onto the 5 cm elevation lattice; sub-cell features (curb, pothole)
  are recovered there by the geometric refinement step, not by the network.

---

## 9. Citation-shaped summary

> FoveaMap: a foveated 2.5-D LiDAR mapping stack.  A shared-weight,
> level-conditioned BEV U-Net segments a three-level foveated pyramid into 11
> classes; predictions are transferred to points and accumulated into a
> five-ring variable-resolution elevation/semantics lattice (5 cm → 50 cm) with
> a geometric refinement stage that recovers curbs and potholes.  On a
> self-generated synthetic dataset of 264 labelled sweeps the map allocates
> 800 k cells instead of 16 M for a uniform 5 cm baseline (20× less memory)
> while retaining near-field detail, and runs at video rate on two CPU cores.
