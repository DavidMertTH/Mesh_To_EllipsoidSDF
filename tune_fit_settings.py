"""Headless tuning harness for the full-object ellipsoid fit of meshes/T-Pose.fbx.

Sweeps fit settings, evaluates every config against ONE fixed high-res reference
SDF (settings-independent RMSE/MAE), and ranks by a balanced 0.5*rmse + 0.5*time
score.  Results are written incrementally to tune_results.csv.

Run with the project venv:
    .venv/Scripts/python.exe tune_fit_settings.py

Does NOT modify any existing source file — only imports + reuses their kernels.
"""
import csv
import os
import time
import traceback

import numpy as np
from PySide6 import QtCore
import warp as wp

wp.init()

from ellipsoid import best_device                       # noqa: E402
from sdf_compute import SdfComputer                     # noqa: E402
from optimization import (                              # noqa: E402
    OptimizationWorker,
    _ellipsoid_sdf_kernel_points,
)
from rig_panel import try_load_rigged                   # noqa: E402

MESH = "meshes/T-Pose.fbx"
MARGIN = 0.5                       # GUI default (slider 50/100)
REF_N = 192                        # reference SDF resolution for evaluation
N_EVAL = 120000                    # number of fixed eval points (uniform subset)
CSV_PATH = os.path.abspath("tune_results.csv")

DEV = best_device()
print("device:", DEV)

# ── load mesh once ────────────────────────────────────────────────────────────
rm = try_load_rigged(MESH)
VERTS = np.asarray(rm.vertices, np.float32).reshape(-1, 3)
FACES = np.asarray(rm.faces, np.int64).reshape(-1, 3)
print("mesh verts/faces:", VERTS.shape, FACES.shape)

app = QtCore.QCoreApplication([])

# One shared mesh-backed SdfComputer (used for both reference + per-n grids and
# passed to the worker for any local-fit path).
SDF = SdfComputer(device=DEV)
SDF.set_mesh(VERTS, FACES)

# ── per-n SDF cache (compute each distinct grid n only once) ──────────────────
_SDF_CACHE: dict[int, tuple] = {}    # n -> (SdfResult, t_sdf)


def get_sdf(n: int):
    if n not in _SDF_CACHE:
        wp.synchronize_device(DEV)
        t0 = time.perf_counter()
        res = SDF.compute_voxel_grid(n=n, margin=MARGIN)
        wp.synchronize_device(DEV)
        _SDF_CACHE[n] = (res, time.perf_counter() - t0)
    return _SDF_CACHE[n]


# ── fixed reference SDF + evaluation points (built ONCE, reused everywhere) ────
print(f"building reference SDF n={REF_N} ...")
REF, _ = get_sdf(REF_N)
ref_grid = REF.grid                               # (nz,ny,nx) float32
nz, ny, nx = ref_grid.shape
dx = float(REF.dx)
origin = np.asarray(REF.origin, np.float32)

# voxel-center world coords for the whole reference grid
iz, iy, ix = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
centers_world = (origin[None, :] + (np.stack(
    [ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(np.float32) + 0.5) * dx)
ref_flat = ref_grid.ravel().astype(np.float32)

# IMPORTANT: the analytic ellipsoid-union "SDF" (MertStein/Quílez) is only a true
# distance NEAR the surface; far from every ellipsoid the Quílez term
# k0*(k0-1)/k1 diverges to ~1e6, so a magnitude RMSE over far exterior points is
# meaningless (it swamps everything and is settings-independent).  We therefore
# score with two divergence-robust metrics instead:
#   * occupancy IoU  — sign agreement (inside/outside), well-defined for the union
#     and the real measure of reconstruction fidelity (under-coverage + protrusion
#     both hurt it);  PRIMARY quality metric.
#   * near-surface clamped MAE — magnitude accuracy in the band where the analytic
#     SDF is valid, clamped to +-0.1 (the training soft-clamp limit).
# Eval points are a UNIFORM random subset of all voxel centres so the IoU reflects
# true volumetric proportions.
rng = np.random.default_rng(12345)
SURF_BAND = 3.0 * dx          # band (world units) where magnitude MAE is computed
CLAMP = 0.1                   # matches training soft_clamp limit
all_idx = np.arange(ref_flat.size, dtype=np.int64)
eval_idx = (all_idx if all_idx.size <= N_EVAL
            else rng.choice(all_idx, size=N_EVAL, replace=False))
EVAL_PTS = np.ascontiguousarray(centers_world[eval_idx], np.float32)
EVAL_REF = np.ascontiguousarray(ref_flat[eval_idx], np.float32)
P = len(EVAL_PTS)
_ref_in = EVAL_REF < 0.0
_band_mask = np.abs(EVAL_REF) <= SURF_BAND
print(f"eval points: {P}  (interior={int(_ref_in.sum())} "
      f"near-surface={int(_band_mask.sum())})")

# upload eval points once
WP_PTS = wp.array(EVAL_PTS, dtype=wp.vec3, device=DEV)
WP_IDX = wp.array(np.arange(P, dtype=np.int32), device=DEV)


def eval_quality(c, r, q):
    """Score the fitted ellipsoid union against the fixed reference SDF.

    Returns (iou, surf_mae):
      * iou      — occupancy intersection-over-union (sign agreement). Higher is
        better; robust to the Quílez far-field magnitude divergence.
      * surf_mae — clamped magnitude MAE in the near-surface band only (where the
        analytic SDF is a valid distance). Lower is better.
    Reuses the project's own ``_ellipsoid_sdf_kernel_points`` (identical math to
    training) so the sign field matches what the fit optimises towards.
    """
    ne = int(len(c))
    if ne == 0:
        return 0.0, float("nan")
    centers = wp.array(np.ascontiguousarray(c, np.float32), dtype=wp.vec3, device=DEV)
    radii = wp.array(np.ascontiguousarray(r, np.float32), dtype=wp.vec3, device=DEV)
    rot_flat = wp.array(np.ascontiguousarray(q, np.float32).reshape(-1),
                        dtype=wp.float32, device=DEV)
    min_d = wp.zeros((P, ne + 1), dtype=wp.float32, device=DEV)
    out = wp.zeros(P, dtype=wp.float32, device=DEV)
    wp.launch(_ellipsoid_sdf_kernel_points, dim=P,
              inputs=[centers, radii, rot_flat, min_d, ne, WP_PTS, WP_IDX, out],
              device=DEV)
    wp.synchronize_device(DEV)
    pred = out.numpy()

    pred_in = pred < 0.0
    inter = int(np.count_nonzero(pred_in & _ref_in))
    union = int(np.count_nonzero(pred_in | _ref_in))
    iou = float(inter / union) if union > 0 else 0.0

    if _band_mask.any():
        cp = np.clip(pred[_band_mask], -CLAMP, CLAMP)
        cr = np.clip(EVAL_REF[_band_mask], -CLAMP, CLAMP)
        surf_mae = float(np.mean(np.abs(cp - cr)))
    else:
        surf_mae = float("nan")
    return iou, surf_mae


def run_fit(cfg):
    """Run one headless fit; return (t_sdf, t_fit, n_ell, c, r, q)."""
    res, t_sdf = get_sdf(cfg["n"])
    captured = {"crq": None}

    def cb(s, l, c, r, q, e):
        captured["crq"] = (np.asarray(c).copy(),
                           np.asarray(r).copy(),
                           np.asarray(q).copy())

    kw = dict(
        sdf_target_np=res.grid, origin=res.origin, dx=res.dx, n=res.n,
        method="adam", report_every=cfg.get("report_every", 50),
        thickness_np=res.thickness, sdf_computer=SDF,
        maintenance_every=0,
        num_ellipsoids=cfg["num_ellipsoids"],
        max_ellipsoids=cfg["max_ellipsoids"],
        num_steps=cfg["num_steps"],
        sdf_mode=cfg["sdf_mode"],
        superfit=cfg["superfit"],
        local_fit=cfg["local_fit"],
        symmetry_enabled=cfg.get("symmetry", False),
        lr_init=cfg.get("lr_init", 0.01),
        lr_final=cfg.get("lr_final", 0.0002),
        lr_decay_k=cfg.get("lr_decay_k", 7.0),
        merge_enabled=cfg.get("merge", True),
        spawn_underrep=cfg.get("spawn", True),
        split_enabled=cfg.get("split", True),
        prune_enabled=cfg.get("prune", True),
    )
    w = OptimizationWorker(**kw)
    w.step_visual.connect(cb)
    wp.synchronize_device(DEV)
    t0 = time.perf_counter()
    w.run()
    wp.synchronize_device(DEV)
    t_fit = time.perf_counter() - t0
    if captured["crq"] is None:
        raise RuntimeError("no ellipsoids emitted")
    c, r, q = captured["crq"]
    return t_sdf, t_fit, int(len(c)), c, r, q


# ── config list ───────────────────────────────────────────────────────────────
# Realistic GUI-derived baseline (scaled to a sweep-friendly budget):
#   GUI defaults: n=512, num_ell=60, max_ell=180, steps=7000, sdf_mode=0(Quílez),
#   superfit ON, local_fit OFF, merge/spawn/split/prune ON.
#   We use n=128 / steps=600 / max_ell=120 as the baseline operating point so the
#   full sweep fits the time budget; sensitivity sweeps move one knob at a time.
BASE = dict(
    n=128, num_ellipsoids=60, max_ellipsoids=120, num_steps=600,
    sdf_mode=4, superfit=True, local_fit=False, report_every=50,
    lr_init=0.01,
)


def cfg(name, **over):
    d = dict(BASE)
    d.update(over)
    d["name"] = name
    return d


CONFIGS = [
    cfg("baseline"),
    # grid n sensitivity
    cfg("n96", n=96),
    cfg("n160", n=160),
    cfg("n192", n=192),
    # num_steps sensitivity
    cfg("steps300", num_steps=300),
    cfg("steps1000", num_steps=1000),
    cfg("steps1500", num_steps=1500),
    # ellipsoid budget, superfit OFF (fixed count)
    cfg("fix40_nosf", superfit=False, num_ellipsoids=40, max_ellipsoids=40),
    cfg("fix60_nosf", superfit=False, num_ellipsoids=60, max_ellipsoids=60),
    cfg("fix90_nosf", superfit=False, num_ellipsoids=90, max_ellipsoids=90),
    # superfit budgets
    cfg("sf_max80", max_ellipsoids=80),
    cfg("sf_max180", max_ellipsoids=180),
    # sdf_mode comparison
    cfg("mode0_quilez", sdf_mode=0),
    cfg("mode1_scmin", sdf_mode=1),
    cfg("mode2_scmean", sdf_mode=2),
    # local fit on/off
    cfg("localfit_on", local_fit=True),
    # lr sensitivity
    cfg("lr_low", lr_init=0.005),
    cfg("lr_high", lr_init=0.02),
]

# focused grid over the most impactful knobs filled in after sensitivity below.

CSV_COLS = [
    "name", "n", "num_ellipsoids", "max_ellipsoids", "num_steps", "sdf_mode",
    "superfit", "local_fit", "lr_init", "reps",
    "t_sdf", "t_fit", "total_time", "eval_iou", "surf_mae",
    "n_ellipsoids", "balanced_score", "error",
]


def write_header():
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLS).writeheader()


def append_row(row):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writerow({k: row.get(k, "") for k in CSV_COLS})
        f.flush()


def run_all(configs, reps=1):
    rows = []
    for i, c in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {c['name']}  "
              f"n={c['n']} steps={c['num_steps']} mode={c['sdf_mode']} "
              f"sf={c['superfit']} lf={c['local_fit']} "
              f"ne={c['num_ellipsoids']}/{c['max_ellipsoids']} lr={c['lr_init']}")
        row = dict(
            name=c["name"], n=c["n"], num_ellipsoids=c["num_ellipsoids"],
            max_ellipsoids=c["max_ellipsoids"], num_steps=c["num_steps"],
            sdf_mode=c["sdf_mode"], superfit=c["superfit"],
            local_fit=c["local_fit"], lr_init=c["lr_init"], reps=reps,
        )
        try:
            best_iou = None
            best_mae = None
            n_ell = None
            t_fits = []
            t_sdf = 0.0
            for rep in range(reps):
                t_sdf, t_fit, ne, cc, rr, qq = run_fit(c)
                t_fits.append(t_fit)
                iou, mae = eval_quality(cc, rr, qq)
                if best_iou is None or iou > best_iou:
                    best_iou, best_mae, n_ell = iou, mae, ne
            t_fit_med = float(np.median(t_fits))
            total = t_sdf + t_fit_med
            row.update(
                t_sdf=round(t_sdf, 3), t_fit=round(t_fit_med, 3),
                total_time=round(total, 3),
                eval_iou=round(best_iou, 6), surf_mae=round(best_mae, 6),
                n_ellipsoids=n_ell,
            )
            print(f"    -> iou={best_iou:.4f} surf_mae={best_mae:.5f} "
                  f"t_fit={t_fit_med:.1f}s total={total:.1f}s n_ell={n_ell}")
            rows.append(row)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print("    !! ERROR:", row["error"])
            traceback.print_exc()
        append_row(row)
    return rows


def score(rows):
    ok = [r for r in rows if r.get("eval_iou") not in ("", None)
          and not (isinstance(r.get("eval_iou"), float)
                   and np.isnan(r["eval_iou"]))]
    if not ok:
        return rows
    # quality error = 1 - IoU (lower is better, same direction as time)
    qerr = np.array([1.0 - r["eval_iou"] for r in ok], float)
    times = np.array([r["total_time"] for r in ok], float)

    def norm(a):
        lo, hi = a.min(), a.max()
        return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo) / (hi - lo)

    qn, tn = norm(qerr), norm(times)
    for r, x, y in zip(ok, qn, tn):
        r["balanced_score"] = round(float(0.5 * x + 0.5 * y), 4)
    return rows


def main():
    write_header()
    t_start = time.time()

    rows = run_all(CONFIGS, reps=1)

    # ── focused grid over the most impactful knobs (n x steps x budget) ──
    focused = [
        cfg("F_n128_s800_sf120", n=128, num_steps=800, max_ellipsoids=120),
        cfg("F_n160_s800_sf120", n=160, num_steps=800, max_ellipsoids=120),
        cfg("F_n128_s1000_sf150", n=128, num_steps=1000, max_ellipsoids=150),
        cfg("F_n128_s400_sf90", n=128, num_steps=400, max_ellipsoids=90),
        cfg("F_n96_s800_sf120", n=96, num_steps=800, max_ellipsoids=120),
    ]
    rows += run_all(focused, reps=1)

    rows = score(rows)
    # rewrite CSV with scores filled
    write_header()
    for r in rows:
        append_row(r)

    # ── shortlist re-timed with 2 reps for stability ──
    scored = [r for r in rows if isinstance(r.get("balanced_score"), float)]
    scored.sort(key=lambda r: r["balanced_score"])
    shortlist_names = [r["name"] for r in scored[:4]]
    print("\nshortlist (re-timing 2 reps):", shortlist_names)
    by_name = {c["name"]: c for c in CONFIGS}
    by_name.update({c["name"]: c for c in focused})
    short_cfgs = [by_name[n] for n in shortlist_names if n in by_name]
    short_rows = run_all(short_cfgs, reps=2)
    # merge re-timed rows back (replace)
    short_map = {r["name"]: r for r in short_rows}
    rows = [short_map.get(r["name"], r) for r in rows]
    rows = score(rows)
    write_header()
    for r in rows:
        append_row(r)

    # ── final ranking print ──
    scored = [r for r in rows if isinstance(r.get("balanced_score"), float)]
    print(f"\n==== total wall time: {(time.time() - t_start) / 60:.1f} min ====")
    print(f"\nCSV: {CSV_PATH}\n")

    by_bal = sorted(scored, key=lambda r: r["balanced_score"])
    by_iou = sorted(scored, key=lambda r: -r["eval_iou"])
    by_time = sorted(scored, key=lambda r: r["total_time"])

    def show(title, lst):
        print(f"--- {title} ---")
        for r in lst[:6]:
            print(f"  {r['name']:>20}  iou={r['eval_iou']:.4f}  "
                  f"surf_mae={r['surf_mae']:.5f}  "
                  f"total={r['total_time']:.1f}s  n_ell={r['n_ellipsoids']}  "
                  f"bal={r['balanced_score']}")

    show("BALANCED (best first)", by_bal)
    show("QUALITY-best (highest IoU)", by_iou)
    show("TIME-best (fastest)", by_time)

    if by_bal:
        win = by_bal[0]
        print("\n*** BALANCED WINNER ***")
        for k in CSV_COLS:
            print(f"  {k}: {win.get(k)}")


if __name__ == "__main__":
    main()
