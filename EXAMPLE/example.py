"""
fortocorrpy usage example: full topographic-correction workflow + verification outputs

This folder (EXAMPLE) layout:
  example.py   <- this script
  data/        <- input data (4_bands.tif, DEM.tif, NDVI.tif, LC.tif, sensing_time.json)
  output/      <- created automatically on run; results are saved here

Reads inputs (image, DEM, forest, acquisition time) and writes the products of
the correction pipeline. Intermediate rasters are computed via the public
modules (solar, terrain, illum) without modifying the package.

Toggle options (True/False) at the top to choose what to output:
  1) Intermediate GeoTIFFs : SZA, SAA, slope, aspect, cos_i
  2) Scatter PNGs          : cos i vs reflectance (before + 6 methods, 2x2 per band)
  3) Numeric CSVs          : coefficients CSV (C etc.) + evaluation CSV (cos i-reflectance, before/after)
  4) Corrected GeoTIFFs    : 6-method corrected results

Run: python example.py
"""

import os
import csv
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fortocorrpy import (Config, correct_image, io, terrain, solar, illum)

# ============================================================
# Paths (relative to this script's location)
#   Anyone can take the EXAMPLE folder and run example.py:
#   it reads data/ in the same folder and creates output/ for results.
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "output")   # created automatically

IMAGE_PATH = os.path.join(DATA_DIR, "4_bands.tif")
DEM_PATH = os.path.join(DATA_DIR, "DEM.tif")
FOREST_PATH = os.path.join(DATA_DIR, "NDVI.tif")   # must match mask_source
SENSING_JSON = os.path.join(DATA_DIR, "sensing_time.json")

MASK_SOURCE = "ndvi"                 # 'ndvi' or 'landcover'
METHODS = ("cosine", "scs", "c", "scsc", "se", "er")
BAND_NAMES = ["Blue", "Green", "Red", "NIR"]

# ============================================================
# Output options (set only what you need to True)
# ============================================================
SAVE_INTERMEDIATE_TIFF = True   # SZA, SAA, slope, aspect, cos_i GeoTIFFs
SAVE_SCATTER_PNG       = True   # cos i vs reflectance scatter (before + 6 methods)
SAVE_METRICS_CSV       = True   # coefficients CSV + evaluation CSV
SAVE_CORRECTED_TIFF    = True   # 6-method corrected images

# Output file names (variables)
F_SZA    = os.path.join(OUT_DIR, "SZA.tif")
F_SAA    = os.path.join(OUT_DIR, "SAA.tif")
F_SLOPE  = os.path.join(OUT_DIR, "slope.tif")
F_ASPECT = os.path.join(OUT_DIR, "aspect.tif")
F_COSI   = os.path.join(OUT_DIR, "cos_i.tif")
F_COEF_CSV   = os.path.join(OUT_DIR, "coefficients.csv")
F_METRIC_CSV = os.path.join(OUT_DIR, "evaluation_metrics.csv")
def f_scatter(tag):  # tag = 'before' or a method name
    return os.path.join(OUT_DIR, f"cosi_{tag}.png")
def f_corrected(method):
    return os.path.join(OUT_DIR, f"corrected_{method}.tif")
# ============================================================


def load_sensing_time():
    with open(SENSING_JSON, "r", encoding="utf-8-sig") as f:
        text = json.load(f)["sensing_time"]
    # datetime.fromisoformat only accepts a "Z" suffix from Python 3.11 on.
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def scatter_panel(ax, x, y, title):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        ax.set_title(title + " (no data)"); return
    a, b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, s=2, alpha=0.3, color="#2c6fbb")
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, a * xs + b, "r-", lw=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("cos i", fontsize=8); ax.set_ylabel("reflectance", fontsize=8)
    ax.text(0.05, 0.95, f"slope={a:.4f}\nr={r:.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))


def scatter_figure(cos_i, refl_stack, mask, band_indices, suptitle, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    ci = cos_i[mask]
    for k, bidx in enumerate(band_indices):
        ax = axes[k // 2, k % 2]
        name = BAND_NAMES[bidx] if bidx < len(BAND_NAMES) else f"band{bidx}"
        scatter_panel(ax, ci, refl_stack[k][mask], name)
    for k in range(len(band_indices), 4):
        axes[k // 2, k % 2].axis("off")
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  PNG: {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    when = load_sensing_time()
    print("Acquisition (sensing) time:", when)

    need_eval = SAVE_METRICS_CSV or SAVE_SCATTER_PNG
    cfg = Config(mask_source=MASK_SOURCE, methods=METHODS)

    print("Running correction (6 methods)...")
    result = correct_image(
        image_path=IMAGE_PATH, datetime_utc=when, dem_path=DEM_PATH,
        forest_path=FOREST_PATH, config=cfg, evaluate=need_eval,
    )
    if result.mask_result.skipped:
        print("skipped: not enough samples.", result.mask_result.quadrant_counts); return
    mask = result.mask_result.mask
    band_indices = result.band_indices
    print("Samples per quadrant:", result.mask_result.quadrant_counts)

    # --- Intermediate values (public modules) ---
    # result.grid is the grid the correction ran on; write every product on it.
    grid = result.grid
    refl0 = io.read_image_on_grid(IMAGE_PATH, grid)
    dem = io.align_to_grid(DEM_PATH, grid, resampling="bilinear")
    hx, hy = grid.pixel_size
    alpha, beta = terrain.slope_aspect(dem, hx, hy)
    theta_s, phi_s = solar.solar_angles(when, grid.transform, grid.crs, grid.shape)
    cos_i, _ = illum.cos_incidence(theta_s, phi_s, alpha, beta)

    # --- 1) Intermediate GeoTIFFs ---
    if SAVE_INTERMEDIATE_TIFF:
        print("\n[Intermediate GeoTIFFs]")
        # The package works in radians; the angle products are written in
        # degrees, which is what these file names imply.
        for arr, fp in [(np.degrees(theta_s), F_SZA), (np.degrees(phi_s), F_SAA),
                        (np.degrees(alpha), F_SLOPE), (np.degrees(beta), F_ASPECT),
                        (cos_i, F_COSI)]:
            io.write_geotiff(fp, arr, grid)
            print(f"  TIFF: {fp}")

    # --- 2) Scatter PNGs ---
    if SAVE_SCATTER_PNG:
        print("\n[Scatter PNGs]")
        scatter_figure(cos_i, refl0, mask, band_indices,
                       "cos i vs reflectance (BEFORE)", f_scatter("before"))
        for method in METHODS:
            scatter_figure(cos_i, result.corrected[method], mask, band_indices,
                           f"cos i vs reflectance (AFTER: {method})", f_scatter(method))

    # --- 3) Numeric CSVs (coefficients + evaluation) ---
    if SAVE_METRICS_CSV:
        print("\n[Numeric CSVs]")
        # Coefficients CSV: coefficients is per-band list[BandCoefficients]
        # (regression fit once, shared across methods)
        with open(F_COEF_CSV, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["band", "a_slope", "b_intercept", "C", "mean_reflectance", "n_samples"])
            coeffs = result.coefficients
            if coeffs is not None:
                for k, bc in enumerate(coeffs):
                    name = BAND_NAMES[band_indices[k]] if band_indices[k] < len(BAND_NAMES) else f"band{band_indices[k]}"
                    w.writerow([name, f"{bc.slope:.6f}", f"{bc.intercept:.6f}",
                                f"{bc.c:.6f}", f"{bc.mean_reflectance:.6f}", bc.n_samples])
        print(f"  CSV: {F_COEF_CSV}")

        # Evaluation CSV (before + each method)
        with open(F_METRIC_CSV, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["method", "band", "slope", "correlation", "mean", "std", "n"])
            def write_metrics(method, mlist):
                for k, m in enumerate(mlist):
                    name = BAND_NAMES[band_indices[k]] if band_indices[k] < len(BAND_NAMES) else f"band{band_indices[k]}"
                    w.writerow([method, name, f"{m.slope:.6f}", f"{m.correlation:.6f}",
                                f"{m.mean:.6f}", f"{m.std:.6f}", m.n_samples])
            if result.metrics_before is not None:
                write_metrics("before", result.metrics_before)
            if result.metrics_after is not None:
                for method, mlist in result.metrics_after.items():
                    write_metrics(method, mlist)
        print(f"  CSV: {F_METRIC_CSV}")

    # --- 4) Corrected GeoTIFFs ---
    if SAVE_CORRECTED_TIFF:
        print("\n[Corrected GeoTIFFs]")
        for method in METHODS:
            fp = f_corrected(method)
            io.write_geotiff(fp, result.corrected[method], grid,
                             band_indices=band_indices)
            print(f"  TIFF: {fp}")

    print(f"\nDone. Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()