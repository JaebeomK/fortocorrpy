# fortocorrpy Design Specification

Version 0.1.0. Requires Python 3.9 or later, with numpy, rasterio and pyproj.

The as-built design. Usage is in `docs/usage.md`, the background to the formulas
in `docs/methods.md`.

---

## 1. Purpose and inputs

Removes the topographic effect from each pixel of an optical satellite image.
Six published methods are available through one interface, applied singly or
together, with reflectance-level metrics optionally reported alongside so that
methods can be compared.

Four things are required of the input.

- A QC-processed reflectance image. Cloud, cloud shadow and other invalid pixels
  must already be set to nodata; the package performs no quality assessment of
  its own.
- A projected CRS in metres. Slope and aspect use the pixel size as a horizontal
  distance.
- A DEM covering the reference grid. Slope and aspect read a 3x3 neighbourhood,
  so a DEM that stops inside the grid leaves the slope and aspect invalid for one
  pixel inside that boundary.
- A UTC acquisition time, passed as a `datetime`. The package does not read
  product metadata files.

Any raster rasterio can read is accepted as input; output is always GeoTIFF.

---

## 2. Correction methods

The six formulas follow their source literature. Slope and aspect use the Horn
(1981) weighted 3x3 finite difference. Coefficients are estimated from a forest
sample, and the correction is applied to every valid pixel it can be applied to.

### 2.1 Notation and conventions

| symbol   | meaning                            |
|----------|------------------------------------|
| θ_s, φ_s | solar zenith and azimuth angle     |
| α, β     | slope and aspect                   |
| i        | local incidence angle              |
| ρ_T, ρ_H | observed and corrected reflectance |
| a, b     | regression slope and intercept     |
| C        | b / a                              |
| ρ̄       | sample-mean reflectance            |

```
cos i   = cos θ_s · cos α + sin θ_s · sin α · cos(φ_s − β)
cos i_h = cos θ_s
```

All angles are radians internally, and φ_s and β are both compass bearings, zero
at north and increasing clockwise. The thresholds a user sets are the exception:
`slope_min_deg` and `aspect_quadrant_edges` are degrees, converted where they are
consumed.

### 2.2 The six formulas

| key      | method                | formula                                   | regression | cos α | source              |
|----------|-----------------------|-------------------------------------------|------------|-------|---------------------|
| `cosine` | Cosine                | ρ_T · cos θ_s / cos i                     | no         | no    | Teillet et al. 1982 |
| `c`      | C-correction          | ρ_T · (cos θ_s + C) / (cos i + C)         | yes        | no    | Teillet et al. 1982 |
| `scs`    | SCS                   | ρ_T · (cos α · cos θ_s) / cos i           | no         | yes   | Gu & Gillespie 1998 |
| `scsc`   | SCS+C                 | ρ_T · (cos α · cos θ_s + C) / (cos i + C) | yes        | yes   | Soenen et al. 2005  |
| `se`     | Statistical-Empirical | ρ_T − (a · cos i + b) + ρ̄                | yes        | no    | Teillet et al. 1982 |
| `er`     | Empirical Rotation    | ρ_T − a · (cos i − cos θ_s)               | yes        | no    | Tan et al. 2013     |

### 2.3 The groups that drive branching

`REGRESSION_METHODS` (`c`, `scsc`, `se`, `er`) need coefficients, so a regression
is fitted per band.

`SLOPE_TC_METHODS` (`scs`, `scsc`) use cos α, so α is kept until the correction
step.

`DENOMINATOR_METHODS` (`cosine`, `scs`, `c`, `scsc`) put cos i in a denominator,
so they are applied only above the illumination threshold and are missing below
it.

All four guard the denominator itself: below 1e-6 in magnitude it yields a
missing value rather than an extreme one. For `cosine` and `scs` the denominator
is cos i, for `c` and `scsc` it is cos i + C, which is why they also have to
guard `cos i = −C`. `se` and `er` have no denominator and so are applied to
every valid pixel, independently of the illumination threshold.

A regression can fail to be defined even when the sample is large enough. Where
`cos i` has no variance over the sample the slope `a` is undefined, and the fit
returns `a = 0`, `C = nan`. The four regression methods then reduce to the
identity: no brightness trend with illumination was observed, so there is
nothing to correct. `cosine` and `scs` do not use the coefficients and are
unaffected. Coefficients are per band and shared by every selected method, so
this does not differ between the regression methods, and the image is not
treated as a failure.

### 2.4 Adding a method

Add the key to `METHODS`, add the formula branch to `T_correct.correct`, and add
the key to whichever groups it belongs to: `REGRESSION_METHODS` if it uses
regression coefficients, `SLOPE_TC_METHODS` if it uses cos α,
`DENOMINATOR_METHODS` if it puts cos i in a denominator. No other module changes.

---

## 3. Design principles

1. Each module's responsibility is complete within that module.
2. Settings are gathered into one object.
3. File access belongs to `io`, and all data are aligned to one reference grid.
4. The correction function returns arrays; writing is a separate function.
5. Memory is managed explicitly.

---

## 4. Module layout

Eleven modules, with one-way dependencies. `config` is referenced by all of them,
`io` is called only by `pipeline`, and the computation modules between them are
pure functions over arrays that do not call one another. `pipeline` assembles.

`config` owns the settings and their validation. It provides `Config`,
`METHODS`, `REGRESSION_METHODS`, `SLOPE_TC_METHODS` and `DENOMINATOR_METHODS`.
It does not read files or hold paths.

`io` owns all disk access. It provides `ImageGrid`, `read_grid`, `read_image`,
`read_image_on_grid`, `read_bands`, `valid_from_band`, `grids_match`,
`lattice_offset`, `align_to_grid` and `write_geotiff`. It does not compute
anything.

`solar` owns the solar position, derived from a grid and a time. It provides
`solar_angles`, `solar_position`, `pixel_lonlat` and `julian_day`. It does not
know about terrain.

`terrain` owns slope and aspect, derived from a DEM. It provides `horn_gradient`
and `slope_aspect`. It does not align the DEM or form `cos i`.

`illum` owns the illumination condition. It provides `cos_incidence` and
`cos_horizontal`. It does not threshold or mask.

`masking` owns the sample and the skip decision. It provides `MaskResult`,
`forest_mask` and `build_sample_mask`. It does not fit anything.

`linear_C` owns the per-band regression. It provides `BandCoefficients`,
`fit_band` and `fit_bands`. It does not re-check sample eligibility.

`T_correct` owns the six formulas. It provides `correct`. It does not choose
methods or build samples.

`evaluation` owns the reflectance-level metrics. It provides `BandMetrics`,
`band_metrics`, `evaluate_before` and `evaluate_after`. It does not judge what
the metrics mean.

`pipeline` owns the assembly, the branching and the release of memory. It
provides `correct_image` and `CorrectionResult`. It does no numerical work.

`__init__` owns the public surface: the re-exports and `__version__`.

---

## 5. Configuration

`Config` is a dataclass validated on construction. It holds settings, never
paths. Every field has a default, so only what a purpose requires is changed.

| field                      | default               | consumed by             | usually                            |
|----------------------------|-----------------------|-------------------------|------------------------------------|
| `methods`                  | `("scsc",)`           | `pipeline`, `T_correct` | set per study                      |
| `mask_source`              | `"ndvi"`              | `masking`               | set per dataset                    |
| `ndvi_threshold`           | `0.5`                 | `masking`               | set per dataset                    |
| `forest_class`             | `None`                | `masking`               | set when `mask_source="landcover"` |
| `band_indices`             | `None`                | `io`, `pipeline`        | set per dataset                    |
| `slope_min_deg`            | `5.0`                 | `masking`               | left at the default                |
| `cos_i_threshold`          | `0.0`                 | `masking`, `T_correct`  | left at the default                |
| `aspect_quadrant_edges`    | `(45, 135, 225, 315)` | `masking`               | left at the default                |
| `min_samples_per_quadrant` | `200`                 | `masking`               | left at the default                |
| `sample_seed`              | `42`                  | `masking`               | left at the default                |
| `n_workers`                | `1`                   | `pipeline`              | set per machine                    |
| `stream`                   | `False`               | `pipeline`              | set per memory budget              |

`cos_i_threshold` serves twice: it bounds the regression sample, and it bounds
where the methods with cos i in a denominator apply.

The properties `needs_regression` and `needs_slope_at_correction` derive the
branching of Section 2.3 from `methods`.

---

## 6. Inputs and grid alignment

One reference grid per call, and everything is placed on it. The grid is
described by `io.ImageGrid` (transform, CRS, width, height, count, nodata, dtype)
and is taken from the image, or from `reference_path` when one is given. The
aligned DEM, slope and aspect, the solar angles, `cos i`, the sample mask and
every corrected array sit on it, and it is returned with the result.

Two notions are distinguished.

Two grids are the same grid when their CRS and pixel geometry agree, their
dimensions are equal, and their origins fall within tolerance of each other.

They share a lattice when their CRS and pixel geometry agree and their origins
differ by a whole number of pixels within tolerance. Their dimensions may differ.

The tolerances are relative for pixel size and rotation (1e-9), and a fraction of
a pixel for the origin (1e-4, three millimetres on a 30 m grid). Below that is
coordinate noise; a larger difference is a different lattice. Exact equality is
not used, because the same grid written by two tools can differ in the last
decimal.

| input                                      | rule                                                                                                                      |
|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| DEM, forest source                         | reprojected and resampled onto the reference grid regardless of CRS or resolution; the DEM must cover the grid completely |
| image on the same grid                     | read as is                                                                                                                |
| image on the same lattice, overlapping     | the overlapping window is placed by slicing; the area it does not cover is left missing                                   |
| image on the same lattice, not overlapping | refused                                                                                                                   |
| image on another lattice                   | refused, never resampled to fit                                                                                           |

Ancillary data may be resampled; reflectance may not, because it is what the
correction acts on.

Ancillary data are aligned to exactly the reference grid, so any margin beyond it
is not used. The slope and aspect of the grid's own outermost row and column are
therefore extrapolated by edge replication, whatever the extent of the DEM. What
the DEM must do is cover the grid without gaps.

Validity comes from the image's own nodata. Nodata becomes NaN on read, and a
pixel is valid where every selected band is finite. The area an image does not
cover, and the pixels a denominator method cannot correct, arrive through the
same channel, so there is one notion of invalid. `io.valid_from_band` reads a
single band and gives the same answer only under the usual condition that QC was
applied across all bands together.

---

## 7. Correction

```
correct_image(image_path, datetime_utc, dem_path, forest_path, config,
              *, evaluate=False, reference_path=None)
```

Passing the image paths and acquisition times as lists invokes bulk processing.
The two lists must be the same length, as must the forest source when it is given
as a list. The DEM is always a single path. `reference_path` names an image
covering the whole area whose grid every result is produced on; without it the
first image's grid is used. With `evaluate=True` the reflectance-level metrics
are computed before and after correction over the regression sample.

Processing begins with grid alignment, derives slope and aspect from the DEM and
the solar geometry from the acquisition time, and combines them into `cos i`.
The forest sample and the skip decision are built for every image. Where a
selected method needs coefficients, a regression is then fitted per band and the
selected methods share those coefficients. Bands are the outer loop and methods
the inner one, so one regression per band serves every method.

Results are returned in memory rather than written. A single request returns one
`CorrectionResult`; a list request returns a list of them, or a generator over
the same results when `stream` is set.

| field                             | contents                                                                                |
|-----------------------------------|-----------------------------------------------------------------------------------------|
| `corrected`                       | the corrected array per method; `None` when skipped or errored                          |
| `coefficients`                    | the regression coefficients per band; elements are `None` when no regression was needed |
| `mask_result`                     | the sampling outcome and the skip decision; `None` on error                             |
| `band_indices`                    | where each output band sat in the input stack                                           |
| `grid`                            | the reference grid the correction ran on                                                |
| `metrics_before`, `metrics_after` | the before and after metrics, with `evaluate=True`                                      |
| `error`                           | the failure, when the image could not be processed                                      |

An image whose sample falls short is left uncorrected for every selected method
and carries only the diagnostics on `mask_result`; an image that could not be
processed carries only `error`. Neither carries corrected arrays. A
failure is recorded on the result only for a list request, so that one bad image
does not cost the rest; a single request raises, having nothing else to protect.

What a bulk call shares and what it derives per image:

| derived                         | scope                                   |
|---------------------------------|-----------------------------------------|
| the reference grid              | once                                    |
| slope and aspect                | once                                    |
| the forest source               | once, or per image when given as a list |
| solar geometry                  | per image                               |
| the sample and the coefficients | per image                               |

`n_workers` sets how many images are processed concurrently in a thread pool and
`stream` sets when results are returned. Both follow the input order: `stream`
turns the list into a generator without reordering it, so a result waits if an
earlier image is still running. Each worker holds its own file handles and
arrays, while the slope, aspect and forest arrays derived once are shared
read-only.

---

## 8. Writing

`io.write_geotiff` writes the output. It takes an array together with the grid
and band indices carried on the result, and produces a GeoTIFF with the CRS,
transform and dimensions of that grid, which are the first image's only when
`reference_path` was not used. Each band records its original position in
the input stack as its description, so a band subset stays identifiable.

An array whose rows and columns do not match the grid is refused, as is a
`band_indices` without one entry per band: the grid's CRS and transform are
written as they are, so an array of another size would be georeferenced into the
wrong place.

Because the correction returns arrays, they may be passed to a subsequent step
without being written at all, and any intermediate product sharing the grid can
be written by the same function.

---

## 9. Memory

An image is held whole; there is no block processing, so an image too large for
memory must be split by the caller. What is managed instead is the number of
arrays alive at once. A file is opened once and the required bands read together,
continuous raster arrays are float32 as a rule (masks are boolean), and
intermediates are released at the step where they stop being needed. Slope is
kept to the correction step only when a selected method requires it.

In bulk, `stream` removes the accumulation of the list. Disabled, every corrected
array stays alive until the call returns; enabled, results are yielded one at a
time in input order and the caller can release each after use, so memory follows
the number in flight rather than the length of the list.