# API Reference

The public API is exposed at the top level of `fortocorrpy`. Most users need
only `Config` and `correct_image`; the computation submodules
(`solar`, `terrain`, `illum`, `linear_C`, `T_correct`, `masking`, `evaluation`,
`io`) are public as well, so intermediate quantities can be produced directly.

Angles are radians throughout the computation modules. Degrees appear only
where a value is named for them: the `lat_deg`/`lon_deg` arguments of
`solar_position`, and the `slope_min_deg` and `aspect_quadrant_edges` settings
in `Config`, which are converted at the point of use.

## High-level API

### correct_image

```python
correct_image(image_path, datetime_utc, dem_path, forest_path, config, *,
              evaluate=False, reference_path=None)
```

Correct one or more images and return the results. All data are placed on a
single reference grid; the DEM and forest source are aligned to it internally.

Parameters:

- `image_path` — path to the QC-processed reflectance image (invalid pixels set to nodata), or a list of paths for bulk processing.
- `datetime_utc` — acquisition time as a `datetime` (UTC; a naive datetime is treated as UTC). With a list of images, a list of the same length.
- `dem_path` — path to a DEM covering the reference grid. A single path, not a list: slope and aspect are computed once and reused.
- `forest_path` — path to the forest source: an NDVI raster (`mask_source="ndvi"`) or a categorical land-cover raster (`mask_source="landcover"`). May be a list, one per image, for a series spanning seasons.
- `config` — a `Config` instance.
- `evaluate` (keyword, default `False`) — if `True`, also compute before/after reflectance-level metrics.
- `reference_path` (keyword, default `None`) — raster whose grid the outputs are placed on. `None` takes the grid from the first image.

Returns a `CorrectionResult` for one image; a list of them in input order for a
list of images, or a generator over the same results when `config.stream` is
set.

An image that is a different window on the reference grid's pixel lattice is
placed on the grid by slicing, so its values are unchanged and the area it does
not cover stays nodata. An image on another lattice (another CRS, another pixel
size, or a fractional-pixel offset) is refused rather than resampled. In a
single-image call this raises an exception; in a bulk call the failure is
recorded in that image's `CorrectionResult.error` and the batch continues.

### CorrectionResult

A dataclass holding the outputs:

- `corrected` — `dict` mapping `method -> ndarray[bands, rows, cols]`, or `None` if the scene was skipped or failed in a bulk call.
- `coefficients` — per-band list, shared across methods: a `BandCoefficients` per band, or `None` in every position when none of the selected methods needs a regression. `None` in place of the list if the scene was skipped or failed.
- `mask_result` — the `MaskResult` for the balanced sample, which carries the skip decision and is used for the regression and for evaluation. Built for every image, whether or not a method needs a regression. `None` if the image did not process.
- `band_indices` — list of original band positions corrected.
- `grid` — the `ImageGrid` the correction ran on, for writing the output.
- `metrics_before` — list of `BandMetrics` (one per band) over the regression sample, or `None`. Shared across methods.
- `metrics_after` — `dict` mapping `method -> list[BandMetrics]`, or `None`.
- `error` — string describing why the image did not process, or `None`.

## Configuration

### Config

A dataclass centralizing thresholds and branching. Validated on construction.

- `mask_source` (`'ndvi'` | `'landcover'`) — forest source type.
- `ndvi_threshold` (default `0.5`) — forest if NDVI ≥ this (NDVI source).
- `forest_class` — land-cover class treated as forest (land-cover source).
- `slope_min_deg` (default `5.0`) — minimum slope for the regression sample.
- `cos_i_threshold` (default `0.0`) — `cos i` above this is illuminated; denominator forms apply only here.
- `aspect_quadrant_edges` (default `(45, 135, 225, 315)`) — quadrant boundaries for aspect balancing.
- `min_samples_per_quadrant` (default `200`) — minimum samples per quadrant, else the scene is skipped.
- `sample_seed` (default `42`) — random seed for balanced subsampling.
- `methods` (default `('scsc',)`) — methods to apply.
- `band_indices` (default `None` = all bands) — 0-based band indices to correct.
- `n_workers` (default `1`) — images processed at once in a bulk call.
- `stream` (default `False`) — if `True`, a bulk call returns a generator yielding one result at a time, in input order, so each can be used and released.

Properties: `needs_regression`, `needs_slope_at_correction`.

### Constants

- `METHODS` — all supported method keys: `cosine`, `scs`, `c`, `scsc`, `se`, `er`.
- `REGRESSION_METHODS` — methods whose coefficients come from the regression: `c`, `scsc`, `se`, `er`.
- `SLOPE_TC_METHODS` — methods needing slope at correction: `scs`, `scsc`.
- `DENOMINATOR_METHODS` — denominator-form methods: `cosine`, `scs`, `c`, `scsc`.

## Solar (`solar`)

- `solar_position(when, lat_deg, lon_deg)` — solar `(zenith, azimuth)` in radians for a time and location given in degrees. Azimuth measured from north, clockwise.
- `solar_angles(when, transform, crs, shape)` — per-pixel `(theta_s, phi_s)` arrays in radians for an image grid.
- `julian_day(when)`, `pixel_lonlat(transform, crs, shape)` — helpers.

## Terrain (`terrain`)

- `slope_aspect(dem, hx, hy)` — `(alpha, beta)` in radians from a DEM, where `hx`, `hy` are pixel sizes in metres. Aspect is the compass bearing of the downslope direction: north 0, clockwise; flat terrain 0. Horn (1981).
- `horn_gradient(dem, hx, hy)` — gradient components (helper).

## Illumination (`illum`)

- `cos_incidence(theta_s, phi_s, alpha, beta)` — `(cos_i, cos_theta_s)` from angles in radians. No thresholding or masking.
- `cos_horizontal(theta_s)` — `cos(theta_s)`, from radians (helper).

## Masking (`masking`)

- `build_sample_mask(cos_i, alpha_rad, beta_rad, valid, forest_source, config)` — build the four-aspect balanced regression sample from slope and aspect in radians. Returns a `MaskResult`.
- `forest_mask(forest_source, config)` — forest boolean from NDVI threshold or land-cover.

### MaskResult

- `mask` — boolean array of the balanced sample.
- `skipped` — `True` if any quadrant had too few samples.
- `quadrant_counts` — per-quadrant candidate counts (diagnostic).
- `n_per_quadrant` — number drawn per quadrant.

## Regression (`linear_C`)

- `fit_band(cos_i, reflectance, mask)` — OLS of `reflectance = a·cos_i + b` over the sample. Returns `BandCoefficients`.
- `fit_bands(cos_i, reflectance, mask)` — per-band fit for a band stack.

### BandCoefficients

- `slope` (a), `intercept` (b), `c` (`b/a`, NaN if `a = 0`), `n_samples`, `mean_reflectance` (sample-mean reflectance, used by `se`).

## Correction (`T_correct`)

- `correct(method, reflectance, cos_i, cos_theta_s, *, cos_alpha=None, coeff=None, cos_i_threshold=0.0)` — apply one method to a single band.
  - `cosine`, `scs` allow `coeff=None`; `c`, `scsc`, `se`, `er` require `coeff` (a `BandCoefficients`).
  - `scs`, `scsc` require `cos_alpha`, the cosine of the slope angle, computed once from the slope raster.
  - Denominator forms apply where `cos i > cos_i_threshold` (NaN elsewhere); subtractive forms apply to all valid pixels regardless of the illumination threshold.

## Evaluation (`evaluation`)

- `evaluate_before(cos_i, reflectance, mask)` — per-band `BandMetrics` before correction (shared across methods).
- `evaluate_after(cos_i, corrected, mask)` — per-band `BandMetrics` after correction.
- `band_metrics(cos_i, reflectance, mask)` — metrics for a single band.

### BandMetrics

- `correlation` (cos i vs reflectance), `slope`, `std`, `mean`, `n_samples`.

## I/O (`io`)

- `read_grid(path)` — grid metadata only. Returns an `ImageGrid`.
- `read_image(path, band_indices=None, *, scale=1.0, offset=0.0)` — read grid + bands in a single open. Returns `(ImageGrid, reflectance_stack)`. nodata → NaN. Optional `scale`/`offset` convert DN to reflectance.
- `read_image_on_grid(path, grid, band_indices=None, *, scale=1.0, offset=0.0)` — read bands onto a given grid. An image on that grid is read as is; a different window on the same pixel lattice is placed by slicing, with the uncovered area left NaN; an image on another lattice raises `ValueError` naming what differs.
- `grids_match(src, dst)` — `True` when two `ImageGrid`s are the same grid, compared with a tolerance that absorbs float round-off.
- `lattice_offset(src, dst)` — `(row, col)` offset of `src` within `dst` when the two share a pixel lattice, else `None`.
- `align_to_grid(path, grid, *, resampling="bilinear", band_index=0)` — resample/reproject auxiliary data onto the reference `grid`; skips warping if already aligned.
- `write_geotiff(path, data, grid, *, nodata=np.nan, band_indices=None)` — write a GeoTIFF using the CRS and transform of `grid`. With `band_indices`, each band records its original 1-based position (`original_band_N`). Raises `ValueError` if the array does not match the grid's shape, or if `band_indices` does not have one entry per band.
- `read_bands`, `valid_from_band` — helpers.

### ImageGrid

- `transform`, `crs`, `width`, `height`, `count`, `nodata`, `dtype`; properties `shape`, `pixel_size`.