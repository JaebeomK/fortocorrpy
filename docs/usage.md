# Usage

## Inputs

Reflectance raster. Must already be QC-processed, with invalid pixels set
to nodata. The package does no cloud or shadow screening of its own; it reads
validity from that nodata and from nothing else. The CRS must be projected in
metres: slope and aspect are derived from the pixel size.

Acquisition time. A `datetime` in UTC. A value carrying `tzinfo` is
converted to UTC; a naive value is taken to be UTC already.

DEM. The source of slope and aspect.

Forest source. Decides which pixels may enter the regression sample. An
NDVI raster (the default) or a land-cover raster.

The DEM and the forest source may be on any CRS or resolution; both are
reprojected and resampled onto the image grid.

## Configuration

Common:

| setting                    | default               | meaning                                                                     |
|----------------------------|-----------------------|-----------------------------------------------------------------------------|
| `methods`                  | `("scsc",)`           | correction methods to apply                                                 |
| `mask_source`              | `"ndvi"`              | forest source type: `"ndvi"` or `"landcover"`                               |
| `ndvi_threshold`           | `0.5`                 | NDVI at or above which a pixel counts as forest                             |
| `forest_class`             | `None`                | forest class value in the land-cover raster                                 |
| `slope_min_deg`            | `5.0`                 | minimum slope for the sample, in degrees                                    |
| `cos_i_threshold`          | `0.0`                 | minimum illumination: bounds the sample and where denominator methods apply |
| `aspect_quadrant_edges`    | `(45, 135, 225, 315)` | aspect quadrant edges, in degrees                                           |
| `min_samples_per_quadrant` | `200`                 | samples required in each quadrant                                           |
| `sample_seed`              | `42`                  | seed for the per-quadrant draw                                              |
| `band_indices`             | `None`                | zero-based bands to correct; `None` is all                                  |

For bulk processing — a list of images in one call:

| setting     | default | meaning                              |
|-------------|---------|--------------------------------------|
| `n_workers` | `1`     | images processed at once             |
| `stream`    | `False` | `True` returns results one at a time |

Both of the last two are about memory. A bulk call holds every corrected array
until it returns, which for full-size tiles and several methods runs to over a
gigabyte per image; `stream=True` yields the results one at a time, in input
order, so each can be used and released before the next arrives, and
`n_workers` sets how many images are in flight at once.

```python
from fortocorrpy import Config
config = Config(methods=("scsc",))
```

`methods` takes one or several of `"cosine"`, `"scs"`, `"c"`, `"scsc"`,
`"se"`, `"er"`. Given several, one call applies them all. `"scsc"` (SCS+C) is
the usual choice over forest; the formulas and what separates them are in
[Methods](methods.md).

## Correcting

`correct_image` returns results in memory, not files. Writing them out is a
separate step, shown at the end of this page.

```python
from datetime import datetime, timezone
from fortocorrpy import correct_image

dt = datetime(2024, 6, 21, 3, 0, tzinfo=timezone.utc)

result = correct_image("scene.tif", dt, "dem.tif", "ndvi.tif", config)

if result.mask_result.skipped:
    print("no correction:", result.mask_result.quadrant_counts)
else:
    arr = result.corrected["scsc"]
```

Not every scene can be corrected, so the skip check belongs in the first thing
you write. Why a scene is left alone is set out under *What comes back*.

Bulk — several images in one call:

```python
results = correct_image(
    ["0621.tif", "0708.tif", "0724.tif"],
    [dt_0621, dt_0708, dt_0724],
    "dem.tif", "ndvi.tif", config,
)

# one result at a time
streaming = Config(methods=("scsc",), stream=True)

for result in correct_image(images, times, "dem.tif", "ndvi.tif", streaming):
    ...
```

A bulk call works on one reference grid: slope and aspect are computed once
from the DEM and shared across the call, which is why the DEM is a single path,
not a list. The reference grid is the first image's, unless `reference_path`
names a raster to take it from — a base image covering the whole area, when the
images are smaller windows of it. Every result comes back on that grid.

An image that is a different window on the same pixel lattice is placed on the
reference grid by slicing, so its values are unchanged and the area it does not
cover stays nodata. An image must still overlap the reference grid: one that
shares the lattice but falls entirely outside it is an error, not an empty
result. An image on a different lattice — another CRS, another pixel size, or a
fractional-pixel offset — is refused rather than resampled to fit:
the reflectance is what the correction acts on, and a tile that merely overlaps
the grid would otherwise come back as a mostly-empty but ordinary-looking
product. Such an image carries `error` and produces nothing to write.

```python
results = correct_image(
    ["0621.tif", "0708.tif"], [dt_0621, dt_0708],
    "dem.tif", "ndvi.tif", config,
    reference_path="base_image.tif",
)
```

The forest source may be a list, for a series that spans seasons.

```python
results = correct_image(
    ["spring.tif", "summer.tif"],
    [dt_spring, dt_summer],
    "dem.tif",
    ["ndvi_spring.tif", "ndvi_summer.tif"],
    config,
)
```

## What comes back

One image returns one `CorrectionResult`; a bulk call returns a list of them,
in input order, or a generator over the same results with `stream=True`.

| attribute        | type                   | contents                                  |
|------------------|------------------------|-------------------------------------------|
| `corrected`      | `dict` or `None`       | one array per method                      |
| `coefficients`   | `list` or `None`       | fitted line per band                      |
| `mask_result`    | `MaskResult` or `None` | how the regression sample turned out      |
| `band_indices`   | `list`                 | where each output band sat in the input   |
| `grid`           | `ImageGrid`            | the grid the correction ran on            |
| `metrics_before` | `list` or `None`       | with `evaluate=True`                      |
| `metrics_after`  | `dict` or `None`       | with `evaluate=True`                      |
| `error`          | `str` or `None`        | the failure, if the image did not process |

```python
result.mask_result.skipped
result.coefficients[0].c
result.metrics_after["scsc"][0].correlation
```

Iterate a bulk result rather than indexing it, so the same code reads whether
or not `stream` is set; `list(...)` restores indexing where it is needed. The
nested types — `MaskResult`, `BandCoefficients`, `BandMetrics`, `ImageGrid` —
are set out in the [API reference](api_reference.md).

An image is left alone when any aspect quadrant holds fewer candidates than
`min_samples_per_quadrant`: the regression would lean on one direction.
`mask_result.skipped` is then true, `corrected` and `coefficients` are `None`,
and `quadrant_counts` says which direction fell short. An image that could not
be processed at all carries `error` instead, and its `mask_result` is `None`.

A single-image call raises instead of setting `error`, so the check below
matters for bulk calls, where one bad file must not discard the rest.

```python
if result.error:
    print(result.error)                         # unreadable file, wrong lattice
elif result.mask_result.skipped:
    print(result.mask_result.quadrant_counts)   # {'N': 812, 'E': 41, ...}
else:
    arr = result.corrected["scsc"]
```

`evaluate=True` — an argument to `correct_image`, not a `Config` setting —
measures the regression sample before and after correction: correlation
between `cos i` and reflectance, regression slope, mean, standard deviation,
and sample size, per band. A correction that has removed the terrain effect
leaves correlation and slope near zero, which is how methods are compared on a
site.

```python
result = correct_image("scene.tif", dt, "dem.tif", "ndvi.tif",
                       Config(methods=("cosine", "scsc", "er")),
                       evaluate=True)

for method, metrics in result.metrics_after.items():
    print(method, [round(m.correlation, 3) for m in metrics])
```

## Writing the output

The array and the grid come off the result; the path is the caller's.

```python
from fortocorrpy.io import write_geotiff

out_path = "corrected/scene_scsc.tif"

write_geotiff(out_path, result.corrected["scsc"], result.grid,
              band_indices=result.band_indices)
```

The output carries the CRS and transform of `result.grid`, the reference grid
the correction ran on, so every result of a call overlays the others exactly;
`band_indices` labels each band with its position in the input stack.

A bulk result

A result carries no path of its own. What ties it to an image is the order: the list, and the generator under stream=True, come back in the order the images went in, so pairing the two with zip names the outputs.

python
from pathlib import Path
from fortocorrpy.io import write_geotiff

images = ["0621.tif", "0708.tif", "0724.tif"]
results = correct_image(images, times, "dem.tif", "ndvi.tif", config)

for path, result in zip(images, results):
    stem = Path(path).stem

    if result.error:                       # unreadable file, wrong lattice
        print(stem, "failed:", result.error)
        continue
    if result.mask_result.skipped:         # sample too thin in one quadrant
        print(stem, "skipped:", result.mask_result.quadrant_counts)
        continue

    for method, arr in result.corrected.items():
        write_geotiff(f"corrected/{stem}_{method}.tif", arr, result.grid,
                      band_indices=result.band_indices)

The two guards come first because corrected is None in both cases. They are not the same outcome: a skipped image met no correction condition and reports its quadrant counts, while a failed one never got that far and reports error.

Several methods in one call give one file per method per image, as the inner loop shows. Every file of the call sits on the same grid, so the whole set stacks directly.

## The example script

`EXAMPLE/example.py` runs the whole sequence on the data in `EXAMPLE/data/`
and writes to `EXAMPLE/output/`: the intermediate rasters (solar zenith and
azimuth, slope, aspect, `cos i`), `cos i` against reflectance as a scatter per
method, the coefficients and evaluation metrics as CSV, and the corrected
images. Toggles at the top of the file select which of those to produce.
