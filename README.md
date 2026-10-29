# fortocorrpy

Topographic correction of optical satellite imagery over rugged terrain.

Over rugged terrain the observed reflectance of a surface depends on how the
ground is illuminated, which follows from the position of the sun relative to
the slope. `fortocorrpy` removes that terrain effect, so that pixels of the
same cover type compare across slopes and aspects. Six correction methods
are available through one interface, and a single call can apply several of
them or process a series of images of the same area.

The package targets forest. Where a method requires them, correction
coefficients are estimated from a forest sample, because the regression of
reflectance against illumination holds only within one land cover. Acquisition geometry is read from the image
grid and the acquisition time, so the package is not tied to any particular
satellite or product.

## Installation

Not on PyPI. Get the source from
[github.com/JaebeomK/fortocorrpy](https://github.com/JaebeomK/fortocorrpy)
and install from the source directory:

```bash
cd fortocorrpy
pip install .
```

For development, install in editable mode so source edits take effect without
reinstalling:

```bash
pip install -e .
```

### Dependencies

Declared without version floors (`numpy`, `rasterio`, `pyproj`). The versions
used during development and verification:

| package  | verified version             |
|----------|------------------------------|
| Python   | 3.9-3.13                     |
| numpy    | 2.4.4                        |
| rasterio | 1.5.0                        |
| pyproj   | 3.7.2                        |

## Quick start

Correcting one image. The inputs are a reflectance raster with invalid pixels
already set to nodata, on a projected CRS in metres, its acquisition time in
UTC, a DEM, and an NDVI raster for identifying forest. The DEM and the NDVI
raster are aligned to the image grid internally.

```python
from datetime import datetime, timezone
from fortocorrpy import Config, correct_image

result = correct_image(
    "scene.tif",
    datetime(2024, 6, 21, 3, 0, tzinfo=timezone.utc),
    "dem.tif", "ndvi.tif",
    Config(methods=("scsc",)),
)

if result.mask_result.skipped:
    print("skipped:", result.mask_result.quadrant_counts)
else:
    corrected = result.corrected["scsc"]      # ndarray[bands, rows, cols]
```

A scene is skipped, rather than corrected, when any of the four aspect
quadrants holds too few forest sample pixels — over gentle terrain, or when
cloud has removed one side of the image. Check `skipped` before using
`corrected`.

Images of the same area from several dates go in as a list — a bulk call.
The images may be different windows on the same pixel lattice, and each must
overlap the reference grid; a window is placed on that grid by slicing, without
resampling. Results come back on the one grid, so they stack directly.
`reference_path` names the raster the grid is taken from; without it the first
image's grid is used. The DEM is aligned once and reused across the call.

```python
results = correct_image(
    ["0621.tif", "0708.tif", "0724.tif"],
    [dt_0621, dt_0708, dt_0724],
    "dem.tif", "ndvi.tif",
    Config(methods=("scsc",)),
)
```

The `methods` keys are `cosine`, `c` (C-correction), `scs`, `scsc` (SCS+C),
`se` (Statistical-Empirical) and `er` (Empirical Rotation); see
[Methods](docs/methods.md) for the formulas.

A long series is better consumed one image at a time; see
[Usage](docs/usage.md).

## Documentation

- [Usage](docs/usage.md) — inputs, calling patterns, and what constrains a run.
- [Methods](docs/methods.md) — what is computed at each step, and the six
  correction formulas.
- [API reference](docs/api_reference.md) — functions, classes, and settings.
- [Design specification](DESIGN_SPEC.md) — module contracts and execution flow.

## License

[MIT](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

## References

Teillet, P.M., Guindon, B., Goodenough, D.G., 1982. On the Slope-Aspect
Correction of Multispectral Scanner Data. *Canadian Journal of Remote Sensing*
8, 84–106. https://doi.org/10.1080/07038992.1982.10855028

Gu, D., and Gillespie, A., 1998. Topographic Normalization of Landsat TM
Images of Forest Based on Subpixel Sun–Canopy–Sensor Geometry. *Remote Sensing
of Environment*, 64(2), p.166–175.
https://doi.org/10.1016/S0034-4257(97)00177-6

Soenen, S.A., Peddle, D.R., Coburn, C.A., 2005. SCS+C: a modified
sun-canopy-sensor topographic correction in forested terrain. *IEEE
Transactions on Geoscience and Remote Sensing* 43, 2148–2159.
https://doi.org/10.1109/TGRS.2005.852480

Tan, B., Wolfe, R., Masek, J., Gao, F., Vermote, E.F., 2010. An illumination
correction algorithm on Landsat-TM data, in: *2010 IEEE International
Geoscience and Remote Sensing Symposium*, 1964–1967.
https://doi.org/10.1109/IGARSS.2010.5653492

Tan, B., Masek, J.G., Wolfe, R., Gao, F., Huang, C., Vermote, E.F.,
Sexton, J.O., Ederer, G., 2013. Improved forest change detection with terrain
illumination corrected Landsat images. *Remote Sensing of Environment* 136,
469–483. https://doi.org/10.1016/j.rse.2013.05.013