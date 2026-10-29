"""Shared fixtures: small synthetic rasters on a projected, metre-unit grid.

Every test raster is written to the same UTM grid unless a test deliberately
asks for a different one, so grid-mismatch behaviour has to be requested
explicitly rather than happening by accident.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

CRS = "EPSG:32652"
PIXEL = 30.0
ORIGIN_X = 300000.0
ORIGIN_Y = 4200000.0


@pytest.fixture
def write_raster(tmp_path):
    """Return ``write(name, data, ...) -> path`` for a float32 GeoTIFF."""

    def _write(name, data, *, nodata=None, crs=CRS,
               origin_x=ORIGIN_X, origin_y=ORIGIN_Y, pixel=PIXEL):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        count, height, width = arr.shape
        path = str(tmp_path / name)
        with rasterio.open(
            path, "w", driver="GTiff", height=height, width=width,
            count=count, dtype="float32", crs=crs,
            transform=from_origin(origin_x, origin_y, pixel, pixel),
            nodata=nodata,
        ) as dst:
            dst.write(arr)
        return path

    return _write


@pytest.fixture
def cone_dem():
    """A cone: every aspect quadrant is populated, slopes well above 5 deg."""

    def _dem(n=120, drop_per_pixel=12.0):
        yy, xx = np.mgrid[0:n, 0:n]
        centre = n / 2
        radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)
        return (1500.0 - radius * drop_per_pixel).astype(np.float32)

    return _dem


@pytest.fixture
def scene(write_raster, cone_dem):
    """A minimal correctable scene: image, DEM, and wall-to-wall forest NDVI."""
    n = 120
    return {
        "n": n,
        "image": write_raster("scene.tif", np.full((4, n, n), 0.30, np.float32)),
        "dem": write_raster("dem.tif", cone_dem(n)),
        "ndvi": write_raster("ndvi.tif", np.full((n, n), 0.80, np.float32)),
    }
