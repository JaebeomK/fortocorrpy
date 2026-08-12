"""Slope and aspect: checked against the geometric definition.

Aspect is the compass bearing of the *downslope* direction, clockwise from
north, matching the solar azimuth convention in :mod:`fortocorrpy.solar`. This
is the same convention GDAL's ``gdaldem aspect`` produces.
"""

import numpy as np
import pytest

from fortocorrpy import terrain


def test_flat_terrain_has_zero_slope():
    alpha, _ = terrain.slope_aspect(np.full((5, 5), 100.0, np.float32), 30.0, 30.0)
    assert np.allclose(alpha, 0.0)


def test_flat_terrain_aspect_is_zero():
    """Aspect is undefined on flat ground; the documented convention is 0."""
    _, beta = terrain.slope_aspect(np.full((5, 5), 100.0, np.float32), 30.0, 30.0)
    assert np.allclose(beta, 0.0)


def test_slope_is_returned_in_radians():
    """Regression: a rise of 30 m over a 30 m pixel is 45 deg = pi/4 radians."""
    dem = np.array([[0, 30, 60]] * 3, dtype=np.float32)
    alpha, _ = terrain.slope_aspect(dem, 30.0, 30.0)
    assert abs(float(alpha[1, 1]) - np.pi / 4) < 1e-4
    assert float(alpha[1, 1]) < np.pi / 2


@pytest.mark.parametrize(
    "name, dem, expected_deg",
    [
        # Elevation decreases eastward -> the downslope direction faces east.
        ("downslope east", np.array([[60, 30, 0]] * 3, dtype=np.float32), 90.0),
        ("downslope west", np.array([[0, 30, 60]] * 3, dtype=np.float32), 270.0),
        ("downslope south",
         np.array([[60] * 3, [30] * 3, [0] * 3], dtype=np.float32), 180.0),
        ("downslope north",
         np.array([[0] * 3, [30] * 3, [60] * 3], dtype=np.float32), 0.0),
    ],
)
def test_aspect_cardinal_directions(name, dem, expected_deg):
    _, beta = terrain.slope_aspect(dem, 30.0, 30.0)
    got = np.rad2deg(float(beta[1, 1])) % 360.0
    assert abs((got - expected_deg + 180.0) % 360.0 - 180.0) < 0.5, name


def test_aspect_stays_within_one_turn():
    dem = np.array([[0, 10, 45], [5, 30, 60], [20, 40, 90]], dtype=np.float32)
    _, beta = terrain.slope_aspect(dem, 30.0, 30.0)
    assert np.all(beta >= 0.0) and np.all(beta < 2.0 * np.pi)


def test_output_is_float32():
    dem = np.array([[0, 30, 60]] * 3, dtype=np.float32)
    alpha, beta = terrain.slope_aspect(dem, 30.0, 30.0)
    assert alpha.dtype == np.float32 and beta.dtype == np.float32


def test_edges_are_computed_not_dropped():
    """Edge padding keeps the output the same shape as the DEM."""
    dem = np.array([[0, 30, 60]] * 4, dtype=np.float32)
    alpha, beta = terrain.slope_aspect(dem, 30.0, 30.0)
    assert alpha.shape == dem.shape and beta.shape == dem.shape
    assert np.isfinite(alpha).all()


def test_anisotropic_pixel_size_is_respected():
    """Halving the north-south pixel size doubles the north-south gradient."""
    dem = np.array([[0] * 3, [30] * 3, [60] * 3], dtype=np.float32)
    coarse, _ = terrain.slope_aspect(dem, 30.0, 30.0)
    fine, _ = terrain.slope_aspect(dem, 30.0, 15.0)
    assert float(fine[1, 1]) > float(coarse[1, 1])


def test_horn_gradient_signs():
    """dz/dx is east-minus-west; dz/dy is the raster row direction (southward)."""
    east_up = np.array([[0, 30, 60]] * 3, dtype=np.float32)
    dzdx, dzdy = terrain.horn_gradient(east_up, 30.0, 30.0)
    assert dzdx[1, 1] > 0 and abs(dzdy[1, 1]) < 1e-6

    south_up = np.array([[0] * 3, [30] * 3, [60] * 3], dtype=np.float32)
    dzdx, dzdy = terrain.horn_gradient(south_up, 30.0, 30.0)
    assert dzdy[1, 1] > 0 and abs(dzdx[1, 1]) < 1e-6
