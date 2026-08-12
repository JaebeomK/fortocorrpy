"""The illumination condition cos i, checked against closed-form geometry.

    cos i = cos(theta_s)cos(alpha) + sin(theta_s)sin(alpha)cos(phi_s - beta)

Two exact identities fall out of this and are used as references below:

* aspect facing the sun (``beta = phi_s``)   -> ``cos i = cos(theta_s - alpha)``
* aspect facing away    (``beta = phi_s+pi``) -> ``cos i = cos(theta_s + alpha)``
"""

import numpy as np
import pytest

from fortocorrpy import illum

d = np.deg2rad


def test_flat_surface_reduces_to_cos_theta_s():
    cos_i, cos_ts = illum.cos_incidence(d(30.0), d(180.0), d(0.0), d(0.0))
    assert np.isclose(float(cos_i), float(cos_ts), atol=1e-6)
    assert np.isclose(float(cos_i), np.cos(d(30.0)), atol=1e-6)


def test_slope_normal_pointing_at_the_sun_gives_one():
    """30 deg slope facing south, sun due south at 30 deg zenith: cos i = 1."""
    cos_i, _ = illum.cos_incidence(d(30.0), d(180.0), d(30.0), d(180.0))
    assert np.isclose(float(cos_i), 1.0, atol=1e-6)


def test_opposing_slope_at_the_complement_is_exactly_zero():
    """Corrected hand calculation.

    Sun at 30 deg zenith from the south; a 60 deg slope facing north gives
    ``cos i = cos(30 + 60) = cos(90) = 0`` -- grazing incidence, not a negative
    value. The sample and the denominator-form methods both use ``cos i > 0``,
    so this is the exact boundary of self-shadow.
    """
    cos_i, _ = illum.cos_incidence(d(30.0), d(180.0), d(60.0), d(0.0))
    assert abs(float(cos_i)) < 1e-6


def test_opposing_slope_beyond_the_complement_is_negative():
    """Past the grazing angle the surface is in self-shadow."""
    cos_i, _ = illum.cos_incidence(d(30.0), d(180.0), d(70.0), d(0.0))
    assert float(cos_i) < 0.0


@pytest.mark.parametrize("alpha_deg", [0.0, 10.0, 25.0, 40.0])
def test_sun_facing_identity(alpha_deg):
    """beta = phi_s  ->  cos i = cos(theta_s - alpha)."""
    theta_s = 35.0
    cos_i, _ = illum.cos_incidence(
        d(theta_s), d(180.0), d(alpha_deg), d(180.0),
    )
    assert np.isclose(float(cos_i), np.cos(d(theta_s - alpha_deg)), atol=1e-6)


@pytest.mark.parametrize("alpha_deg", [0.0, 10.0, 25.0, 40.0])
def test_sun_opposing_identity(alpha_deg):
    """beta = phi_s + 180  ->  cos i = cos(theta_s + alpha)."""
    theta_s = 35.0
    cos_i, _ = illum.cos_incidence(d(theta_s), d(180.0), d(alpha_deg), d(0.0))
    assert np.isclose(float(cos_i), np.cos(d(theta_s + alpha_deg)), atol=1e-6)


def test_result_is_clipped_to_the_unit_interval():
    alpha = d(np.array([0.0, 45.0, 89.0], dtype=np.float32))
    beta = d(np.array([180.0, 180.0, 0.0], dtype=np.float32))
    cos_i, _ = illum.cos_incidence(d(30.0), d(180.0), alpha, beta)
    assert cos_i.min() >= -1.0 and cos_i.max() <= 1.0
    assert cos_i.dtype == np.float32


def test_scalar_solar_angles_broadcast_over_terrain_rasters():
    alpha = np.full((4, 6), d(20.0), dtype=np.float32)
    beta = np.full((4, 6), d(180.0), dtype=np.float32)
    cos_i, cos_ts = illum.cos_incidence(d(30.0), d(180.0), alpha, beta)
    assert cos_i.shape == (4, 6)
    assert np.ndim(cos_ts) == 0


def test_per_pixel_solar_angles_give_per_pixel_reference():
    theta_s = np.full((3, 3), d(30.0), dtype=np.float32)
    phi_s = np.full((3, 3), d(180.0), dtype=np.float32)
    alpha = np.zeros((3, 3), dtype=np.float32)
    beta = np.zeros((3, 3), dtype=np.float32)
    cos_i, cos_ts = illum.cos_incidence(theta_s, phi_s, alpha, beta)
    assert cos_ts.shape == (3, 3)
    assert np.allclose(cos_i, cos_ts, atol=1e-6)


def test_cos_horizontal_matches_cos_of_zenith():
    assert np.isclose(illum.cos_horizontal(d(30.0)), np.cos(d(30.0)), atol=1e-6)
    arr = illum.cos_horizontal(np.full((2, 2), d(45.0), dtype=np.float32))
    assert arr.dtype == np.float32
    assert np.allclose(arr, np.cos(d(45.0)), atol=1e-6)
