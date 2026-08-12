"""Solar geometry, checked against independent astronomical references.

Expected values come from the NOAA solar position calculator, not from this
package's own output.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fortocorrpy import solar

SEOUL_LAT, SEOUL_LON = 37.5665, 126.9780
KST = timezone(timedelta(hours=9))


def test_angles_are_returned_in_radians():
    """Regression: the public angle API is radians, not degrees.

    A degrees-based caller would silently keep working for zenith (both are
    positive numbers) but produce nonsense downstream, so pin the range.
    """
    zenith, azimuth = solar.solar_position(
        datetime(2024, 6, 21, 3, 30, tzinfo=timezone.utc), SEOUL_LAT, SEOUL_LON,
    )
    assert 0.0 <= float(zenith) <= np.pi / 2
    assert 0.0 <= float(azimuth) < 2.0 * np.pi


def test_seoul_summer_solstice_local_noon():
    """NOAA reference: Seoul, 2024-06-21 12:30 KST -> zenith ~14.1 deg, due south."""
    zenith, azimuth = solar.solar_position(
        datetime(2024, 6, 21, 3, 30, tzinfo=timezone.utc), SEOUL_LAT, SEOUL_LON,
    )
    assert abs(np.rad2deg(float(zenith)) - 14.1) < 0.5
    assert abs(np.rad2deg(float(azimuth)) - 180.0) < 5.0


def test_equator_equinox_noon_sun_near_zenith():
    """Equator at equinox, solar noon: the sun is nearly overhead."""
    zenith, _ = solar.solar_position(
        datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc), 0.0, 0.0,
    )
    assert np.rad2deg(float(zenith)) < 3.0


def test_azimuth_moves_east_to_west_through_the_day():
    """Morning sun east of south (<180 deg), afternoon sun west of south."""
    morning = solar.solar_position(  # 09:00 KST
        datetime(2024, 6, 21, 0, 0, tzinfo=timezone.utc), SEOUL_LAT, SEOUL_LON,
    )[1]
    afternoon = solar.solar_position(  # 16:00 KST
        datetime(2024, 6, 21, 7, 0, tzinfo=timezone.utc), SEOUL_LAT, SEOUL_LON,
    )[1]
    assert np.rad2deg(float(morning)) < 180.0
    assert np.rad2deg(float(afternoon)) > 180.0


def test_naive_datetime_treated_as_utc():
    naive = datetime(2024, 6, 21, 3, 30)
    aware = datetime(2024, 6, 21, 3, 30, tzinfo=timezone.utc)
    z1, a1 = solar.solar_position(naive, SEOUL_LAT, SEOUL_LON)
    z2, a2 = solar.solar_position(aware, SEOUL_LAT, SEOUL_LON)
    assert abs(float(z1) - float(z2)) < 1e-6
    assert abs(float(a1) - float(a2)) < 1e-6


def test_aware_datetime_is_converted_to_utc():
    """12:30 KST and 03:30 UTC are the same instant and must agree."""
    z_kst, a_kst = solar.solar_position(
        datetime(2024, 6, 21, 12, 30, tzinfo=KST), SEOUL_LAT, SEOUL_LON,
    )
    z_utc, a_utc = solar.solar_position(
        datetime(2024, 6, 21, 3, 30, tzinfo=timezone.utc), SEOUL_LAT, SEOUL_LON,
    )
    assert abs(float(z_kst) - float(z_utc)) < 1e-6
    assert abs(float(a_kst) - float(a_utc)) < 1e-6


def test_julian_day_at_j2000_epoch():
    """J2000.0 is 2000-01-01 12:00 UT = JD 2451545.0 by definition."""
    assert abs(solar.julian_day(datetime(2000, 1, 1, 12, 0)) - 2451545.0) < 1e-6


def test_julian_day_advances_by_one_per_day():
    d0 = solar.julian_day(datetime(2024, 3, 1, 0, 0))
    d1 = solar.julian_day(datetime(2024, 3, 2, 0, 0))
    assert abs((d1 - d0) - 1.0) < 1e-9


@pytest.mark.parametrize("shape", [(5, 7), (1, 1)])
def test_solar_angles_returns_per_pixel_rasters(shape):
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    theta_s, phi_s = solar.solar_angles(
        datetime(2024, 8, 14, 2, 27, 9, tzinfo=timezone.utc),
        from_origin(300000, 4200000, 30, 30), CRS.from_epsg(32652), shape,
    )
    assert theta_s.shape == shape and phi_s.shape == shape
    assert theta_s.dtype == np.float32 and phi_s.dtype == np.float32


def test_solar_angles_vary_across_a_large_footprint():
    """Per-pixel evaluation must actually vary; a scene-constant would be a bug."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    theta_s, _ = solar.solar_angles(
        datetime(2024, 8, 14, 2, 27, 9, tzinfo=timezone.utc),
        from_origin(300000, 4200000, 30, 30), CRS.from_epsg(32652), (2000, 2000),
    )
    assert np.rad2deg(theta_s.max() - theta_s.min()) > 0.1
