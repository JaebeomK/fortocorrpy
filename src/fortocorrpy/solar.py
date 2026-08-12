"""Per-pixel solar geometry from a reference image grid and acquisition time.

Topographic correction requires the solar zenith angle ``theta_s`` and solar
azimuth angle ``phi_s`` at every pixel in order to form the local illumination
condition ``cos i``. No sensor-specific metadata is parsed: the inputs are the
geometry of a reference image grid (affine transform, CRS, shape) and one
acquisition time in UTC, and the outputs are full per-pixel angle rasters.

Implementation
--------------
The Sun's ecliptic coordinates follow the low-precision formulae of The
Astronomical Almanac, stated to 0.01 degree through 2050 (Michalsky 1988,
Solar Energy 40, 227-235). Greenwich mean sidereal time uses the equivalent
degree-based form of Meeus, *Astronomical Algorithms*, eq. 12.4. Both are
written in NumPy; ``pyproj`` is used only for the projected-to-geographic
coordinate transform.

Angles are evaluated at every pixel centre. Over a large footprint the solar
zenith varies across the scene by one to a few degrees, so a per-pixel
evaluation keeps the downstream ``cos i`` from being scene-averaged.

The longitude and latitude arrays are needed only to evaluate the angles and
are released once the angle rasters are formed.

Angles are evaluated at every pixel centre. Over a large footprint the solar
zenith varies across the scene by one to a few degrees, so a per-pixel
evaluation keeps the downstream ``cos i`` from being scene-averaged.

The longitude and latitude arrays are needed only to evaluate the angles and
are released once the angle rasters are formed.

Angle conventions
-----------------
* ``theta_s`` (solar zenith) is measured from the local vertical, in radians:
  0 at the zenith, pi/2 at the horizon.
* ``phi_s`` (solar azimuth) is measured clockwise from geographic North, in
  radians: 0 = North, pi/2 = East, pi = South, 3*pi/2 = West.

These conventions must match the terrain aspect convention used when forming
``cos i`` (see :mod:`fortocorrpy.terrain`).
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

__all__ = [
    "julian_day",
    "solar_position",
    "pixel_lonlat",
    "solar_angles",
]


def julian_day(when: datetime) -> float:
    """Return the Julian Day for a UTC datetime.

    Parameters
    ----------
    when : datetime.datetime
        Acquisition time. A timezone-aware value is converted to UTC; a naive
        value is assumed to already be UTC.

    Returns
    -------
    float
        Julian Day (including the fractional day).
    """
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc)

    year, month = when.year, when.month
    day_frac = (
        when.day
        + (when.hour + (when.minute + when.second / 60.0) / 60.0) / 24.0
    )

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4

    return (
        np.floor(365.25 * (year + 4716))
        + np.floor(30.6001 * (month + 1))
        + day_frac
        + b
        - 1524.5
    )


def solar_position(when: datetime, lat_deg, lon_deg):
    r"""Solar zenith and azimuth at given geographic coordinates and UTC time.

    Parameters
    ----------
    when : datetime.datetime
        Acquisition time (UTC).
    lat_deg, lon_deg : float or numpy.ndarray
        Latitude and longitude in degrees (WGS-84). Scalars or arrays of equal
        shape; arrays yield a per-pixel result.

    Returns
    -------
    theta_s, phi_s : numpy.ndarray
        Solar zenith angle and solar azimuth angle in radians, ``float32``,
        broadcast to the shape of the inputs. ``phi_s`` is measured clockwise
        from North (0 = N, pi/2 = E).

    Notes
    -----
    The result is the apparent geometric position; atmospheric refraction is
    not applied. Its effect on ``cos i`` is negligible for topographic
    correction away from the horizon.
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)

    n = julian_day(when) - 2451545.0  # days since the J2000.0 epoch

    # Mean longitude (L) and mean anomaly (g) of the Sun, in degrees.
    mean_longitude = (280.460 + 0.9856474 * n) % 360.0
    mean_anomaly = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)

    # Ecliptic longitude (lambda) and obliquity of the ecliptic (epsilon).
    ecliptic_longitude = np.deg2rad(
        mean_longitude
        + 1.915 * np.sin(mean_anomaly)
        + 0.020 * np.sin(2.0 * mean_anomaly)
    )
    obliquity = np.deg2rad(23.439 - 0.0000004 * n)

    # Right ascension and declination of the Sun.
    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude),
        np.cos(ecliptic_longitude),
    )
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_longitude))

    # Greenwich mean sidereal time, then the local hour angle.
    gmst = (280.46061837 + 360.98564736629 * n) % 360.0
    local_sidereal = np.deg2rad((gmst + lon) % 360.0)
    hour_angle = local_sidereal - right_ascension

    lat_rad = np.deg2rad(lat)
    sin_altitude = np.clip(
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle),
        -1.0,
        1.0,
    )
    zenith = np.pi / 2.0 - np.arcsin(sin_altitude)

    # Azimuth clockwise from North.
    azimuth = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(lat_rad)
        - np.tan(declination) * np.cos(lat_rad),
    )
    # arctan2 returns [-pi, pi] in math convention; shift to compass bearing
    # [0, 2*pi) clockwise from North, directly in radians.
    azimuth = np.mod(azimuth + np.pi, 2.0 * np.pi)

    theta_s = zenith.astype(np.float32)
    phi_s = azimuth.astype(np.float32)
    return theta_s, phi_s


def pixel_lonlat(transform, crs, shape):
    """Return per-pixel longitude/latitude (WGS-84) for an image grid.

    Pixel-centre coordinates on the projected image grid are transformed to
    geographic coordinates. These arrays are intermediate products for the
    solar-angle computation and can be released once the angles are formed.

    Parameters
    ----------
    transform : affine.Affine
        Affine transform of the reference image grid (projected CRS).
    crs : rasterio.crs.CRS or str
        Coordinate reference system of the image grid.
    shape : tuple of int
        ``(rows, cols)`` of the image grid.

    Returns
    -------
    lon, lat : numpy.ndarray
        Longitude and latitude in degrees, shape ``(rows, cols)``, ``float64``.
    """
    from pyproj import Transformer

    rows, cols = shape

    col_centres = np.arange(cols, dtype=np.float64) + 0.5
    row_centres = np.arange(rows, dtype=np.float64) + 0.5
    cc, rr = np.meshgrid(col_centres, row_centres)

    x = transform.a * cc + transform.b * rr + transform.c
    y = transform.d * cc + transform.e * rr + transform.f
    del cc, rr

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    del x, y

    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def solar_angles(when: datetime, transform, crs, shape):
    """Return per-pixel solar zenith/azimuth rasters for an image grid.

    This is the public entry point for the input-preparation stage. It builds
    per-pixel geographic coordinates from the reference grid, evaluates the
    solar position at the acquisition time, and releases the intermediate
    coordinate arrays before returning.

    Parameters
    ----------
    when : datetime.datetime
        Acquisition time (UTC).
    transform : affine.Affine
        Affine transform of the reference image grid.
    crs : rasterio.crs.CRS or str
        Image CRS.
    shape : tuple of int
        ``(rows, cols)``.

    Returns
    -------
    theta_s, phi_s : numpy.ndarray
        Per-pixel solar zenith and azimuth rasters in radians, ``float32``,
        shape ``(rows, cols)``. ``phi_s`` is clockwise from North.
    """
    lon, lat = pixel_lonlat(transform, crs, shape)
    theta_s, phi_s = solar_position(when, lat, lon)
    del lon, lat
    return theta_s, phi_s
