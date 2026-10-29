"""Slope and aspect from a DEM using Horn's (1981) third-order method.

This module derives the two terrain-geometry rasters needed downstream: the
slope angle ``alpha`` and the aspect (slope direction) ``beta``. It does not
compute ``cos i`` (that is :mod:`fortocorrpy.illum`) and it does not reproject
or resample the DEM (that is :mod:`fortocorrpy.io`). The DEM passed in is
assumed to be already on the reference image grid, in a projected, metric CRS.

Method
------
The gradient components are estimated with the weighted third-order finite
difference of Horn (1981, "Hill Shading and the Reflectance Map",
Proceedings of the IEEE, section IX). In the 3x3 neighbourhood

    a b c
    d e f
    g h i

with west-east pixel size ``hx`` and south-north pixel size ``hy``, the
gradient components are

    dz/dx = [(c + 2f + i) - (a + 2d + g)] / (8 * hx)
    dz/dy = [(g + 2h + i) - (a + 2b + c)] / (8 * hy)

Edge-adjacent (orthogonal) neighbours are weighted twice as heavily as
diagonal neighbours, so local elevation errors contribute less to the slope
than in a plain central difference. GDAL's ``gdaldem`` uses the same
estimator.

Conventions
-----------
* ``alpha`` (slope) is the angle of the surface from horizontal, in radians:
  0 on flat ground, pi/2 on a vertical face.
* ``beta`` (aspect) is the compass bearing of the downslope direction, in
  radians: 0 = North, increasing clockwise (pi/2 = East, pi = South,
  3*pi/2 = West). This matches the solar azimuth convention in
  :mod:`fortocorrpy.solar` (after degree-to-radian conversion), which is
  required for ``cos i`` to be correct. Flat pixels (zero gradient) are
  assigned aspect 0.
"""

from __future__ import annotations

import numpy as np

__all__ = ["horn_gradient", "slope_aspect"]


def horn_gradient(dem, hx, hy):
    r"""Return the Horn (1981) gradient components ``(dz/dx, dz/dy)``.

    Parameters
    ----------
    dem : numpy.ndarray
        Two-dimensional elevation array (metres), shape ``(rows, cols)``.
    hx, hy : float
        West-east and south-north pixel size in metres (the same length unit
        as the elevations).

    Returns
    -------
    dzdx, dzdy : numpy.ndarray
        Gradient components in the west-to-east and south-to-north
        directions, same shape as ``dem``, ``float32``.

    Notes
    -----
    The DEM is edge-padded by replication so the output keeps the input shape.
    Row index increases southward (raster convention), so the south-to-north
    component uses the bottom rows minus the top rows.
    """
    z = np.pad(np.asarray(dem, dtype=np.float32), 1, mode="edge")

    # 3x3 neighbourhood (row index increases downward = southward):
    #   a b c   (north row)
    #   d e f
    #   g h i   (south row)
    a = z[:-2, :-2]
    b = z[:-2, 1:-1]
    c = z[:-2, 2:]
    d = z[1:-1, :-2]
    f = z[1:-1, 2:]
    g = z[2:, :-2]
    h = z[2:, 1:-1]
    i = z[2:, 2:]

    # Reciprocal pixel sizes as float32 so the products stay in float32.
    # The gradient depends only on local elevation *differences* over a 3x3
    # neighbourhood, so float32 carries them without quantisation loss while
    # roughly halving time and memory versus float64.
    inv8hx = np.float32(1.0 / (8.0 * hx))
    inv8hy = np.float32(1.0 / (8.0 * hy))

    dzdx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) * inv8hx
    # north row is at the top (smaller row index); south-to-north is
    # (south row) - (north row) = (g,h,i) - (a,b,c)
    dzdy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) * inv8hy

    return dzdx.astype(np.float32), dzdy.astype(np.float32)


def slope_aspect(dem, hx, hy):
    r"""Return slope ``alpha`` and aspect ``beta`` rasters from a DEM.

    Parameters
    ----------
    dem : numpy.ndarray
        Two-dimensional elevation array (metres) on the reference image grid.
    hx, hy : float
        West-east and south-north pixel size in metres.

    Returns
    -------
    alpha, beta : numpy.ndarray
        Slope angle and aspect in radians, ``float32``, same shape as ``dem``.
        ``beta`` is the compass bearing of the downslope direction (0 = North,
        clockwise). Flat pixels are assigned ``beta = 0``.

    Notes
    -----
    Slope is :math:`\alpha = \arctan\sqrt{(dz/dx)^2 + (dz/dy)^2}`.

    Aspect is derived from the gradient and then expressed as a compass
    bearing to match the solar azimuth convention. The downslope direction
    points opposite to the gradient (which points uphill); the conversion
    ``beta = (pi/2 - atan2(dzdy, -dzdx)) mod 2*pi`` maps the
    math-convention angle (0 = East, counter-clockwise) to the compass
    bearing (0 = North, clockwise).

    All downstream modules (illum, masking, T_correct) receive radians
    directly, avoiding per-pixel degree conversions on full rasters.
    Scalar thresholds (e.g. slope_min_deg in Config) are converted to
    radians at the point of use, which is negligible.
    """
    dzdx, dzdy = horn_gradient(dem, hx, hy)

    # sqrt(p^2 + q^2) is faster than hypot (which adds overflow-guard logic
    # that is unnecessary for bounded terrain gradients).
    alpha = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)).astype(np.float32)

    # Math-convention angle of the gradient, then convert to compass bearing.
    # Done directly in radians: no intermediate degree array needed.
    aspect_math = np.arctan2(dzdy, -dzdx)
    del dzdx, dzdy  # no longer needed; free before beta computation
    beta = np.mod(np.float32(np.pi / 2.0) - aspect_math,
                  np.float32(2.0 * np.pi))

    # Flat pixels have no defined aspect; assign 0.
    beta = np.where(alpha == 0.0, np.float32(0.0), beta).astype(np.float32)

    return alpha, beta