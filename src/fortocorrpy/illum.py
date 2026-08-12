"""Local illumination condition (cos i) from solar and terrain geometry.

This module forms the single raster that every correction method depends on:
the cosine of the local solar incidence angle, ``cos i``. It is the meeting
point of the solar geometry (zenith ``theta_s`` and azimuth ``phi_s``, from
:mod:`fortocorrpy.solar`) and the terrain geometry (slope ``alpha`` and aspect
``beta``, from :mod:`fortocorrpy.terrain`).

The module computes ``cos i`` and nothing else. Aspect quadrant labels, pixel
validity and the four-direction sample decision belong to
:mod:`fortocorrpy.masking`.

Formula
-------
.. math::

    \\cos i = \\cos\\theta_s \\cos\\alpha
             + \\sin\\theta_s \\sin\\alpha \\cos(\\phi_s - \\beta)

This is the standard local-incidence form (Teillet et al. 1982). All angles
share the convention that azimuth and aspect are measured clockwise from
North, so ``phi_s - beta`` is the correct relative azimuth.

Conventions and units
---------------------
* All angle inputs and outputs are in radians. ``theta_s`` and ``phi_s``
  (from :mod:`fortocorrpy.solar`) and ``alpha`` and ``beta`` (from
  :mod:`fortocorrpy.terrain`) arrive in radians, so no per-pixel unit
  conversion is needed inside this module.
* ``theta_s`` and ``phi_s`` may be per-pixel rasters or scalars; NumPy
  broadcasting makes the formula identical either way.
* Output ``cos i`` is ``float32`` in the range [-1, 1].
* Self-shadow (``cos i <= 0``) is represented by the value of ``cos i``
  itself. Cast shadow, which would require ray tracing over the terrain, is
  not modelled.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cos_incidence", "cos_horizontal"]


def cos_incidence(theta_s, phi_s, alpha, beta):
    r"""Return the illumination condition ``cos i`` and the reference ``cos theta_s``.

    Both are produced together because ``cos i`` already requires
    ``cos(theta_s)``; returning it avoids recomputing (and avoids keeping
    ``theta_s`` alive past this call just to form the horizontal reference used
    by the correction step).

    Parameters
    ----------
    theta_s, phi_s : float or numpy.ndarray
        Solar zenith and solar azimuth in radians. Scalars or per-pixel
        arrays. Azimuth is clockwise from North.
    alpha, beta : numpy.ndarray
        Slope and aspect in radians, shape ``(rows, cols)``, from
        :func:`fortocorrpy.terrain.slope_aspect`. Aspect is clockwise from
        North, matching ``phi_s``.

    Returns
    -------
    cos_i : numpy.ndarray
        ``cos i`` raster, ``float32``, shape ``(rows, cols)``, in [-1, 1].
    cos_theta_s : numpy.ndarray or float
        Horizontal-surface reference ``cos(theta_s)``. A per-pixel
        ``float32`` array if ``theta_s`` is an array, else a float.

    Notes
    -----
    The result carries the self-shadow information implicitly: pixels with
    ``cos i <= 0`` receive no direct illumination. Downstream masking applies
    ``config.cos_i_threshold``; this function does not threshold.
    """
    theta_s = np.asarray(theta_s, dtype=np.float32)
    phi_s = np.asarray(phi_s, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    cos_theta_s = np.cos(theta_s)

    cos_i = (
        cos_theta_s * np.cos(alpha)
        + np.sin(theta_s) * np.sin(alpha) * np.cos(phi_s - beta)
    )

    cos_i = np.clip(cos_i, -1.0, 1.0).astype(np.float32)

    if np.ndim(cos_theta_s) == 0:
        cos_theta_s = float(cos_theta_s)
    else:
        cos_theta_s = cos_theta_s.astype(np.float32)

    return cos_i, cos_theta_s


def cos_horizontal(theta_s):
    r"""Return the horizontal-surface illumination condition ``cos i_h``.

    For a horizontal surface (slope = 0), ``cos i`` reduces to the cosine of
    the solar zenith angle: :math:`\cos i_h = \cos\theta_s`. This is the
    reference illumination used by the Cosine, C, SCS, SCS+C and ER correction
    formulas. SE re-centres on the regression-sample mean and does not use it.

    Parameters
    ----------
    theta_s : float or numpy.ndarray
        Solar zenith angle in radians. Scalar or per-pixel array.

    Returns
    -------
    float or numpy.ndarray
        ``cos(theta_s)``: a scalar for scalar input, a ``float32`` array for
        array input.
    """
    out = np.cos(np.asarray(theta_s, dtype=np.float32))
    return float(out) if np.ndim(out) == 0 else out.astype(np.float32)