"""Per-band regression of reflectance against the illumination condition.

For the methods that need it (``config.REGRESSION_METHODS``), the correction
coefficients come from a linear regression of observed reflectance on the
illumination condition over the regression sample:

.. math::

    \\rho_T = a \\cos i + b

fit per band, with the C parameter defined as :math:`C = b / a` (Teillet et
al. 1982; Soenen et al. 2005). The slope ``a`` is used directly by SE and ER;
``C`` is used by the C-correction and SCS+C denominators.

This module is pure statistics. The sample is decided upstream
(:mod:`fortocorrpy.masking`): it receives a boolean mask and extracts the
sample values at those pixels. Sample eligibility (slope, illumination,
forest, validity) and the four-direction balance are already applied, so this
module does not re-check them.

No additional regression diagnostics, such as the standard error of ``C``, are
produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BandCoefficients", "fit_band", "fit_bands"]


@dataclass
class BandCoefficients:
    """Regression coefficients for one band.

    Attributes
    ----------
    slope : float
        Regression slope ``a`` in ``rho_T = a * cos i + b``.
    intercept : float
        Regression intercept ``b``.
    c : float
        The C parameter, ``C = b / a``. ``nan`` if the slope is zero.
    n_samples : int
        Number of sample pixels used in the fit.
    mean_reflectance : float
        Mean observed reflectance over the sample, used by ``se`` as the
        re-centring reference.
    """

    slope: float
    intercept: float
    c: float
    n_samples: int
    mean_reflectance: float


def _fit_line(x, y):
    """Ordinary least-squares fit ``y = a*x + b``; return ``(a, b)``.

    Uses the closed-form normal equations in float64 for numerical stability,
    independent of the float32 storage of the input rasters.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x_mean = x.mean()
    y_mean = y.mean()
    dx = x - x_mean
    sxx = np.dot(dx, dx)
    sxy = np.dot(dx, y - y_mean)

    if sxx == 0.0:
        # cos i is constant over the sample: slope undefined.
        return 0.0, float(y_mean)

    a = sxy / sxx
    b = y_mean - a * x_mean
    return float(a), float(b)


def fit_band(cos_i, reflectance, mask):
    """Fit ``rho_T = a * cos i + b`` for a single band over the sample.

    Parameters
    ----------
    cos_i : numpy.ndarray
        Illumination condition raster.
    reflectance : numpy.ndarray
        Single-band reflectance raster, same shape as ``cos_i``.
    mask : numpy.ndarray
        Boolean sample mask (from :mod:`fortocorrpy.masking`); ``True`` marks
        the pixels to include.

    Returns
    -------
    BandCoefficients
        Slope ``a``, intercept ``b``, ``C = b / a``, and the sample size.
    """
    m = np.asarray(mask, dtype=bool)
    x = np.asarray(cos_i)[m]
    y = np.asarray(reflectance)[m]

    a, b = _fit_line(x, y)
    c = b / a if a != 0.0 else float("nan")
    mean_reflectance = float(np.asarray(y, dtype=np.float64).mean()) if x.size else float("nan")
    return BandCoefficients(
        slope=a, intercept=b, c=c, n_samples=int(x.size),
        mean_reflectance=mean_reflectance,
    )


def fit_bands(cos_i, reflectance, mask):
    """Fit the regression for every band of a stack.

    Parameters
    ----------
    cos_i : numpy.ndarray
        Illumination condition raster, shape ``(rows, cols)``.
    reflectance : numpy.ndarray
        Reflectance stack, shape ``(bands, rows, cols)`` or a single band as
        ``(rows, cols)``.
    mask : numpy.ndarray
        Boolean sample mask, shape ``(rows, cols)``.

    Returns
    -------
    list of BandCoefficients
        One entry per band, in band order.
    """
    refl = np.asarray(reflectance)
    if refl.ndim == 2:
        refl = refl[np.newaxis, ...]

    return [fit_band(cos_i, refl[b], mask) for b in range(refl.shape[0])]