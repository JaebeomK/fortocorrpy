"""Reflectance-based evaluation of topographic-correction quality.

Correction and its assessment run in one call: the pipeline can apply several
methods and report how each one performed over the same sample, so the methods
are directly comparable.

Metrics
-------
Over the evaluation sample, for one band:

* ``correlation``: Pearson correlation between ``cos i`` and reflectance.
  Topographic effect shows up as a positive correlation (brighter on
  sun-facing slopes); a good correction drives it toward 0.
* ``slope``: ordinary-least-squares slope of reflectance on ``cos i``.
  Like the correlation, it should fall toward 0 after correction.
* ``std``: standard deviation of reflectance. Removing slope-driven
  brightness variation typically reduces it.
* ``mean``: mean reflectance. A correction should largely preserve overall
  brightness, so the mean should stay close to its pre-correction value.

before / after and multiple methods
-----------------------------------
The pre-correction ("before") metrics depend only on the original
reflectance, so they are computed once and shared across methods. The
post-correction ("after") metrics are computed per method, per band, since
each method yields different corrected reflectance. The pipeline pairs them so
the reduction in correlation/slope is directly visible.

Evaluation sample
-----------------
Metrics are computed over the forest regression sample (the mask from
:mod:`fortocorrpy.masking`), consistent with how the coefficients were fit.
NaN pixels (self-shadow, invalid) are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BandMetrics", "band_metrics", "evaluate_before", "evaluate_after"]


@dataclass
class BandMetrics:
    """Reflectance-based metrics for one band over the evaluation sample.

    Attributes
    ----------
    correlation : float
        Pearson correlation between ``cos i`` and reflectance.
    slope : float
        OLS slope of reflectance regressed on ``cos i``.
    std : float
        Standard deviation of reflectance.
    mean : float
        Mean reflectance.
    n_samples : int
        Number of valid sample pixels used.
    """

    correlation: float
    slope: float
    std: float
    mean: float
    n_samples: int


def band_metrics(cos_i, reflectance, mask):
    """Compute the standard metrics for one band over the masked sample.

    Parameters
    ----------
    cos_i : numpy.ndarray
        Illumination condition raster.
    reflectance : numpy.ndarray
        Single-band reflectance raster, same shape as ``cos_i``.
    mask : numpy.ndarray
        Boolean evaluation-sample mask (typically the forest regression mask).

    Returns
    -------
    BandMetrics
        Correlation, slope, std, mean, and the sample size. Fields are ``nan``
        when undefined (e.g. fewer than two valid pixels, or zero variance in
        ``cos i``).
    """
    m = np.asarray(mask, dtype=bool)
    # Extract masked pixels first (small array), then promote to float64
    # for numerical stability. Avoids creating a full-raster float64 copy.
    x = np.asarray(cos_i)[m].astype(np.float64)
    y = np.asarray(reflectance)[m].astype(np.float64)

    # Drop pixels where either is NaN (self-shadow / invalid).
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    n = x.size

    if n < 2:
        return BandMetrics(
            correlation=float("nan"), slope=float("nan"),
            std=float("nan"), mean=float("nan"), n_samples=int(n),
        )

    mean = float(y.mean())
    std = float(y.std(ddof=0))

    x_mean = x.mean()
    dx = x - x_mean
    sxx = float(np.dot(dx, dx))

    if sxx == 0.0:
        correlation = float("nan")
        slope = float("nan")
    else:
        dy = y - y.mean()
        sxy = float(np.dot(dx, dy))
        slope = sxy / sxx
        sy = float(np.dot(dy, dy))
        correlation = sxy / np.sqrt(sxx * sy) if sy > 0.0 else float("nan")

    return BandMetrics(
        correlation=float(correlation), slope=float(slope),
        std=std, mean=mean, n_samples=int(n),
    )


def evaluate_before(cos_i, reflectance, mask):
    """Per-band metrics on the original (pre-correction) reflectance.

    Computed once and shared across methods, since the input reflectance does
    not depend on the correction method.

    Parameters
    ----------
    cos_i : numpy.ndarray
        Illumination condition raster.
    reflectance : numpy.ndarray
        Reflectance stack ``(bands, rows, cols)`` or a single band.
    mask : numpy.ndarray
        Boolean evaluation-sample mask.

    Returns
    -------
    list of BandMetrics
        One entry per band, in band order.
    """
    refl = np.asarray(reflectance)
    if refl.ndim == 2:
        refl = refl[np.newaxis, ...]
    return [band_metrics(cos_i, refl[b], mask) for b in range(refl.shape[0])]


def evaluate_after(cos_i, corrected, mask):
    """Per-band metrics on corrected reflectance for one method.

    Parameters
    ----------
    cos_i : numpy.ndarray
        Illumination condition raster.
    corrected : numpy.ndarray
        Corrected reflectance stack ``(bands, rows, cols)`` for one method, or
        a single band.
    mask : numpy.ndarray
        Boolean evaluation-sample mask (same as used for ``before``).

    Returns
    -------
    list of BandMetrics
        One entry per band, in band order.
    """
    corr = np.asarray(corrected)
    if corr.ndim == 2:
        corr = corr[np.newaxis, ...]
    return [band_metrics(cos_i, corr[b], mask) for b in range(corr.shape[0])]