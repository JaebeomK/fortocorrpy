"""Topographic correction methods, implemented per the source literature.

Six methods operate on the observed (sloped) reflectance ``rho_T`` and share
the local illumination condition ``cos i``. Cosine, C, SCS, SCS+C and ER refer
the result to horizontal-terrain illumination ``cos i_h = cos(theta_s)``; SE
removes the fitted illumination trend and re-centres on the regression-sample
mean instead. The regression coefficients, where a method uses them, are
estimated from the forest sample upstream; the formulas themselves run over
the whole raster.

Methods and formulas (see DESIGN_SPEC, "Correction methods")
------------------------------------------------------------
* ``cosine``       : rho_H = rho_T * cos(theta_s) / cos i
* ``scs``          : rho_H = rho_T * (cos(alpha) * cos(theta_s)) / cos i
* ``c``            : rho_H = rho_T * (cos(theta_s) + C) / (cos i + C)
* ``scsc``         : rho_H = rho_T * (cos(alpha) * cos(theta_s) + C) / (cos i + C)
* ``se``           : rho_H = rho_T - (a * cos i + b) + mean(rho)
* ``er``           : rho_H = rho_T - a * (cos i - cos(theta_s))

Two groups by structure:

* Denominator-form (cosine, scs, c, scsc): ``cos i`` is in the denominator, so
  the correction factor diverges as it approaches zero. These are applied where
  ``cos i`` exceeds ``cos_i_threshold`` (0 by default); at or below it the
  result is NaN, self-shadow having no direct-illumination basis.
* Subtractive-form (se, er): no denominator, so they do not diverge and are
  applied to every valid pixel.

Coefficient requirement
------------------------
``cosine`` and ``scs`` need no regression; they may be called with
``coeff=None``. ``c`` and ``scsc`` need ``C``; ``se`` and ``er`` need the slope
``a``. The pipeline decides whether to run the regression at all based on the
selected methods.

``cos i = -C`` guard: ``c`` and ``scsc`` divide by ``cos i + C``. A minimal
numerical guard against a near-zero denominator is applied: pixels whose
denominator falls below 1e-6 in magnitude become NaN. The floor is a fixed
conservative value rather than one derived from the band's noise level; it
prevents blow-up without affecting well-illuminated pixels.
"""

from __future__ import annotations

import numpy as np

from .config import DENOMINATOR_METHODS

__all__ = ["correct"]

# Denominators with absolute value below this are treated as singular (NaN),
# guarding the cos i + C and cos i divisions against blow-up.
_DENOM_FLOOR = 1e-6


def _apply_denominator_mask(cos_i, cos_i_threshold):
    """Return a boolean array, True where denominator-form methods apply.

    Denominator-form methods are valid only in direct light (``cos i`` above
    the threshold, default 0). Elsewhere the correction has no basis and the
    output is NaN.
    """
    return np.asarray(cos_i) > cos_i_threshold


def _safe_divide(numerator, denominator):
    """Divide, returning NaN where the denominator is near zero."""
    denom = np.asarray(denominator, dtype=np.float32)
    out = np.full(np.broadcast(numerator, denom).shape, np.nan, dtype=np.float32)
    ok = np.abs(denom) >= _DENOM_FLOOR
    np.divide(numerator, denom, out=out, where=ok)
    return out


def correct(method, reflectance, cos_i, cos_theta_s, *, cos_alpha=None,
            coeff=None, cos_i_threshold=0.0):
    r"""Apply one correction method to a single band.

    Parameters
    ----------
    method : str
        One of ``cosine``, ``scs``, ``c``, ``scsc``, ``se``, ``er``.
    reflectance : numpy.ndarray
        Single-band observed reflectance ``rho_T``.
    cos_i : numpy.ndarray
        Local illumination condition, same shape as ``reflectance``.
    cos_theta_s : float or numpy.ndarray
        Horizontal reference ``cos(theta_s)``; scalar or per-pixel.
    cos_alpha : numpy.ndarray, optional
        Cosine of slope angle, pre-computed once from
        ``np.cos(alpha)``. Required for ``scs`` and ``scsc`` (the
        slope-TC group), unused otherwise.
    coeff : BandCoefficients, optional
        Regression coefficients for this band. Required for ``c``, ``scsc``
        (uses ``C``) and ``se``, ``er`` (uses slope ``a``). May be ``None`` for
        ``cosine`` and ``scs``.
    cos_i_threshold : float, default 0.0
        Lower bound on ``cos i`` for denominator-form methods; pixels at or
        below it become NaN.

    Returns
    -------
    numpy.ndarray
        Corrected reflectance ``rho_H``, ``float32``, same shape as
        ``reflectance``.
    """
    rho = np.asarray(reflectance, dtype=np.float32)
    ci = np.asarray(cos_i, dtype=np.float32)
    cos_ts = np.asarray(cos_theta_s, dtype=np.float32)

    if method == "cosine":
        rho_h = rho * _safe_divide(cos_ts, ci)

    elif method == "scs":
        if cos_alpha is None:
            raise ValueError("scs requires cos_alpha (cosine of slope)")
        cos_a = np.asarray(cos_alpha, dtype=np.float32)
        rho_h = rho * _safe_divide(cos_a * cos_ts, ci)

    elif method == "c":
        if coeff is None:
            raise ValueError("c (C-correction) requires coeff with C")
        c = np.float32(coeff.c)
        if not np.isfinite(c):
            # slope a = 0: no illumination trend to correct. Use identity here,
            # but still let the self-shadow (denominator) mask below apply.
            rho_h = rho.astype(np.float32).copy()
        else:
            rho_h = rho * _safe_divide(cos_ts + c, ci + c)

    elif method == "scsc":
        if cos_alpha is None:
            raise ValueError("scsc requires cos_alpha (cosine of slope)")
        if coeff is None:
            raise ValueError("scsc requires coeff with C")
        cos_a = np.asarray(cos_alpha, dtype=np.float32)
        c = np.float32(coeff.c)
        if not np.isfinite(c):
            # slope a = 0: identity here, but the self-shadow mask below applies.
            rho_h = rho.astype(np.float32).copy()
        else:
            rho_h = rho * _safe_divide(cos_a * cos_ts + c, ci + c)

    elif method == "se":
        if coeff is None:
            raise ValueError("se requires coeff with slope, intercept, and mean_reflectance")
        a = np.float32(coeff.slope)
        b = np.float32(coeff.intercept)
        mean_rho = np.float32(coeff.mean_reflectance)
        # Statistical-Empirical: remove the regression trend, re-centre on the
        # sample-mean reflectance. Subtractive form: no denominator mask.
        return (rho - (a * ci + b) + mean_rho).astype(np.float32)

    elif method == "er":
        if coeff is None:
            raise ValueError("er requires coeff with slope a")
        a = np.float32(coeff.slope)
        # Empirical Rotation: remove the regression trend, re-centre on the
        # horizontal reference cos(theta_s). The intercept cancels. Subtractive
        # form: no denominator mask.
        return (rho - a * (ci - cos_ts)).astype(np.float32)

    else:
        raise ValueError(f"unknown method: {method!r}")

    # Denominator-form: restrict to direct-light pixels; NaN elsewhere.
    if method in DENOMINATOR_METHODS:
        apply = _apply_denominator_mask(ci, cos_i_threshold)
        rho_h = np.where(apply, rho_h, np.nan).astype(np.float32)

    return rho_h