"""Build the regression-sample mask and decide scene eligibility.

This module produces the single mask that :mod:`fortocorrpy.linear_C` uses
to fit the reflectance-cos i regression. It is the integration point for every
sample condition, applied in a fixed logical order, followed by the
four-direction balancing draw that also decides whether the scene can be
corrected at all.

All inputs arrive as in-memory arrays from the pipeline; this module reads no
files.

Logical order
-------------
1. ``M_target = (alpha >= slope_min_rad) & (cos i > cos_i_threshold) & forest``
   selects the pixels where a correction is meaningful.
2. ``candidate = M_target & valid`` restricts to pixels with valid
   reflectance across all processed bands (a single combined validity mask).
3. Label candidates by aspect quadrant; count per quadrant.
4. If any quadrant has fewer than ``min_samples_per_quadrant`` candidates, the
   scene is skipped. Otherwise draw, from every quadrant, the same number of
   pixels (the smallest quadrant count) at random, giving an aspect-balanced
   sample. The drawn pixels form the final regression mask.

Step 4 comes last because the per-quadrant counts are only meaningful once all
prior conditions (slope, illumination, forest, validity) have been applied.

The forest source is interpreted by ``config.mask_source``: an external
growing-season NDVI raster (thresholded) or an external land-cover raster
(the forest class is selected). NDVI here is a separate wall-to-wall
product, not the scene being corrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

__all__ = ["MaskResult", "forest_mask", "build_sample_mask"]

# Canonical quadrant order for labelling and diagnostics.
_QUADRANTS = ("N", "E", "S", "W")


@dataclass
class MaskResult:
    """Outcome of building the regression-sample mask.

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean array, ``True`` at the pixels drawn for the regression sample
        (aspect-balanced). All ``False`` if the scene is skipped.
    skipped : bool
        ``True`` if any aspect quadrant had fewer than the required minimum
        candidates, so the scene cannot be corrected.
    quadrant_counts : dict
        Candidate count per quadrant ("N", "E", "S", "W") *before* the balanced
        draw. Useful for diagnosing why a scene was skipped.
    n_per_quadrant : int
        Number of pixels drawn from each quadrant (0 if skipped). The smallest
        quadrant count when not skipped.
    """

    mask: np.ndarray
    skipped: bool
    quadrant_counts: dict
    n_per_quadrant: int


def forest_mask(forest_source, config: Config):
    """Return a boolean forest mask from the external forest source.

    Parameters
    ----------
    forest_source : numpy.ndarray
        Either a growing-season NDVI raster (float, ``mask_source='ndvi'``) or
        a land-cover raster (``mask_source='landcover'``); for land cover, the
        pixels equal to ``forest_class`` are selected.
    config : Config
        Settings; ``mask_source`` selects the interpretation and provides the
        relevant threshold or class.

    Returns
    -------
    numpy.ndarray
        Boolean forest mask, same shape as ``forest_source``.
    """
    arr = np.asarray(forest_source)

    if config.mask_source == "ndvi":
        return arr >= config.ndvi_threshold

    # land cover: select the forest class only. align_to_grid returns float32,
    # so compare by value (an integer class survives as e.g. 5.0).
    return np.isclose(arr, config.forest_class)


def _quadrant_labels(beta_rad, edges_rad):
    """Label each pixel by aspect quadrant.

    Parameters
    ----------
    beta_rad : numpy.ndarray
        Aspect in radians (0 = North, clockwise).
    edges_rad : sequence of float
        Four bin edges in radians. With the default Config edges
        ``(45, 135, 225, 315)`` degrees, the caller converts them to radians
        before passing.

    Returns
    -------
    numpy.ndarray
        Integer labels 0..3 for N, E, S, W, and ``-1`` where the aspect is
        undefined (NaN, e.g. over DEM nodata). Shape matches ``beta_rad``.

    Notes
    -----
    The array is initialised to ``-1`` rather than left uninitialised: a NaN
    aspect satisfies none of the four bin comparisons, so an uninitialised
    buffer would leave arbitrary memory as that pixel's quadrant label. Such
    pixels cannot reach the sample today (a NaN aspect implies a NaN slope,
    which fails the slope threshold), but ``-1`` makes them unambiguously
    unclassified instead of relying on that.
    """
    e0, e1, e2, e3 = edges_rad
    TWO_PI = np.float32(2.0 * np.pi)
    b = np.mod(np.asarray(beta_rad, dtype=np.float32), TWO_PI)

    label = np.full(b.shape, -1, dtype=np.int8)
    # N wraps across 0: [e3, 2*pi) U [0, e0)
    label[(b >= e3) | (b < e0)] = 0  # N
    label[(b >= e0) & (b < e1)] = 1  # E
    label[(b >= e1) & (b < e2)] = 2  # S
    label[(b >= e2) & (b < e3)] = 3  # W
    return label


def build_sample_mask(cos_i, alpha_rad, beta_rad, valid, forest_source, config: Config):
    """Build the aspect-balanced regression-sample mask and skip decision.

    Parameters
    ----------
    cos_i : numpy.ndarray
        Local illumination condition (from :mod:`fortocorrpy.illum`).
    alpha_rad, beta_rad : numpy.ndarray
        Slope and aspect in radians (from :mod:`fortocorrpy.terrain`). Aspect
        is clockwise from North.
    valid : numpy.ndarray
        Boolean array, ``True`` where reflectance is valid across all processed
        bands (the combined no-data mask, built by the pipeline).
    forest_source : numpy.ndarray
        External NDVI or land-cover raster (see :func:`forest_mask`).
    config : Config
        Settings.

    Returns
    -------
    MaskResult
        The drawn regression mask, skip flag, and per-quadrant diagnostics.

    Notes
    -----
    The four-direction balancing draws the same number of pixels from each
    quadrant, the smallest quadrant's candidate count, using ``numpy``'s random
    generator seeded by ``config.sample_seed`` for reproducibility.
    """
    shape = np.asarray(cos_i).shape

    # Convert the scalar threshold to radians (negligible cost) instead of
    # converting the full alpha array to degrees.
    # Cast to float32 to match alpha's dtype and preserve boundary behaviour.
    slope_min_rad = np.float32(np.deg2rad(config.slope_min_deg))
    m_target = (
        (np.asarray(alpha_rad) >= slope_min_rad)
        & (np.asarray(cos_i) > config.cos_i_threshold)
        & forest_mask(forest_source, config)
    )

    candidate = m_target & np.asarray(valid, dtype=bool)

    # Label candidates by aspect quadrant and count.
    # Convert the 4 scalar edges to radians (float32); the beta array stays in radians.
    edges_rad = tuple(np.float32(np.deg2rad(e)) for e in config.aspect_quadrant_edges)
    labels = _quadrant_labels(beta_rad, edges_rad)
    # Flat indices of candidates, grouped by quadrant.
    cand_flat = np.flatnonzero(candidate)
    cand_labels = labels.ravel()[cand_flat]

    quadrant_counts = {}
    per_quadrant_idx = {}
    for q, name in enumerate(_QUADRANTS):
        idx = cand_flat[cand_labels == q]
        per_quadrant_idx[name] = idx
        quadrant_counts[name] = int(idx.size)

    # Check eligibility, then draw a balanced sample.
    smallest = min(quadrant_counts.values())
    mask = np.zeros(shape, dtype=bool)

    if smallest < config.min_samples_per_quadrant:
        return MaskResult(
            mask=mask,
            skipped=True,
            quadrant_counts=quadrant_counts,
            n_per_quadrant=0,
        )

    rng = np.random.default_rng(config.sample_seed)
    flat_mask = mask.ravel()
    for name in _QUADRANTS:
        idx = per_quadrant_idx[name]
        # draw `smallest` pixels without replacement from this quadrant
        chosen = rng.choice(idx, size=smallest, replace=False)
        flat_mask[chosen] = True

    return MaskResult(
        mask=mask,
        skipped=False,
        quadrant_counts=quadrant_counts,
        n_per_quadrant=smallest,
    )