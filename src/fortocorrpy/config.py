"""Central configuration for the topographic-correction pipeline.

:class:`Config` holds the method choices, thresholds, sampling settings and
bulk-processing options. Every field has a default, and validation runs at
construction time so an invalid value is reported before a long batch run
rather than inside it.

Settings files are not read here. A caller keeping parameters in JSON or
similar converts them to a mapping and passes ``Config(**mapping)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = ["Config", "METHODS", "REGRESSION_METHODS", "SLOPE_TC_METHODS",
           "DENOMINATOR_METHODS"]

# Correction-method keys.
METHODS = ("cosine", "scs", "c", "scsc", "se", "er")

# Methods whose coefficients come from the reflectance-cos i regression.
# Naming the set here rather than inlining it keeps the three method groups in
# one place, so adding a method means adding it to the groups it belongs to.
REGRESSION_METHODS = ("c", "scsc", "se", "er")

# Methods whose formula uses cos(alpha) separately from cos i.
# These require the slope raster to survive until the correction step.
SLOPE_TC_METHODS = ("scs", "scsc")

# Methods whose formula has cos i in the denominator.
# Applied only above Config.cos_i_threshold.
DENOMINATOR_METHODS = ("cosine", "scs", "c", "scsc")


@dataclass
class Config:
    """Settings for a topographic correction run, single image or bulk.

    Parameters
    ----------
    mask_source : {'ndvi', 'landcover'}
        Source used to identify the target (forest) cover for the regression
        sample. ``'ndvi'`` thresholds an external growing-season NDVI raster;
        ``'landcover'`` selects a class from an external land-cover raster.
    ndvi_threshold : float
        Lower NDVI bound for forest, used when ``mask_source='ndvi'``.
    forest_class : int or None
        Land-cover class value treated as forest, used when
        ``mask_source='landcover'``.
    slope_min_deg : float
        Minimum slope (degrees) for a pixel to enter the regression sample.
    cos_i_threshold : float
        Lower bound on cos i for the regression sample and for applying the
        denominator-form methods. ``0.0`` excludes self-shadow (cos i <= 0).
    aspect_quadrant_edges : sequence of float
        Aspect bin edges (degrees, clockwise from North) used to label
        quadrants for the four-direction sample check. The default splits
        N, E, S, W centred on the cardinal directions.
    min_samples_per_quadrant : int
        Minimum number of sample pixels required in every aspect quadrant.
        If any quadrant falls short, the scene is skipped.
    sample_seed : int or None
        Seed for the per-quadrant random draw that balances the regression
        sample across aspect quadrants. A fixed seed makes the draw (and thus
        the resulting coefficients) reproducible: the same seed always selects
        the same sample. ``None`` draws a different sample each run.
    methods : sequence of str
        Correction methods to apply. Must be a subset of :data:`METHODS`.
    band_indices : sequence of int or None
        Zero-based indices of the reflectance bands to correct. ``None`` means
        all bands in the image.
    n_workers : int
        Number of threads for bulk processing. ``1`` (default) processes
        images sequentially. Values above 1 process images in parallel using
        a thread pool; each worker opens its own file handles and allocates
        its own arrays, so peak memory is roughly ``n_workers`` times the
        single-image footprint. Set to the number of available cores or fewer.
    stream : bool
        When a bulk call hands results back. ``False`` (default) processes the
        whole list and returns it, so every corrected array is alive at once:
        the cost is ``images x methods x bands x rows x cols x 4`` bytes.
        ``True`` returns a generator that yields the results one at a time, in
        input order, so the caller can consume and release each before the
        next arrives and peak memory follows ``n_workers`` instead of the
        length of the list. Ignored for a single image.

        The setting changes the type of the return value, not the results
        themselves. Iterating works either way; only indexing
        (``results[0]``) and ``len()`` need the list form, which ``list(...)``
        restores.
    """

    # --- forest (target) identification ---
    mask_source: str = "ndvi"
    ndvi_threshold: float = 0.5
    forest_class: int | None = None

    # --- illumination / sample geometry ---
    slope_min_deg: float = 5.0
    cos_i_threshold: float = 0.0

    # --- four-direction (aspect quadrant) balance ---
    aspect_quadrant_edges: Sequence[float] = field(
        default_factory=lambda: (45.0, 135.0, 225.0, 315.0)
    )
    min_samples_per_quadrant: int = 200
    sample_seed: int | None = 42

    # --- correction methods ---
    methods: Sequence[str] = field(default_factory=lambda: ("scsc",))

    # --- band selection ---
    band_indices: Sequence[int] | None = None

    # --- parallelism / result delivery ---
    n_workers: int = 1
    stream: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.mask_source not in ("ndvi", "landcover"):
            raise ValueError(
                f"mask_source must be 'ndvi' or 'landcover', got "
                f"{self.mask_source!r}"
            )
        if self.mask_source == "landcover" and self.forest_class is None:
            raise ValueError(
                "forest_class must be set when mask_source='landcover'"
            )

        if self.slope_min_deg < 0.0:
            raise ValueError(
                f"slope_min_deg must be non-negative, got {self.slope_min_deg}"
            )

        if self.cos_i_threshold < 0.0:
            raise ValueError(
                f"cos_i_threshold must be >= 0.0 to exclude self-shadow, "
                f"got {self.cos_i_threshold}"
            )
        if self.min_samples_per_quadrant < 1:
            raise ValueError(
                f"min_samples_per_quadrant must be >= 1, got "
                f"{self.min_samples_per_quadrant}"
            )

        edges = tuple(self.aspect_quadrant_edges)
        if len(edges) != 4:
            raise ValueError(
                "aspect_quadrant_edges must contain exactly four edges"
            )
        if any(not 0.0 <= e < 360.0 for e in edges):
            raise ValueError(
                "aspect_quadrant_edges must all lie in [0, 360)"
            )
        if any(edges[i] >= edges[i + 1] for i in range(3)):
            raise ValueError(
                "aspect_quadrant_edges must be strictly increasing"
            )

        if not self.methods:
            raise ValueError("methods must contain at least one method")
        unknown = [m for m in self.methods if m not in METHODS]
        if unknown:
            raise ValueError(
                f"unknown correction method(s): {unknown}; "
                f"valid methods are {METHODS}"
            )

        if self.n_workers < 1:
            raise ValueError(
                f"n_workers must be >= 1, got {self.n_workers}"
            )

    @property
    def needs_regression(self) -> bool:
        """True if any selected method requires the reflectance-cos i regression."""
        return any(m in REGRESSION_METHODS for m in self.methods)

    @property
    def needs_slope_at_correction(self) -> bool:
        """True if any selected method uses cos(alpha) at the correction step.

        Determines whether the slope raster must survive until T_correct
        (slope-TC group) or may be released after cos i is formed.
        """
        return any(m in SLOPE_TC_METHODS for m in self.methods)