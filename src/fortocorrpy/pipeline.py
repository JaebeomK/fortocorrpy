"""Topographic-correction pipeline (the assembly entry point).

``correct_image`` calls the modules in order, branches on the selected
methods, and releases arrays as they stop being needed. It does no numeric
work of its own. The step order and the result contract are set out in
DESIGN_SPEC, "Correction".

Parallelism
-----------
Within an image the per-band loop is sequential. Across images, ``n_workers``
above 1 runs the per-image correction in a thread pool: each worker opens its
own file handles and allocates its own arrays, while the terrain (alpha, beta)
and the forest source are shared read-only. Peak memory is roughly
``n_workers x single-image footprint + shared arrays``.

Images on different pixel lattices are separate calls, so parallelism across
tiles is
left to the caller (a job array, ``multiprocessing``, and so on).

Array lifetimes
---------------
theta_s and phi_s are released once cos_i and cos_theta_s are formed, beta
once the quadrant labels are built, and alpha after illum unless a selected
method needs cos(alpha) at the correction step. Reflectance, cos_i and
cos_theta_s live until the correction. The corrected arrays are held until the
call returns unless ``Config.stream`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import io, solar, terrain, illum, masking, linear_C, T_correct, evaluation
from .config import Config

__all__ = ["CorrectionResult", "correct_image"]


@dataclass
class CorrectionResult:
    """Outcome of correcting one image.

    Attributes
    ----------
    corrected : dict or None
        Mapping ``method -> ndarray[bands, rows, cols]`` of corrected
        reflectance. ``None`` if the image was skipped or if it failed.
    coefficients : list or None
        Per band, the :class:`fortocorrpy.linear_C.BandCoefficients` used (or
        ``None`` if no regression was needed). ``None`` if skipped.
    mask_result : fortocorrpy.masking.MaskResult
        The sample-mask outcome, including skip flag and quadrant diagnostics.
        ``None`` only if the image failed before masking.
    grid : fortocorrpy.io.ImageGrid
        The reference grid the correction ran on, to be passed to
        :func:`fortocorrpy.io.write_geotiff` when writing the arrays out.
    band_indices : list
        Zero-based positions of the output bands in the original input stack,
        preserving spectral identity (especially when a subset was selected).
        For full-image runs this is ``[0, 1, ..., n-1]``.
    metrics_before : list or None
        Per-band reflectance-level metrics on the uncorrected reflectance over
        the regression sample. ``None`` if not evaluated or if skipped. Shared
        across methods because the input does not depend on the method.
    metrics_after : dict or None
        Mapping ``method -> list[BandMetrics]`` on the corrected reflectance
        over the same regression sample. ``None`` if not evaluated or skipped.
    error : str or None
        The exception raised while processing this image, if any. Only set on
        a bulk call, where one unreadable file must not cost the batch the
        images that are fine; a single-image call raises instead. ``None`` on
        success.

    Notes
    -----
    ``skipped`` and ``error`` are different outcomes. A skipped image is
    normal: the scene did not meet the four-quadrant sample requirement, and
    ``mask_result.quadrant_counts`` says why. An errored image is a failure:
    the file could not be read, its CRS was unusable, or the computation
    raised.
    """

    corrected: dict | None
    coefficients: list | None
    mask_result: object
    band_indices: list
    grid: object
    metrics_before: list | None = None
    metrics_after: dict | None = None
    error: str | None = None


def _validate_grid(grid):
    """Check that the image CRS is projected with metre units."""
    if grid.crs is None:
        raise ValueError("image CRS is required")
    if not grid.crs.is_projected:
        raise ValueError(
            "image CRS must be projected (metre units) for slope/aspect"
        )
    unit_name, unit_factor = grid.crs.linear_units_factor
    if abs(unit_factor - 1.0) > 1e-9:
        raise ValueError(
            f"image CRS linear unit must be metre; got {unit_name!r} "
            f"with factor {unit_factor}"
        )


def _correct_single(image_path, datetime_utc, grid, alpha, beta,
                     forest_source, config, band_indices, *, evaluate=False):
    """Correct one image using pre-computed terrain and forest data.

    This is the inner loop body shared by single and bulk execution paths.
    """
    reflectance = io.read_image_on_grid(image_path, grid, band_indices)

    n_bands = reflectance.shape[0]
    resolved_bands = band_indices if band_indices is not None else list(range(n_bands))

    # Solar angles for this image's acquisition time.
    theta_s, phi_s = solar.solar_angles(
        datetime_utc, grid.transform, grid.crs, grid.shape,
    )

    # cos i and the horizontal reference cos(theta_s); release solar angles.
    cos_i, cos_theta_s = illum.cos_incidence(theta_s, phi_s, alpha, beta)
    del theta_s, phi_s

    # Build the regression-sample mask and skip decision.
    mask_result = masking.build_sample_mask(
        cos_i, alpha, beta, _valid_from_reflectance(reflectance),
        forest_source, config,
    )

    if mask_result.skipped:
        del cos_i, cos_theta_s, reflectance
        return CorrectionResult(
            corrected=None, coefficients=None,
            mask_result=mask_result, band_indices=resolved_bands,
            grid=grid,
        )

    # Determine whether cos(alpha) is needed at the correction step.
    # Precompute once instead of recalculating per band per method.
    keep_alpha = config.needs_slope_at_correction
    cos_alpha = np.cos(alpha).astype(np.float32) if keep_alpha else None

    # Per-band, sequential: regress (if needed) then correct.
    methods = list(config.methods)
    corrected = {
        m: np.empty((n_bands,) + grid.shape, dtype=np.float32) for m in methods
    }
    band_coeffs = []

    for b in range(n_bands):
        refl_b = reflectance[b]

        if config.needs_regression:
            coeff_b = linear_C.fit_band(cos_i, refl_b, mask_result.mask)
        else:
            coeff_b = None
        band_coeffs.append(coeff_b)

        for m in methods:
            corrected[m][b] = T_correct.correct(
                m, refl_b, cos_i, cos_theta_s,
                cos_alpha=cos_alpha,
                coeff=coeff_b,
                cos_i_threshold=config.cos_i_threshold,
            )

    # Evaluation over the SAME regression-sample mask.
    metrics_before = None
    metrics_after = None
    if evaluate:
        metrics_before = evaluation.evaluate_before(
            cos_i, reflectance, mask_result.mask,
        )
        metrics_after = {
            m: evaluation.evaluate_after(cos_i, corrected[m], mask_result.mask)
            for m in methods
        }

    del reflectance, cos_i, cos_theta_s

    return CorrectionResult(
        corrected=corrected,
        coefficients=band_coeffs,
        mask_result=mask_result,
        band_indices=resolved_bands,
        grid=grid,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )


def correct_image(image_path, datetime_utc, dem_path, forest_path,
                  config: Config, *, evaluate=False, reference_path=None):
    """Correct one or more images and return the results.

    Parameters
    ----------
    image_path : str or list of str
        Path(s) to the QC-processed reflectance raster(s). A single str
        processes one image; a list processes multiple images of the same area.
        An image that is a different window on the same pixel lattice is placed
        on the reference grid by slicing, without resampling; an image on a
        different lattice is refused, never resampled to fit.
    datetime_utc : datetime or list of datetime
        Acquisition time(s) (UTC). Must match ``image_path`` in length when
        both are lists.
    dem_path : str
        Path to a DEM. A single path only: a batch shares one grid, so one DEM
        covers it, and slope/aspect are computed once and reused. Correct
        images on different grids with separate calls.
    forest_path : str or list of str
        Path to the forest source (NDVI or land-cover raster). A single str
        aligns once and is shared; a list aligns per image, so the forest
        source can follow the season.
    config : Config
        Settings controlling masking, sampling, and the methods to apply.
    evaluate : bool, default False
        If True, compute reflectance-level metrics over the regression sample
        for the uncorrected and corrected reflectance per image.
    reference_path : str or None, default None
        Raster whose grid every result is produced on. ``None`` uses the first
        image. Give a base image covering the whole area when the images are
        smaller windows of it, so that all results share one grid and stack
        directly. Only its grid is read; its pixel values are never used.

    Returns
    -------
    CorrectionResult, list of CorrectionResult, or generator
        A single ``CorrectionResult`` when ``image_path`` is a str. For a
        list, either a list of results or, when ``config.stream`` is True, a
        generator over the same results in the same order. Iterating works the
        same way for both.

    Examples
    --------
    Single image:

    >>> result = correct_image(
    ...     "scene.tif",
    ...     datetime(2024, 6, 21, 3, 0, tzinfo=timezone.utc),
    ...     "dem.tif", "ndvi.tif", config,
    ... )

    Bulk (same DEM, same forest source):

    >>> results = correct_image(
    ...     ["scene_01.tif", "scene_02.tif", "scene_03.tif"],
    ...     [dt_01, dt_02, dt_03],
    ...     "dem.tif", "ndvi.tif", config,
    ... )

    Bulk (same DEM, per-image forest source):

    >>> results = correct_image(
    ...     ["scene_01.tif", "scene_02.tif"],
    ...     [dt_01, dt_02],
    ...     "dem.tif",
    ...     ["ndvi_spring.tif", "ndvi_summer.tif"],
    ...     config,
    ... )

    A long series, consumed one image at a time:

    >>> config = Config(methods=("scsc",), stream=True)
    >>> for result in correct_image(images, times, "dem.tif", "ndvi.tif", config):
    ...     if result.error or result.mask_result.skipped:
    ...         continue
    ...     for method, arr in result.corrected.items():
    ...         io.write_geotiff(f"{method}.tif", arr, result.grid,
    ...                          band_indices=result.band_indices)

    Notes
    -----
    Writing is the caller's step, through
    :func:`fortocorrpy.io.write_geotiff` on the returned arrays.

    Without ``config.stream`` every corrected array stays alive until the call
    returns, so a batch costs ``images x methods x bands x rows x cols x 4``
    bytes: a 3660x3660 four-band tile with six methods is about 1.3 GB per
    image. With ``config.stream`` only the images in flight are held, so the
    cost follows ``n_workers``.

    On a list input a failure is recorded on the result rather than raised, so
    the remaining images are still processed. Check ``error`` before
    ``corrected``.

    Every result is produced on the reference grid, the first image's or
    ``reference_path``'s, so the results of a batch stack directly. An image
    covering only part of that grid leaves the rest ``nan``, which counts as
    invalid and stays out of the regression sample.
    """
    # --- Normalize inputs to lists ---
    single_input = isinstance(image_path, str)
    if single_input:
        image_path = [image_path]
        datetime_utc = [datetime_utc]

    n_images = len(image_path)

    if len(datetime_utc) != n_images:
        raise ValueError(
            f"image_path ({n_images}) and datetime_utc "
            f"({len(datetime_utc)}) must have the same length"
        )

    if not isinstance(dem_path, str):
        raise TypeError(
            "dem_path must be a single path: a batch shares one grid, so one "
            "DEM covers it. Correct images on different grids with separate "
            f"correct_image() calls. Got {type(dem_path).__name__}."
        )

    forest_fixed = isinstance(forest_path, str)

    if not forest_fixed and len(forest_path) != n_images:
        raise ValueError(
            f"forest_path list ({len(forest_path)}) must match "
            f"image_path ({n_images}) in length"
        )

    band_indices = (
        list(config.band_indices) if config.band_indices is not None else None
    )
    forest_resampling = (
        "bilinear" if config.mask_source == "ndvi" else "nearest"
    )

    # --- Terrain: computed once from the reference grid ---
    grid = io.read_grid(reference_path if reference_path else image_path[0])
    _validate_grid(grid)
    dem = io.align_to_grid(dem_path, grid, resampling="bilinear")
    alpha, beta = terrain.slope_aspect(dem, *grid.pixel_size)
    del dem

    forest_source = None
    if forest_fixed:
        forest_source = io.align_to_grid(
            forest_path, grid, resampling=forest_resampling,
        )

    # --- Process each image ---
    def _process_one(i):
        """Process the i-th image. Designed to run in a thread worker."""
        # Per-image forest source (when forest_path is a list). Aligned here
        # rather than up front so a long list holds one raster, not n.
        forest_i = forest_source
        if not forest_fixed:
            forest_i = io.align_to_grid(
                forest_path[i], grid, resampling=forest_resampling,
            )

        return _correct_single(
            image_path[i], datetime_utc[i], grid, alpha, beta,
            forest_i, config, band_indices, evaluate=evaluate,
        )

    def _process_guarded(i):
        """Process one image, returning the failure rather than raising it.

        The exception is recorded on the result and the batch continues, so one
        unreadable file does not discard the images that are fine. A
        single-image call has nothing to protect and raises normally.
        """
        try:
            return _process_one(i)
        except Exception as exc:                       # noqa: BLE001
            return CorrectionResult(
                corrected=None, coefficients=None, mask_result=None,
                band_indices=band_indices or [],
                grid=grid,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _iter_results():
        """Yield results in input order, one image at a time.

        In parallel, at most ``n_workers + 1`` images are submitted at a time.
        ThreadPoolExecutor.map would queue the whole list at once and run to
        the end of the batch however slowly results were consumed, which would
        leave ``Config.stream`` bounding nothing.
        """
        n_workers = min(config.n_workers, n_images)

        if n_workers <= 1:
            for i in range(n_images):
                yield _process_guarded(i)
            return

        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            in_flight = deque()
            submitted = 0
            while submitted < n_images and len(in_flight) <= n_workers:
                in_flight.append(pool.submit(_process_guarded, submitted))
                submitted += 1
            while in_flight:
                result = in_flight.popleft().result()
                if submitted < n_images:
                    in_flight.append(pool.submit(_process_guarded, submitted))
                    submitted += 1
                yield result

    if single_input:
        return _process_one(0)
    if config.stream:
        return _iter_results()
    return list(_iter_results())


def _valid_from_reflectance(reflectance):
    """Validity from the reflectance stack: pixels finite across all bands.

    Nodata became NaN on read, so a pixel is valid where no band is NaN. QC is
    usually common across bands, in which case this agrees with a single-band
    check; it stays correct when the bands differ.
    """
    return ~np.isnan(reflectance).any(axis=0)