"""fortocorrpy: topographic correction of optical satellite imagery.

A satellite- and format-agnostic toolkit for topographic correction (TC) of
optical imagery over mountainous terrain. It applies several correction
methods through one interface and, optionally, reports the standard
reflectance-level metrics for each method in the same call, so methods can be
compared without re-running bespoke evaluation code.

Quick start
-----------
Single image:

>>> from fortocorrpy import Config, correct_image, io
>>> from datetime import datetime, timezone
>>>
>>> cfg = Config(
...     mask_source="ndvi",
...     methods=("cosine", "scsc", "er"),
... )
>>> result = correct_image(
...     image_path="scene.tif",
...     datetime_utc=datetime(2024, 6, 21, 3, 0, tzinfo=timezone.utc),
...     dem_path="dem.tif",
...     forest_path="ndvi.tif",
...     config=cfg,
...     evaluate=True,
... )
>>>
>>> if not result.mask_result.skipped:
...     for method, arr in result.corrected.items():
...         io.write_geotiff(
...             f"scene_{method}.tif", arr, result.grid,
...             band_indices=result.band_indices,
...         )

Time series on one pixel lattice. The images may be different overlapping
windows; one DEM and one forest raster cover the reference grid:

>>> images = ["scene_01.tif", "scene_02.tif", "scene_03.tif"]
>>> times  = [dt_01, dt_02, dt_03]
>>> results = correct_image(images, times, "dem.tif", "ndvi.tif", cfg)

Within one image the band loop is sequential. Across images, set
``Config.n_workers`` above 1 to process several at once in a thread pool. Pass
``stream=True`` to receive the results one at a time, in input order, instead
of the whole list at the end, so a long series can be consumed and released one
image at a time rather than held in full. Images on different pixel lattices
are not a batch: call ``correct_image`` once per lattice.
"""

from __future__ import annotations

__version__ = "0.1.0"

# High-level entry point and configuration (the common API).
from .config import (
    Config, METHODS, REGRESSION_METHODS, SLOPE_TC_METHODS,
    DENOMINATOR_METHODS,
)
from .pipeline import correct_image, CorrectionResult

# Result/record types returned by the pipeline and lower-level functions.
from .masking import MaskResult
from .linear_C import BandCoefficients
from .evaluation import BandMetrics
from .io import ImageGrid

# Submodules, for direct use of the lower-level building blocks.
from . import (
    io,
    solar,
    terrain,
    illum,
    masking,
    linear_C,
    T_correct,
    evaluation,
    pipeline,
    config,
)

__all__ = [
    "__version__",
    # high-level
    "Config",
    "correct_image",
    "CorrectionResult",
    # constants
    "METHODS",
    "REGRESSION_METHODS",
    "SLOPE_TC_METHODS",
    "DENOMINATOR_METHODS",
    # record types
    "MaskResult",
    "BandCoefficients",
    "BandMetrics",
    "ImageGrid",
    # submodules
    "io",
    "solar",
    "terrain",
    "illum",
    "masking",
    "linear_C",
    "T_correct",
    "evaluation",
    "pipeline",
    "config",
]