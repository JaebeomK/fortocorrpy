"""All disk input/output for the package.

Every file read and write in the package happens here. The other modules take
arrays from the pipeline and return arrays; none of them opens a file.

Auxiliary rasters (DEM, external NDVI or land cover) are reprojected and
resampled onto the reference grid. Reflectance is not: it is read as is when
it already matches the grid, sliced when it is another window on the same
pixel lattice, and refused otherwise.

Format policy
-------------
Input format is delegated to rasterio (GDAL underneath), so effectively any
GDAL-readable raster is accepted; GeoTIFF is not special-cased on input.
Output is always GeoTIFF.

Validity / QC
-------------
Validity comes from the reflectance image's own nodata. The image is assumed to
be QC-processed by the user beforehand: invalid pixels (cloud, cloud shadow,
etc.) must already be set to nodata.

The pipeline's contract is that a pixel is valid where every selected band is
finite, which it derives from the stack after nodata has become NaN.
``valid_from_band`` is a cheaper helper for callers who only need one band; it
gives the same answer only under the usual condition that QC was applied across
all bands as one product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window

# Tolerances for comparing two grids, since the same grid written by two tools
# can differ in the last decimal.
#
# Pixel size and rotation are compared relatively: they multiply across the
# raster, so a 0.1% difference drifts by a hundred metres over a 3660-pixel
# tile. The origin is compared as a fraction of a pixel, since it only shifts
# the raster by that amount. A ten-thousandth of a pixel is 3 mm on a 30 m
# grid, which covers float round-off and millimetre-rounded coordinates but
# not a real misregistration.
_SCALE_RTOL = 1e-9
_ORIGIN_FRAC = 1e-4

__all__ = [
    "ImageGrid",
    "read_grid",
    "read_image",
    "read_image_on_grid",
    "read_bands",
    "valid_from_band",
    "grids_match",
    "lattice_offset",
    "align_to_grid",
    "write_geotiff",
]


@dataclass
class ImageGrid:
    """Reference grid of the input image.

    Attributes
    ----------
    transform : affine.Affine
        Affine transform of the image grid.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    width, height : int
        Grid dimensions in pixels.
    count : int
        Number of bands.
    nodata : float or None
        Image nodata value (used to derive validity).
    dtype : str
        Band data type.
    """

    transform: object
    crs: object
    width: int
    height: int
    count: int
    nodata: float | None
    dtype: str

    @property
    def shape(self):
        """``(height, width)`` of the grid."""
        return (self.height, self.width)

    @property
    def pixel_size(self):
        """``(hx, hy)`` pixel size in CRS units (metres for a projected CRS)."""
        return (abs(self.transform.a), abs(self.transform.e))


def _grid_from(src):
    """Build an :class:`ImageGrid` from an open rasterio dataset."""
    return ImageGrid(
        transform=src.transform,
        crs=src.crs,
        width=src.width,
        height=src.height,
        count=src.count,
        nodata=src.nodata,
        dtype=src.dtypes[0],
    )


def _nodata_to_nan(raw, nodata):
    """Cast a raw band stack to float32 with nodata pixels set to NaN.

    The comparison is made on the raw values, before the cast, so an integer
    nodata is matched exactly rather than after a lossy conversion.
    """
    arr = raw.astype(np.float32)
    if nodata is not None:
        if np.issubdtype(raw.dtype, np.floating) and np.isnan(nodata):
            pass  # already NaN where nodata
        else:
            arr[raw == nodata] = np.nan
    return arr


def read_grid(path):
    """Read only the grid metadata of a raster (cheap, no pixel values).

    Parameters
    ----------
    path : str
        Path to a raster readable by rasterio/GDAL.

    Returns
    -------
    ImageGrid
        Grid description used to drive solar-angle, terrain, and illumination
        computation before any pixel values are loaded.
    """
    with rasterio.open(path) as src:
        return _grid_from(src)


def _same_pixel_geometry(t_src, t_dst):
    """True when two transforms have the same pixel size and rotation."""
    px, py = abs(t_dst.a), abs(t_dst.e)
    return (
        abs(t_src.a - t_dst.a) <= _SCALE_RTOL * px
        and abs(t_src.e - t_dst.e) <= _SCALE_RTOL * py
        and abs(t_src.b - t_dst.b) <= _SCALE_RTOL * px
        and abs(t_src.d - t_dst.d) <= _SCALE_RTOL * py
    )


def grids_match(src, dst):
    """True when two grids describe the same pixels.

    Compared with a tolerance rather than for exact equality: the same grid
    written by different tools can differ in the last decimal of the transform,
    which is not a real difference. Pixel size and rotation are compared
    relatively; the origin within a ten-thousandth of a pixel; the shape
    exactly.

    Parameters
    ----------
    src, dst : ImageGrid
        Grids to compare.

    Returns
    -------
    bool
    """
    if src.crs != dst.crs or src.shape != dst.shape:
        return False
    t_src, t_dst = src.transform, dst.transform
    if not _same_pixel_geometry(t_src, t_dst):
        return False
    px, py = abs(t_dst.a), abs(t_dst.e)
    return (
        abs(t_src.c - t_dst.c) <= _ORIGIN_FRAC * px
        and abs(t_src.f - t_dst.f) <= _ORIGIN_FRAC * py
    )


def lattice_offset(src, dst):
    """Offset of ``src`` within ``dst`` when both sit on the same lattice.

    Two grids share a lattice when their CRS, pixel size, and rotation agree
    and their origins differ by a whole number of pixels. Such grids are
    different *windows* on one pixel lattice, so one can be placed into the
    other by slicing, without resampling and so without changing the values.

    Parameters
    ----------
    src, dst : ImageGrid
        Grids to compare.

    Returns
    -------
    tuple of int, or None
        ``(row_off, col_off)`` of ``src``'s first pixel within ``dst``, or
        ``None`` when the two do not share a lattice.
    """
    if src.crs != dst.crs:
        return None
    t_src, t_dst = src.transform, dst.transform
    if not _same_pixel_geometry(t_src, t_dst):
        return None
    if t_dst.b != 0 or t_dst.d != 0:
        return None  # rotated grids: offsets are not row/column aligned

    col = (t_src.c - t_dst.c) / t_dst.a
    row = (t_src.f - t_dst.f) / t_dst.e
    if abs(col - round(col)) > _ORIGIN_FRAC or abs(row - round(row)) > _ORIGIN_FRAC:
        return None
    return int(round(row)), int(round(col))


def _lattice_difference(src, dst):
    """Name what stops ``src`` from sharing ``dst``'s lattice, for an error."""
    if src.crs != dst.crs:
        return f"CRS {src.crs} vs {dst.crs}"
    if not _same_pixel_geometry(src.transform, dst.transform):
        return (f"pixel size {abs(src.transform.a):g} m vs "
                f"{abs(dst.transform.a):g} m")
    col = (src.transform.c - dst.transform.c) / dst.transform.a
    row = (src.transform.f - dst.transform.f) / dst.transform.e
    return (f"origin offset {row:.4f} rows, {col:.4f} columns, which is not a "
            "whole number of pixels")


def read_image(path, band_indices=None, *, scale=1.0, offset=0.0):
    """Read grid metadata and the requested reflectance bands in one open.

    This is the pipeline's entry read: it opens the file once and returns both
    the :class:`ImageGrid` and the reflectance stack, reading all requested
    bands in a single ``rasterio`` call to avoid per-band I/O. Nodata pixels
    become ``nan`` so invalidity propagates; the validity mask is derived from
    those NaNs downstream in masking.

    Parameters
    ----------
    path : str
        Path to the reflectance raster.
    band_indices : sequence of int or None
        Zero-based band indices to read. ``None`` reads all bands.
    scale, offset : float
        Linear conversion ``reflectance = DN * scale + offset`` applied on read.

    Returns
    -------
    grid : ImageGrid
        Grid metadata of the image.
    reflectance : numpy.ndarray
        Reflectance stack, shape ``(bands, rows, cols)``, ``float32``, with
        nodata pixels set to ``nan``.
    """
    with rasterio.open(path) as src:
        grid = _grid_from(src)
        if band_indices is None:
            bands_1based = list(range(1, src.count + 1))
        else:
            bands_1based = [b + 1 for b in band_indices]

        raw = src.read(bands_1based)  # single I/O for all requested bands
        nodata = src.nodata

    arr = _nodata_to_nan(raw, nodata)
    del raw  # free the original buffer; only the float32 copy is needed

    if scale != 1.0 or offset != 0.0:
        arr = arr * np.float32(scale) + np.float32(offset)

    return grid, arr


def read_image_on_grid(path, grid: ImageGrid, band_indices=None, *,
                       scale=1.0, offset=0.0):
    """Read a reflectance image placed on a reference grid.

    Two cases are accepted, and both are lossless:

    1. The image already sits on ``grid`` (within tolerance): read as is.
    2. The image is a different window on the same pixel lattice, overlapping
       ``grid`` by at least one pixel: the overlapping window is read and
       placed into ``grid`` by slicing. Values are unchanged; the part of
       ``grid`` the image does not cover stays ``nan`` and counts as invalid
       downstream.

    Anything else raises: an image on another lattice, and an image that
    shares the lattice but lies entirely outside the grid. Reflectance is
    never resampled to fit, because it is what the correction acts on.
    Correct images on another lattice with a separate call.

    Parameters
    ----------
    path : str
        Path to the reflectance raster.
    grid : ImageGrid
        Reference grid to place the image on.
    band_indices : sequence of int or None
        Zero-based band indices to read. ``None`` reads all bands.
    scale, offset : float
        Linear conversion ``reflectance = DN * scale + offset`` applied on read.

    Returns
    -------
    numpy.ndarray
        Reflectance stack on ``grid``, shape ``(bands, rows, cols)``,
        ``float32``, nodata and uncovered pixels set to ``nan``.

    Raises
    ------
    ValueError
        If the image does not share the reference grid's pixel lattice, or
        shares it but lies entirely outside the grid.
    """
    height, width = grid.shape

    with rasterio.open(path) as src:
        src_grid = _grid_from(src)
        if band_indices is None:
            bands_1based = list(range(1, src.count + 1))
        else:
            bands_1based = [b + 1 for b in band_indices]
        nodata = src.nodata

        if grids_match(src_grid, grid):
            arr = _nodata_to_nan(src.read(bands_1based), nodata)

        elif (rc := lattice_offset(src_grid, grid)) is not None:
            row_off, col_off = rc
            dr0, dc0 = max(row_off, 0), max(col_off, 0)
            dr1 = min(row_off + src_grid.height, height)
            dc1 = min(col_off + src_grid.width, width)

            if dr1 <= dr0 or dc1 <= dc0:
                raise ValueError(
                    f"{path} shares the reference grid's lattice but lies "
                    f"entirely outside it (offset {row_off} rows, {col_off} "
                    f"columns into a {height} x {width} grid), so it would "
                    "contribute no pixel. Correct it against its own area."
                )

            arr = np.full((len(bands_1based), height, width), np.nan, np.float32)
            raw = src.read(
                bands_1based,
                window=Window(dc0 - col_off, dr0 - row_off,
                              dc1 - dc0, dr1 - dr0),
            )
            arr[:, dr0:dr1, dc0:dc1] = _nodata_to_nan(raw, nodata)
            del raw

        else:
            raise ValueError(
                f"{path} does not share the reference grid's pixel lattice: "
                f"{_lattice_difference(src_grid, grid)}. Correct images on "
                "another lattice with a separate correct_image() call."
            )

    if scale != 1.0 or offset != 0.0:
        arr = arr * np.float32(scale) + np.float32(offset)

    return arr


def read_bands(path, band_indices=None, *, scale=1.0, offset=0.0):
    """Read reflectance bands as a float32 stack.

    Parameters
    ----------
    path : str
        Path to the reflectance raster.
    band_indices : sequence of int or None
        Zero-based band indices to read. ``None`` reads all bands.
    scale, offset : float
        Linear conversion ``reflectance = DN * scale + offset`` applied on read.

    Returns
    -------
    numpy.ndarray
        Reflectance stack, shape ``(bands, rows, cols)``, ``float32``. Nodata
        pixels are set to ``nan`` so they propagate as invalid downstream.
    """
    with rasterio.open(path) as src:
        if band_indices is None:
            bands_1based = list(range(1, src.count + 1))
        else:
            bands_1based = [b + 1 for b in band_indices]

        raw = src.read(bands_1based)
        nodata = src.nodata

    # Mark nodata pixels as NaN so invalidity propagates.
    arr = _nodata_to_nan(raw, nodata)
    del raw  # free the original buffer

    if scale != 1.0 or offset != 0.0:
        arr = arr * np.float32(scale) + np.float32(offset)

    return arr


def valid_from_band(path, band_index=0):
    """Derive the validity mask from one band's nodata.

    The image is assumed QC-processed (invalid pixels already nodata) and QC is
    common across bands, so a single band defines scene validity.

    Parameters
    ----------
    path : str
        Path to the reflectance raster.
    band_index : int
        Zero-based index of the band to inspect.

    Returns
    -------
    numpy.ndarray
        Boolean array, ``True`` where the pixel is valid (not nodata), shape
        ``(rows, cols)``.
    """
    with rasterio.open(path) as src:
        band = src.read(band_index + 1)
        nodata = src.nodata

    if nodata is None:
        # No nodata declared: treat all finite pixels as valid.
        return np.isfinite(band) if np.issubdtype(band.dtype, np.floating) else np.ones(band.shape, dtype=bool)

    if np.issubdtype(band.dtype, np.floating) and np.isnan(nodata):
        return ~np.isnan(band)
    return band != nodata


def align_to_grid(path, grid: ImageGrid, *, resampling="bilinear", band_index=0):
    """Reproject/resample an auxiliary raster onto the reference image grid.

    The image grid is the absolute reference; the auxiliary raster (DEM,
    external NDVI, land cover) at any CRS/resolution is reprojected and
    resampled onto it. If the source already matches the grid (same CRS,
    transform, and size) it is read without warping.

    Parameters
    ----------
    path : str
        Path to the auxiliary raster.
    grid : ImageGrid
        Reference grid to align onto.
    resampling : str
        Resampling method name: ``"bilinear"`` for continuous data (DEM, NDVI)
        and ``"nearest"`` for a categorical land-cover raster (the forest class
        is selected later).
    band_index : int
        Zero-based band of the source to use.

    Returns
    -------
    numpy.ndarray
        Aligned raster on the image grid, shape ``(rows, cols)``, ``float32``.
    """
    method = getattr(Resampling, resampling)

    with rasterio.open(path) as src:
        src_nodata = src.nodata
        if grids_match(_grid_from(src), grid):
            arr = src.read(band_index + 1).astype(np.float32)
            if src_nodata is not None and not (
                isinstance(src_nodata, float) and np.isnan(src_nodata)
            ):
                arr[arr == np.float32(src_nodata)] = np.nan
            return arr

        destination = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band_index + 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            init_dest_nodata=True,
            resampling=method,
        )
    return destination


def write_geotiff(path, data, grid: ImageGrid, *, nodata=np.nan, band_indices=None):
    """Write a corrected raster to GeoTIFF on the reference grid.

    Parameters
    ----------
    path : str
        Output path (GeoTIFF).
    data : numpy.ndarray
        Result array, shape ``(bands, rows, cols)`` or ``(rows, cols)``.
    grid : ImageGrid
        Reference grid; its CRS and transform are written so the output
        overlays the input exactly.
    nodata : float
        Nodata value to record (default ``nan``, matching the NaN used for
        self-shadow and invalid pixels).
    band_indices : sequence of int or None
        Zero-based positions of these bands in the original input stack. When
        given, each output band is labelled with its original position so the
        spectral band stays identifiable even after selecting a subset. The
        label is written 1-based (``"original_band_3"``). ``None`` labels the
        bands by their own order.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or 3-D, if its rows and columns do not match
        ``grid``, or if ``band_indices`` does not have one entry per band.

    Notes
    -----
    Spectral identity is carried by stack position rather than by copying the
    input metadata, which differs from scene to scene.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"data must be 2-D or 3-D, got shape {arr.shape}")
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    # The CRS and transform written are the grid's, so an array of another size
    # would be georeferenced as if it started at the grid's origin: a file that
    # opens cleanly and sits in the wrong place.
    if arr.shape[-2:] != grid.shape:
        raise ValueError(
            f"data is {arr.shape[-2:]} but the grid is {grid.shape}; the "
            "output would carry the grid's transform and land in the wrong "
            "place."
        )

    count, height, width = arr.shape

    # Without one entry per band the labelling silently goes out of step:
    # surplus bands are left undescribed, and the rest are mislabelled.
    if band_indices is not None and len(band_indices) != count:
        raise ValueError(
            f"band_indices has {len(band_indices)} entries but the data has "
            f"{count} bands"
        )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        # Label each band with its original 1-based stack position.
        if band_indices is not None:
            for out_pos, orig0 in enumerate(band_indices, start=1):
                dst.set_band_description(out_pos, f"original_band_{orig0 + 1}")
        else:
            for out_pos in range(1, count + 1):
                dst.set_band_description(out_pos, f"original_band_{out_pos}")