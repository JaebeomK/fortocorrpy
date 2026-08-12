"""Disk I/O: reading, alignment of auxiliary rasters, and writing."""

import numpy as np
import rasterio

from fortocorrpy import io


def test_read_grid_returns_metadata_only(write_raster):
    path = write_raster("g.tif", np.zeros((3, 4, 5), np.float32))
    grid = io.read_grid(path)
    assert grid.shape == (4, 5)
    assert grid.count == 3
    assert grid.pixel_size == (30.0, 30.0)
    assert grid.crs.is_projected


def test_read_image_returns_grid_and_stack(write_raster):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    grid, arr = io.read_image(write_raster("img.tif", data))
    assert grid.shape == (3, 4)
    assert arr.shape == (2, 3, 4)
    assert arr.dtype == np.float32
    assert np.allclose(arr, data)


def test_read_image_selects_bands_by_zero_based_index(write_raster):
    data = np.stack([
        np.full((3, 3), 1.0), np.full((3, 3), 2.0), np.full((3, 3), 3.0),
    ]).astype(np.float32)
    _, arr = io.read_image(write_raster("img.tif", data), band_indices=[0, 2])
    assert arr.shape == (2, 3, 3)
    assert np.allclose(arr[0], 1.0) and np.allclose(arr[1], 3.0)


def test_read_image_converts_nodata_to_nan(write_raster):
    data = np.array([[[0.1, -9999.0], [0.3, 0.4]]], np.float32)
    _, arr = io.read_image(write_raster("img.tif", data, nodata=-9999.0))
    assert np.isnan(arr[0, 0, 1])
    assert np.isclose(arr[0, 0, 0], 0.1)


def test_read_image_applies_scale_and_offset(write_raster):
    """HLS-style DN -> reflectance conversion."""
    data = np.array([[[1000.0, 2500.0]]], np.float32)
    _, arr = io.read_image(write_raster("img.tif", data), scale=0.0001)
    assert np.allclose(arr[0, 0], [0.1, 0.25], atol=1e-6)


def test_read_bands_matches_read_image(write_raster):
    data = np.array([[[0.1, 0.2], [0.3, 0.4]]], np.float32)
    path = write_raster("img.tif", data)
    _, from_image = io.read_image(path)
    assert np.allclose(io.read_bands(path), from_image)


def test_valid_from_band_uses_nodata(write_raster):
    data = np.array([[[0.1, -9999.0], [0.3, 0.4]]], np.float32)
    valid = io.valid_from_band(write_raster("img.tif", data, nodata=-9999.0))
    assert valid.tolist() == [[True, False], [True, True]]


def test_align_to_grid_skips_warping_when_already_aligned(write_raster):
    data = np.array([[100.0, 200.0], [150.0, 300.0]], np.float32)
    path = write_raster("dem.tif", data)
    aligned = io.align_to_grid(path, io.read_grid(path))
    assert np.allclose(aligned, data)
    assert aligned.dtype == np.float32


def test_align_to_grid_converts_source_nodata_to_nan(write_raster):
    """Regression: an untouched -9999 would become a cliff in the slope raster."""
    data = np.array([[100.0, 200.0, -9999.0], [150.0, -9999.0, 300.0]], np.float32)
    path = write_raster("dem.tif", data, nodata=-9999.0)
    aligned = io.align_to_grid(path, io.read_grid(path))
    assert np.isnan(aligned[0, 2]) and np.isnan(aligned[1, 1])
    assert aligned[0, 0] == 100.0 and aligned[1, 2] == 300.0


def test_align_to_grid_resamples_a_coarser_source(write_raster):
    fine = write_raster("ref.tif", np.zeros((8, 8), np.float32))
    coarse = write_raster("dem.tif", np.full((4, 4), 500.0, np.float32), pixel=60.0)
    aligned = io.align_to_grid(coarse, io.read_grid(fine))
    assert aligned.shape == (8, 8)
    assert np.allclose(aligned[np.isfinite(aligned)], 500.0)


def test_align_to_grid_returns_nan_where_the_source_does_not_reach(write_raster):
    ref = write_raster("ref.tif", np.zeros((4, 4), np.float32))
    far = write_raster("far.tif", np.full((4, 4), 1.0, np.float32),
                       origin_x=900000.0)
    aligned = io.align_to_grid(far, io.read_grid(ref))
    assert np.isnan(aligned).all()


def test_write_geotiff_preserves_crs_and_transform(write_raster, tmp_path):
    grid = io.read_grid(write_raster("ref.tif", np.zeros((3, 4), np.float32)))
    out = str(tmp_path / "out.tif")
    io.write_geotiff(out, np.full((2, 3, 4), 0.5, np.float32), grid)
    with rasterio.open(out) as src:
        assert src.crs == grid.crs
        assert src.transform == grid.transform
        assert src.count == 2 and src.dtypes[0] == "float32"
        assert np.isnan(src.nodata)


def test_write_geotiff_accepts_a_two_dimensional_array(write_raster, tmp_path):
    grid = io.read_grid(write_raster("ref.tif", np.zeros((3, 4), np.float32)))
    out = str(tmp_path / "out.tif")
    io.write_geotiff(out, np.full((3, 4), 0.5, np.float32), grid)
    with rasterio.open(out) as src:
        assert src.count == 1


def test_write_geotiff_labels_bands_with_original_positions(write_raster, tmp_path):
    """A band subset must stay identifiable after writing."""
    grid = io.read_grid(write_raster("ref.tif", np.zeros((3, 4), np.float32)))
    out = str(tmp_path / "out.tif")
    io.write_geotiff(out, np.zeros((2, 3, 4), np.float32), grid,
                     band_indices=[1, 3])
    with rasterio.open(out) as src:
        assert src.descriptions == ("original_band_2", "original_band_4")


# --- placing an image on a reference grid --------------------------------

def test_grids_match_tolerates_last_decimal_noise(write_raster):
    """The same grid written by two tools can differ in the last decimal.

    Exact float equality would call these different grids and refuse the
    image, so the comparison is made with a tolerance.
    """
    a = io.read_grid(write_raster("a.tif", np.zeros((4, 4), np.float32)))
    b = io.read_grid(write_raster(
        "b.tif", np.zeros((4, 4), np.float32),
        origin_x=300000.0000001, origin_y=4199999.9999999,
    ))
    assert a.transform != b.transform      # not equal as floats
    assert io.grids_match(a, b)            # but the same grid


def test_grids_match_rejects_a_pixel_size_difference(write_raster):
    """A 0.1% pixel-size difference drifts metres away across a tile."""
    a = io.read_grid(write_raster("a.tif", np.zeros((4, 4), np.float32)))
    b = io.read_grid(write_raster("b.tif", np.zeros((4, 4), np.float32),
                                  pixel=30.03))
    assert not io.grids_match(a, b)


def test_lattice_offset_finds_the_window(write_raster):
    ref = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    sub = io.read_grid(write_raster(
        "sub.tif", np.zeros((4, 4), np.float32),
        origin_x=300000.0 + 3 * 30.0, origin_y=4200000.0 - 2 * 30.0,
    ))
    assert io.lattice_offset(sub, ref) == (2, 3)


def test_lattice_offset_rejects_a_half_pixel_shift(write_raster):
    ref = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    half = io.read_grid(write_raster("half.tif", np.zeros((4, 4), np.float32),
                                     origin_x=300000.0 + 15.0))
    assert io.lattice_offset(half, ref) is None


def test_read_image_on_grid_is_lossless_for_a_window(write_raster):
    """A different window on the same lattice is sliced, not resampled, so
    every value must survive the placement bit for bit."""
    ref = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4) / 16.0
    sub = write_raster("sub.tif", values,
                       origin_x=300000.0 + 3 * 30.0,
                       origin_y=4200000.0 - 2 * 30.0)

    placed = io.read_image_on_grid(sub, ref)

    assert placed.shape == (1, 10, 10)
    assert np.array_equal(placed[0, 2:6, 3:7], values[0])
    outside = placed[0].copy()
    outside[2:6, 3:7] = np.nan
    assert np.isnan(outside).all()


def test_read_image_on_grid_clips_a_window_that_hangs_over_the_edge(write_raster):
    ref = io.read_grid(write_raster("ref.tif", np.zeros((6, 6), np.float32)))
    sub = write_raster("sub.tif", np.full((1, 4, 4), 0.5, np.float32),
                       origin_x=300000.0 + 4 * 30.0,
                       origin_y=4200000.0 - 4 * 30.0)

    placed = io.read_image_on_grid(sub, ref)

    assert np.allclose(placed[0, 4:6, 4:6], 0.5)
    assert np.isnan(placed[0, :4, :]).all()


def test_read_image_on_grid_refuses_another_lattice(write_raster):
    """A half-pixel shift cannot be sliced. The reflectance is not resampled to
    fit: correcting interpolated values, or producing a mostly-empty product
    from a tile that merely overlaps, is worse than refusing the image."""
    import pytest
    ref = io.read_grid(write_raster("ref.tif", np.zeros((6, 6), np.float32)))
    shifted = write_raster("shifted.tif", np.full((1, 6, 6), 0.4, np.float32),
                           origin_x=300000.0 + 15.0)

    with pytest.raises(ValueError, match="whole number of pixels"):
        io.read_image_on_grid(shifted, ref)


def test_lattice_error_names_what_differs(write_raster):
    """The message has to say why, or the user cannot fix the input."""
    import pytest
    ref = io.read_grid(write_raster("ref.tif", np.zeros((6, 6), np.float32)))
    coarse = write_raster("coarse.tif", np.full((1, 3, 3), 0.4, np.float32),
                          pixel=60.0)

    with pytest.raises(ValueError, match="pixel size 60 m vs 30 m"):
        io.read_image_on_grid(coarse, ref)


def test_read_image_on_grid_does_not_warn_for_a_plain_window(write_raster,
                                                             recwarn):
    ref = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    sub = write_raster("sub.tif", np.full((1, 4, 4), 0.5, np.float32),
                       origin_x=300000.0 + 3 * 30.0,
                       origin_y=4200000.0 - 2 * 30.0)
    io.read_image_on_grid(sub, ref)
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_read_image_on_grid_refuses_a_window_outside_the_grid(write_raster):
    """A tile on the same lattice but elsewhere would contribute no pixel.

    Refused here rather than left to the sampling step: that route reports a
    foreign image as a scene short of forest, and only works while the sample
    is built at all -- which methods needing no regression may not require.
    """
    import pytest
    ref = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    far = write_raster("far.tif", np.full((1, 10, 10), 0.5, np.float32),
                       origin_x=300000.0 + 500 * 30.0)

    with pytest.raises(ValueError, match="entirely outside"):
        io.read_image_on_grid(far, ref)


def test_read_image_on_grid_keeps_a_one_pixel_overlap(write_raster):
    """One overlapping pixel is data, so the image is placed, not refused."""
    ref = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    corner = write_raster("corner.tif", np.full((1, 4, 4), 0.5, np.float32),
                          origin_x=300000.0 + 9 * 30.0,
                          origin_y=4200000.0 - 9 * 30.0)

    placed = io.read_image_on_grid(corner, ref)
    assert placed[0, 9, 9] == np.float32(0.5)
    assert int(np.isfinite(placed).sum()) == 1


def test_write_geotiff_refuses_an_array_that_does_not_fit_the_grid(write_raster,
                                                                   tmp_path):
    """The grid supplies the CRS and transform, so an array of another size
    would be written as if it started at the grid's origin -- a file that opens
    cleanly and sits in the wrong place."""
    import pytest
    grid = io.read_grid(write_raster("ref.tif", np.zeros((10, 10), np.float32)))
    with pytest.raises(ValueError, match="wrong place"):
        io.write_geotiff(str(tmp_path / "out.tif"),
                         np.zeros((5, 5), np.float32), grid)


def test_write_geotiff_requires_one_band_index_per_band(write_raster, tmp_path):
    """Otherwise the labelling goes out of step and a band loses its identity."""
    import pytest
    grid = io.read_grid(write_raster("ref.tif", np.zeros((3, 4), np.float32)))
    with pytest.raises(ValueError, match="band_indices has 1 entries"):
        io.write_geotiff(str(tmp_path / "out.tif"),
                         np.zeros((2, 3, 4), np.float32), grid,
                         band_indices=[4])
