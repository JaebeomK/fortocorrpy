"""End-to-end assembly: single image, batches on one grid, and failure modes."""

from datetime import datetime, timezone

import numpy as np
import pytest
import rasterio

from fortocorrpy import Config, CorrectionResult, correct_image, io

WHEN = datetime(2024, 8, 14, 2, 27, 9, tzinfo=timezone.utc)


def cfg(**kw):
    kw.setdefault("mask_source", "ndvi")
    kw.setdefault("methods", ("cosine", "scsc", "se", "er"))
    return Config(**kw)


# --- single image -------------------------------------------------------

def test_single_image_returns_one_result(scene):
    result = correct_image(scene["image"], WHEN, scene["dem"], scene["ndvi"], cfg())
    assert isinstance(result, CorrectionResult)
    assert not result.mask_result.skipped
    assert set(result.corrected) == {"cosine", "scsc", "se", "er"}
    assert result.corrected["scsc"].shape == (4, scene["n"], scene["n"])
    assert len(result.coefficients) == 4
    assert result.error is None


def test_band_subset_preserves_original_positions(scene):
    result = correct_image(
        scene["image"], WHEN, scene["dem"], scene["ndvi"],
        cfg(methods=("scsc",), band_indices=[1, 3]),
    )
    assert result.band_indices == [1, 3]
    assert result.corrected["scsc"].shape[0] == 2
    assert len(result.coefficients) == 2


def test_evaluate_uses_the_regression_sample(scene):
    result = correct_image(
        scene["image"], WHEN, scene["dem"], scene["ndvi"], cfg(), evaluate=True,
    )
    assert result.metrics_before is not None
    assert set(result.metrics_after) == set(result.corrected)
    assert result.metrics_before[0].n_samples == result.coefficients[0].n_samples


def test_skipped_scene_reports_diagnostics_and_no_output(scene, write_raster):
    """Not enough forest: a normal outcome, not an error."""
    bare = write_raster("bare.tif", np.full((scene["n"], scene["n"]), 0.1, np.float32))
    result = correct_image(scene["image"], WHEN, scene["dem"], bare, cfg())
    assert result.mask_result.skipped
    assert result.corrected is None
    assert result.error is None
    assert set(result.mask_result.quadrant_counts) == {"N", "E", "S", "W"}


# --- batches on one grid ------------------------------------------------

def test_batch_returns_one_result_per_image(scene, write_raster):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3 + 0.01 * i, np.float32))
        for i in range(3)
    ]
    results = correct_image(images, [WHEN] * 3, scene["dem"], scene["ndvi"], cfg())
    assert isinstance(results, list) and len(results) == 3
    assert all(r.error is None for r in results)


def test_single_element_list_still_returns_a_list(scene):
    results = correct_image(
        [scene["image"]], [WHEN], scene["dem"], scene["ndvi"], cfg(),
    )
    assert isinstance(results, list) and len(results) == 1


def test_terrain_and_forest_are_aligned_once_for_the_whole_batch(
    scene, write_raster, monkeypatch,
):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(4)
    ]
    calls = []
    original = io.align_to_grid

    def counting(path, grid, **kw):
        calls.append(path)
        return original(path, grid, **kw)

    monkeypatch.setattr(io, "align_to_grid", counting)
    correct_image(images, [WHEN] * 4, scene["dem"], scene["ndvi"],
                  cfg(methods=("scsc",)))
    assert calls == [scene["dem"], scene["ndvi"]]


def test_per_image_validity_drives_the_sample(scene, write_raster):
    """A shared forest source says *where forest is*; each image's own nodata
    says *where that image can see it*. Two images over the same forest must
    therefore get different samples when their QC masks differ."""
    n = scene["n"]
    nodata = -9999.0
    clear = np.full((4, n, n), 0.3, np.float32)
    clouded = clear.copy()
    clouded[:, :, 3 * n // 4:] = nodata  # eastern quarter removed by QC

    images = [
        write_raster("clear.tif", clear, nodata=nodata),
        write_raster("clouded.tif", clouded, nodata=nodata),
    ]
    a, b = correct_image(images, [WHEN] * 2, scene["dem"], scene["ndvi"],
                         cfg(methods=("scsc",)))
    assert not a.mask_result.skipped and not b.mask_result.skipped
    assert b.mask_result.n_per_quadrant < a.mask_result.n_per_quadrant
    assert not b.mask_result.mask[:, 3 * n // 4:].any()


def test_workers_do_not_change_the_result(scene, write_raster):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3 + 0.02 * i, np.float32))
        for i in range(3)
    ]
    serial = correct_image(images, [WHEN] * 3, scene["dem"], scene["ndvi"],
                           cfg(methods=("scsc",), n_workers=1))
    threaded = correct_image(images, [WHEN] * 3, scene["dem"], scene["ndvi"],
                             cfg(methods=("scsc",), n_workers=3))
    for s, t in zip(serial, threaded):
        assert np.allclose(s.corrected["scsc"], t.corrected["scsc"],
                           equal_nan=True)


# --- one grid per batch -------------------------------------------------

def test_a_smaller_window_on_the_same_lattice_is_corrected(scene, write_raster):
    """Images downloaded against a base image often cover different windows of
    it. They sit on one lattice, so they are placed on the reference grid by
    slicing and corrected like any other image."""
    n = scene["n"]
    sub = write_raster(
        "sub.tif", np.full((4, 80, 80), 0.3, np.float32),
        origin_x=300000.0 + 20 * 30.0, origin_y=4200000.0 - 20 * 30.0,
    )
    full, part = correct_image([scene["image"], sub], [WHEN] * 2,
                               scene["dem"], scene["ndvi"],
                               cfg(methods=("scsc",)))

    assert full.error is None and part.error is None
    assert part.corrected["scsc"].shape == (4, n, n)   # on the reference grid
    inside = part.corrected["scsc"][0, 20:100, 20:100]
    assert np.isfinite(inside).any()                   # the window was filled
    assert np.isnan(part.corrected["scsc"][0, :20, :]).all()   # rest untouched


def test_the_uncovered_area_stays_out_of_the_sample(scene, write_raster):
    """Pixels the image does not cover are nan, which is what invalid means
    here, so they must not enter the regression sample."""
    sub = write_raster(
        "sub.tif", np.full((4, 80, 80), 0.3, np.float32),
        origin_x=300000.0 + 20 * 30.0, origin_y=4200000.0 - 20 * 30.0,
    )
    result, = correct_image([sub], [WHEN], scene["dem"], scene["ndvi"],
                            cfg(methods=("scsc",)), reference_path=scene["image"])
    assert not result.mask_result.mask[:20, :].any()
    assert not result.mask_result.mask[:, :20].any()


def test_reference_path_fixes_the_output_grid(scene, write_raster):
    """With a base image as the reference, every result comes back on that
    grid, so a batch of differently sized windows stacks directly."""
    n = scene["n"]
    subs = [
        write_raster(f"sub{i}.tif", np.full((4, 80 - 10 * i, 80 - 10 * i),
                                            0.3, np.float32),
                     origin_x=300000.0 + 20 * 30.0,
                     origin_y=4200000.0 - 20 * 30.0)
        for i in range(2)
    ]
    results = correct_image(subs, [WHEN] * 2, scene["dem"], scene["ndvi"],
                            cfg(methods=("scsc",)),
                            reference_path=scene["image"])
    assert {r.grid.shape for r in results} == {(n, n)}
    assert all(r.corrected["scsc"].shape == (4, n, n) for r in results)


def test_last_decimal_noise_in_the_transform_is_not_a_mismatch(scene, write_raster):
    """Regression: the grid comparison used exact float equality, so an image
    whose transform differed in the last decimal -- the same grid, written by a
    different tool -- was refused."""
    n = scene["n"]
    noisy = write_raster(
        "noisy.tif", np.full((4, n, n), 0.3, np.float32),
        origin_x=300000.0000001, origin_y=4199999.9999999,
    )
    ok, also_ok = correct_image([scene["image"], noisy], [WHEN] * 2,
                                scene["dem"], scene["ndvi"],
                                cfg(methods=("scsc",)))
    assert ok.error is None and also_ok.error is None
    assert np.allclose(ok.corrected["scsc"], also_ok.corrected["scsc"],
                       equal_nan=True)


def test_dem_path_list_is_rejected(scene):
    """Regression: a per-image DEM implied per-image grids, and the single
    forest raster was then aligned to the first image only -- masking every
    later image with the wrong window."""
    with pytest.raises(TypeError, match="dem_path must be a single path"):
        correct_image([scene["image"]], [WHEN], [scene["dem"]],
                      scene["ndvi"], cfg())


def test_forest_path_list_length_must_match(scene, write_raster):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="forest_path .* must match image_path"):
        correct_image(images, [WHEN] * 2, scene["dem"],
                      [scene["ndvi"]] * 3, cfg())


def test_forest_source_can_follow_the_season(scene, write_raster):
    """A per-image forest source is safe here: the reference grid is fixed for
    the whole batch, so every forest raster is aligned onto the same grid. The
    seasonal case is the reason to allow it."""
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(2)
    ]
    # Spring: the eastern quarter has not leafed out yet. Summer: all forest.
    spring = np.full((n, n), 0.8, np.float32)
    spring[:, 3 * n // 4:] = 0.1
    forests = [
        write_raster("ndvi_spring.tif", spring),
        write_raster("ndvi_summer.tif", np.full((n, n), 0.8, np.float32)),
    ]

    a, b = correct_image(images, [WHEN] * 2, scene["dem"], forests,
                         cfg(methods=("scsc",)))
    assert not a.mask_result.skipped and not b.mask_result.skipped
    assert not a.mask_result.mask[:, 3 * n // 4:].any()  # spring: no sample there
    assert b.mask_result.mask[:, 3 * n // 4:].any()      # summer: sampled
    assert a.mask_result.n_per_quadrant < b.mask_result.n_per_quadrant


def test_listed_forest_sources_are_aligned_once_per_image(
    scene, write_raster, monkeypatch,
):
    """Not eagerly up front: a long list would hold one aligned raster each."""
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(3)
    ]
    forests = [
        write_raster(f"ndvi{i}.tif", np.full((n, n), 0.8, np.float32))
        for i in range(3)
    ]
    calls = []
    original = io.align_to_grid

    def counting(path, grid, **kw):
        calls.append(path)
        return original(path, grid, **kw)

    monkeypatch.setattr(io, "align_to_grid", counting)
    correct_image(images, [WHEN] * 3, scene["dem"], forests,
                  cfg(methods=("scsc",)))
    assert calls == [scene["dem"], *forests]


def test_forest_source_that_misses_the_scene_skips_that_image(scene, write_raster):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(3)
    ]
    good = write_raster("good.tif", np.full((n, n), 0.8, np.float32))
    far = write_raster("far.tif", np.full((20, 20), 0.8, np.float32),
                       origin_x=900000.0)

    results = correct_image(images, [WHEN] * 3, scene["dem"],
                            [good, far, good], cfg(methods=("scsc",)))
    assert not results[0].mask_result.skipped
    assert not results[2].mask_result.skipped
    # A forest raster that misses the scene aligns to all-NaN, so nothing
    # passes the forest test and the image is skipped with empty quadrants.
    assert results[1].mask_result.skipped
    assert all(v == 0 for v in results[1].mask_result.quadrant_counts.values())


def test_unprojected_crs_is_rejected(write_raster):
    image = write_raster("ll.tif", np.full((4, 10, 10), 0.3, np.float32),
                         crs="EPSG:4326", origin_x=127.0, origin_y=38.0,
                         pixel=0.0003)
    dem = write_raster("lldem.tif", np.full((10, 10), 100.0, np.float32),
                       crs="EPSG:4326", origin_x=127.0, origin_y=38.0,
                       pixel=0.0003)
    with pytest.raises(ValueError, match="projected"):
        correct_image(image, WHEN, dem, dem, cfg())


# --- argument validation ------------------------------------------------

def test_length_mismatch_is_rejected(scene):
    with pytest.raises(ValueError, match="same length"):
        correct_image([scene["image"]] * 2, [WHEN] * 3,
                      scene["dem"], scene["ndvi"], cfg())


# --- per-image failure --------------------------------------------------

def test_one_bad_image_does_not_destroy_the_batch(scene, write_raster, monkeypatch):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(3)
    ]
    original = io.read_image_on_grid

    def flaky(path, *a, **kw):
        if path == images[1]:
            raise RuntimeError("corrupt tile")
        return original(path, *a, **kw)

    monkeypatch.setattr(io, "read_image_on_grid", flaky)
    results = correct_image(images, [WHEN] * 3, scene["dem"], scene["ndvi"],
                            cfg(methods=("scsc",)))

    assert results[0].error is None and results[2].error is None
    assert results[1].error is not None
    assert "corrupt tile" in results[1].error
    assert results[1].corrected is None


def test_single_image_failure_propagates(scene, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("corrupt tile")

    monkeypatch.setattr(io, "read_image_on_grid", boom)
    with pytest.raises(RuntimeError, match="corrupt tile"):
        correct_image(scene["image"], WHEN, scene["dem"], scene["ndvi"], cfg())


# --- stream: hand results back one at a time -----------------------------

@pytest.fixture
def series(scene, write_raster):
    n = scene["n"]
    return [
        write_raster(f"t{i:02d}.tif", np.full((4, n, n), 0.30 + 0.005 * i, np.float32))
        for i in range(10)
    ]


def test_stream_yields_the_same_results_as_the_list(scene, write_raster):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3 + 0.02 * i, np.float32))
        for i in range(3)
    ]
    args = (images, [WHEN] * 3, scene["dem"], scene["ndvi"], cfg(methods=("scsc",)))
    eager = correct_image(*args)
    streamed = list(correct_image(*[*args[:4], cfg(methods=("scsc",), stream=True)]))

    assert len(streamed) == len(eager)
    for a, b in zip(eager, streamed):
        assert np.allclose(a.corrected["scsc"], b.corrected["scsc"], equal_nan=True)
        assert a.mask_result.n_per_quadrant == b.mask_result.n_per_quadrant


def test_stream_is_ignored_for_a_single_image(scene):
    result = correct_image(scene["image"], WHEN, scene["dem"], scene["ndvi"],
                           cfg(methods=("scsc",), stream=True))
    assert isinstance(result, CorrectionResult)
    assert result.corrected is not None


def test_stream_computes_lazily(scene, series, monkeypatch):
    """Nothing is corrected until a result is asked for."""
    started = []
    original = io.read_image_on_grid

    def counting(path, *a, **kw):
        started.append(path)
        return original(path, *a, **kw)

    monkeypatch.setattr(io, "read_image_on_grid", counting)
    it = correct_image(series, [WHEN] * 10, scene["dem"], scene["ndvi"],
                       cfg(methods=("scsc",), n_workers=1, stream=True))
    assert started == []
    next(it)
    assert len(started) == 1
    it.close()


def test_stream_does_not_run_far_ahead_of_the_consumer(scene, series, monkeypatch):
    """The whole point: memory must not grow with the length of the list.
    ThreadPoolExecutor.map would queue all ten and process them regardless."""
    import time

    started = []
    original = io.read_image_on_grid

    def counting(path, *a, **kw):
        started.append(path)
        return original(path, *a, **kw)

    monkeypatch.setattr(io, "read_image_on_grid", counting)
    it = correct_image(series, [WHEN] * 10, scene["dem"], scene["ndvi"],
                       cfg(methods=("scsc",), n_workers=2, stream=True))
    next(it)
    time.sleep(0.5)  # ample time to run away if the bound were missing
    assert len(started) <= 4  # 1 consumed + at most n_workers + 1 in flight
    it.close()

    started.clear()
    correct_image(series, [WHEN] * 10, scene["dem"], scene["ndvi"],
                  cfg(methods=("scsc",), n_workers=2))
    assert len(started) == 10  # the eager form processes everything


def test_stream_preserves_input_order_under_parallelism(scene, series):
    eager = correct_image(series, [WHEN] * 10, scene["dem"], scene["ndvi"],
                          cfg(methods=("scsc",), n_workers=1))
    streamed = list(correct_image(series, [WHEN] * 10, scene["dem"], scene["ndvi"],
                                  cfg(methods=("scsc",), n_workers=3, stream=True)))
    for a, b in zip(eager, streamed):
        assert np.allclose(a.corrected["scsc"], b.corrected["scsc"], equal_nan=True)


def test_stream_records_per_image_failures(scene, write_raster, monkeypatch):
    n = scene["n"]
    images = [
        write_raster(f"s{i}.tif", np.full((4, n, n), 0.3, np.float32))
        for i in range(3)
    ]
    original = io.read_image_on_grid

    def flaky(path, *a, **kw):
        if path == images[1]:
            raise RuntimeError("corrupt tile")
        return original(path, *a, **kw)

    monkeypatch.setattr(io, "read_image_on_grid", flaky)
    results = list(correct_image(images, [WHEN] * 3, scene["dem"], scene["ndvi"],
                                 cfg(methods=("scsc",), stream=True)))
    assert [r.error is None for r in results] == [True, False, True]
    assert "corrupt tile" in results[1].error


def test_accumulating_a_mosaic_needs_only_three_arrays(scene, series):
    """A pixel-wise mean does not need the list: streaming gives the same
    mosaic from an accumulator, which is why stream costs nothing here."""
    args = (series, [WHEN] * 10, scene["dem"], scene["ndvi"], cfg(methods=("scsc",)))

    eager = correct_image(*args)
    stack = np.stack([r.corrected["scsc"] for r in eager])
    reference = np.nanmean(stack, axis=0)

    acc = np.zeros((4, scene["n"], scene["n"]), np.float64)
    cnt = np.zeros_like(acc)
    for corr in correct_image(*[*args[:4], cfg(methods=("scsc",), stream=True)]):
        if corr.corrected is None:
            continue
        arr = corr.corrected["scsc"]
        finite = np.isfinite(arr)
        acc += np.where(finite, arr, 0.0)
        cnt += finite
    accumulated = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)

    assert np.allclose(accumulated, reference, equal_nan=True)


def test_result_carries_the_grid_it_ran_on(scene):
    """Writing the output needs nothing but the result itself."""
    result = correct_image(scene["image"], WHEN, scene["dem"], scene["ndvi"],
                           cfg(methods=("scsc",)))
    reference = io.read_grid(scene["image"])
    assert result.grid.shape == reference.shape
    assert result.grid.transform == reference.transform
    assert result.grid.crs == reference.crs


def test_grid_is_set_on_skipped_and_failed_results(scene, write_raster, monkeypatch):
    bare = write_raster("bare.tif", np.full((scene["n"], scene["n"]), 0.1, np.float32))
    skipped = correct_image(scene["image"], WHEN, scene["dem"], bare,
                            cfg(methods=("scsc",)))
    assert skipped.mask_result.skipped and skipped.grid is not None

    original = io.read_image_on_grid
    monkeypatch.setattr(io, "read_image_on_grid",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    failed, = correct_image([scene["image"]], [WHEN], scene["dem"], scene["ndvi"],
                            cfg(methods=("scsc",)))
    monkeypatch.setattr(io, "read_image_on_grid", original)
    assert failed.error is not None and failed.grid is not None


def test_writing_output_from_the_result_alone(scene, tmp_path):
    """The usage.md pattern: array + grid + band_indices, all off the result."""
    result = correct_image(scene["image"], WHEN, scene["dem"], scene["ndvi"],
                           cfg(methods=("scsc",), band_indices=[1, 3]))
    out = str(tmp_path / "scene_scsc.tif")
    io.write_geotiff(out, result.corrected["scsc"], result.grid,
                     band_indices=result.band_indices)
    with rasterio.open(out) as src:
        assert src.crs == result.grid.crs
        assert src.transform == result.grid.transform
        assert src.descriptions == ("original_band_2", "original_band_4")


def test_a_different_lattice_is_an_error_not_a_product(scene, write_raster):
    """A tile that merely overlaps the reference grid must not come back as a
    mostly-empty but otherwise ordinary product: it is refused, so nothing is
    written for it, and the rest of the batch still runs."""
    coarse = write_raster("coarse.tif", np.full((4, 60, 60), 0.3, np.float32),
                          pixel=60.0)
    ok, bad = correct_image([scene["image"], coarse], [WHEN] * 2,
                            scene["dem"], scene["ndvi"], cfg(methods=("scsc",)))
    assert ok.error is None and ok.corrected is not None
    assert bad.error is not None and "pixel lattice" in bad.error
    assert bad.corrected is None            # nothing to write out
