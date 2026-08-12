"""The regression-sample mask: conditions, aspect balancing, and the skip rule."""

import numpy as np
import pytest

from fortocorrpy import Config
from fortocorrpy.masking import _quadrant_labels, build_sample_mask, forest_mask

EDGES_RAD = tuple(np.float32(np.deg2rad(e)) for e in (45.0, 135.0, 225.0, 315.0))


@pytest.fixture
def uniform_scene():
    """40x40 with aspects spread evenly over all four quadrants.

    Slope, illumination, forest, and validity all pass everywhere, so any test
    that turns one condition off isolates that condition.
    """
    n = 40
    beta = np.deg2rad(
        np.linspace(0.0, 359.9, n * n, dtype=np.float32)
    ).reshape(n, n).astype(np.float32)
    return {
        "n": n,
        "cos_i": np.full((n, n), 0.5, np.float32),
        "alpha": np.full((n, n), np.deg2rad(20.0), np.float32),
        "beta": beta,
        "valid": np.ones((n, n), bool),
        "forest": np.full((n, n), 0.8, np.float32),
    }


def cfg(**kw):
    kw.setdefault("mask_source", "ndvi")
    kw.setdefault("min_samples_per_quadrant", 100)
    return Config(**kw)


# --- forest_mask --------------------------------------------------------

def test_ndvi_threshold_is_inclusive():
    ndvi = np.array([0.3, 0.5, 0.7], np.float32)
    mask = forest_mask(ndvi, cfg(ndvi_threshold=0.5))
    assert np.array_equal(mask, [False, True, True])


def test_ndvi_nodata_is_excluded():
    """align_to_grid returns NaN outside the source; NaN >= t is False."""
    ndvi = np.array([0.8, np.nan, 0.9], np.float32)
    mask = forest_mask(ndvi, cfg(ndvi_threshold=0.5))
    assert np.array_equal(mask, [True, False, True])


def test_landcover_class_selected_even_when_stored_as_float():
    """Regression: align_to_grid returns float32, so class 5 arrives as 5.0."""
    lc = np.array([[1.0, 5.0, 7.0], [5.0, 2.0, 5.0]], np.float32)
    mask = forest_mask(lc, cfg(mask_source="landcover", forest_class=5))
    assert np.array_equal(mask, lc == 5)


# --- quadrant labelling -------------------------------------------------

@pytest.mark.parametrize(
    "aspect_deg, expected", [(0.0, 0), (90.0, 1), (180.0, 2), (270.0, 3),
                             (350.0, 0), (44.9, 0), (45.0, 1)],
)
def test_quadrant_labels_cardinal(aspect_deg, expected):
    labels = _quadrant_labels(
        np.array([np.deg2rad(aspect_deg)], np.float32), EDGES_RAD,
    )
    assert labels[0] == expected


def test_undefined_aspect_is_labelled_minus_one_not_garbage():
    """Regression: the label array must be initialised, not np.empty.

    A NaN aspect matches none of the four bins, so an uninitialised buffer
    left arbitrary memory as the quadrant label.
    """
    labels = _quadrant_labels(np.full(64, np.nan, np.float32), EDGES_RAD)
    assert np.all(labels == -1)


# --- build_sample_mask --------------------------------------------------

def test_draw_is_balanced_across_quadrants(uniform_scene):
    s = uniform_scene
    result = build_sample_mask(
        s["cos_i"], s["alpha"], s["beta"], s["valid"], s["forest"], cfg(),
    )
    assert not result.skipped
    labels = _quadrant_labels(s["beta"], EDGES_RAD)
    drawn = labels[result.mask]
    counts = [int((drawn == q).sum()) for q in range(4)]
    assert counts == [result.n_per_quadrant] * 4


def test_scene_is_skipped_when_a_quadrant_is_short(uniform_scene):
    s = uniform_scene
    valid = s["valid"].copy()
    labels = _quadrant_labels(s["beta"], EDGES_RAD)
    valid[labels == 0] = False  # wipe out the north quadrant
    result = build_sample_mask(
        s["cos_i"], s["alpha"], s["beta"], valid, s["forest"], cfg(),
    )
    assert result.skipped
    assert result.quadrant_counts["N"] == 0
    assert result.n_per_quadrant == 0
    assert not result.mask.any()


def test_validity_mask_restricts_the_sample(uniform_scene):
    """The per-image nodata mask, not just the shared forest source, decides."""
    s = uniform_scene
    full = build_sample_mask(
        s["cos_i"], s["alpha"], s["beta"], s["valid"], s["forest"], cfg(),
    )
    valid = s["valid"].copy()
    valid[:, : s["n"] // 2] = False  # half the scene is nodata in this image
    partial = build_sample_mask(
        s["cos_i"], s["alpha"], s["beta"], valid, s["forest"], cfg(),
    )
    assert partial.n_per_quadrant < full.n_per_quadrant
    assert not partial.mask[:, : s["n"] // 2].any()


def test_slope_threshold_is_applied(uniform_scene):
    s = uniform_scene
    alpha = np.full_like(s["alpha"], np.deg2rad(2.0))  # below the 5 deg default
    result = build_sample_mask(
        s["cos_i"], alpha, s["beta"], s["valid"], s["forest"], cfg(),
    )
    assert result.skipped
    assert all(v == 0 for v in result.quadrant_counts.values())


def test_self_shadow_is_excluded(uniform_scene):
    s = uniform_scene
    cos_i = np.full_like(s["cos_i"], -0.2)
    result = build_sample_mask(
        cos_i, s["alpha"], s["beta"], s["valid"], s["forest"], cfg(),
    )
    assert result.skipped


def test_non_forest_pixels_are_excluded(uniform_scene):
    s = uniform_scene
    forest = np.full_like(s["forest"], 0.1)  # below the NDVI threshold
    result = build_sample_mask(
        s["cos_i"], s["alpha"], s["beta"], s["valid"], forest, cfg(),
    )
    assert result.skipped


def test_equal_quadrants_are_taken_whole(uniform_scene):
    """When every quadrant is the same size the draw keeps every candidate."""
    s = uniform_scene
    result = build_sample_mask(
        s["cos_i"], s["alpha"], s["beta"], s["valid"], s["forest"], cfg(),
    )
    assert result.n_per_quadrant == min(result.quadrant_counts.values())
    assert int(result.mask.sum()) == sum(result.quadrant_counts.values())


@pytest.fixture
def unbalanced_scene(uniform_scene):
    """One quadrant deliberately smaller, forcing a strict random subsample.

    With equal quadrants the balanced draw takes every candidate, so the seed
    cannot change the outcome; testing reproducibility needs a real subsample.
    """
    s = dict(uniform_scene)
    valid = s["valid"].copy()
    north = np.flatnonzero(_quadrant_labels(s["beta"], EDGES_RAD).ravel() == 0)
    valid.flat[north[:-150]] = False  # leave 150 candidates in the N quadrant
    s["valid"] = valid
    return s


def test_same_seed_draws_the_same_sample(unbalanced_scene):
    s = unbalanced_scene
    args = (s["cos_i"], s["alpha"], s["beta"], s["valid"], s["forest"])
    a = build_sample_mask(*args, cfg(sample_seed=7))
    b = build_sample_mask(*args, cfg(sample_seed=7))
    assert a.n_per_quadrant == 150
    assert np.array_equal(a.mask, b.mask)


def test_different_seeds_draw_different_samples(unbalanced_scene):
    s = unbalanced_scene
    args = (s["cos_i"], s["alpha"], s["beta"], s["valid"], s["forest"])
    a = build_sample_mask(*args, cfg(sample_seed=7))
    b = build_sample_mask(*args, cfg(sample_seed=8))
    assert not np.array_equal(a.mask, b.mask)
    assert a.mask.sum() == b.mask.sum()
