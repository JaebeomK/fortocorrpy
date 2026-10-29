"""Reflectance-level evaluation metrics."""

import numpy as np

from fortocorrpy import evaluation


def test_perfect_topographic_effect_gives_correlation_one():
    cos_i = np.linspace(0.1, 0.9, 100).astype(np.float32)
    refl = (0.5 * cos_i + 0.05).astype(np.float32)
    m = evaluation.band_metrics(cos_i, refl, np.ones(100, bool))
    assert np.isclose(m.correlation, 1.0, atol=1e-6)
    assert np.isclose(m.slope, 0.5, atol=1e-5)
    assert m.n_samples == 100


def test_correlation_and_slope_fall_to_zero_after_a_perfect_correction():
    cos_i = np.linspace(0.1, 0.9, 100).astype(np.float32)
    refl = (0.5 * cos_i + 0.05).astype(np.float32)
    corrected = np.full(100, refl.mean(), np.float32)
    m = evaluation.band_metrics(cos_i, corrected, np.ones(100, bool))
    assert np.isnan(m.correlation)  # zero variance in y
    assert np.isclose(m.slope, 0.0, atol=1e-9)


def test_std_drops_when_the_illumination_trend_is_removed():
    cos_i = np.linspace(0.1, 0.9, 200).astype(np.float32)
    refl = (0.5 * cos_i + 0.05).astype(np.float32)
    corrected = (refl - 0.5 * (cos_i - 0.5)).astype(np.float32)  # er, a = 0.5
    before = evaluation.band_metrics(cos_i, refl, np.ones(200, bool))
    after = evaluation.band_metrics(cos_i, corrected, np.ones(200, bool))
    assert after.std < before.std
    assert abs(after.mean - before.mean) < 1e-3  # brightness preserved


def test_nan_pixels_are_dropped_not_propagated():
    """Denominator methods leave NaN in self-shadow; metrics must ignore them."""
    cos_i = np.array([0.2, 0.4, 0.6, 0.8], np.float32)
    refl = np.array([0.1, np.nan, 0.3, 0.4], np.float32)
    m = evaluation.band_metrics(cos_i, refl, np.ones(4, bool))
    assert m.n_samples == 3
    assert np.isfinite(m.mean)


def test_fewer_than_two_samples_gives_nan_metrics():
    cos_i = np.array([0.5, 0.6], np.float32)
    refl = np.array([0.3, np.nan], np.float32)
    m = evaluation.band_metrics(cos_i, refl, np.ones(2, bool))
    assert m.n_samples == 1
    assert np.isnan(m.correlation) and np.isnan(m.slope)
    assert np.isnan(m.mean) and np.isnan(m.std)


def test_constant_cos_i_gives_nan_correlation_and_slope():
    cos_i = np.full(10, 0.5, np.float32)
    refl = np.linspace(0.1, 0.5, 10).astype(np.float32)
    m = evaluation.band_metrics(cos_i, refl, np.ones(10, bool))
    assert np.isnan(m.correlation) and np.isnan(m.slope)
    assert np.isfinite(m.mean) and np.isfinite(m.std)


def test_mask_restricts_the_evaluation_sample():
    cos_i = np.linspace(0.1, 0.9, 100).astype(np.float32)
    refl = (0.5 * cos_i + 0.05).astype(np.float32)
    mask = np.zeros(100, bool)
    mask[:40] = True
    assert evaluation.band_metrics(cos_i, refl, mask).n_samples == 40


def test_evaluate_before_and_after_return_one_entry_per_band():
    cos_i = np.linspace(0.1, 0.9, 36).astype(np.float32).reshape(6, 6)
    stack = np.stack([cos_i * 0.5, cos_i * 0.3, cos_i * 0.2]).astype(np.float32)
    mask = np.ones((6, 6), bool)
    assert len(evaluation.evaluate_before(cos_i, stack, mask)) == 3
    assert len(evaluation.evaluate_after(cos_i, stack, mask)) == 3


def test_single_band_raster_is_accepted():
    cos_i = np.linspace(0.1, 0.9, 36).astype(np.float32).reshape(6, 6)
    assert len(evaluation.evaluate_before(cos_i, cos_i, np.ones((6, 6), bool))) == 1
