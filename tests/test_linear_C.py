"""Per-band regression of reflectance on cos i, and the C parameter."""

import numpy as np

from fortocorrpy.linear_C import fit_band, fit_bands


def test_exact_recovery_of_a_noiseless_line():
    cos_i = np.array([0.2, 0.4, 0.6, 0.8], np.float32)
    refl = (2.0 * cos_i + 1.0).astype(np.float32)  # a = 2, b = 1
    coeff = fit_band(cos_i, refl, np.ones(4, bool))
    assert np.isclose(coeff.slope, 2.0, atol=1e-5)
    assert np.isclose(coeff.intercept, 1.0, atol=1e-5)
    assert np.isclose(coeff.c, 0.5, atol=1e-5)  # C = b / a
    assert coeff.n_samples == 4


def test_c_is_intercept_over_slope():
    cos_i = np.array([0.1, 0.5, 0.9], np.float32)
    refl = (0.4 * cos_i + 0.08).astype(np.float32)
    coeff = fit_band(cos_i, refl, np.ones(3, bool))
    assert np.isclose(coeff.c, coeff.intercept / coeff.slope, atol=1e-6)


def test_sample_mean_reflectance_is_reported():
    """`se` re-centres on this value, so it must come from the sample."""
    cos_i = np.array([0.2, 0.4, 0.6, 0.8], np.float32)
    refl = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    coeff = fit_band(cos_i, refl, np.ones(4, bool))
    assert np.isclose(coeff.mean_reflectance, 0.25, atol=1e-6)


def test_mask_restricts_the_fit():
    cos_i = np.array([0.2, 0.4, 0.6, 0.8, 0.9], np.float32)
    refl = np.array([1.4, 1.8, 2.2, 2.6, 99.0], np.float32)  # last is an outlier
    mask = np.array([True, True, True, True, False])
    coeff = fit_band(cos_i, refl, mask)
    assert coeff.n_samples == 4
    assert np.isclose(coeff.slope, 2.0, atol=1e-5)


def test_constant_cos_i_gives_zero_slope_and_nan_c():
    """With no illumination range the slope is undefined; C must not blow up."""
    cos_i = np.full(5, 0.5, np.float32)
    refl = np.array([0.1, 0.2, 0.3, 0.4, 0.5], np.float32)
    coeff = fit_band(cos_i, refl, np.ones(5, bool))
    assert coeff.slope == 0.0
    assert np.isnan(coeff.c)
    assert np.isclose(coeff.intercept, 0.3, atol=1e-6)


def test_noise_recovers_the_underlying_slope():
    rng = np.random.default_rng(0)
    cos_i = rng.uniform(0.1, 0.9, 5000).astype(np.float32)
    refl = (0.5 * cos_i + 0.1 + rng.normal(0, 0.01, 5000)).astype(np.float32)
    coeff = fit_band(cos_i, refl, np.ones(5000, bool))
    assert abs(coeff.slope - 0.5) < 0.01
    assert abs(coeff.intercept - 0.1) < 0.01


def test_fit_bands_returns_one_entry_per_band():
    cos_i = np.array([[0.2, 0.4], [0.6, 0.8]], np.float32)
    stack = np.stack([
        (2.0 * cos_i + 1.0),
        (4.0 * cos_i + 2.0),
        (1.0 * cos_i + 0.0),
    ]).astype(np.float32)
    coeffs = fit_bands(cos_i, stack, np.ones((2, 2), bool))
    assert len(coeffs) == 3
    assert np.isclose(coeffs[0].slope, 2.0, atol=1e-4)
    assert np.isclose(coeffs[1].slope, 4.0, atol=1e-4)
    assert np.isclose(coeffs[2].slope, 1.0, atol=1e-4)


def test_fit_bands_accepts_a_single_band_raster():
    cos_i = np.array([[0.2, 0.4], [0.6, 0.8]], np.float32)
    refl = (2.0 * cos_i + 1.0).astype(np.float32)
    coeffs = fit_bands(cos_i, refl, np.ones((2, 2), bool))
    assert len(coeffs) == 1
    assert np.isclose(coeffs[0].slope, 2.0, atol=1e-4)
