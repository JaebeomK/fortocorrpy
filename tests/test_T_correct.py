"""The six correction formulas, checked against hand calculations.

The correction step takes ``cos_alpha`` (the cosine of the slope), not the
slope angle: the pipeline computes it once per image rather than per band per
method.
"""

import numpy as np
import pytest

from fortocorrpy import T_correct, config
from fortocorrpy.linear_C import BandCoefficients

RHO = np.array([0.3], dtype=np.float32)


def coeff(slope=1.0, intercept=0.2, c=0.2, mean=0.3, n=300):
    return BandCoefficients(
        slope=slope, intercept=intercept, c=c, n_samples=n, mean_reflectance=mean,
    )


# --- cosine ------------------------------------------------------------

def test_cosine_is_identity_on_a_horizontal_surface():
    out = T_correct.correct(
        "cosine", RHO, np.array([0.5], np.float32), 0.5,
    )
    assert np.isclose(out[0], 0.3, atol=1e-6)


def test_cosine_hand_value():
    # 0.3 * 0.8 / 0.4 = 0.6
    out = T_correct.correct(
        "cosine", RHO, np.array([0.4], np.float32), 0.8,
    )
    assert np.isclose(out[0], 0.6, atol=1e-6)


# --- scs ---------------------------------------------------------------

def test_scs_hand_value():
    # 0.3 * (0.5 * 0.8) / 0.4 = 0.3
    out = T_correct.correct(
        "scs", RHO, np.array([0.4], np.float32), 0.8,
        cos_alpha=np.array([0.5], np.float32),
    )
    assert np.isclose(out[0], 0.3, atol=1e-6)


def test_scs_requires_cos_alpha():
    with pytest.raises(ValueError, match="cos_alpha"):
        T_correct.correct("scs", RHO, np.array([0.5], np.float32), 0.5)


# --- c -----------------------------------------------------------------

def test_c_hand_value():
    # 0.3 * (0.8 + 0.2) / (0.4 + 0.2) = 0.5
    out = T_correct.correct(
        "c", RHO, np.array([0.4], np.float32), 0.8, coeff=coeff(),
    )
    assert np.isclose(out[0], 0.5, atol=1e-6)


def test_c_requires_coeff():
    with pytest.raises(ValueError):
        T_correct.correct("c", RHO, np.array([0.5], np.float32), 0.5)


def test_c_moderates_the_cosine_overcorrection_at_low_cos_i():
    """The whole point of C: a smaller boost than cosine on weakly lit slopes."""
    ci = np.array([0.05], np.float32)
    cosine = T_correct.correct("cosine", RHO, ci, 0.8)
    c_corr = T_correct.correct("c", RHO, ci, 0.8, coeff=coeff())
    assert c_corr[0] < cosine[0]


# --- scsc --------------------------------------------------------------

def test_scsc_hand_value():
    # 0.3 * (0.5 * 0.8 + 0.2) / (0.4 + 0.2) = 0.3
    out = T_correct.correct(
        "scsc", RHO, np.array([0.4], np.float32), 0.8,
        cos_alpha=np.array([0.5], np.float32), coeff=coeff(),
    )
    assert np.isclose(out[0], 0.3, atol=1e-6)


def test_scsc_requires_cos_alpha_and_coeff():
    with pytest.raises(ValueError):
        T_correct.correct("scsc", RHO, np.array([0.5], np.float32), 0.5,
                          coeff=coeff())
    with pytest.raises(ValueError):
        T_correct.correct("scsc", RHO, np.array([0.5], np.float32), 0.5,
                          cos_alpha=np.array([0.5], np.float32))


# --- se / er -----------------------------------------------------------

def test_se_and_er_are_not_identical_for_general_coefficients():
    """Regression: se re-centres on the sample mean, er on cos(theta_s)."""
    rho = np.array([0.3, 0.4, 0.5], np.float32)
    ci = np.array([0.4, 0.6, 0.8], np.float32)
    cos_ts = np.float32(0.7)
    bc = coeff(slope=0.2, intercept=0.1, c=0.5, mean=0.4)

    se = T_correct.correct("se", rho, ci, cos_ts, coeff=bc)
    er = T_correct.correct("er", rho, ci, cos_ts, coeff=bc)

    assert not np.allclose(se, er)
    assert np.allclose(se, rho - (0.2 * ci + 0.1) + 0.4, atol=1e-6)
    assert np.allclose(er, rho - 0.2 * (ci - 0.7), atol=1e-6)


def test_er_is_identity_where_cos_i_equals_cos_theta_s():
    """The rotation pivots on the horizontal reference."""
    rho = np.array([0.3, 0.4], np.float32)
    ci = np.array([0.7, 0.7], np.float32)
    out = T_correct.correct("er", rho, ci, np.float32(0.7),
                            coeff=coeff(slope=0.2))
    assert np.allclose(out, rho, atol=1e-6)


def test_se_requires_coeff():
    with pytest.raises(ValueError):
        T_correct.correct("se", RHO, np.array([0.5], np.float32), 0.5)


def test_er_requires_coeff():
    with pytest.raises(ValueError):
        T_correct.correct("er", RHO, np.array([0.5], np.float32), 0.5)


# --- self-shadow / denominator behaviour --------------------------------

@pytest.mark.parametrize("method", ["cosine", "scs", "c", "scsc"])
def test_denominator_methods_return_nan_in_self_shadow(method):
    rho = np.array([0.3, 0.3], np.float32)
    ci = np.array([0.5, -0.2], np.float32)
    out = T_correct.correct(
        method, rho, ci, 0.5,
        cos_alpha=np.array([0.9, 0.9], np.float32), coeff=coeff(),
    )
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


@pytest.mark.parametrize("method", ["se", "er"])
def test_subtractive_methods_stay_finite_in_self_shadow(method):
    rho = np.array([0.3, 0.3], np.float32)
    ci = np.array([0.5, -0.2], np.float32)
    out = T_correct.correct(method, rho, ci, 0.5, coeff=coeff(slope=0.2))
    assert np.all(np.isfinite(out))


def test_cos_i_threshold_is_applied_to_denominator_methods():
    rho = np.array([0.3, 0.3], np.float32)
    ci = np.array([0.05, 0.5], np.float32)
    out = T_correct.correct("cosine", rho, ci, 0.5, cos_i_threshold=0.1)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_near_singular_denominator_becomes_nan():
    """cos i = -C would divide by zero; the guard returns NaN instead."""
    ci = np.array([0.5], np.float32)
    out = T_correct.correct("c", RHO, ci, 0.8, coeff=coeff(c=-0.5))
    assert np.isnan(out[0])


# --- degenerate regression (slope a = 0, so C is undefined) --------------

@pytest.mark.parametrize("method", ["c", "scsc"])
def test_zero_regression_slope_is_identity_on_lit_pixels(method):
    rho = np.array([0.3, 0.4], np.float32)
    ci = np.array([0.5, 0.7], np.float32)
    bc = coeff(slope=0.0, intercept=0.3, c=float("nan"))
    out = T_correct.correct(
        method, rho, ci, 0.6,
        cos_alpha=np.array([0.9, 0.9], np.float32), coeff=bc,
    )
    assert np.allclose(out, rho, atol=1e-6)


@pytest.mark.parametrize("method", ["c", "scsc"])
def test_zero_regression_slope_still_masks_self_shadow(method):
    rho = np.array([0.3, 0.4], np.float32)
    ci = np.array([0.5, -0.2], np.float32)
    bc = coeff(slope=0.0, intercept=0.3, c=float("nan"))
    out = T_correct.correct(
        method, rho, ci, 0.6,
        cos_alpha=np.array([0.9, 0.9], np.float32), coeff=bc,
    )
    assert np.isclose(out[0], 0.3, atol=1e-6)
    assert np.isnan(out[1])


# --- misc ---------------------------------------------------------------

def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        T_correct.correct("nope", RHO, np.array([0.5], np.float32), 0.5)


@pytest.mark.parametrize("method", ["cosine", "scs", "c", "scsc", "se", "er"])
def test_output_is_float32(method):
    out = T_correct.correct(
        method, RHO, np.array([0.5], np.float32), 0.5,
        cos_alpha=np.array([0.9], np.float32), coeff=coeff(),
    )
    assert out.dtype == np.float32


def test_the_method_groups_are_owned_by_config():
    """`T_correct` uses DENOMINATOR_METHODS but does not re-export it.

    The three method groups are declared in one place, `config`, so that adding
    a method means editing one file. Re-exporting a group from the module that
    consumes it would advertise two owners for the same name.
    """
    assert "DENOMINATOR_METHODS" not in T_correct.__all__
    assert T_correct.DENOMINATOR_METHODS is config.DENOMINATOR_METHODS
