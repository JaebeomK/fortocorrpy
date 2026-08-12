"""Config validation: an invalid setting must fail at construction."""

import pytest

from fortocorrpy import Config, DENOMINATOR_METHODS, METHODS, SLOPE_TC_METHODS


def test_defaults_construct():
    cfg = Config()
    assert cfg.mask_source == "ndvi"
    assert cfg.methods == ("scsc",)
    assert cfg.n_workers == 1


def test_unknown_mask_source_rejected():
    with pytest.raises(ValueError, match="mask_source"):
        Config(mask_source="lidar")


def test_landcover_requires_forest_class():
    with pytest.raises(ValueError, match="forest_class"):
        Config(mask_source="landcover", forest_class=None)


def test_negative_slope_threshold_rejected():
    with pytest.raises(ValueError, match="slope_min_deg"):
        Config(slope_min_deg=-1.0)


def test_negative_cos_i_threshold_rejected():
    """A negative threshold would admit self-shadow into the regression."""
    with pytest.raises(ValueError, match="cos_i_threshold"):
        Config(cos_i_threshold=-0.5)


def test_quadrant_edges_must_be_four():
    with pytest.raises(ValueError, match="four edges"):
        Config(aspect_quadrant_edges=(90.0, 180.0))


def test_quadrant_edges_must_increase():
    with pytest.raises(ValueError, match="increasing"):
        Config(aspect_quadrant_edges=(45.0, 45.0, 225.0, 315.0))


def test_quadrant_edges_must_be_within_one_turn():
    with pytest.raises(ValueError, match=r"\[0, 360\)"):
        Config(aspect_quadrant_edges=(45.0, 135.0, 225.0, 400.0))


def test_min_samples_must_be_positive():
    with pytest.raises(ValueError, match="min_samples_per_quadrant"):
        Config(min_samples_per_quadrant=0)


def test_methods_must_not_be_empty():
    with pytest.raises(ValueError, match="at least one method"):
        Config(methods=())


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown correction method"):
        Config(methods=("scsc", "minnaert"))


def test_n_workers_must_be_positive():
    with pytest.raises(ValueError, match="n_workers"):
        Config(n_workers=0)


# --- derived branching --------------------------------------------------

@pytest.mark.parametrize("method, expected", [
    ("cosine", False), ("scs", False),
    ("c", True), ("scsc", True), ("se", True), ("er", True),
])
def test_needs_regression(method, expected):
    assert Config(methods=(method,)).needs_regression is expected


@pytest.mark.parametrize("method, expected", [
    ("cosine", False), ("c", False), ("se", False), ("er", False),
    ("scs", True), ("scsc", True),
])
def test_needs_slope_at_correction(method, expected):
    assert Config(methods=(method,)).needs_slope_at_correction is expected


# --- constants ----------------------------------------------------------

def test_six_methods_are_exposed():
    assert len(METHODS) == 6
    assert set(METHODS) == {"cosine", "scs", "c", "scsc", "se", "er"}


def test_method_groups_are_subsets_of_methods():
    assert set(SLOPE_TC_METHODS) <= set(METHODS)
    assert set(DENOMINATOR_METHODS) <= set(METHODS)
    assert set(SLOPE_TC_METHODS) == {"scs", "scsc"}
    assert set(DENOMINATOR_METHODS) == {"cosine", "scs", "c", "scsc"}
