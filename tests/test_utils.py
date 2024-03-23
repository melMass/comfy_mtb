import pytest
from setup_test import setup_test

setup_test()

from comfy_mtb_utils.nodes.graph_utils import FitNumber
from comfy_mtb_utils.utils import apply_easing


# - apply_easing
@pytest.mark.parametrize(
    "value, easing_type, expected",
    [
        (0.5, "Linear", 0.5),
        (0, "Linear", 0),
        (1, "Linear", 1),
        (0.5, "Sine In", 0.2928932188134524),
    ],
)
def test_apply_easing(value: float, easing_type: str, expected: float):
    assert apply_easing(value, easing_type) == pytest.approx(
        expected
    ), "Easing function did not return expected value"


@pytest.mark.parametrize(
    "val, easing_type",
    [
        (0.5, "NonExistentEasing"),
    ],
)
def test_apply_easing_error_handling(val, easing_type):
    with pytest.raises(ValueError):
        apply_easing(val, easing_type)


@pytest.mark.parametrize(
    "val, easing_type, expected",
    [
        (0, "Linear", 0),
        (1, "Linear", 1),
        (0.5, "Linear", 0.5),
    ],
)
def test_apply_easing_corner_cases(val, easing_type, expected):
    assert apply_easing(val, easing_type) == expected


# - FitNumber
@pytest.fixture
def fit_number_instance():
    return FitNumber()


@pytest.mark.parametrize(
    "value, clamp, source_min, source_max, target_min, target_max, easing, expected",
    [
        (0.5, False, 0, 1, 0, 1, "Linear", (0.5,)),
    ],
)
def test_fit_number_set_range(
    fit_number_instance,
    value,
    clamp,
    source_min,
    source_max,
    target_min,
    target_max,
    easing,
    expected,
):
    assert (
        fit_number_instance.set_range(
            value=value,
            clamp=clamp,
            source_min=source_min,
            source_max=source_max,
            target_min=target_min,
            target_max=target_max,
            easing=easing,
        )
        == expected
    ), "Set range method did not return expected value"


@pytest.mark.parametrize(
    "value, clamp, source_min, source_max, target_min, target_max, easing, expected",
    [
        (0.5, False, 0, 1, 0, 1, "Linear", (0.5,)),
        (0.4, False, 1.0, 0.4, 0.4, 0.7, "Linear", (0.7,)),
        # Add more test cases covering various scenarios, edge cases, and easing types
    ],
)
def test_fit_number_inverted_ranges(
    fit_number_instance,
    value,
    clamp,
    source_min,
    source_max,
    target_min,
    target_max,
    easing,
    expected,
):
    assert (
        fit_number_instance.set_range(
            value=value,
            clamp=clamp,
            source_min=source_min,
            source_max=source_max,
            target_min=target_min,
            target_max=target_max,
            easing=easing,
        )
        == expected
    ), "Set range method did not return expected value"
