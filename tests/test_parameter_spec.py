import pytest

from epidemic_sandbox.parameter_spec import ParameterSpec, resolve_parameters

CONTINUOUS = ParameterSpec("Rate", "A rate.", 0.5, 0.0, 1.0, 0.1)
WHOLE = ParameterSpec("Count", "A count.", 5, 0, 10, 1)


def test_valid_value_is_returned():
    assert CONTINUOUS.validate("rate", 0.25) == 0.25


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_out_of_range_value_is_rejected(value):
    with pytest.raises(ValueError, match="out of range"):
        CONTINUOUS.validate("rate", value)


@pytest.mark.parametrize("value", ["0.5", None, True, [0.5]])
def test_non_number_is_rejected(value):
    with pytest.raises(ValueError, match="must be a number"):
        CONTINUOUS.validate("rate", value)


def test_whole_step_setting_accepts_only_whole_numbers_and_returns_int():
    assert WHOLE.validate("count", 4.0) == 4
    assert isinstance(WHOLE.validate("count", 4.0), int)
    with pytest.raises(ValueError, match="whole number"):
        WHOLE.validate("count", 4.5)


def test_resolve_fills_defaults_and_validates():
    assert resolve_parameters({"rate": CONTINUOUS, "count": WHOLE}, {"count": 7}) == {
        "rate": 0.5,
        "count": 7,
    }


def test_resolve_rejects_unknown_parameters_and_non_objects():
    with pytest.raises(ValueError, match="Unknown parameters"):
        resolve_parameters({"rate": CONTINUOUS}, {"speed": 1})
    with pytest.raises(TypeError):
        resolve_parameters({"rate": CONTINUOUS}, 5)
