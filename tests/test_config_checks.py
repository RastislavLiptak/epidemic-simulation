import pytest

from epidemic_sandbox.config_checks import (
    require,
    require_not_negative,
    require_positive,
    require_share,
    require_weights,
)


def test_valid_values_pass():
    require(True, "never raised")
    require_positive("a", 0.1)
    require_not_negative("b", 0)
    require_share("c", 1)
    require_weights("d", [0, 2])


@pytest.mark.parametrize(
    ("check", "args", "message"),
    [
        (require, (False, "plain message"), "plain message"),
        (require_positive, ("a", 0), "a must be positive"),
        (require_not_negative, ("b", -0.1), "b must not be negative"),
        (require_share, ("c", 1.1), "c must be between 0 and 1"),
        (require_share, ("c", -0.1), "c must be between 0 and 1"),
        (require_weights, ("d", [0, 0]), "d must be non-negative"),
        (require_weights, ("d", [1, -1]), "d must be non-negative"),
        (require_weights, ("d", []), "d must be non-negative"),
    ],
)
def test_invalid_values_raise_a_clear_error(check, args, message):
    with pytest.raises(ValueError, match=message):
        check(*args)
