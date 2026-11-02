"""Small checks that keep nonsense values out of the config files."""


def require(condition: bool, message: str) -> None:
    """Raise ValueError with `message` unless `condition` holds."""
    if not condition:
        raise ValueError(message)


def require_positive(name: str, value: float) -> None:
    require(value > 0, f"{name} must be positive, not {value}")


def require_not_negative(name: str, value: float) -> None:
    require(value >= 0, f"{name} must not be negative, not {value}")


def require_share(name: str, value: float) -> None:
    require(0 <= value <= 1, f"{name} must be between 0 and 1, not {value}")


def require_weights(name: str, weights) -> None:
    """Weights of a choice: none negative and not all zero."""
    require(
        len(weights) > 0 and min(weights) >= 0 and sum(weights) > 0,
        f"{name} must be non-negative and not all zero",
    )
