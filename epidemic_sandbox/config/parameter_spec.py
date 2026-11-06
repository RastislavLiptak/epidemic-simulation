import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSpec:
    """A setting: default, allowed range and how the UI explains it.

    `control` is the UI widget kind: "slider" or "number". A setting without a `maximum` has no
    upper bound.
    """

    label: str
    description: str
    default: float
    minimum: float
    maximum: float | None
    step: float
    control: str = "slider"

    def validate(self, name: str, value: float) -> float:
        """Return `value` (a whole number for whole-step settings) or raise ValueError."""
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Parameter {name}={value!r} must be a number")
        too_big = self.maximum is not None and value > self.maximum
        if not math.isfinite(value) or value < self.minimum or too_big:
            allowed = (
                f"at least {self.minimum}"
                if self.maximum is None
                else f"<{self.minimum}, {self.maximum}>"
            )
            raise ValueError(f"Parameter {name}={value} out of range, {allowed}")
        if float(self.step).is_integer():
            if not float(value).is_integer():
                raise ValueError(f"Parameter {name}={value} must be a whole number")
            return int(value)
        return value


def resolve_parameters(specs: Mapping[str, ParameterSpec], given: Mapping) -> dict:
    """Validate `given` against `specs` and fill missing values with the defaults."""
    if not isinstance(given, Mapping):
        raise TypeError("Parameters must be an object")
    unknown = given.keys() - specs.keys()
    if unknown:
        raise ValueError(f"Unknown parameters {sorted(unknown)}, available: {sorted(specs)}")
    return {
        name: spec.validate(name, given.get(name, spec.default)) for name, spec in specs.items()
    }
