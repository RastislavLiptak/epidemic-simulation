from collections.abc import Mapping
from dataclasses import dataclass

EFFECTS = {
    "susceptibility": "how easily a person catches the target disease",
    "infectiousness": "how infectious a person with the target disease is",
}
MAX_FACTOR = 100.0


@dataclass(frozen=True)
class Interaction:
    """While a person is in any of `states` of `source`, `target` is changed by `factor`.

    With effect "susceptibility" the person's chance of catching `target` is multiplied by
    `factor` (below 1 protects, above 1 makes the person more vulnerable). With effect
    "infectiousness" a person who has `target` infects others `factor` times as much.
    """

    source: str
    states: tuple[str, ...]
    target: str
    effect: str
    factor: float

    @classmethod
    def from_definition(cls, raw: Mapping) -> "Interaction":
        if not isinstance(raw, Mapping):
            raise TypeError("An interaction must be an object")
        states = raw["states"]
        if isinstance(states, str) or not all(isinstance(state, str) for state in states):
            raise TypeError("Interaction states must be a list of names")
        factor = raw["factor"]
        if isinstance(factor, bool) or not isinstance(factor, int | float):
            raise ValueError(f"Interaction factor {factor!r} must be a number")
        if not 0 <= factor <= MAX_FACTOR:
            raise ValueError(f"Interaction factor {factor} out of range <0, {MAX_FACTOR:g}>")
        if raw["effect"] not in EFFECTS:
            raise ValueError(f"Unknown effect '{raw['effect']}', available: {sorted(EFFECTS)}")
        return cls(raw["source"], tuple(states), raw["target"], raw["effect"], float(factor))

    def to_definition(self) -> dict:
        return {
            "source": self.source,
            "states": list(self.states),
            "target": self.target,
            "effect": self.effect,
            "factor": self.factor,
        }
