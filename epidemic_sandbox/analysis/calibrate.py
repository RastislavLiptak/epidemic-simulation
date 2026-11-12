"""Measure a disease in the default town and compare it with the targets of its file.

    python -m epidemic_sandbox.analysis.calibrate covid19
    python -m epidemic_sandbox.analysis.calibrate covid19 --set transmission=0.1 --runs 32
    python -m epidemic_sandbox.analysis.calibrate covid19 --fit transmission=household_attack_rate

A disease file names, in `[targets]`, the published values its Fitted parameters were fitted to
and those its behaviour is checked against, each with a value, a range and a source, and
optionally the conditions of the measurement (`with`: parameters changed for it, such as nobody
immune for a reproduction number in a population without immunity):

    [targets.household_attack_rate]
    value = 0.166
    low = 0.140
    high = 0.193
    source = "Madewell et al. 2020, JAMA Netw Open"
    with = { initial_immune = 0.0 }

A target named differently from its measurement names it in `measure`. A household target gives
the follow-up of its study in `days` (28 if not) and the ages of its first cases and of the members
it followed in `index_ages` and `contact_ages` ([from, to) years); an epidemic target the number of
first cases in `initial`; an incidence target the time within which two infections of a person
are one episode in `episode_days`. The measurements are those of
`epidemic_sandbox.analysis.measure`; every set of conditions is measured separately. A disease
that stays in the town (incidence, prevalence) is started from many cases (`ENDEMIC_START`, a
share of the town), so that it does not die out by chance and is near its settled level before the
counting starts.

`--fit knob=target` fits a knob to a target by bisection with common random numbers (the same
seeds for every value tried, so that the comparison of two values is not blurred by chance): a
parameter (`transmission`) or the multiplier of an age range of an age effect
(`age.susceptibility.0-19`). Several `--fit` are fitted in turn, in rounds, until each target is
met within its measurement's standard error (or the rounds run out); then everything is measured.
"""

import argparse
import math
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from epidemic_sandbox.analysis.measure import (
    EPIDEMIC_POPULATION,
    FIRST_GENERATION,
    HOUSEHOLD_DAYS,
    Experiment,
    endemic,
    epidemic,
    household_attack_rate,
)
from epidemic_sandbox.config.app import load_app_config
from epidemic_sandbox.config.experiment import load_library
from epidemic_sandbox.paths import CONFIG_DIRECTORY

ENDEMIC = ("incidence", "case_incidence", "prevalence")  # measured in a long run
HOUSEHOLD = (
    "household_attack_rate",
    "household_case_rate",
    "household_direct_rate",
    "household_chain_rate",
)
ENDEMIC_START = 0.05  # share of the town infected at the start of a long run

TARGET_KEYS = {"value", "low", "high", "source"}
OPTIONAL_KEYS = {
    "with",
    "measure",
    "days",
    "index_ages",
    "contact_ages",
    "initial",
    "first_generation",
    "episode_days",
    "fit",
}
AGE_KNOB = "age."


def load_targets(path: Path) -> dict[str, dict]:
    """The `[targets]` of a disease file, checked: every target has a value inside its range and
    a source."""
    with path.open("rb") as disease_file:
        targets = tomllib.load(disease_file).get("targets", {})
    for name, target in targets.items():
        if not TARGET_KEYS <= set(target) <= TARGET_KEYS | OPTIONAL_KEYS:
            raise ValueError(
                f"{path}: target {name} needs {sorted(TARGET_KEYS)} and may have {OPTIONAL_KEYS}"
            )
        if not target["low"] <= target["value"] <= target["high"]:
            raise ValueError(f"{path}: target {name} has a value outside its range")
        if not target["low"] <= target.get("fit", target["value"]) <= target["high"]:
            raise ValueError(f"{path}: target {name} is to be fitted to a value outside its range")
        if not str(target["source"]).strip():
            raise ValueError(f"{path}: target {name} needs a source")
        for key in ("index_ages", "contact_ages"):
            ages = target.get(key)
            if ages is not None and not (len(ages) == 2 and 0 <= ages[0] < ages[1]):
                raise ValueError(f"{path}: target {name} needs {key} = [from, to)")
    return targets


@dataclass(frozen=True)
class Settings:
    """How the measurements are run."""

    seeds: list
    household_seeds: list | None = None  # the households are many more runs of a small town
    days: int = 120
    initial: int = 10
    burn_in: int = 100
    endemic_days: int = 300
    endemic_initial: int | None = None
    population: int = EPIDEMIC_POPULATION
    workers: int | None = None


def condition_of(target: dict) -> tuple:
    return tuple(sorted(target.get("with", {}).items()))


def measure(library, experiment: Experiment, targets: dict, settings: Settings) -> dict:
    """Every measurement for the targets measured under the conditions of `experiment` (all of
    them share them): the household attack rate, the epidemic and, if a target needs it, the
    long run of a disease that stays."""
    measured = household(library, experiment, targets, settings)
    measured.update(early(library, experiment, targets, settings))
    if any(_measure_name(name, t).startswith(ENDEMIC) for name, t in targets.items()):
        measured.update(long_run(library, experiment, targets, settings))
    return measured


def household(library, experiment, targets, settings) -> dict:
    options = _options(targets, HOUSEHOLD)
    return household_attack_rate(
        library,
        experiment,
        settings.household_seeds or settings.seeds,
        days=int(options.get("days", HOUSEHOLD_DAYS)),
        workers=settings.workers,
        index_ages=_ages(options.get("index_ages")),
        contact_ages=_ages(options.get("contact_ages")),
    )


def early(library, experiment, targets, settings) -> dict:
    options = _options(targets, None, excluding=HOUSEHOLD + ENDEMIC)
    return epidemic(
        library,
        experiment.of_size(settings.population),
        settings.seeds,
        int(options.get("initial", settings.initial)),
        settings.days,
        settings.workers,
        int(options.get("first_generation", FIRST_GENERATION)),
    )


def long_run(library, experiment, targets, settings) -> dict:
    options = _options(targets, ENDEMIC)
    large = experiment.of_size(settings.population)
    return endemic(
        library,
        large,
        settings.seeds,
        settings.endemic_initial or round(ENDEMIC_START * large.population),
        settings.endemic_days,
        settings.burn_in,
        settings.workers,
        float(options.get("episode_days", 0.0)),
    )


def _measure_name(name: str, target: dict) -> str:
    return target.get("measure", name)


def _options(targets: dict, prefixes, excluding=()) -> dict:
    """The measurement options of the targets of one kind (those whose measurement starts with one
    of `prefixes`, or all but `excluding`); they must agree."""
    options = {}
    for name, target in targets.items():
        measured = _measure_name(name, target)
        if prefixes is not None and not measured.startswith(prefixes):
            continue
        if excluding and measured.startswith(excluding):
            continue
        for key in (
            "days",
            "index_ages",
            "contact_ages",
            "initial",
            "first_generation",
            "episode_days",
        ):
            if key in target:
                if key in options and options[key] != target[key]:
                    raise ValueError(f"targets measured together need the same {key}")
                options[key] = target[key]
    return options


def _ages(ages) -> tuple[float, float] | None:
    return None if ages is None else (float(ages[0]), float(ages[1]))


# --- fitting ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Knob:
    """A parameter, or the multiplier of the age range [low, high) of an age effect."""

    name: str
    effect: str | None = None
    low: float = 0.0
    high: float = 0.0

    @classmethod
    def parse(cls, text: str) -> "Knob":
        if not text.startswith(AGE_KNOB):
            return cls(text)
        _, effect, ages = text.split(".", 2)
        low, high = (float(age) for age in ages.split("-"))
        return cls(text, effect, low, high)

    def value(self, library, experiment: Experiment) -> float:
        if self.effect is None:
            disease = library.catalog[experiment.disease]
            given = experiment.overrides.get(self.name, disease.parameters.get(self.name, 0.0))
            return float(given)
        for low, high, multiplier in _rows(library, experiment, self.effect):
            if (low, high) == (self.low, self.high):
                return float(multiplier)
        return 1.0

    def set(self, library, experiment: Experiment, value: float) -> Experiment:
        if self.effect is None:
            return replace(experiment, overrides={**experiment.overrides, self.name: value})
        rows = [
            row
            for row in _rows(library, experiment, self.effect)
            if row[:2] != [self.low, self.high]
        ]
        rows.append([self.low, self.high, value])
        rows.sort()
        effects = {**(experiment.age_effects or {}), self.effect: rows}
        return replace(experiment, age_effects=effects)

    def bounds(self, library, experiment: Experiment) -> tuple[float, float]:
        """Where to look: a probability between 0 and 1, anything else from a hundredth to a
        hundred times its value."""
        if self.name == "transmission" or self.name.endswith(("_share", "_coverage")):
            return 1e-4, 0.999
        if self.name == "room_exposure":
            return 1e-3, 1000.0
        current = max(self.value(library, experiment), 1e-3)
        return current / 100, current * 100


def _rows(library, experiment: Experiment, effect: str) -> list:
    if experiment.age_effects and effect in experiment.age_effects:
        return [list(row) for row in experiment.age_effects[effect]]
    return [list(row) for row in library.catalog[experiment.disease].age_effects.get(effect, ())]


def fit(
    library,
    experiment: Experiment,
    pairs: list[tuple[Knob, str]],
    targets: dict,
    settings: Settings,
    rounds: int = 3,
    steps: int = 12,
    report=print,
) -> Experiment:
    """Fit each knob to its target in turn (bisection with the same seeds for every value), over
    `rounds` rounds or until every target is met within its measurement's standard error."""
    for round_number in range(rounds):
        met = True
        moved = False
        for knob, name in pairs:
            target = targets[name]
            aim = target.get("fit", target["value"])  # a value inside the range, if one is given
            conditions = dict(condition_of(target))
            here = {n: t for n, t in targets.items() if condition_of(t) == condition_of(target)}
            low, high = knob.bounds(library, experiment)
            value = knob.value(library, experiment)
            if not low < value < high:  # a knob that is off or out of range starts in the middle
                value = math.sqrt(low * high)
            for _ in range(steps):
                trial = knob.set(library, experiment, value)
                measured, error = _measure_one(library, trial, name, here, conditions, settings)
                report(
                    f"  round {round_number + 1}: {knob.name} = {value:.5g} -> "
                    f"{name} {measured:.4g}"
                )
                if not math.isfinite(measured):
                    break
                if abs(measured - aim) <= max(error, 1e-9):
                    break
                if measured < aim:
                    low = value
                else:
                    high = value
                if high / low < 1.01:  # the bracket is narrower than any measurement tells
                    break
                value = math.sqrt(low * high)
            moved |= not math.isclose(value, knob.value(library, experiment), rel_tol=0.02)
            experiment = knob.set(library, experiment, value)
            met &= math.isfinite(measured) and abs(measured - aim) <= max(error, 1e-9)
        # a knob met before the next one moved may not be met after it: stop when a round moved
        # none of them (at once, with a single knob)
        if met and (not moved or len(pairs) == 1):
            break
    return experiment


def _measure_one(library, experiment, name, targets, conditions, settings) -> tuple[float, float]:
    """The measurement of one target and its standard error (0 when none is known)."""
    measured_name = _measure_name(name, targets[name])
    trial = replace(experiment, overrides={**experiment.overrides, **conditions})
    if measured_name.startswith(HOUSEHOLD):
        result = household(library, trial, targets, settings)
        return result[measured_name], result.get("standard_error", 0.0)
    if measured_name.startswith(ENDEMIC):
        result = long_run(library, trial, targets, settings)
        return result[measured_name], 0.0
    result = early(library, trial, targets, settings)
    error = 0.0
    if measured_name == "reproduction_number":
        error = result.get("reproduction_number_se", 0.0)
    return result[measured_name], error


# --- the command --------------------------------------------------------------------------------


def _value(text: str) -> float:
    return float(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("disease")
    parser.add_argument("--set", action="append", default=[], help="parameter=value")
    parser.add_argument(
        "--fit",
        action="append",
        default=[],
        help="knob=target, e.g. transmission=household_attack_rate",
    )
    parser.add_argument("--rounds", type=int, default=3, help="rounds of fitting")
    parser.add_argument(
        "--fit-only", action="store_true", help="print the fitted values, measure nothing else"
    )
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument(
        "--household-runs", type=int, default=None, help="runs of the household measurement"
    )
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--initial", type=int, default=10, help="first cases of an epidemic")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    parser.add_argument(
        "--burn-in", type=int, default=100, help="days left out of a long run (endemic diseases)"
    )
    parser.add_argument("--endemic-days", type=int, default=300, help="length of a long run")
    parser.add_argument(
        "--endemic-initial",
        type=int,
        default=None,
        help=f"first cases of a long run (default {ENDEMIC_START * 100:g} %% of the town)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=EPIDEMIC_POPULATION,
        help="people of the town of the epidemic measurements (with the density of the default "
        "town); the household attack rate is measured in the default town",
    )
    arguments = parser.parse_args()
    app = load_app_config(CONFIG_DIRECTORY / "app.toml")
    library = load_library(CONFIG_DIRECTORY / "experiment.toml", app.limits)
    overrides = dict(
        (name, _value(value)) for name, value in (item.split("=", 1) for item in arguments.set)
    )
    settings = Settings(
        seeds=list(range(arguments.first_seed, arguments.first_seed + arguments.runs)),
        household_seeds=(
            list(range(arguments.first_seed, arguments.first_seed + arguments.household_runs))
            if arguments.household_runs
            else None
        ),
        days=arguments.days,
        initial=arguments.initial,
        burn_in=arguments.burn_in,
        endemic_days=arguments.endemic_days,
        endemic_initial=arguments.endemic_initial,
        population=arguments.population,
        workers=arguments.workers,
    )
    targets = load_targets(CONFIG_DIRECTORY / "diseases" / f"{arguments.disease}.toml")
    base = Experiment(arguments.disease, overrides)
    if arguments.fit:
        pairs = []
        for item in arguments.fit:
            knob, name = item.split("=", 1)
            if name not in targets:
                raise SystemExit(f"no target {name} in the disease file")
            pairs.append((Knob.parse(knob), name))
        base = fit(library, base, pairs, targets, settings, arguments.rounds)
        print(f"fitted: {base.overrides} {base.age_effects or ''}", flush=True)
        if arguments.fit_only:
            return
    conditions = {()} | {condition_of(t) for t in targets.values()}
    for condition in sorted(conditions):
        experiment = replace(base, overrides={**base.overrides, **dict(condition)})
        here = {
            _measure_name(name, target): (name, target)
            for name, target in targets.items()
            if condition_of(target) == condition
        }
        chosen = {
            name: target for name, target in targets.items() if condition_of(target) == condition
        }
        measured = measure(library, experiment, chosen, settings)
        print(
            f"{arguments.disease} {experiment.overrides or ''}, {arguments.runs} runs "
            f"of {arguments.days} days"
        )
        for measurement, value in measured.items():
            if measurement not in here:
                print(f"  {measurement:34s} {_format(value)}")
                continue
            name, target = here[measurement]
            inside = target["low"] <= value <= target["high"]
            print(
                f"  {name:34s} {_format(value):>10s}  target {target['value']} "
                f"[{target['low']}, {target['high']}] {'ok' if inside else 'OUTSIDE'}"
                f"  ({target['source']})"
            )


def _format(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key} {share:.3f}" for key, share in value.items())
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    main()
