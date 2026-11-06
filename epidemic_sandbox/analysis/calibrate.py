"""Measure a disease in the default town and compare it with the targets of its file.

    python -m epidemic_sandbox.analysis.calibrate covid19
    python -m epidemic_sandbox.analysis.calibrate covid19 --set transmission=0.1 --runs 32

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

A target named differently from its measurement names it in `measure`, and a household target
gives the follow-up of its study in `days` (28 if not). The measurements are those of
`epidemic_sandbox.analysis.measure`; every set of conditions is measured separately. A disease
that stays in the town (incidence, prevalence) is started from many cases (`ENDEMIC_START`, a
share of the town), so that it does not die out by chance and is near its settled level before the
counting starts.
"""

import argparse
import tomllib
from pathlib import Path

from epidemic_sandbox.analysis.measure import (
    EPIDEMIC_POPULATION,
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
ENDEMIC_START = 0.05  # share of the town infected at the start of a long run

TARGET_KEYS = {"value", "low", "high", "source"}
OPTIONAL_KEYS = {"with", "measure", "days"}


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
        if not str(target["source"]).strip():
            raise ValueError(f"{path}: target {name} needs a source")
    return targets


def _value(text: str) -> float:
    return float(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("disease")
    parser.add_argument("--set", action="append", default=[], help="parameter=value")
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--initial", type=int, default=10, help="first cases of an epidemic")
    parser.add_argument("--first-seed", type=int, default=1)
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
    seeds = list(range(arguments.first_seed, arguments.first_seed + arguments.runs))
    targets = load_targets(CONFIG_DIRECTORY / "diseases" / f"{arguments.disease}.toml")
    conditions = {()} | {tuple(sorted(t.get("with", {}).items())) for t in targets.values()}
    for condition in sorted(conditions):
        experiment = Experiment(arguments.disease, {**overrides, **dict(condition)})
        large = experiment.of_size(arguments.population)
        here = {
            target.get("measure", name): (name, target)
            for name, target in targets.items()
            if tuple(sorted(target.get("with", {}).items())) == condition
        }
        follow_up = [t["days"] for _, t in here.values() if "days" in t]
        days = int(follow_up[0]) if follow_up else HOUSEHOLD_DAYS
        measured = household_attack_rate(library, experiment, seeds, days=days)
        measured.update(epidemic(library, large, seeds, arguments.initial, arguments.days))
        if any(target.get("measure", name).startswith(ENDEMIC) for name, target in targets.items()):
            measured.update(
                endemic(
                    library,
                    large,
                    seeds,
                    arguments.endemic_initial or round(ENDEMIC_START * large.population),
                    arguments.endemic_days,
                    arguments.burn_in,
                )
            )
        print(
            f"{arguments.disease} {experiment.overrides or ''}, {arguments.runs} runs "
            f"of {arguments.days} days from {arguments.initial} first cases "
            f"(households followed for {days} days)"
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
