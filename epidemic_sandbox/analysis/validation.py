"""Validation of the simulated population against published data.

Two kinds of tests, both against files in `data/reference/`:

* equivalence tests (TOST) say whether a simulated share or mean lies within an accepted margin
  of the published value; the margin reflects how much two official sources differ,
* permutation tests say whether the age structure of simulated contacts (who meets whom, by age)
  agrees with the published contact matrices more than chance alignment of age groups would,
* the contacts of a week of the town are compared with the contact survey by place and by how long
  they last, within the distance at which infections pass (2 m), as each person would report them
  (a visitor's contact with the host at "other", the host's at "home").
"""

import argparse
import csv
import math
import sys
import tomllib
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from epidemic_sandbox.engine.spatial_kernels import build_grid, find_pairs
from epidemic_sandbox.paths import CONFIG_DIRECTORY, REFERENCE_DIRECTORY
from epidemic_sandbox.world.population import (
    AGES,
    FEMALE,
    MALE,
    PRESCHOOL,
    REMOTE,
    RESIDENT,
    SCHOOL,
    UNIVERSITY,
    WORKER,
    Population,
    assign_roles,
    generate_population,
    read_people,
)
from epidemic_sandbox.world.schedule import (
    HOME,
    WORK,
    destination_codes_at,
    draw_day_schedule,
)
from epidemic_sandbox.world.town import (
    DAYS_PER_WEEK,
    WEEKEND_FROM_DAY,
    TownParameters,
    build_town,
    load_town_parameters,
)

DEFAULT_TOWN_PATH = CONFIG_DIRECTORY / "town.toml"
ALPHA = 0.05
CONTACT_AGE_GROUPS = 16
CONTACT_GROUP_WIDTH = 5
LAYERS = ("home", "school", "work", "others")
PERMUTATIONS = 5000
MIN_CORRELATION = 0.5
PERMUTATION_ALPHA = 0.01
AGE_GROUP_MARGIN = 0.004  # the household fit is exact on its pool, the rest is sampling error
OLDEST_FROM = 90
AGE_GROUPS = 18  # 0-4, 5-9, ..., 80-84, 85+
POPULATION_SOURCE = "Eurostat demo_pjan (czechia_population_2019.csv)"
# Published mean ages count everybody as having lived half of the year of age they are in, while
# the simulation knows the completed years.
MEAN_AGE_WITHIN_YEAR = 0.5
# Somebody at least this much younger than the oldest of a household counts as their child.
PARENT_GAP = 15
INFANT_YEARS = 3  # children this young tell the age of their mother at birth, with enough of them


@dataclass(frozen=True)
class Check:
    """One validation result: what was measured, what it was compared with, and the verdict."""

    name: str
    observed: float
    expected: str
    p_value: float
    passed: bool
    source: str


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def equivalence_p_value(
    observed: float, standard_error: float, reference: float, margin: float
) -> float:
    """TOST p-value: small when `observed` is within `margin` of `reference`, given the error."""
    standard_error = max(standard_error, 1e-12)
    below = 1.0 - normal_cdf((observed - (reference - margin)) / standard_error)
    above = 1.0 - normal_cdf(((reference + margin) - observed) / standard_error)
    return max(below, above)


def share_check(name, share, count, reference, margin, source, alpha=ALPHA) -> Check:
    standard_error = math.sqrt(max(share * (1.0 - share), 1e-12) / count)
    p_value = equivalence_p_value(share, standard_error, reference, margin)
    return Check(name, share, f"{reference:.3f} ± {margin:.3f}", p_value, p_value < alpha, source)


def mean_check(name, values, reference, margin, source) -> Check:
    standard_error = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    observed = float(np.mean(values))
    p_value = equivalence_p_value(observed, standard_error, reference, margin)
    return Check(
        name, observed, f"{reference:.2f} ± {margin:.2f}", p_value, p_value < ALPHA, source
    )


def rank(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, tied values sharing their average rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    ranks = np.corrcoef(rank(first.ravel()), rank(second.ravel()))
    return float(ranks[0, 1])


def permutation_test(simulated: np.ndarray, reference: np.ndarray, rng) -> tuple[float, float]:
    """Spearman correlation of two matrices and the chance of one at least as high by luck.

    The null hypothesis keeps the structure of the simulated matrix but shuffles which age group
    is which (rows and columns together).
    """
    observed = spearman(simulated, reference)
    groups = len(simulated)
    at_least = 0
    for _ in range(PERMUTATIONS):
        order = rng.permutation(groups)
        at_least += spearman(simulated[np.ix_(order, order)], reference) >= observed
    return observed, (1 + at_least) / (PERMUTATIONS + 1)


def read_reference(name: str) -> dict:
    with (REFERENCE_DIRECTORY / name).open("rb") as reference_file:
        return tomllib.load(reference_file)


def validate_population(params: TownParameters, count: int = 200_000, seed: int = 0) -> list[Check]:
    """Compare the people, households and what they do with published data."""
    reference = read_reference("czechia_reference.toml")
    rng = np.random.default_rng(seed)
    people = generate_population(count, params.population, rng)
    role = assign_roles(people, params.population, rng)
    return [
        *_age_checks(people, reference),
        *_household_checks(people, reference),
        *_family_checks(people, reference),
        *_activity_checks(people, role, params, reference),
    ]


def _age_checks(people: Population, reference: dict) -> list[Check]:
    age, sex, count = people.age, people.sex, len(people.age)
    source = reference["age"]["source"]
    checks = [
        share_check(name, mask.mean(), count, expected, 0.012, source)
        for name, mask, expected in (
            ("age under 15", age < 15, reference["age"]["under_15"]),
            ("age 15 to 64", (age >= 15) & (age < 65), reference["age"]["from_15_to_64"]),
            ("age 65 and over", age >= 65, reference["age"]["from_65"]),
        )
    ]
    checks.append(
        mean_check(
            "mean age", age + MEAN_AGE_WITHIN_YEAR, reference["age"]["mean_age"], 0.8, source
        )
    )
    table = _read_population()
    shares = table.sum(axis=1) / table.sum()
    for group in range(AGE_GROUPS):
        start, end = group * 5, (group + 1) * 5 if group < AGE_GROUPS - 1 else AGES
        label = f"{start}-{end - 1}" if group < AGE_GROUPS - 1 else f"{start}+"
        checks.append(
            share_check(
                f"age group {label}",
                ((age >= start) & (age < end)).mean(),
                count,
                shares[start:end].sum(),
                AGE_GROUP_MARGIN,
                POPULATION_SOURCE,
                alpha=ALPHA / AGE_GROUPS,
            )
        )
    checks.append(
        share_check(
            "age 90 and over",
            (age >= OLDEST_FROM).mean(),
            count,
            shares[OLDEST_FROM:].sum(),
            AGE_GROUP_MARGIN,
            POPULATION_SOURCE,
        )
    )
    source = reference["sex"]["source"]
    checks.append(
        share_check(
            "men", (sex == MALE).mean(), count, reference["sex"]["male_share"], 0.01, source
        )
    )
    seniors = age >= 65
    checks.append(
        share_check(
            "women of those over 65",
            (sex[seniors] == FEMALE).mean(),
            int(seniors.sum()),
            reference["sex"]["women_share_from_65"],
            0.02,
            source,
        )
    )
    return checks


def _household_checks(people: Population, reference: dict) -> list[Check]:
    """Households are the people of a dwelling; their sizes are a target of the fit."""
    private = people.household[~people.resident]
    sizes = np.bincount(private)[:-1]  # the last household may be cut
    households = reference["households"]
    source = households["source"]
    checks = [
        mean_check(
            "mean household size", sizes.astype(float), households["mean_size"], 0.05, source
        )
    ]
    largest = households["sizes"][-1]
    for size, expected in zip(households["sizes"], households["size_shares"], strict=True):
        mask = sizes >= size if size == largest else sizes == size
        name = f"{size} or more persons" if size == largest else f"{size}-person households"
        margin = max(0.05 * expected, 0.004)
        checks.append(share_check(name, mask.mean(), len(sizes), expected, margin, source))
    return checks


def _family_checks(people: Population, reference: dict) -> list[Check]:
    """Living arrangements. Only those alone at 55-64 and over 65 are shares of the fit's kinds."""
    age, household = people.age, people.household
    in_household = np.bincount(household)[household]
    alone = (in_household == 1) & ~people.resident
    table, source = reference["living_alone"], reference["living_alone"]["source"]
    checks = []
    for name, mask, expected in (
        ("alone 18-24", (age >= 18) & (age < 25), table["age_18_24"]),
        ("alone 25-49", (age >= 25) & (age < 50), table["age_25_49"]),
        ("alone 55-64", (age >= 55) & (age < 65), table["age_55_64"]),
        ("alone 65+", age >= 65, table["from_65"]),
    ):
        checks.append(
            share_check(name, alone[mask].mean(), int(mask.sum()), expected, 0.03, source)
        )
    oldest = np.zeros(int(household.max()) + 1, np.int16)
    np.maximum.at(oldest, household, age)
    adults = np.bincount(household, weights=(age >= 18).astype(float))
    table, source = reference["family"], reference["family"]["source"]
    young = (age >= 18) & (age < 25)
    children = age < 18
    checks += [
        share_check(
            "with parents 18-24",
            (oldest[household] - age >= PARENT_GAP)[young].mean(),
            int(young.sum()),
            table["young_with_parents_18_24"],
            0.08,
            source,
        ),
        share_check(
            "children with one adult",
            (adults[household] == 1)[children].mean(),
            int(children.sum()),
            table["children_with_one_parent"],
            0.03,
            source,
        ),
    ]
    return checks + _mother_checks(people, reference)


def _mother_checks(people: Population, reference: dict) -> list[Check]:
    """How old the mothers of infants are, against the ages of mothers at birth.

    The mother is the oldest woman of a childbearing age: a younger one is a sister of the child.
    """
    age, household = people.age, people.household
    women = (people.sex == FEMALE) & (age >= 15) & (age < 50) & ~people.resident
    mother = np.zeros(int(household.max()) + 1, np.int16)
    np.maximum.at(mother, household[women], age[women])
    infants = age < INFANT_YEARS
    of_infants = (mother[household[infants]] - age[infants])[mother[household[infants]] > 0]
    table, source = reference["births"], reference["births"]["source"]
    return [
        share_check(name, mask.mean(), len(of_infants), expected, 0.05, source)
        for name, mask, expected in (
            ("mothers under 25", of_infants < 25, table["mother_under_25"]),
            (
                "mothers 25-34",
                (of_infants >= 25) & (of_infants < 35),
                table["mother_25_34"],
            ),
            ("mothers 35 and over", of_infants >= 35, table["mother_from_35"]),
        )
    ]


def _activity_checks(people, role, params: TownParameters, reference: dict) -> list[Check]:
    age, sex = people.age, people.sex
    employed = np.isin(role, (WORKER, REMOTE))
    in_education = np.isin(role, (PRESCHOOL, SCHOOL, UNIVERSITY))
    table, source = reference["employment"], reference["employment"]["source"]
    checks = []
    for name, mask, expected in (
        ("employment 15-64", (age >= 15) & (age < 65), table["age_15_64"]),
        ("employment 15-24", (age >= 15) & (age < 25), table["age_15_24"]),
        ("employment 25-49", (age >= 25) & (age < 50), table["age_25_49"]),
        ("employment 50-64", (age >= 50) & (age < 65), table["age_50_64"]),
        ("employment men 15-64", (age >= 15) & (age < 65) & (sex == MALE), table["men_15_64"]),
        (
            "employment women 15-64",
            (age >= 15) & (age < 65) & (sex == FEMALE),
            table["women_15_64"],
        ),
    ):
        checks.append(
            share_check(name, employed[mask].mean(), int(mask.sum()), expected, 0.02, source)
        )
    youth, source = reference["youth"], reference["youth"]["source"]
    for name, low, high, expected in (
        ("neither work nor study 15-29", 15, 30, youth["neet_15_29"]),
        ("neither work nor study 20-24", 20, 25, youth["neet_20_24"]),
    ):
        band = (age >= low) & (age < high)
        idle = ~employed & ~in_education
        checks.append(share_check(name, idle[band].mean(), int(band.sum()), expected, 0.03, source))
    youngest = np.full(int(people.household.max()) + 1, 200, np.int16)
    np.minimum.at(youngest, people.household, age)
    mothers = (
        (sex == FEMALE)
        & (age >= 25)
        & (age < 50)
        & (youngest[people.household] < 6)
        & (age - youngest[people.household] >= params.population.parent_gap_min)
    )
    checks.append(
        share_check(
            "mothers of under-6s at work",
            employed[mothers].mean(),
            int(mothers.sum()),
            reference["mothers"]["youngest_under_6"],
            0.04,
            reference["mothers"]["source"],
        )
    )
    care = reference["care_homes"]
    residents = role == RESIDENT
    checks += [
        share_check(
            "people in care homes",
            residents.mean(),
            len(age),
            care["share_of_population"],
            0.0005,
            care["source"],
        ),
        share_check(
            "women of the residents",
            (sex[residents] == FEMALE).mean(),
            int(residents.sum()),
            care["women_share"],
            0.05,
            care["source"],
        ),
    ]
    return checks


TOWNS_FOR_PLACES = 4  # towns whose places are pooled (see `validate_town`)


def validate_town(params: TownParameters, population: int = 60_000, seed: int = 0) -> list[Check]:
    """Compare the places of a town large enough to have all of them with published data: the
    workplaces by the size of the local unit, the care homes."""
    reference = read_reference("czechia_reference.toml")
    units = reference["local_units"]
    sizes = []
    # the shares of the employees in the few large local units differ by about two percentage
    # points from one town to the next (measured, six seeds), as much as the margin: a town is
    # not enough, the shares are those of several towns
    for offset in range(TOWNS_FOR_PLACES):
        town = build_town(
            population,
            2000.0 * math.sqrt(population / 3000),
            params,
            np.random.default_rng(seed + offset),
        )
        in_unit = (town.role == WORKER) & (town.place_kind == WORKER) & ~town.activity_staff
        in_unit &= town.work_people > 1
        sizes.append(town.work_people[in_unit])
    size = np.concatenate(sizes)
    checks = [
        share_check(name, mask.mean(), len(size), units[key], 0.03, units["source"])
        for name, mask, key in (
            ("employees in local units of 1-10", size <= 10, "up_to_10"),
            ("employees in local units of 11-49", (size > 10) & (size < 50), "from_11_to_49"),
            ("employees in local units of 50+", size >= 50, "from_50"),
        )
    ]
    return checks


# The week the contacts are measured: a week of the school term in which the universities teach
# (their winter term begins in the second half of September).
VALIDATION_MONDAY = date(2018, 10, 1)
CONTACT_DISTANCE = 2.0  # the distance at which droplet and aerosol diseases pass (m)
NEAR_DISTANCE = 1.0  # within it a contact counts fully (see the diseases' near_distance)
SETTINGS = ("home", "school", "work", "other")
LONG = {"1h": 2, "4h": 8}  # contacts lasting 1 hour or more, 4 hours or more, in half-hour steps
# Stehle et al. 2011 (PLoS ONE 6:e23176) measured face-to-face contact within 1-1.5 m in a primary
# school: 3.2 times as much time with classmates as with children of other classes, 2.4 to 5.3 in
# the ten classes (Table 3). Grades 1-5 are ages 6 to 10.
SCHOOL_MIXING = {"distance": 1.5, "ages": (6, 11), "range": (2.4, 5.3), "value": 3.2}


@dataclass
class Week:
    """What a week of the town shows (see `observe_week`): per person and day, by the place as
    the person would report it, the different people within 2 m (`count`) and those within 2 m
    for 1 hour or more and 4 hours or more, the exposure (hours times the proximity weight); the
    share of the pairs at work and at school met again the next weekday; the time the pairs of a
    household spend within 2 m at home; who meets whom by age (contact matrices); the time young
    pupils spend near classmates and near others."""

    age: np.ndarray
    role: np.ndarray
    contacts: dict[str, np.ndarray]
    repeated: dict[str, float]
    household_hours: np.ndarray
    household_pairs: int
    together: np.ndarray
    matrices: dict[str, np.ndarray]
    school_mixing: float


_WEEKS: dict[tuple, Week] = {}


def _validation_parameters(params: TownParameters) -> TownParameters:
    calendar = replace(params.schedule.calendar, first_day=VALIDATION_MONDAY)
    return replace(params, schedule=replace(params.schedule, calendar=calendar))


def observe_week(params: TownParameters, population: int = 6000, seed: int = 0) -> Week:
    """Run the town without diseases for a week of October 2018 and record its contacts every
    half hour (kept for later calls with the same parameters)."""
    key = (params.fingerprint, id(params), population, seed)
    if key not in _WEEKS:
        _WEEKS[key] = _observe(_validation_parameters(params), population, seed)
    return _WEEKS[key]


def _observe(params: TownParameters, population: int, seed: int) -> Week:
    from epidemic_sandbox.config.experiment import (
        SQUARE_METERS_PER_SQUARE_KILOMETER,
        STEPS_PER_DAY,
        ExperimentConfig,
    )
    from epidemic_sandbox.engine.epidemic_model import EpidemicModel

    arena = 2000.0 * math.sqrt(population / 3000)
    world = {
        "population": population,
        "area_km2": arena**2 / SQUARE_METERS_PER_SQUARE_KILOMETER,
        "arena_size": arena,
        "seed": seed,
        "steps_per_day": STEPS_PER_DAY,
        "max_days": 0,
        "runs": 1,
    }
    model = EpidemicModel(ExperimentConfig(world, {}), params)
    town = model.town
    group = np.minimum(town.age // CONTACT_GROUP_WIDTH, CONTACT_AGE_GROUPS - 1)
    contacts = {
        name: np.zeros((len(SETTINGS), population)) for name in ("count", "1h", "4h", "exposure")
    }
    matrices = {layer: np.zeros((CONTACT_AGE_GROUPS,) * 2) for layer in LAYERS}
    pairs_by_day = {"school": [], "work": []}
    household_hours = []
    together = np.zeros(population)
    mixing = np.zeros(2)
    pupil = np.isin(town.role, (PRESCHOOL, SCHOOL, UNIVERSITY))
    young = pupil & (town.age >= SCHOOL_MIXING["ages"][0]) & (town.age < SCHOOL_MIXING["ages"][1])
    for day in range(DAYS_PER_WEEK):
        records = []
        for _step in range(STEPS_PER_DAY):
            model.step()
            setting = _settings(model, pupil)
            together += _together(model) * 0.5 / DAYS_PER_WEEK
            first, second = _pairs(model, CONTACT_DISTANCE, model.venue >= 0)
            distance = np.hypot(model.x[first] - model.x[second], model.y[first] - model.y[second])
            one, other = np.concatenate([first, second]), np.concatenate([second, first])
            near = np.concatenate([distance, distance])
            weight = np.minimum(1.0, (NEAR_DISTANCE / np.maximum(near, 1e-9)) ** 2)
            records.append((one, other, setting[one], weight))
            if day < WEEKEND_FROM_DAY:
                both = pupil[first] & pupil[second] & (model._code[first] == WORK)
                both &= model._code[second] == WORK
                close = both & (distance <= SCHOOL_MIXING["distance"])
                ends = young[first[close]].astype(int) + young[second[close]].astype(int)
                classmates = town.work_venue[first[close]] == town.work_venue[second[close]]
                mixing += [(ends * classmates).sum(), (ends * ~classmates).sum()]
        columns = zip(*records, strict=True)
        one, other, setting, weight = (np.concatenate(values) for values in columns)
        key = (one.astype(np.int64) * population + other) * len(SETTINGS) + setting
        unique, inverse, steps = np.unique(key, return_inverse=True, return_counts=True)
        exposure = np.bincount(inverse, weights=weight) * 0.5  # hours at the proximity weight
        place = unique % len(SETTINGS)
        person = (unique // len(SETTINGS)) // population
        partner = (unique // len(SETTINGS)) % population
        for name, chosen in (
            ("count", steps >= 1),
            ("1h", steps >= LONG["1h"]),
            ("4h", steps >= LONG["4h"]),
        ):
            np.add.at(contacts[name], (place[chosen], person[chosen]), 1.0 / DAYS_PER_WEEK)
        np.add.at(contacts["exposure"], (place, person), exposure / DAYS_PER_WEEK)
        for index, layer in enumerate(LAYERS):
            chosen = place == index
            np.add.at(
                matrices[layer], (group[person[chosen]], group[partner[chosen]]), 1 / DAYS_PER_WEEK
            )
        at_home = (place == SETTINGS.index("home")) & (
            town.household[person] == town.household[partner]
        )
        household_hours.append(steps[at_home] * 0.5)
        if day < WEEKEND_FROM_DAY:  # the pairs of a place together for an hour or more
            for name in pairs_by_day:
                chosen = (place == SETTINGS.index(name)) & (steps >= LONG["1h"])
                pairs_by_day[name].append(set(unique[chosen] // len(SETTINGS)))
    in_group = np.bincount(group, minlength=CONTACT_AGE_GROUPS)
    repeated = {}
    for name, days in pairs_by_day.items():
        again = [len(days[d] & days[d - 1]) / max(len(days[d]), 1) for d in range(1, len(days))]
        repeated[name] = float(np.mean(again))
    private = town.role != RESIDENT
    size = np.bincount(town.household)
    return Week(
        town.age,
        town.role,
        contacts,
        repeated,
        np.concatenate(household_hours) if household_hours else np.zeros(0),
        int((size[town.household[private]] - 1).sum()) * DAYS_PER_WEEK,
        together,
        {layer: matrix / np.maximum(in_group, 1)[:, None] for layer, matrix in matrices.items()},
        float(mixing[0] / max(mixing[1], 1)),
    )


def _together(model) -> np.ndarray:
    """Who is awake at home in the common room of the dwelling with another member of the
    household there."""
    town, dwelling = model.town, model.town_parameters.dwelling
    if dwelling is None or dwelling.asleep(model.hour):
        return np.zeros(model.population, np.bool_)
    home = (model._code == HOME) & (model.venue == town.household) & (town.role != RESIDENT)
    distance = np.hypot(model.x - town.home_center_x, model.y - town.home_center_y)
    common = home & (distance <= town.home_radius + 1e-6)
    present = np.bincount(town.household, weights=common, minlength=int(town.household.max()) + 1)
    return common & (present[town.household] >= 2)


def _settings(model, pupil: np.ndarray) -> np.ndarray:
    """The place of everybody as they would report it in a contact diary: home (also a resident
    of a care home and the host of a visitor), school (a pupil at school), work (at work, also a
    teacher), other (errands, leisure, lunch, a visit to somebody's home)."""
    code = model._code.copy()
    code[model._detour[0]] = HOME
    code[model._detour[1]] = WORK
    return np.select(
        [code == HOME, (code == WORK) & pupil, code == WORK], [0, 1, 2], len(SETTINGS) - 1
    )


def _pairs(model, radius: float, member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arena = float(model.world["arena_size"])
    cells = max(1, min(int(arena // radius), 2 * math.isqrt(model.population)))
    grid = build_grid(model.x, model.y, arena, cells)
    return find_pairs(model.x, model.y, member, model.venue, *grid, cells, radius)


def read_polymod_durations() -> list[dict]:
    """Contacts per day lasting 1 hour or more and 4 hours or more by place, and the share of the
    contacts met daily, in Germany and Poland (POLYMOD), with their standard errors; see
    data/reference/derivations/polymod_durations.py."""
    with (REFERENCE_DIRECTORY / "polymod_durations_de_pl.csv").open(newline="") as table:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(table)]


# The age groups compared (the survey's bands joined so that the means are stable): every place for
# children, adults and the old, school for pupils, work for the ages of work.
DURATION_GROUPS = {
    "home": ((0, 15), (15, 65), (65, 120)),
    "school": ((5, 20),),
    "work": ((20, 60),),
    "other": ((0, 20), (20, 70)),
}
# A simulated mean passes when it is within 35 % of the survey's or within two standard errors of
# it, whichever is wider: the survey counts conversations and touch at any distance, the town
# everybody within 2 m, which is close to them for long contacts but not the same.
DURATION_TOLERANCE = 0.35


def validate_contact_durations(
    params: TownParameters, population: int = 6000, seed: int = 0
) -> list[Check]:
    """Long contacts by place and age, within 2 m, against the contact survey (POLYMOD, Germany
    and Poland): per person and day, contacts that lasted 1 hour or more and 4 hours or more."""
    week = observe_week(params, population, seed)
    survey = read_polymod_durations()
    checks = []
    for place, groups in DURATION_GROUPS.items():
        index = SETTINGS.index(place)
        for low, high in groups:
            rows = [row for row in survey if row["age_from"] >= low and row["age_to"] <= high]
            weights = np.array([row["participants"] for row in rows])
            for kind in ("1h", "4h"):
                values = np.array([row[f"{place}_{kind}"] for row in rows])
                errors = np.array([row[f"{place}_{kind}_se"] for row in rows])
                expected = float((weights * values).sum() / weights.sum())
                error = float(np.sqrt((weights**2 * errors**2).sum()) / weights.sum())
                band = (week.age >= low) & (week.age < high)
                observed = float(week.contacts[kind][index, band].mean())
                margin = max(DURATION_TOLERANCE * expected, 2 * error)
                checks.append(
                    Check(
                        f"{place} {kind}+ contacts {low}-{high - 1}",
                        observed,
                        f"{expected:.2f} ± {margin:.2f}",
                        math.nan,
                        abs(observed - expected) <= margin,
                        "POLYMOD, Germany and Poland (polymod_durations_de_pl.csv)",
                    )
                )
    return checks


# Contacts met again: the share of the contacts at work and at school lasting 1 hour or more that
# the survey's participants meet daily or almost daily (polymod_durations_de_pl.csv, the adults'
# and the pupils' bands), against the share of the pairs within 2 m for an hour or more on a
# weekday who were so the weekday before.
REPEAT_GROUPS = {"work": (20, 60), "school": (5, 20)}
REPEAT_MARGIN = 0.20


def validate_repetition(params: TownParameters, population: int = 6000, seed: int = 0) -> list:
    """Whether people meet the same people at work and at school day after day."""
    week = observe_week(params, population, seed)
    survey = read_polymod_durations()
    checks = []
    for place, (low, high) in REPEAT_GROUPS.items():
        rows = [row for row in survey if row["age_from"] >= low and row["age_to"] <= high]
        weights = np.array([row["participants"] * row[f"{place}_1h"] for row in rows])
        shares = np.array([row[f"{place}_daily_share"] for row in rows])
        expected = float((weights * shares).sum() / weights.sum())
        observed = week.repeated[place]
        checks.append(
            Check(
                f"{place} contacts met again",
                observed,
                f"{expected:.2f} ± {REPEAT_MARGIN:.2f}",
                math.nan,
                abs(observed - expected) <= REPEAT_MARGIN,
                "POLYMOD, Germany and Poland (polymod_durations_de_pl.csv)",
            )
        )
    return checks


def validate_household_proximity(
    params: TownParameters, population: int = 6000, seed: int = 0
) -> list[Check]:
    """How long the people of a household are close at home, against the survey: of the daily
    contacts at home (the people one lives with) of adults, 63 % lasted more than 4 hours and 21 %
    1-4 hours (POLYMOD, Germany and Poland; czechia_reference.toml [household_contacts])."""
    week = observe_week(params, population, seed)
    table = read_reference("czechia_reference.toml")["household_contacts"]
    hours = week.household_hours
    pairs = max(week.household_pairs, 1)
    checks = []
    for name, observed, key in (
        ("household pairs close over 4 h", (hours > 4).sum() / pairs, "over_4_hours"),
        ("household pairs close 1-4 h", ((hours >= 1) & (hours <= 4)).sum() / pairs, "hours_1_4"),
    ):
        checks.append(
            Check(
                name,
                float(observed),
                f"{table[key]:.2f} ± {table['margin']:.2f}",
                math.nan,
                abs(observed - table[key]) <= table["margin"],
                table["source"],
            )
        )
    return checks


def validate_time_together(
    params: TownParameters, population: int = 6000, seed: int = 0
) -> list[Check]:
    """Hours a day awake with the members of one's household, against the time-use survey
    (czechia_reference.toml [time_together]); the town counts the time at home only."""
    week = observe_week(params, population, seed)
    table = read_reference("czechia_reference.toml")["time_together"]
    private = week.role != RESIDENT
    checks = []
    for (low, high), expected in zip(table["age_groups"], table["hours"], strict=True):
        band = (week.age >= low) & (week.age < high) & private
        observed = float(week.together[band].mean())
        lowest, highest = table["lowest"] * expected, table["highest"] * expected
        checks.append(
            Check(
                f"hours with the household {low}-{high - 1}",
                observed,
                f"{expected:.2f} ({lowest:.2f} to {highest:.2f})",
                math.nan,
                lowest <= observed <= highest,
                table["source"],
            )
        )
    return checks


def read_reference_contacts() -> dict[str, np.ndarray]:
    """The published contact matrices of Czechia, rows are the age group of the person."""
    matrices = {layer: np.zeros((CONTACT_AGE_GROUPS,) * 2) for layer in LAYERS}
    with (REFERENCE_DIRECTORY / "czechia_contacts_prem2021.csv").open(newline="") as table:
        for row in csv.DictReader(table):
            if row["location"] in matrices:
                person = int(row["age_group"].split()[0].rstrip("+"))
                other = int(row["contact_age_group"].split()[0].rstrip("+"))
                matrices[row["location"]][person // 5, other // 5] = float(
                    row["mean_contacts_per_day"]
                )
    return matrices


def validate_contacts(
    params: TownParameters, population: int = 6000, radius: float = CONTACT_DISTANCE, seed: int = 0
) -> list[Check]:
    """Compare who meets whom, by age and kind of place, with the published contact matrices
    (their structure: rank correlation of all cells), the contacts within 2 m of a week counted
    as each person would report them."""
    week = observe_week(params, population, seed)
    reference = read_reference_contacts()
    rng = np.random.default_rng(seed)
    source = "Prem et al. 2021 (czechia_contacts_prem2021.csv)"
    checks = []
    for layer in LAYERS:
        correlation, p_value = permutation_test(week.matrices[layer], reference[layer], rng)
        passed = correlation >= MIN_CORRELATION and p_value < PERMUTATION_ALPHA
        checks.append(
            Check(
                f"contact matrix {layer}",
                correlation,
                f"Spearman ≥ {MIN_CORRELATION}, p < {PERMUTATION_ALPHA}",
                p_value,
                passed,
                source,
            )
        )
    return checks


def validate_school_mixing(
    params: TownParameters, population: int = 6000, seed: int = 0
) -> list[Check]:
    """Time young pupils spend with classmates against time with pupils of other classes."""
    ratio = observe_week(params, population, seed).school_mixing
    low, high = SCHOOL_MIXING["range"]
    return [
        Check(
            "school: classmates to others",
            float(ratio),
            f"{SCHOOL_MIXING['value']} ({low} to {high})",
            math.nan,
            low <= ratio <= high,
            "Stehle et al. 2011, PLoS ONE 6:e23176, Table 3",
        )
    ]


def at_home_shares(
    params: TownParameters, hour: float, groups, population: int = 20_000, seed: int = 0
) -> np.ndarray:
    """The share of the people of each age group ([from, to) years) whose plan puts them at home
    at `hour` of the ordinary weekdays of a year (a hook for the national travel survey, see
    `validate_whereabouts`)."""
    town = build_town(
        population, 2000.0 * math.sqrt(population / 3000), params, np.random.default_rng(seed)
    )
    calendar = params.schedule.calendar
    rng = np.random.default_rng(seed + 1)
    shares = np.zeros(len(groups))
    days = [day for day in range(0, 364, 7) if not calendar.is_day_off(day)]
    for day in days:
        plan = draw_day_schedule(
            town.role, town.traits, params.schedule, False, rng, calendar.closed_places(day)
        )
        home = destination_codes_at(plan, hour, params.schedule.max_events) == HOME
        for index, (low, high) in enumerate(groups):
            band = (town.age >= low) & (town.age < high) & (town.role != RESIDENT)
            shares[index] += home[band].mean() / len(days)
    return shares


def validate_whereabouts(params: TownParameters, population: int = 20_000, seed: int = 0):
    """People at home at 10 a.m. of a workday by age, against the national travel survey (Cesko v
    pohybu, czechia_reference.toml [whereabouts]); when people go out is set by the schedules."""
    table = read_reference("czechia_reference.toml")["whereabouts"]
    groups = [tuple(group) for group in table["age_groups"]]
    shares = at_home_shares(params, table["hour"], groups, population, seed)
    return [
        Check(
            f"at home at {table['hour']:g} h, {low}-{high - 1}",
            float(observed),
            f"{expected:.2f} ± {table['margin']:.2f}",
            math.nan,
            abs(observed - expected) <= table["margin"],
            table["source"],
        )
        for (low, high), observed, expected in zip(groups, shares, table["at_home"], strict=True)
    ]


def format_report(checks: list[Check]) -> str:
    lines = [f"{'check':36s} {'observed':>9s}  {'expected':24s} {'p':>8s}  result"]
    for check in checks:
        verdict = "ok" if check.passed else "FAIL"
        lines.append(
            f"{check.name:36s} {check.observed:9.3f}  {check.expected:24s} "
            f"{check.p_value:8.4f}  {verdict}"
        )
    return "\n".join(lines)


def _read_population() -> np.ndarray:
    """Reference people by year of age and sex."""
    return read_people(REFERENCE_DIRECTORY / "czechia_population_2019.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the simulated population against published data"
    )
    parser.add_argument("--town", type=Path, default=DEFAULT_TOWN_PATH)
    parser.add_argument("--population", type=int, default=200_000)
    parser.add_argument("--contact-population", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    params = load_town_parameters(arguments.town)
    people, seed = arguments.contact_population, arguments.seed
    checks = validate_population(params, arguments.population, seed)
    checks += validate_town(params, seed=seed)
    checks += validate_contact_durations(params, people, seed)
    checks += validate_repetition(params, people, seed)
    checks += validate_household_proximity(params, people, seed)
    checks += validate_time_together(params, people, seed)
    checks += validate_contacts(params, people, seed=seed)
    checks += validate_school_mixing(params, people, seed)
    checks += validate_whereabouts(params, seed=seed)
    print(format_report(checks))
    failed = [check.name for check in checks if not check.passed]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
