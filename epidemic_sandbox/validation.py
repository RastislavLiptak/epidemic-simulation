"""Validation of the simulated population against published data.

Two kinds of tests, both against files in `data/reference/`:

* equivalence tests (TOST) say whether a simulated share or mean lies within an accepted margin
  of the published value; the margin reflects how much two official sources differ,
* permutation tests say whether the age structure of simulated contacts (who meets whom, by age)
  agrees with the published contact matrices more than chance alignment of age groups would.
"""

import argparse
import csv
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epidemic_sandbox.population import (
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
from epidemic_sandbox.schedule import (
    HOME,
    LUNCH,
    WORK,
    destination_codes_at,
    draw_day_schedule,
)
from epidemic_sandbox.spatial_kernels import build_grid, find_pairs
from epidemic_sandbox.town import (
    DAYS_PER_WEEK,
    HOURS_PER_DAY,
    WEEKEND_FROM_DAY,
    TownParameters,
    build_town,
    destination_coordinates,
    destination_venues,
    load_town_parameters,
)

REFERENCE_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "reference"
DEFAULT_TOWN_PATH = Path(__file__).resolve().parent.parent / "configs" / "town.toml"
ALPHA = 0.05
CONTACT_AGE_GROUPS = 16
CONTACT_GROUP_WIDTH = 5
LAYERS = ("home", "school", "work", "others")
PERMUTATIONS = 5000
VENUE_RADIUS = 10.0
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
    private = people.household[~people.resident]
    sizes = np.bincount(private)[:-1]  # the last household may be cut
    households = reference["households"]
    source = households["source"]
    checks = [
        mean_check("mean household size", sizes.astype(float), households["mean_size"], 0.1, source)
    ]
    largest = households["sizes"][-1]
    for size, expected in zip(households["sizes"], households["size_shares"], strict=True):
        mask = sizes >= size if size == largest else sizes == size
        name = f"{size} or more persons" if size == largest else f"{size}-person households"
        margin = max(0.15 * expected, 0.004)
        checks.append(share_check(name, mask.mean(), len(sizes), expected, margin, source))
    return checks


def _family_checks(people: Population, reference: dict) -> list[Check]:
    """Living arrangements. Only those alone at 55-64 and over 65 are shares of the fit's kinds."""
    age, sex, household = people.age, people.sex, people.household
    in_household = np.bincount(household)[household]
    alone = (in_household == 1) & ~people.resident
    table, source = reference["living_alone"], reference["living_alone"]["source"]
    checks = []
    for name, mask, expected in (
        ("alone 18-24", (age >= 18) & (age < 25), table["age_18_24"]),
        ("alone 25-49", (age >= 25) & (age < 50), table["age_25_49"]),
        ("alone 55-64", (age >= 55) & (age < 65), table["age_55_64"]),
        ("alone 65+", age >= 65, table["from_65"]),
        ("alone 65+, men", (age >= 65) & (sex == MALE), table["men_from_65"]),
        ("alone 65+, women", (age >= 65) & (sex == FEMALE), table["women_from_65"]),
    ):
        checks.append(
            share_check(name, alone[mask].mean(), int(mask.sum()), expected, 0.04, source)
        )
    oldest = np.zeros(int(household.max()) + 1, np.int16)
    np.maximum.at(oldest, household, age)
    adults = np.bincount(household, weights=(age >= 18).astype(float))
    table, source = reference["family"], reference["family"]["source"]
    young = (age >= 18) & (age < 25)
    children = age < 18
    with_children = np.bincount(household[~people.resident], weights=children[~people.resident]) > 0
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
        share_check(
            "households with a child",
            with_children.mean(),
            len(with_children),
            table["households_with_children"],
            0.04,
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
    checks.append(
        share_check(
            "people in care homes",
            (role == RESIDENT).mean(),
            len(age),
            reference["care_homes"]["share_of_population"],
            0.001,
            reference["care_homes"]["source"],
        )
    )
    return checks


def validate_town(params: TownParameters, population: int = 150_000, seed: int = 0) -> list[Check]:
    """Compare the places of a town large enough to have all of them with published data."""
    reference = read_reference("czechia_reference.toml")["enterprises"]
    town = build_town(
        population,
        2000.0 * math.sqrt(population / 3000),
        params,
        np.random.default_rng(seed),
    )
    in_firm = (town.role == WORKER) & (town.place_kind == WORKER) & ~town.activity_staff
    size = town.work_people[in_firm]
    return [
        share_check(
            "workers in firms under 10",
            (size < 10).mean(),
            len(size),
            reference["under_10"],
            0.08,
            reference["source"],
        ),
        share_check(
            "workers in firms from 250",
            (size >= 250).mean(),
            len(size),
            reference["from_250"],
            0.10,
            reference["source"],
        ),
    ]


def simulated_contact_matrices(
    params: TownParameters, population: int, arena: float, radius: float, rng
) -> dict[str, np.ndarray]:
    """Mean number of distinct people of each age group within `radius` metres of a person per day.

    Everybody is placed at the destination of their schedule for every hour of a weekday and of a
    weekend day; a contact is attributed to the kind of place both people are at.
    """
    town = build_town(population, arena, params, rng)
    group = np.minimum(town.age // CONTACT_GROUP_WIDTH, CONTACT_AGE_GROUPS - 1)
    in_group = np.bincount(group, minlength=CONTACT_AGE_GROUPS)
    education = town.at_school
    people = np.arange(population)
    cells = max(1, min(int(arena // radius), 2 * math.isqrt(population)))
    matrices = {layer: np.zeros((CONTACT_AGE_GROUPS,) * 2) for layer in LAYERS}
    for weekend, weight in ((False, WEEKEND_FROM_DAY), (True, DAYS_PER_WEEK - WEEKEND_FROM_DAY)):
        schedule = draw_day_schedule(town.role, town.traits, params.schedule, weekend, rng)
        pairs = {layer: [] for layer in LAYERS}
        for hour in range(HOURS_PER_DAY):
            codes = destination_codes_at(schedule, hour + 0.5, params.schedule.max_events)
            x, y = destination_coordinates(town, codes, people, schedule)
            venue = destination_venues(town, codes, people, schedule)
            layer_of = np.select(
                [
                    codes == HOME,
                    (codes == WORK) & education,
                    codes == WORK,
                    codes >= LUNCH,
                ],
                [0, 1, 2, 3],
            )
            grid = build_grid(x, y, arena, cells)
            for index, layer in enumerate(LAYERS):
                first, second = find_pairs(x, y, layer_of == index, venue, *grid, cells, radius)
                pairs[layer].append(first.astype(np.int64) * population + second)
        for layer in LAYERS:
            unique = np.unique(np.concatenate(pairs[layer]))
            first, second = unique // population, unique % population
            np.add.at(matrices[layer], (group[first], group[second]), weight / DAYS_PER_WEEK)
            np.add.at(matrices[layer], (group[second], group[first]), weight / DAYS_PER_WEEK)
    return {layer: matrix / np.maximum(in_group, 1)[:, None] for layer, matrix in matrices.items()}


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
    params: TownParameters, population: int = 15000, radius: float = VENUE_RADIUS, seed: int = 0
) -> list[Check]:
    """Compare who meets whom, by age and kind of place, with the published contact matrices.

    The published matrices count conversations in a venue, so the simulated people are counted
    within `radius` (venue scale), not within an infection distance of a couple of metres. Only
    the structure is compared (rank correlation of all cells); the levels differ by definition.
    """
    rng = np.random.default_rng(seed)
    arena = 2000.0 * math.sqrt(population / 3000)
    simulated = simulated_contact_matrices(params, population, arena, radius, rng)
    reference = read_reference_contacts()
    source = "Prem et al. 2021 (czechia_contacts_prem2021.csv)"
    checks = []
    for layer in LAYERS:
        correlation, p_value = permutation_test(simulated[layer], reference[layer], rng)
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


# Conversations happen within the personal and social distance of people, up to about 3.7 m
# (Hall 1966, The Hidden Dimension: social distance 1.2-3.7 m); the contact surveys count a
# conversation in the presence of somebody without a distance.
CONVERSATION_DISTANCE = (1.2, 3.7)
# Stehle et al. 2011 (PLoS ONE 6:e23176) measured face-to-face contact within 1-1.5 m in a primary
# school: 3.2 times as much time with classmates as with children of other classes, 2.4 to 5.3 in
# the ten classes (Table 3). Grades 1-5 are ages 6 to 10.
SCHOOL_MIXING = {"distance": 1.5, "ages": (6, 11), "range": (2.4, 5.3), "value": 3.2}


def _town_week(params: TownParameters, population: int, arena: float, seed: int, observe) -> None:
    """Run the town without diseases for a week, from a Monday of the school term, and call
    `observe(model, day)` after every step."""
    from epidemic_sandbox.epidemic_model import EpidemicModel
    from epidemic_sandbox.experiment_config import STEPS_PER_DAY, ExperimentConfig

    world = {
        "population": population,
        "arena_area": arena**2,
        "arena_size": arena,
        "seed": seed,
        "steps_per_day": STEPS_PER_DAY,
        "max_days": 0,
        "runs": 1,
    }
    model = EpidemicModel(ExperimentConfig(world, {}), params)
    for day in range(DAYS_PER_WEEK):
        for _step in range(STEPS_PER_DAY):
            model.step()
            observe(model, day)


def _pairs(model, radius: float, member: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arena = float(model.world["arena_size"])
    cells = max(1, min(int(arena // radius), 2 * math.isqrt(model.population)))
    grid = build_grid(model.x, model.y, arena, cells)
    return find_pairs(model.x, model.y, member, model.venue, *grid, cells, radius)


def simulated_contact_levels(
    params: TownParameters, population: int, arena: float, radii, seed: int
) -> tuple[np.ndarray, dict]:
    """Mean number of different people a person is within each radius of per day, by the kind of
    place as the person would write it in a contact diary (home 0, school 1, work 2, other 3: a
    resident of a care home is at home, the teacher of a class at work, the pupil at school), over
    a week (5 weekdays and 2 weekend days, the 5:2 weighting); returns {radius: [place, person]}
    and the ages."""
    levels = {radius: np.zeros((len(LAYERS), population)) for radius in radii}
    pairs = {radius: [[] for _ in LAYERS] for radius in radii}
    ages = []

    def observe(model, day):
        town = model.town
        ages[:] = [town.age]
        code = model._code.copy()
        code[model._detour[0]] = HOME
        code[model._detour[1]] = WORK
        layer = np.select(
            [
                (code == HOME) & (model.venue == town.household),
                (code == WORK) & town.at_school & (town.place_kind == town.role),
                code == WORK,
            ],
            [0, 1, 2],
            3,
        )
        for radius in radii:
            first, second = _pairs(model, radius, model.venue >= 0)
            one, other = np.concatenate([first, second]), np.concatenate([second, first])
            for index in range(len(LAYERS)):
                chosen = layer[one] == index  # the contact as `one` sees it
                key = one[chosen].astype(np.int64) * population + other[chosen]
                pairs[radius][index].append(key)
        if model.steps % model.world["steps_per_day"] == 0:  # the end of a day
            for radius in radii:
                for index in range(len(LAYERS)):
                    unique = np.unique(np.concatenate(pairs[radius][index]))
                    levels[radius][index] += (
                        np.bincount(unique // population, minlength=population) / DAYS_PER_WEEK
                    )
                    pairs[radius][index].clear()

    _town_week(params, population, arena, seed, observe)
    return ages[0], levels


def read_polymod_levels() -> list[dict]:
    """Contacts per day by age band and place in Germany and Poland (POLYMOD), with their standard
    errors; see data/reference/derivations/polymod_contacts.py."""
    with (REFERENCE_DIRECTORY / "polymod_contacts_de_pl.csv").open(newline="") as table:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(table)]


def validate_contact_levels(
    params: TownParameters, population: int = 3000, seed: int = 0
) -> list[Check]:
    """How many different people the people of the town meet in a day, against the contact survey.

    The number of different people within the near end of conversation distance should not be
    above what the survey found, nor the number within the far end below it: the check passes when
    the range of the town overlaps the 95 % confidence interval of the survey's mean. All
    contacts: the survey of 2006 in Germany and Poland had other kindergarten attendance and
    employment of the old than Czechia in 2019, so the places are not compared one by one.
    """
    near, far = CONVERSATION_DISTANCE
    age, levels = simulated_contact_levels(
        params, population, 2000.0 * math.sqrt(population / 3000), (near, far), seed
    )
    checks = []
    for row in read_polymod_levels():
        band = (age >= row["age_from"]) & (age < row["age_to"])
        low, high = (float(levels[radius].sum(axis=0)[band].mean()) for radius in (near, far))
        margin = 1.96 * row["all_se"]
        checks.append(
            Check(
                f"contacts a day {int(row['age_from'])}-{int(row['age_to']) - 1}",
                row["all"],
                f"town {low:.1f} to {high:.1f} (survey ±{margin:.1f})",
                math.nan,
                low <= row["all"] + margin and row["all"] - margin <= high,
                "POLYMOD, Germany and Poland (polymod_contacts_de_pl.csv)",
            )
        )
    return checks


def validate_school_mixing(
    params: TownParameters, population: int = 6000, seed: int = 0
) -> list[Check]:
    """Time young pupils spend with classmates against time with pupils of other classes."""
    distance, (young_from, young_to) = SCHOOL_MIXING["distance"], SCHOOL_MIXING["ages"]
    counts = np.zeros(2)

    def observe(model, day):
        town = model.town
        if day >= WEEKEND_FROM_DAY:
            return
        pupil = town.role == SCHOOL
        first, second = _pairs(model, distance, pupil & (model._code == WORK) & (model.venue >= 0))
        young = (town.age >= young_from) & (town.age < young_to) & pupil
        ends = young[first].astype(int) + young[second].astype(int)
        classmates = town.work_venue[first] == town.work_venue[second]
        counts[0] += (ends * classmates).sum()
        counts[1] += (ends * ~classmates).sum()

    _town_week(params, population, 2000.0 * math.sqrt(population / 3000), seed, observe)
    ratio = counts[0] / max(counts[1], 1)
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


def format_report(checks: list[Check]) -> str:
    lines = [f"{'check':32s} {'observed':>9s}  {'expected':24s} {'p':>8s}  result"]
    for check in checks:
        verdict = "ok" if check.passed else "FAIL"
        lines.append(
            f"{check.name:32s} {check.observed:9.3f}  {check.expected:24s} "
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
    checks = validate_population(params, arguments.population, arguments.seed)
    checks += validate_town(params, seed=arguments.seed)
    checks += validate_contacts(params, arguments.contact_population, seed=arguments.seed)
    checks += validate_contact_levels(params, seed=arguments.seed)
    checks += validate_school_mixing(params, seed=arguments.seed)
    print(format_report(checks))
    failed = [check.name for check in checks if not check.passed]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
