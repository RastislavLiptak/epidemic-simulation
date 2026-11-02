"""Synthetic population: ages, sexes, households and what everybody does during the day.

Households are generated first (size, kind, age of the head, the other members from the kind),
as in synthetic population generators such as SynthPops. The households are then reweighted so
that the people by age and sex, the household sizes and the household kinds match the census.
People in care homes live outside private households and are made separately.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from epidemic_sandbox.config_checks import (
    require,
    require_not_negative,
    require_positive,
    require_share,
    require_weights,
)

ROLES = (
    "homebound",
    "preschool",
    "school",
    "university",
    "worker",
    "remote",
    "retired",
    "resident",
)
HOMEBOUND, PRESCHOOL, SCHOOL, UNIVERSITY, WORKER, REMOTE, RETIRED, RESIDENT = range(len(ROLES))
MALE, FEMALE = 0, 1
SEXES = ("male", "female")
# Numerical constants of the fit of the household weights, not facts about the world: the pool is
# large enough that its sampling error is below the validation margins, the iterations and steps
# are enough for the shares to settle, the limits keep one update or one weight from running away,
# and the seed makes the fit the same in every run.
FIT_POOL_HOUSEHOLDS = 300_000
FIT_ITERATIONS = 200
FIT_STEP = 0.3
FIT_MAX_STEP = 1.0
FIT_MAX_LOG_WEIGHT = 6.0
FIT_SEED = 12345
# The fit follows single years of age; the few people above this age share one cell, since a cell
# holding only a handful of people of the pool would be noise.
FIT_OPEN_AGE = 95
GENERATION_CHUNK = 400_000  # households made at once, a bound on memory
# Adulthood in Czechia begins at 18 (Civil Code); "other" households and partners are adults.
ADULT_AGE = 18
# The oldest age made; the table of the population collects everybody older in this year of age.
MAX_AGE = 99
AGES = MAX_AGE + 1
# The year of the population table: a mother's age tells when she was born, which sets when she
# began to have children.
REFERENCE_YEAR = 2019
# Model: a woman uses the first-birth age of the year in which her cohort reached this age. The
# published means are of the births of one year, and while women postpone children those lag behind
# the women who have them, so the year of the typical first birth of the cohort is taken.
COHORT_FIRST_BIRTH_AGE = 28
CHILD_AGE_ATTEMPTS = 8  # redraws of a family whose children do not fit
# Source: births to mothers under 15 are a handful a year (Eurostat demo_fasec).
MIN_FIRST_BIRTH_AGE = 15
# Assumption: nine months of pregnancy and a year of recovery; twins are not modelled.
MIN_SIBLING_SPACING = 1.5
# Source: births to mothers of 50 or more were 11 of 111,793 in 2021 (Eurostat demo_fasec), so a
# mother's last child is taken as born before she is 50.
LAST_BIRTH_AGE = 50
# Model: how many later or earlier children are considered beyond the ones living at home, because
# the oldest child at home is not always the first born (the older ones have left).
EXTRA_CHILDREN = 2
# Households above this percentile of weight are always kept; the bias from that is far below the
# validation margins and it avoids a very slow rejection sampling.
WEIGHT_CAP_PERCENTILE = 99.9
# Assumption: members of an "other" household (flatmates, siblings) are of similar age; 4 years is
# the spread around the head, no source.
OTHER_AGE_SPREAD = 4.0

KIND_SIZES = {
    "alone": lambda size: size == 1,
    "alone_older": lambda size: size == 1,
    "alone_senior": lambda size: size == 1,
    "couple": lambda size: size == 2,
    "parent_with_children": lambda size: size >= 2,
    "couple_with_children": lambda size: size >= 3,
    "parent_with_adult_children": lambda size: size >= 2,
    "couple_with_adult_children": lambda size: size >= 3,
    "multigenerational": lambda size: size >= 4,
    "other": lambda size: size >= 2,
}
CHILD_KINDS = (
    "parent_with_children",
    "couple_with_children",
    "parent_with_adult_children",
    "couple_with_adult_children",
)


@dataclass(frozen=True, eq=False)
class PopulationParameters:
    """People, households and roles, loaded from `configs/town.toml`.

    The education and employment tables hold a share per year of age and sex.
    """

    fingerprint: str
    age_group_width: int
    age_shares: np.ndarray
    people: np.ndarray
    household_sizes: np.ndarray
    household_weights: np.ndarray
    household_radius: float
    first_birth: np.ndarray
    sibling_spacing_mean: float
    sibling_spacing_sd: float
    parent_gap_min: float
    partner_gap_mean: float
    partner_gap_sd: float
    lone_father_share: float
    male_birth_share: float
    kinds: dict[int, tuple[tuple[str, ...], np.ndarray]]
    kind_names: tuple[str, ...]
    kind_shares: np.ndarray
    head_age: dict[str, tuple[int, np.ndarray]]
    child_ages: dict[str, tuple[int, int]]
    preschool: np.ndarray
    school: np.ndarray
    university: np.ndarray
    employment: np.ndarray
    employment_bands: np.ndarray
    remote_share: float
    retired_from: int
    mother_employment: np.ndarray
    mother_ages: tuple[int, int]
    care_home_rates: np.ndarray
    care_home_size: float

    @property
    def mean_household_size(self) -> float:
        return float((self.household_sizes * self.household_weights).sum())


@dataclass
class Population:
    """Age, sex and household of everybody; people of a household are next to each other.

    Care-home residents come last, every home being one household.
    """

    age: np.ndarray
    sex: np.ndarray
    household: np.ndarray
    resident: np.ndarray


def read_people(source: Path) -> np.ndarray:
    """Shares of people by year of age and sex from a table of age, male, female."""
    counts = np.zeros((AGES, len(SEXES)))
    with source.open(newline="") as table:
        for row in csv.DictReader(table):
            age = min(int(row["age"]), MAX_AGE)
            counts[age] += (float(row["male"]), float(row["female"]))
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("the people by age and sex must be non-negative and not all zero")
    return counts / counts.sum()


def group_ages(people: np.ndarray, group_width: int, open_from: int) -> np.ndarray:
    """Shares of people in age groups; the last group is open."""
    groups = open_from // group_width + 1
    return np.bincount(
        np.minimum(np.arange(AGES) // group_width, groups - 1),
        weights=people.sum(axis=1),
        minlength=groups,
    )


def _by_age_and_sex(rows, name: str) -> np.ndarray:
    """A table of rows [from age, to age, share of men, share of women] as a share per year."""
    shares = np.zeros((AGES, len(SEXES)))
    table = np.array(rows, dtype=float).reshape(-1, 4)
    if ((table[:, 0] >= table[:, 1]) | (table[:, 2:] < 0).any(axis=1)).any():
        raise ValueError(f"invalid age range or share in {name}")
    if (table[:, 2:] > 1).any():
        raise ValueError(f"invalid age range or share in {name}")
    for start, end, male, female in table:
        shares[int(start) : int(end)] = (male, female)
    return shares


def _load_kinds(households: dict, sizes: np.ndarray) -> dict:
    kinds = {}
    for size in sizes:
        names = tuple(households["kinds"][str(size)])
        weights = np.array([households["kinds"][str(size)][name] for name in names], float)
        unknown = [name for name in names if name not in KIND_SIZES]
        invalid = [name for name in names if name in KIND_SIZES and not KIND_SIZES[name](size)]
        if unknown or invalid or weights.sum() <= 0:
            raise ValueError(f"invalid household kinds {unknown + invalid} for size {size}")
        kinds[int(size)] = (names, weights / weights.sum())
    return kinds


def _check_households(households: dict) -> None:
    for name, table in households["head_age"].items():
        require_weights(f"head_age.{name}", table["weights"])
        require(table["start"] >= 0, f"head_age.{name}: start must not be negative")
    for key in ("radius", "sibling_spacing_mean", "sibling_spacing_sd"):
        require_positive(f"households.{key}", households[key])
    require_not_negative("households.partner_age_gap_sd", households["partner_age_gap_sd"])
    require_not_negative("households.parent_age_gap_min", households["parent_age_gap_min"])
    require_share("households.lone_father_share", households["lone_father_share"])
    require_share("households.male_birth_share", households["male_birth_share"])


def load_population_parameters(
    raw: dict, fingerprint: str, directory: Path
) -> PopulationParameters:
    """Build and validate the population parameters from the parsed town file."""
    age, households, employment = raw["age"], raw["households"], raw["employment"]
    sizes = np.array(households["sizes"], dtype=int)
    weights = np.array(households["weights"], dtype=float)
    require(
        age["group_width"] >= 1 and age["open_from"] >= age["group_width"], "invalid age groups"
    )
    people = read_people(directory / age["source"])
    if len(sizes) != len(weights) or (sizes < 1).any() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid household sizes or weights")
    kinds = _load_kinds(households, sizes)
    _check_households(households)
    head_age = {
        name: (table["start"], np.array(table["weights"], float) / sum(table["weights"]))
        for name, table in households["head_age"].items()
    }
    used = {name for names, _ in kinds.values() for name in names}
    missing = used - head_age.keys()
    if missing:
        raise ValueError(f"missing head_age for household kinds {sorted(missing)}")
    kind_names = tuple(sorted(used))
    unexplained = used ^ set(households["kind_shares"])
    if unexplained:
        raise ValueError(f"kind_shares must list the household kinds, see {sorted(unexplained)}")
    kind_shares = np.array([households["kind_shares"][name] for name in kind_names], float)
    require_weights("households.kind_shares", kind_shares)
    require_share("employment.remote_share", employment["remote_share"])
    low, high = employment["mother_ages"]
    require(0 <= low < high, "employment.mother_ages must be an increasing age range")
    tables = {
        name: _by_age_and_sex(raw["education"][name], name)
        for name in ("preschool", "school", "university")
    }
    if (sum(tables.values()) > 1.0 + 1e-9).any():
        raise ValueError("invalid age range or share: education shares exceed 1 at some age")
    mothers = np.array(employment["mother_employment"], float).reshape(-1, 2)
    if ((mothers[:, 1] < 0) | (mothers[:, 1] > 1)).any() or (np.diff(mothers[:, 0]) <= 0).any():
        raise ValueError("employment.mother_employment needs rising ages and shares of 0 to 1")
    care = raw["care_homes"]
    rates = _by_age_and_sex(
        [[start, end, rate, rate] for start, end, rate in care["rates"]], "care_homes.rates"
    )
    require_positive("care_homes.mean_size", care["mean_size"])
    rows = np.array(employment["rates"], dtype=float).reshape(-1, 4)
    return PopulationParameters(
        fingerprint=fingerprint,
        age_group_width=age["group_width"],
        age_shares=group_ages(people, age["group_width"], age["open_from"]),
        people=people,
        household_sizes=sizes,
        household_weights=weights / weights.sum(),
        household_radius=float(households["radius"]),
        first_birth=np.array(households["first_birth"], float).reshape(-1, 3),
        sibling_spacing_mean=households["sibling_spacing_mean"],
        sibling_spacing_sd=households["sibling_spacing_sd"],
        parent_gap_min=households["parent_age_gap_min"],
        partner_gap_mean=households["partner_age_gap_mean"],
        partner_gap_sd=households["partner_age_gap_sd"],
        lone_father_share=households["lone_father_share"],
        male_birth_share=households["male_birth_share"],
        kinds=kinds,
        kind_names=kind_names,
        kind_shares=kind_shares / kind_shares.sum(),
        head_age=head_age,
        child_ages={name: tuple(ages) for name, ages in households["child_ages"].items()},
        employment=_by_age_and_sex(rows, "employment"),
        employment_bands=rows[:, :2].astype(int),
        remote_share=employment["remote_share"],
        retired_from=employment["retired_from"],
        mother_employment=mothers,
        mother_ages=(low, high),
        care_home_rates=rates,
        care_home_size=care["mean_size"],
        **tables,
    )


def generate_population(count: int, params: PopulationParameters, rng) -> Population:
    """Create `count` people; their ages, sexes and households follow the parameters."""
    in_care = params.people * params.care_home_rates
    residents = round(count * in_care.sum())
    private = count - residents
    log_weights, weight_cap = _fit_household_weights(params)
    sizes_kept, ages_kept, sexes_kept, people = [], [], [], 0
    while people < private:
        batch = int(
            np.clip((private - people) / params.mean_household_size * 3, 1000, GENERATION_CHUNK)
        )
        sizes, ages, sexes, kinds, valid = _generate_households(batch, params, rng)
        cells = _household_cells(sizes, ages, sexes, kinds, params)
        keep = valid & (rng.random(batch) < np.exp(log_weights[cells].sum(axis=1)) / weight_cap)
        sizes_kept.append(sizes[keep])
        ages_kept.append(ages[keep])
        sexes_kept.append(sexes[keep])
        people += int(sizes[keep].sum())
    sizes, ages, sexes = (np.concatenate(kept) for kept in (sizes_kept, ages_kept, sexes_kept))
    households = np.searchsorted(np.cumsum(sizes), private) + 1
    present = ~np.isnan(ages[:households])
    care_age, care_sex, care_home = _care_home_residents(residents, in_care, params, rng)
    return Population(
        np.concatenate([np.floor(ages[:households][present])[:private], care_age]).astype(np.int16),
        np.concatenate([sexes[:households][present][:private], care_sex]).astype(np.int8),
        np.concatenate(
            [np.repeat(np.arange(households), sizes[:households])[:private], households + care_home]
        ).astype(np.int32),
        np.arange(count) >= private,
    )


def _care_home_residents(count: int, in_care: np.ndarray, params: PopulationParameters, rng):
    """Age, sex and home of `count` residents, sorted by home."""
    if not count:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    cells = rng.choice(in_care.size, count, p=in_care.ravel() / in_care.sum())
    homes = rng.integers(0, max(1, round(count / params.care_home_size)), count)
    order = np.argsort(homes, kind="stable")
    _, home = np.unique(homes[order], return_inverse=True)
    return cells[order] // 2, cells[order] % 2, home


def assign_roles(population: Population, params: PopulationParameters, rng) -> np.ndarray:
    """Role of everybody from age and sex: education, work, retirement, a care home."""
    age, sex, count = population.age, population.sex, len(population.age)
    role = np.where(age >= params.retired_from, RETIRED, HOMEBOUND).astype(np.int8)
    preschool = params.preschool[age, sex]
    school = preschool + params.school[age, sex]
    university = school + params.university[age, sex]
    draw = rng.random(count)
    role[draw < university] = UNIVERSITY
    role[draw < school] = SCHOOL
    role[draw < preschool] = PRESCHOOL
    role[population.resident] = RESIDENT
    free = np.isin(role, (HOMEBOUND, RETIRED))
    working = free & (rng.random(count) < _chance_of_work(population, params, free))
    role[working] = WORKER
    role[working & (rng.random(count) < params.remote_share)] = REMOTE
    return role


def _chance_of_work(population, params, free: np.ndarray) -> np.ndarray:
    """Chance of a job for everybody free to have one, so that every band has its published rate.

    The employment rates count everybody of an age and sex, so those who are free to work take up
    the jobs of those who study. Mothers of small children work less: they have their own rate and
    the rest of the band makes up for it.
    """
    age, sex = population.age, population.sex
    chance = np.zeros(len(age))
    mother_rate = _mother_rates(population, params)
    is_mother = ~np.isnan(mother_rate)
    chance[is_mother & free] = mother_rate[is_mother & free]
    everybody = ~population.resident
    for start, end in params.employment_bands:
        in_band = (age >= start) & (age < end) & everybody
        for code in range(len(SEXES)):
            band = in_band & (sex == code)
            jobs = params.employment[start, code] * band.sum()
            others = band & free & ~is_mother
            left = jobs - chance[band & is_mother].sum()
            chance[others] = np.clip(left / max(others.sum(), 1), 0.0, 1.0)
    return chance


def _mother_rates(population, params) -> np.ndarray:
    """The employment rate of mothers of small children by the youngest child; NaN for others."""
    age, household = population.age, population.household
    youngest = np.full(int(household.max()) + 1, 200, np.int16)
    np.minimum.at(youngest, household, age)
    low, high = params.mother_ages
    mother = (
        (population.sex == FEMALE)
        & (age >= low)
        & (age < high)
        & (age - youngest[household] >= params.parent_gap_min)
        & ~population.resident
    )
    rates = np.full(len(age), np.nan)
    tier = np.searchsorted(params.mother_employment[:, 0], youngest[household], side="right")
    known = mother & (tier < len(params.mother_employment))
    rates[known] = params.mother_employment[tier[known], 1]
    return rates


def _generate_households(count: int, params: PopulationParameters, rng):
    """Sizes, member ages (NaN where a household has fewer members), sexes, kinds, feasibility."""
    sizes = rng.choice(params.household_sizes, count, p=params.household_weights)
    width = int(params.household_sizes.max())
    ages = np.full((count, width), np.nan)
    sexes = np.zeros((count, width), np.int8)
    kinds = np.zeros(count, np.int64)
    valid = np.ones(count, np.bool_)
    for size in params.household_sizes:
        rows = np.flatnonzero(sizes == size)
        names, weights = params.kinds[int(size)]
        chosen = rng.choice(len(names), len(rows), p=weights)
        for index, name in enumerate(names):
            group = rows[chosen == index]
            if len(group):
                ages[group, :size], sexes[group, :size], valid[group] = _members(
                    name, int(size), len(group), params, rng
                )
                kinds[group] = params.kind_names.index(name)
    return sizes, ages, sexes, kinds, valid


def _members(kind: str, size: int, count: int, params: PopulationParameters, rng):
    """Ages and sexes of `count` households of one kind and size, and which of them are possible.

    In a couple the head is the man; a family that cannot exist, such as a mother too old for her
    children or a partner under age, is discarded and not repaired.
    """
    start, weights = params.head_age[kind]
    chosen = rng.choice(len(weights), count, p=weights)
    head = start + (chosen + rng.random(count)) * params.age_group_width
    fits = head < AGES
    random_sex = (rng.random((count, size)) < 0.5).astype(np.int8)
    if kind in ("alone", "alone_older", "alone_senior", "other"):
        columns = [head]
        for _ in range(size - 1):
            other = head + rng.normal(0, OTHER_AGE_SPREAD, count)
            fits &= (other >= ADULT_AGE) & (other < AGES)
            columns.append(other)
        return np.stack(columns, axis=1), random_sex, fits
    if kind == "multigenerational":
        return _three_generations(head, size, fits, params, rng)
    couple = kind.startswith("couple")
    sexes = _child_sexes(count, size, params, rng)
    sexes[:, 0] = MALE
    columns, mother = [head], head
    if couple:
        mother, adult = _partner(head, params, rng)
        fits &= adult
        columns.append(mother)
        sexes[:, 1] = FEMALE
    else:
        sexes[:, 0] = (rng.random(count) >= params.lone_father_share).astype(np.int8)
    if kind == "couple":
        return np.stack(columns, axis=1), sexes, fits
    lowest, highest = params.child_ages[kind]
    children, born = _child_ages(mother, size - len(columns), lowest, highest, params, rng)
    fits &= born & (np.minimum(head, mother) - children.max(axis=1) >= params.parent_gap_min)
    if not couple:
        sexes[children.min(axis=1) < params.mother_employment[0, 0], 0] = FEMALE
    return np.concatenate([np.stack(columns, axis=1), children], axis=1), sexes, fits


def _three_generations(head, size, fits, params, rng):
    """Grandparents, a parent or a couple of the middle generation and the grandchildren."""
    count = len(head)
    grandmother, adult = _partner(head, params, rng)
    middle, born = _child_ages(grandmother, 1, ADULT_AGE, MAX_AGE, params, rng)
    middle = middle[:, 0]
    fits &= adult & born & (head - middle >= params.parent_gap_min)
    columns = [head, grandmother, middle]
    sexes = _child_sexes(count, size, params, rng)
    sexes[:, :3] = (MALE, FEMALE, FEMALE)
    if size >= 5:
        columns.append(middle - rng.normal(params.partner_gap_mean, params.partner_gap_sd, count))
        fits &= (columns[-1] >= ADULT_AGE) & (columns[-1] < AGES)
        sexes[:, 3] = MALE
    lowest, highest = params.child_ages["couple_with_children"]
    children, born = _child_ages(middle, size - len(columns), lowest, highest, params, rng)
    fits &= born & (middle - children.max(axis=1) >= params.parent_gap_min)
    return np.concatenate([np.stack(columns, axis=1), children], axis=1), sexes, fits


def _child_sexes(count: int, size: int, params: PopulationParameters, rng) -> np.ndarray:
    return (rng.random((count, size)) >= params.male_birth_share).astype(np.int8)


def _partner(head: np.ndarray, params: PopulationParameters, rng):
    """Age of the partner of everybody in `head`, and whether the partner is an adult."""
    partner = head + rng.normal(params.partner_gap_mean, params.partner_gap_sd, len(head))
    return partner, (partner >= ADULT_AGE) & (partner < AGES)


def _child_ages(
    mother: np.ndarray, count: int, lowest: int, highest: int, params: PopulationParameters, rng
):
    """Ages of `count` children at home of each mother, and whether she can have them.

    A mother bore her children one after another from an age typical of her cohort. The children
    at home are the oldest ones between `lowest` and `highest`: the older ones have left home. A
    family is drawn again a few times when it does not fit, because a woman who is young for a
    mother of her cohort had her child young, which is rare but no reason to discard her.
    """
    ages = np.zeros((len(mother), count))
    pending = np.arange(len(mother))
    for _ in range(CHILD_AGE_ATTEMPTS):
        drawn, fits = _draw_children(mother[pending], count, lowest, highest, params, rng)
        ages[pending[fits]] = drawn[fits]
        pending = pending[~fits]
        if not len(pending):
            break
    born = np.ones(len(mother), np.bool_)
    born[pending] = False
    return ages, born


def _draw_children(mother, count, lowest, highest, params, rng):
    years, means, spreads = params.first_birth.T
    cohort = REFERENCE_YEAR - mother + COHORT_FIRST_BIRTH_AGE
    first = rng.normal(np.interp(cohort, years, means), np.interp(cohort, years, spreads))
    above_minimum = params.sibling_spacing_mean - MIN_SIBLING_SPACING
    shape = (above_minimum / params.sibling_spacing_sd) ** 2
    spacing = MIN_SIBLING_SPACING + rng.gamma(
        shape, above_minimum / shape, (len(mother), count + EXTRA_CHILDREN - 1)
    )
    gaps = np.concatenate([np.zeros((len(mother), 1)), spacing.cumsum(axis=1)], axis=1)
    all_children = (mother - first)[:, None] - gaps
    at_home = (
        (all_children >= lowest)
        & (all_children < highest + 1)
        & (mother[:, None] - all_children < LAST_BIRTH_AGE)
    )
    start = at_home.argmax(axis=1)
    rows = np.arange(len(mother))[:, None]
    chosen = start[:, None] + np.arange(count)[None, :]
    return (
        all_children[rows, np.minimum(chosen, all_children.shape[1] - 1)],
        (at_home.sum(axis=1) >= count)
        & (start + count <= all_children.shape[1])
        & (first >= MIN_FIRST_BIRTH_AGE),
    )


def _household_cells(sizes, ages, sexes, kinds, params: PopulationParameters) -> np.ndarray:
    """Per household the fit cells it counts in: one per member, its size and its kind.

    A member's cell is its age and sex; the last cell, which stands for "nobody" in a household
    smaller than the largest, has a weight of zero.
    """
    people_cells = (FIT_OPEN_AGE + 1) * len(SEXES)
    size_cells = people_cells + np.searchsorted(params.household_sizes, sizes)
    kind_cells = people_cells + len(params.household_sizes) + kinds
    empty = _cell_count(params)
    present = ~np.isnan(ages)
    years = np.minimum(np.where(present, ages, 0), FIT_OPEN_AGE).astype(np.int64)
    members = np.where(present, years * len(SEXES) + sexes, empty)
    return np.column_stack([members, size_cells, kind_cells])


def _cell_count(params: PopulationParameters) -> int:
    return (FIT_OPEN_AGE + 1) * len(SEXES) + len(params.household_sizes) + len(params.kind_names)


def _targets(params: PopulationParameters) -> np.ndarray:
    """What the fit aims at per household: people in every cell, then shares of sizes and kinds."""
    private = params.people * (1.0 - params.care_home_rates)
    private = private / private.sum() * params.mean_household_size
    by_age = np.vstack([private[:FIT_OPEN_AGE], private[FIT_OPEN_AGE:].sum(axis=0)])
    return np.concatenate([by_age.ravel(), params.household_weights, params.kind_shares])


_FITS: dict[str, tuple[np.ndarray, float]] = {}


def _fit_household_weights(params: PopulationParameters) -> tuple[np.ndarray, float]:
    """Exponential tilting of households so that people, sizes and kinds match the targets.

    Returns the log-weights per cell (the extra last one is zero) and the weight above which
    households are always kept.
    """
    if params.fingerprint not in _FITS:
        rng = np.random.default_rng(FIT_SEED)
        sizes, ages, sexes, kinds, valid = _generate_households(FIT_POOL_HOUSEHOLDS, params, rng)
        cells = _household_cells(sizes, ages, sexes, kinds, params)[valid]
        width = _cell_count(params)
        target = _targets(params)
        log_weights = np.zeros(width + 1)
        for _ in range(FIT_ITERATIONS):
            score = log_weights[cells].sum(axis=1)
            weight = np.exp(score - score.max())
            weight /= weight.sum()
            expected = np.bincount(
                cells.ravel(), np.repeat(weight, cells.shape[1]), minlength=width + 1
            )[:width]
            step = np.clip(
                np.log(target + 1e-9) - np.log(expected + 1e-9), -FIT_MAX_STEP, FIT_MAX_STEP
            )
            log_weights[:width] = np.clip(
                log_weights[:width] + FIT_STEP * step, -FIT_MAX_LOG_WEIGHT, FIT_MAX_LOG_WEIGHT
            )
        _FITS[params.fingerprint] = (
            log_weights,
            float(np.percentile(np.exp(log_weights[cells].sum(axis=1)), WEIGHT_CAP_PERCENTILE)),
        )
    return _FITS[params.fingerprint]
