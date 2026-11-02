from pathlib import Path

import numpy as np
import pytest
from config_factory import CONFIG_DIRECTORY, LIBRARY

from epidemic_sandbox import population as population_module
from epidemic_sandbox.population import (
    FEMALE,
    HOMEBOUND,
    MALE,
    PRESCHOOL,
    REMOTE,
    RESIDENT,
    RETIRED,
    ROLES,
    SCHOOL,
    UNIVERSITY,
    WORKER,
    assign_roles,
    generate_population,
)
from epidemic_sandbox.town import load_town_parameters

PARAMS = LIBRARY.town.population
REFERENCE_MEAN_AGE = 42.3
COUNT = 120_000


@pytest.fixture(scope="module")
def population():
    return generate_population(COUNT, PARAMS, np.random.default_rng(1))


@pytest.fixture(scope="module")
def roles(population):
    return assign_roles(population, PARAMS, np.random.default_rng(2))


def household_sizes(population):
    return np.bincount(population.household)


def test_people_by_age_and_sex_match_the_table_within_sampling_error(population):
    shares = np.zeros((100, 2))
    np.add.at(shares, (population.age, population.sex), 1.0 / COUNT)
    assert np.abs(shares - PARAMS.people).max() < 0.001
    five_years = shares.sum(axis=1).reshape(20, 5).sum(axis=1)
    assert np.abs(five_years - PARAMS.people.sum(axis=1).reshape(20, 5).sum(axis=1)).max() < 0.003


def test_single_years_of_age_have_no_artificial_steps(population):
    by_year = np.bincount(population.age, minlength=100)[:80]
    assert np.abs(by_year / (PARAMS.people.sum(axis=1)[:80] * COUNT) - 1).max() < 0.15


def test_broad_age_groups_and_mean_age_match_the_2019_population(population):
    age = population.age
    assert (age < 15).mean() == pytest.approx(0.159, abs=0.01)
    assert ((age >= 15) & (age < 65)).mean() == pytest.approx(0.645, abs=0.01)
    assert (age >= 65).mean() == pytest.approx(0.196, abs=0.01)
    assert (age + 0.5).mean() == pytest.approx(REFERENCE_MEAN_AGE, abs=0.4)


def test_old_people_are_mostly_women(population):
    assert (population.sex[population.age >= 85] == FEMALE).mean() > 0.65
    assert (population.sex[population.age < 15] == MALE).mean() == pytest.approx(0.513, abs=0.015)


def private_sizes(population):
    return np.bincount(population.household[~population.resident])


def test_household_sizes_match_the_census(population):
    sizes = private_sizes(population)[:-1]  # the last household may be cut
    shares = np.bincount(sizes, minlength=9)[1:9] / len(sizes)
    assert shares == pytest.approx(PARAMS.household_weights, abs=0.005)
    assert (~population.resident).sum() / len(sizes) == pytest.approx(2.15, abs=0.03)


def test_household_kinds_keep_their_shares_through_the_fit():
    rng = np.random.default_rng(1)
    log_weights, _ = population_module._fit_household_weights(PARAMS)
    sizes, ages, sexes, kinds, valid = population_module._generate_households(200_000, PARAMS, rng)
    cells = population_module._household_cells(sizes, ages, sexes, kinds, PARAMS)
    weight = np.exp(log_weights[cells].sum(axis=1)) * valid
    shares = np.bincount(kinds, weights=weight) / weight.sum()
    assert shares == pytest.approx(PARAMS.kind_shares, abs=0.003)


def test_people_of_a_household_are_next_to_each_other(population):
    assert (np.diff(population.household) >= 0).all() and len(population.household) == COUNT


def test_every_household_has_an_adult_and_children_live_with_a_parent_generation(population):
    household, age = population.household, population.age
    oldest = np.zeros(household.max() + 1)
    np.maximum.at(oldest, household, age)
    assert (oldest >= 18).all()
    child = age < 18
    assert (oldest[household][child] - age[child]).min() >= PARAMS.parent_gap_min - 1


def mothers_of_infants(population):
    """Age of the oldest woman of a childbearing age with a child under 3, at the birth."""
    household, age = population.household, population.age
    women = (population.sex == FEMALE) & (age >= 15) & (age < 50)
    mother = np.zeros(household.max() + 1)
    np.maximum.at(mother, household[women], age[women])
    infants = age < 3
    return (mother[household] - age)[infants], mother[household][infants] > 0


def test_infants_have_a_mother_of_a_possible_age_who_lives_with_them(population):
    born_to, has_mother = mothers_of_infants(population)
    assert has_mother.mean() > 0.995
    assert born_to[has_mother].min() >= 15
    assert (born_to[has_mother] >= 50).sum() == 0


def test_the_ages_of_mothers_are_the_published_ones(population):
    born_to, has_mother = mothers_of_infants(population)
    born_to = born_to[has_mother]
    assert (born_to + 0.5).mean() == pytest.approx(30.4, abs=0.7)  # Eurostat demo_fasec 2019
    assert (born_to < 25).mean() == pytest.approx(0.126, abs=0.03)
    assert (born_to >= 35).mean() == pytest.approx(0.218, abs=0.05)


def test_older_mothers_had_their_children_younger(population):
    """Mothers of today's adults had children at about 23, those of today's infants at 29."""
    household, age = population.household, population.age
    women = (population.sex == FEMALE) & (age >= 15)
    mother = np.zeros(household.max() + 1)
    np.maximum.at(mother, household[women], age[women])
    grown = (age >= 25) & (age < 45) & (mother[household] - age >= 17)
    small = (age < 5) & (mother[household] - age >= 17) & (mother[household] < 50)
    assert (mother[household] - age)[grown].mean() < (mother[household] - age)[small].mean() - 2


def test_lone_parents_of_infants_are_mothers(population):
    household, age = population.household, population.age
    adults = np.bincount(household, weights=(age >= 18).astype(float))
    lone = (adults[household] == 1) & (age >= 18)
    youngest = np.full(household.max() + 1, 200)
    np.minimum.at(youngest, household, age)
    assert (population.sex[lone & (youngest[household] < 3)] == FEMALE).all()


def households_with_children(population):
    """Ages of (adults, children under 18) for households of three or more with two adults."""
    household, age = population.household, population.age
    start = np.searchsorted(household, np.arange(household.max() + 1))
    sizes = household_sizes(population)
    found = []
    for index in range(len(sizes) - 1):  # the last household may be cut
        members = age[start[index] : start[index] + sizes[index]]
        adults, children = np.sort(members[members >= 18]), members[members < 18]
        if len(adults) == 2 and len(children):
            found.append((adults, children))
    return found


def test_both_parents_are_old_enough_and_the_younger_was_about_28_at_the_first_birth(population):
    couples = [
        (adults, children)
        for adults, children in households_with_children(population)
        if adults[1] - adults[0] < 12  # a couple, not a parent with an adult child
    ]
    gaps = np.array([adults[0] - children.max() for adults, children in couples])
    assert (gaps < PARAMS.parent_gap_min - 1).sum() == 0
    assert gaps.mean() == pytest.approx(28.0, abs=2.0)  # CZSO 2022: mothers 28.8 at the first child


def test_siblings_are_born_a_few_years_apart(population):
    spacing = np.concatenate(
        [
            np.diff(np.sort(children))
            for _, children in households_with_children(population)
            if len(children) > 1
        ]
    )
    assert (spacing <= 1).mean() < 0.2
    assert (spacing >= 8).mean() < 0.15
    assert 2.8 < spacing.mean() < 4.2


def test_the_oldest_old_are_present_in_the_census_numbers(population):
    expected = PARAMS.people[90:].sum()
    assert (population.age >= 90).mean() == pytest.approx(expected, abs=0.0015)


def test_couples_are_close_in_age(population):
    sizes = private_sizes(population)
    pairs = np.flatnonzero(sizes == 2)[:-1]
    ages = np.array(
        [population.age[population.household == household] for household in pairs[:3000]]
    )
    adults = (ages >= 18).all(axis=1)
    gap = np.abs(ages[:, 0] - ages[:, 1])[adults]
    assert np.median(gap) < 7


def test_lone_households_skew_old_as_in_reality(population):
    alone = (household_sizes(population)[population.household] == 1) & ~population.resident
    assert (population.age[alone] >= 65).mean() == pytest.approx(0.37, abs=0.03)
    assert (population.age[alone] < 18).sum() == 0


def test_more_old_women_live_alone_than_old_men(population):
    alone = (household_sizes(population)[population.household] == 1) & ~population.resident
    old = population.age >= 65
    assert alone[old & (population.sex == FEMALE)].mean() == pytest.approx(0.40, abs=0.04)
    assert alone[old & (population.sex == MALE)].mean() == pytest.approx(0.21, abs=0.04)


def test_care_home_residents_come_last_and_are_old(population, roles):
    assert population.resident.mean() == pytest.approx(0.0055, abs=0.0005)
    assert population.resident[-1] and not population.resident[0]
    assert (np.diff(population.resident.astype(int)) >= 0).all()
    assert (population.age[population.resident] >= 50).all()
    assert (roles[population.resident] == RESIDENT).all()
    in_care = population.resident[population.age >= 86].mean()
    assert in_care == pytest.approx(0.135, abs=0.03)


@pytest.mark.parametrize("count", [100, 3000])
def test_a_small_town_has_a_care_home_of_its_size(count):
    population = generate_population(count, PARAMS, np.random.default_rng(5))
    assert len(population.age) == count
    assert population.resident.sum() == round(count * 0.0055)


def test_roles_follow_age(population, roles):
    age = population.age
    assert (roles[(age >= 8) & (age < 15)] == SCHOOL).all()
    assert (roles[age < 2] == HOMEBOUND).all()
    assert not np.isin(roles[age >= 65], (PRESCHOOL, SCHOOL, UNIVERSITY)).any()
    assert not np.isin(roles[age < 15], (WORKER, REMOTE, RETIRED, UNIVERSITY)).any()
    assert (roles[age >= PARAMS.retired_from] != HOMEBOUND).all()


@pytest.mark.parametrize(
    ("age", "expected"), [(2, 0.176), (3, 0.776), (4, 0.871), (5, 0.944), (6, 0.512), (7, 0.021)]
)
def test_preschool_attendance(population, roles, age, expected):
    assert (roles[population.age == age] == PRESCHOOL).mean() == pytest.approx(expected, abs=0.05)


def test_half_of_the_six_year_olds_are_still_in_kindergarten(population, roles):
    six = roles[population.age == 6]
    assert (six == SCHOOL).mean() == pytest.approx(0.477, abs=0.05)
    assert (six == HOMEBOUND).mean() < 0.06


def test_nobody_is_in_two_levels_of_education_and_attendance_adds_up(population, roles):
    in_education = np.isin(roles, (PRESCHOOL, SCHOOL, UNIVERSITY))
    assert in_education[(population.age >= 7) & (population.age < 15)].mean() == pytest.approx(
        1.0, abs=0.001
    )


def test_many_nineteen_year_olds_still_go_to_school(population, roles):
    nineteen = roles[population.age == 19]
    assert (nineteen == SCHOOL).mean() == pytest.approx(0.47, abs=0.06)
    assert (nineteen == UNIVERSITY).mean() == pytest.approx(0.24, abs=0.06)


def test_more_women_than_men_study_at_university(population, roles):
    young = (population.age >= 20) & (population.age < 25)
    women = (roles[young & (population.sex == FEMALE)] == UNIVERSITY).mean()
    men = (roles[young & (population.sex == MALE)] == UNIVERSITY).mean()
    assert women > men + 0.1


def test_secondary_school_attendance(population, roles):
    teenagers = (population.age >= 15) & (population.age < 18)
    assert (roles[teenagers] == SCHOOL).mean() == pytest.approx(0.976, abs=0.02)


def test_employment_rates_match_the_labour_force_survey_by_age_and_sex(population, roles):
    age, sex = population.age, population.sex
    working = np.isin(roles, (WORKER, REMOTE))
    for start, end in PARAMS.employment_bands:
        for code in (MALE, FEMALE):
            band = (age >= start) & (age < end) & (sex == code)
            assert working[band].mean() == pytest.approx(PARAMS.employment[start, code], abs=0.03)


def test_a_person_without_a_job_is_retired_from_sixty(population, roles):
    idle = ~np.isin(roles, (WORKER, REMOTE))
    assert (roles[idle & (population.age >= 60) & ~population.resident] == RETIRED).all()
    assert (roles[idle & (population.age >= 25) & (population.age < 60)] != RETIRED).all()


def test_a_share_of_workers_work_from_home(roles):
    workers = np.isin(roles, (WORKER, REMOTE))
    assert (roles[workers] == REMOTE).mean() == pytest.approx(PARAMS.remote_share, abs=0.02)


def test_mothers_of_infants_are_mostly_at_home_and_fathers_are_not(population, roles):
    household, age, sex = population.household, population.age, population.sex
    youngest = np.full(household.max() + 1, 200)
    np.minimum.at(youngest, household, age)
    parents = (age >= 25) & (age < 45) & (youngest[household] < 3)
    working = np.isin(roles, (WORKER, REMOTE))
    assert working[parents & (sex == FEMALE)].mean() < 0.3
    assert working[parents & (sex == MALE)].mean() > 0.85


def test_every_role_is_a_known_role(roles):
    assert roles.min() >= 0 and roles.max() < len(ROLES)


def test_population_is_reproducible_from_the_seed():
    first = generate_population(5000, PARAMS, np.random.default_rng(3))
    second = generate_population(5000, PARAMS, np.random.default_rng(3))
    assert np.array_equal(first.age, second.age) and np.array_equal(
        first.household, second.household
    )


@pytest.mark.parametrize("count", [50, 51, 1000])
def test_any_population_size_is_produced_exactly(count):
    population = generate_population(count, PARAMS, np.random.default_rng(4))
    assert len(population.age) == len(population.household) == count


def write_town(tmp_path, old=None, new=None, age_table=None) -> Path:
    """A copy of the town config in a temporary tree with the reference data it points to."""
    reference = tmp_path / "data" / "reference"
    reference.mkdir(parents=True)
    table = (
        CONFIG_DIRECTORY.parent / "data" / "reference" / "czechia_population_2019.csv"
    ).read_text()
    (reference / "czechia_population_2019.csv").write_text(age_table or table)
    text = (CONFIG_DIRECTORY / "town.toml").read_text()
    if old is not None:
        assert old in text
        text = text.replace(old, new, 1)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "town.toml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("weights = [0.39068, 0.29524", "weights = [0.29524", "invalid household sizes"),
        ("couple = 0.702", "alone = 0.702", "invalid household kinds"),
        ("couple = 0.702", "nonsense = 0.702", "invalid household kinds"),
        ("preschool = [[2, 3, 0.171", "preschool = [[3, 2, 0.171", "invalid age range"),
        ("[8, 15, 1.0, 1.0]", "[8, 15, 1.5, 1.0]", "invalid age range"),
        ("[6, 7, 0.421, 0.536]", "[6, 7, 0.9, 0.9]", "education shares exceed 1"),
        ("[households.head_age.other]", "[households.head_age.unused]", "missing head_age"),
        ("alone_older = 0.0611\n", "", "kind_shares must list"),
        ('first_day = "2018-09-10"', 'first_day = "2018-09-11"', "must be a Monday"),
        ('first_day = "2018-09-10"', 'first_day = "September"', "Invalid isoformat"),
        ("mother_ages = [18, 50]", "mother_ages = [50, 18]", "increasing age range"),
        ("[3, 0.20], [6, 0.88]", "[6, 0.20], [3, 0.88]", "mother_employment"),
        ("accompanied_under = 10", "accompanied_under = -1", "accompanied_under"),
        ("staff_per_resident = 0.33", "staff_per_resident = -1", "staff_per_resident"),
        ("[50, 66, 0.00143]", "[66, 50, 0.00143]", "invalid age range"),
    ],
)
def test_invalid_population_settings_are_rejected(tmp_path, old, new, message):
    with pytest.raises(ValueError, match=message):
        load_town_parameters(write_town(tmp_path, old, new))


def test_age_structure_is_read_from_the_reference_table(tmp_path):
    table = "age,male,female\n0,50,50\n5,60,40\n10,100,100\n85,50,50\n"
    params = load_town_parameters(write_town(tmp_path, age_table=table)).population
    assert params.age_shares[:3].tolist() == [0.2, 0.2, 0.4] and params.age_shares[-1] == 0.2
    assert params.people[5].tolist() == [0.12, 0.08]


def test_negative_age_counts_are_rejected(tmp_path):
    table = "age,male,female\n0,100,100\n5,-100,-100\n"
    with pytest.raises(ValueError, match="must be non-negative"):
        load_town_parameters(write_town(tmp_path, age_table=table))
