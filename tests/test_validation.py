from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from epidemic_sandbox import validation
from epidemic_sandbox.town import load_town_parameters

TOWN_PATH = Path(__file__).resolve().parent.parent / "configs" / "town.toml"


@pytest.fixture(scope="module")
def town_parameters():
    return load_town_parameters(TOWN_PATH)


def test_equivalence_accepts_values_inside_the_margin_and_rejects_outside():
    inside = validation.equivalence_p_value(0.20, 0.002, 0.20, 0.02)
    outside = validation.equivalence_p_value(0.26, 0.002, 0.20, 0.02)
    assert inside < 0.001
    assert outside > 0.99


def test_equivalence_cannot_be_shown_when_sampling_error_exceeds_the_margin():
    assert validation.equivalence_p_value(0.20, 0.05, 0.20, 0.02) > 0.3


def test_rank_averages_ties():
    assert validation.rank(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_spearman_of_monotone_transform_is_one():
    values = np.arange(1.0, 20.0)
    assert validation.spearman(values, values**3) == pytest.approx(1.0)


def test_permutation_test_accepts_matching_structure_and_rejects_unrelated_one():
    ages = np.arange(16)
    reference = np.exp(-np.abs(ages[:, None] - ages[None, :]) / 3.0)
    rng = np.random.default_rng(0)
    matching = reference * (1 + 0.05 * rng.random(reference.shape))
    correlation, p_value = validation.permutation_test(matching, reference, rng)
    assert correlation > 0.95 and p_value < 0.01

    unrelated = rng.random(reference.shape)
    correlation, p_value = validation.permutation_test(unrelated, reference, rng)
    assert abs(correlation) < 0.3 and p_value > 0.05


def test_permutation_test_detects_a_matrix_with_shifted_age_groups():
    ages = np.arange(16)
    reference = np.exp(-np.abs(ages[:, None] - ages[None, :]) / 3.0)
    shifted = np.roll(reference, 6, axis=0)
    _, p_value = validation.permutation_test(shifted, reference, np.random.default_rng(0))
    assert p_value > 0.01


def test_default_town_agrees_with_published_population_data(town_parameters):
    failed = [c.name for c in validation.validate_population(town_parameters) if not c.passed]
    assert failed == []


def test_default_town_agrees_with_published_contact_structure(town_parameters):
    failed = [c.name for c in validation.validate_contacts(town_parameters) if not c.passed]
    assert failed == []


def test_validation_fails_for_a_population_with_the_wrong_age_structure(town_parameters):
    young = replace(
        town_parameters,
        population=replace(
            town_parameters.population, age_shares=town_parameters.population.age_shares[::-1]
        ),
    )
    checks = validation.validate_population(young, count=50_000)
    assert any(not check.passed for check in checks if check.name.startswith("age"))


def test_people_of_every_age_meet_about_as_many_others_a_day_as_the_survey_found(town_parameters):
    failed = [c.name for c in validation.validate_contact_levels(town_parameters) if not c.passed]
    assert not failed, failed


def test_young_pupils_spend_most_of_their_time_at_school_with_their_class(town_parameters):
    failed = [c.name for c in validation.validate_school_mixing(town_parameters) if not c.passed]
    assert not failed, failed


def test_the_contact_levels_would_catch_a_town_where_everybody_stays_at_one_spot(town_parameters):
    # Who meets few different people (nobody moves about) fails the check for the young, who
    # otherwise meet many at school.
    frozen = replace(town_parameters, dwell_hours=1e9)
    checks = validation.validate_contact_levels(frozen)
    assert not all(check.passed for check in checks)
