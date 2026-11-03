"""The SVIR model against its definition: structure, mechanisms, textbook results, preview numbers.

The dynamics are compared with the SVIR equations in a well-mixed population (tests/mean_field.py).
With `duration_shape=1` and identical people the average course of many runs follows

    dS/dt = -lambda S - nu S + w V        dV/dt = nu S - (1 - VE) lambda V - w V
    dI/dt = lambda (S + (1 - VE) V) - g I    dR/dt = g I

and, when nobody is vaccinated or loses protection during the epidemic, the share `a` ever infected
solves the final size relation of a leaky vaccine (integrate the force of infection: its integral is
R0 times `a`, because every infected person is infectious for 1/g on average)

    a = i0 + s0 (1 - exp(-R0 a)) + v0 (1 - exp(-(1 - VE) R0 a)).

Its threshold is R0 (s0 + (1 - VE) v0) = 1, that is, the critical coverage (1 - 1/R0) / VE.
"""

import math
from functools import cache

import numpy as np
import pytest
from cited_numbers import check_default_numbers, check_demo_numbers
from config_factory import disease_definition, make_model
from disease_helpers import (
    ONE_DAY,
    SURE,
    UNIFORM,
    advance,
    count_in,
    new_state,
    place,
    state_names,
)
from mean_field import final_size, integrate, run_well_mixed

from epidemic_sandbox.disease_models.disease_model_abstract import (
    DEAD,
    INFECTIOUS,
    LIFELONG_DAYS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.svir import VACCINATED, SvirModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
NO_CAMPAIGN = {"vaccination_rate": 0.0, "initial_coverage": 0.0}
FROM_THE_START = {"vaccination_start_day": 0.0, "initial_coverage": 0.0}
KEPT_FOR_LIFE = {"vaccine_days": float(LIFELONG_DAYS), "immunity_days": float(LIFELONG_DAYS)}
STEP = 1 / 24  # the step of the well-mixed driver

# The well-mixed epidemic used for the textbook comparisons: R0 = hazard * contacts * mean stay.
TRANSMISSION, CONTACTS, INFECTIOUS_DAYS = 0.3, 2.0, 4.0
BETA = -math.log(1 - TRANSMISSION) * CONTACTS
# A stay is the continuous stay rounded up to whole steps: half a step longer on average.
R0 = BETA * (INFECTIOUS_DAYS + STEP / 2)
WELL_MIXED = {
    "transmission": TRANSMISSION,
    "infectious_days": INFECTIOUS_DAYS,
    "duration_shape": 1.0,
    "contacts": CONTACTS,
    **IDENTICAL,
    **KEPT_FOR_LIFE,
}


def four_sigma(share: float, people: int) -> float:
    """Four standard deviations of the share of `people` independent draws at chance `share`."""
    return 4 * math.sqrt(share * (1 - share) / people)


def leaky_final_size(r0: float, infected: float, vaccinated: float, efficacy: float) -> float:
    """Share ever infected, solving the final size relation of the docstring by iteration."""
    susceptible = 1 - infected - vaccinated
    size = 0.5
    for _ in range(2000):
        size = (
            infected
            + susceptible * (1 - math.exp(-r0 * size))
            + vaccinated * (1 - math.exp(-(1 - efficacy) * r0 * size))
        )
    return size


def ever_infected(runs: int, population: int, cases: int, days: float, **parameters) -> float:
    """Mean share ever infected in the well-mixed town (infectious now, recovered and dead)."""
    finished = []
    for seed in range(runs):
        _, counts = run_well_mixed(
            SvirModel, population=population, cases=cases, days=days, seed=seed, **parameters
        )
        assert counts[-1, 2] < 0.01 * population, "the epidemic is over at the end of the run"
        finished.append(counts[-1, [2, 3, 4]].sum() / population)
    return float(np.mean(finished))


FINAL_SIZE_POPULATION, FINAL_SIZE_CASES, FINAL_SIZE_RUNS = 6000, 120, 4
# One run deviates from the mean final size by 0.003 to 0.016 (12 runs of 8,000 people in the
# configurations below, the larger values near the threshold) and by 1/sqrt(population) more in
# smaller towns, so the mean of 4 runs of 6,000 people by at most 4 x 0.016 x sqrt(8/6) / 2 = 0.037.
FINAL_SIZE_TOLERANCE = 0.03


@cache
def final_size_in_a_well_mixed_town(efficacy: float, coverage: float, days: int = 80) -> float:
    return ever_infected(
        FINAL_SIZE_RUNS,
        FINAL_SIZE_POPULATION,
        FINAL_SIZE_CASES,
        days,
        **WELL_MIXED,
        vaccine_efficacy=efficacy,
        initial_coverage=coverage,
        vaccination_rate=0.0,
    )


def predicted_final_size(efficacy: float, coverage: float) -> float:
    infected = FINAL_SIZE_CASES / FINAL_SIZE_POPULATION
    return leaky_final_size(R0, infected, coverage * (1 - infected), efficacy)


# --- structure: who can become what ------------------------------------------------------------


def test_the_states_are_the_svir_compartments_and_only_the_infectious_spread():
    assert list(SvirModel.STATE_COLORS) == [SUSCEPTIBLE, VACCINATED, INFECTIOUS, RECOVERED, DEAD]
    model = SvirModel(ONE_DAY)
    weights = {state: model.state_infectiousness(state) for state in SvirModel.STATE_COLORS}
    assert weights == {SUSCEPTIBLE: 0, VACCINATED: 0, INFECTIOUS: 1, RECOVERED: 0, DEAD: 0}


def test_an_infection_runs_from_infectious_to_recovered_and_vaccination_is_not_part_of_it():
    assert SvirModel.course_states() == {INFECTIOUS, RECOVERED}
    model = SvirModel(ONE_DAY, fatality=0.1)
    ends = {course.end: course.probability for course in model.courses()}
    assert ends == pytest.approx({RECOVERED: 0.9, DEAD: 0.1})


def test_the_moves_a_run_makes_are_exactly_those_of_the_scheme():
    model = SvirModel(
        1 / 4,
        transmission=0.8,
        infectious_days=2.0,
        vaccination_rate=0.3,
        initial_coverage=0.2,
        vaccine_days=6.0,
        vaccine_efficacy=0.5,
        immunity_days=8.0,
        immunity_protection=0.5,
        fatality=0.2,
    )
    names = list(model.STATE_COLORS)
    disease_state, rng = new_state(model, population=3000, cases=np.arange(100))
    seen = set()
    for _ in range(400):
        before = disease_state.state.copy()
        advance(model, disease_state, rng, exposure=0.5)
        seen |= {
            (names[a], names[b]) for a, b in zip(before, disease_state.state, strict=True) if a != b
        }
    assert seen == {
        (SUSCEPTIBLE, VACCINATED),
        (VACCINATED, SUSCEPTIBLE),
        (SUSCEPTIBLE, INFECTIOUS),
        (VACCINATED, INFECTIOUS),
        (INFECTIOUS, RECOVERED),
        (INFECTIOUS, DEAD),
        (RECOVERED, SUSCEPTIBLE),
        (RECOVERED, INFECTIOUS),
    }


def test_nobody_is_vaccinated_after_infection_and_the_recovered_stay_recovered_for_life():
    model = SvirModel(ONE_DAY, vaccination_rate=1.0, **FROM_THE_START, **KEPT_FOR_LIFE)
    disease_state, rng = new_state(model, population=500, cases=np.arange(100))
    place(model, disease_state, rng, np.arange(100, 200), RECOVERED)
    for _ in range(30):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, RECOVERED) == 200
    assert count_in(model, disease_state, VACCINATED) == 300
    assert count_in(model, disease_state, SUSCEPTIBLE) == 0


def test_the_infectious_are_not_vaccinated_while_they_are_ill():
    model = SvirModel(
        ONE_DAY, vaccination_rate=1.0, **FROM_THE_START, infectious_days=60.0, **UNIFORM
    )
    population, cases = 100_000, 10_000
    disease_state, rng = new_state(model, population=population, cases=np.arange(cases))
    advance(model, disease_state, rng)
    assert count_in(model, disease_state, INFECTIOUS) == cases
    share = 1 - math.exp(-1.0)
    assert count_in(model, disease_state, VACCINATED) / (population - cases) == pytest.approx(
        share, abs=four_sigma(share, population - cases)
    )


def test_only_the_infectious_react_to_being_ill():
    model = SvirModel(ONE_DAY, stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, population=len(model.STATE_COLORS))
    for person, state in enumerate(model.STATE_COLORS):
        if state != DEAD:
            place(model, disease_state, rng, [person], state)
    stays = dict(zip(model.STATE_COLORS, disease_state.stays_home.tolist(), strict=True))
    assert stays == {
        SUSCEPTIBLE: False,
        VACCINATED: False,
        INFECTIOUS: True,
        RECOVERED: False,
        DEAD: False,
    }


# --- the leaky vaccine -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "protection"), [(SUSCEPTIBLE, 0.0), (VACCINATED, 0.7), (RECOVERED, 0.4)]
)
def test_each_state_catches_the_infection_at_the_hazard_its_protection_leaves(state, protection):
    model = SvirModel(
        ONE_DAY,
        transmission=0.5,
        vaccine_efficacy=0.7,
        immunity_protection=0.4,
        **NO_CAMPAIGN,
        **UNIFORM,
    )
    population = 200_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), state)
    advance(model, disease_state, rng, exposure=1)
    expected = 1 - 0.5 ** (1 - protection)
    assert disease_state.cumulative_infections / population == pytest.approx(
        expected, abs=four_sigma(expected, population)
    )


def test_the_vaccine_lowers_the_hazard_and_does_not_remove_a_share_of_the_people():
    """Leaky, not all-or-nothing: a stronger exposure still gets through to the vaccinated."""
    model = SvirModel(
        ONE_DAY, transmission=0.5, vaccine_efficacy=0.8, **NO_CAMPAIGN, **UNIFORM, vaccine_days=60
    )
    population = 200_000
    for exposure in (0.5, 4.0):
        disease_state, rng = new_state(model, population=population)
        place(model, disease_state, rng, np.arange(population), VACCINATED)
        advance(model, disease_state, rng, exposure=exposure)
        expected = 1 - 0.5 ** (exposure * 0.2)
        assert disease_state.cumulative_infections / population == pytest.approx(
            expected, abs=four_sigma(expected, population)
        )


def test_a_vaccine_with_full_efficacy_protects_completely_and_one_without_does_nothing():
    complete = SvirModel(ONE_DAY, **SURE, vaccine_efficacy=1.0, **NO_CAMPAIGN)
    disease_state, rng = new_state(complete, population=100)
    place(complete, disease_state, rng, np.arange(100), VACCINATED)
    assert not complete.exposable_mask(disease_state).any()
    advance(complete, disease_state, rng, exposure=50)
    assert disease_state.cumulative_infections == 0

    useless = SvirModel(ONE_DAY, **SURE, vaccine_efficacy=0.0, **NO_CAMPAIGN)
    disease_state, rng = new_state(useless, population=100)
    place(useless, disease_state, rng, np.arange(100), VACCINATED)
    advance(useless, disease_state, rng, exposure=0.001)
    assert disease_state.cumulative_infections == 100


def test_an_infected_vaccinated_person_is_infectious_and_ill_like_anybody_else():
    model = SvirModel(
        ONE_DAY,
        **SURE,
        vaccine_efficacy=0.5,
        stay_home=1.0,
        **NO_CAMPAIGN,
        **UNIFORM,
        infectious_days=0.5,
        duration_shape=50.0,
        fatality=0.3,
    )
    population = 20_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), VACCINATED)
    advance(model, disease_state, rng, exposure=1)
    assert count_in(model, disease_state, INFECTIOUS) == population
    assert disease_state.stays_home.all()
    assert disease_state.ever_infected.all()
    for _ in range(3):
        advance(model, disease_state, rng)
    assert disease_state.deaths / population == pytest.approx(
        0.3, abs=four_sigma(0.3, population)
    ), "the vaccine does not make a breakthrough case milder: the framework has no such effect"


# --- vaccination: rate, start, initial coverage, waning ------------------------------------------


@pytest.mark.parametrize("step_days", [1.0, 0.25])
def test_coverage_grows_as_one_minus_exp_of_rate_times_time(step_days):
    model = SvirModel(step_days, vaccination_rate=0.3, **FROM_THE_START, **KEPT_FOR_LIFE)
    population = 200_000
    disease_state, rng = new_state(model, population=population)
    elapsed = 0.0
    for days in (2.0, 5.0):
        for _ in range(round((days - elapsed) / step_days)):
            advance(model, disease_state, rng)
        elapsed = days
        expected = 1 - math.exp(-0.3 * days)
        assert count_in(model, disease_state, VACCINATED) / population == pytest.approx(
            expected, abs=four_sigma(expected, population)
        )


def test_the_rate_is_a_hazard_per_susceptible_not_a_share_of_the_town_per_day():
    model = SvirModel(ONE_DAY, vaccination_rate=0.4, **FROM_THE_START, **KEPT_FOR_LIFE)
    population = 200_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population // 2), RECOVERED)
    advance(model, disease_state, rng)
    expected = 0.5 * (1 - math.exp(-0.4))
    assert count_in(model, disease_state, VACCINATED) / population == pytest.approx(
        expected, abs=four_sigma(expected, population)
    )
    assert count_in(model, disease_state, RECOVERED) == population // 2


def test_the_campaign_starts_on_its_day():
    model = SvirModel(
        1 / 4,
        vaccination_rate=1.0,
        vaccination_start_day=3.0,
        initial_coverage=0.0,
        **KEPT_FOR_LIFE,
    )
    disease_state, rng = new_state(model, population=10_000)
    for _ in range(12):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) == 0
    for _ in range(4):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) > 6000, "63 % in a day"


def test_a_share_of_the_people_but_not_the_first_cases_starts_vaccinated():
    model = SvirModel(ONE_DAY, initial_coverage=0.3, vaccination_rate=0.0)
    population, cases = 100_000, 1000
    disease_state, _ = new_state(model, population=population, cases=np.arange(cases))
    assert (disease_state.state[:cases] == list(model.STATE_COLORS).index(INFECTIOUS)).all()
    vaccinated = disease_state.state[cases:] == list(model.STATE_COLORS).index(VACCINATED)
    others = population - cases
    assert vaccinated.mean() == pytest.approx(0.3, abs=four_sigma(0.3, others))
    assert not disease_state.ever_infected[cases:].any()
    assert (disease_state.timer[cases:][vaccinated] > 0).all(), "their protection runs out"
    everybody = SvirModel(ONE_DAY, initial_coverage=1.0)
    disease_state, _ = new_state(everybody, population=1000, cases=[5])
    assert count_in(everybody, disease_state, VACCINATED) == 999


def test_protection_ends_after_the_given_time_on_average_and_the_person_is_susceptible_again():
    step_days = 1 / 4
    model = SvirModel(step_days, vaccine_days=20.0, **NO_CAMPAIGN)
    population = 20_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), VACCINATED)
    steps_protected = np.zeros(population)
    for _ in range(round(120 / step_days)):
        steps_protected += disease_state.state == list(model.STATE_COLORS).index(VACCINATED)
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, SUSCEPTIBLE) == population
    days = steps_protected * step_days
    error = 5 * days.std() / math.sqrt(population)
    assert days.mean() == pytest.approx(20.0 + step_days / 2, abs=error + 0.005)


def test_protection_for_life_does_not_wane():
    model = SvirModel(ONE_DAY, vaccine_days=LIFELONG_DAYS, **NO_CAMPAIGN)
    disease_state, rng = new_state(model, population=500)
    place(model, disease_state, rng, np.arange(500), VACCINATED)
    for _ in range(400):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) == 500


def test_a_campaign_keeps_the_run_going_and_an_endless_protection_without_one_does_not():
    running = SvirModel(ONE_DAY, vaccination_rate=0.1, initial_coverage=0.0, **KEPT_FOR_LIFE)
    finished = SvirModel(ONE_DAY, **NO_CAMPAIGN, **KEPT_FOR_LIFE)
    for model, expected in ((running, False), (finished, True)):
        disease_state, rng = new_state(model, population=10)
        place(model, disease_state, rng, [1, 2], VACCINATED)
        assert model.is_quiescent(disease_state) is expected


# --- age effects -------------------------------------------------------------------------------


def test_age_changes_the_susceptibility_of_the_unvaccinated_and_the_vaccinated_alike():
    model = SvirModel(
        ONE_DAY,
        transmission=0.5,
        vaccine_efficacy=0.6,
        **NO_CAMPAIGN,
        **UNIFORM,
        age_effects={"susceptibility": [[0, 15, 0.5]]},
    )
    population = 200_000
    for state, hazard in ((SUSCEPTIBLE, 1.0), (VACCINATED, 0.4)):
        for age, relative in ((5, 0.5), (40, 1.0)):
            disease_state, rng = new_state(model, population=population, age=age)
            place(model, disease_state, rng, np.arange(population), state)
            advance(model, disease_state, rng, exposure=1)
            expected = 1 - 0.5 ** (hazard * relative)
            assert disease_state.cumulative_infections / population == pytest.approx(
                expected, abs=four_sigma(expected, population)
            ), (state, age)


def test_age_changes_the_chance_of_dying_of_the_infection():
    model = SvirModel(
        ONE_DAY,
        fatality=0.2,
        infectious_days=0.5,
        duration_shape=50.0,
        age_effects={"fatality": [[0, 15, 0.5], [65, 120, 2.0]]},
    )
    population = 40_000
    for age, risk in ((5, 0.1), (40, 0.2), (80, 0.4)):
        disease_state, rng = new_state(
            model, population=population, cases=np.arange(population), age=age
        )
        for _ in range(3):
            advance(model, disease_state, rng)
        assert disease_state.deaths / population == pytest.approx(
            risk, abs=four_sigma(risk, population)
        ), age


# --- preview numbers ---------------------------------------------------------------------------


def test_the_preview_numbers_describe_the_infection_of_an_unvaccinated_town():
    shape, days, transmission = 4.0, 5.0, 0.3
    model = SvirModel(
        ONE_DAY,
        transmission=transmission,
        infectious_days=days,
        duration_shape=shape,
        fatality=0.4,
        vaccine_efficacy=0.9,
        initial_coverage=0.5,
        vaccination_rate=0.1,
    )
    hazard = -math.log(1 - transmission)
    assert model.infectious_days() == pytest.approx(days), "also those who die spread for a stay"
    assert model.mean_field_r0(3.0) == pytest.approx(hazard * 3.0 * days)
    assert model.pair_transmission_probability() == pytest.approx(
        1 - (1 + hazard * days / shape) ** -shape
    )


# --- the textbook equations in a well-mixed population ------------------------------------------


def svir_equations(
    nu: float, start: float, efficacy: float, protection_days: float, step_days: float = STEP
):
    """Right-hand side of the SVIR equations for the state (S, V, I, R, clock)."""
    recovery = 1 / (INFECTIOUS_DAYS + step_days / 2)
    waning = 1 / (protection_days + step_days / 2)

    def derivative(y):
        s, v, i, _, clock = y
        force = BETA * i
        vaccination = nu * s if clock >= start else 0.0
        return np.array(
            [
                -force * s - vaccination + waning * v,
                vaccination - (1 - efficacy) * force * v - waning * v,
                force * (s + (1 - efficacy) * v) - recovery * i,
                recovery * i,
                1.0,
            ]
        )

    return derivative


def test_the_mean_course_follows_the_svir_equations_with_a_campaign_and_waning():
    population, cases, days = 12_000, 600, 50
    nu, start, efficacy, coverage, protection_days = 0.02, 5.0, 0.8, 0.1, 40.0
    runs = np.array(
        [
            run_well_mixed(
                SvirModel,
                population=population,
                cases=cases,
                days=days,
                seed=seed,
                **{**WELL_MIXED, "vaccine_days": protection_days},
                vaccine_efficacy=efficacy,
                vaccination_rate=nu,
                vaccination_start_day=start,
                initial_coverage=coverage,
            )[1]
            for seed in range(5)
        ]
    )
    mean = runs.mean(axis=0)[:, :4] / population
    infected = cases / population
    vaccinated = coverage * (1 - infected)
    ode = integrate(
        svir_equations(nu, start, efficacy, protection_days),
        np.array([1 - infected - vaccinated, vaccinated, infected, 0.0, 0.0]),
        days,
    )
    # one run deviates from the average by at most 0.01 (measured over 30 runs), so the mean of 5
    # runs by 4 x 0.01 / sqrt(5) = 0.018; the equations carry no bias beyond 0.005 (30 runs)
    for column, name in enumerate("SVIR"):
        assert np.abs(mean[:, column] - ode[:, column]).max() < 0.02, name
    assert mean[:, 2].max() == pytest.approx(ode[:, 2].max(), abs=0.02)
    assert mean[:, 2].argmax() * STEP == pytest.approx(ode[:, 2].argmax() * STEP, abs=0.75)


@pytest.mark.parametrize(
    ("efficacy", "coverage"),
    [(0.8, 0.0), (0.8, 0.3), (0.8, 0.9), (0.5, 0.5), (0.5, 0.8), (0.0, 0.5)],
)
def test_the_final_size_follows_the_relation_of_a_leaky_vaccine(efficacy, coverage):
    assert final_size_in_a_well_mixed_town(efficacy, coverage) == pytest.approx(
        predicted_final_size(efficacy, coverage), abs=FINAL_SIZE_TOLERANCE
    )


def test_with_a_perfect_vaccine_the_final_size_is_that_of_an_sir_epidemic_with_a_lower_r0():
    population, cases, coverage = FINAL_SIZE_POPULATION, FINAL_SIZE_CASES, 0.5
    unvaccinated = cases + (population - cases) * (1 - coverage)
    sir = final_size(R0 * unvaccinated / population, cases / unvaccinated)
    observed = final_size_in_a_well_mixed_town(1.0, coverage) * population / unvaccinated
    assert observed == pytest.approx(sir, abs=FINAL_SIZE_TOLERANCE * population / unvaccinated)


def test_the_epidemic_takes_off_below_the_critical_coverage_and_stays_small_above_it():
    efficacy, below, above = 0.8, 0.5, 0.9
    critical = (1 - 1 / R0) / efficacy
    assert below < critical < above
    r_above = R0 * (1 - above * efficacy)
    assert r_above < 1 < R0 * (1 - below * efficacy)
    assert final_size_in_a_well_mixed_town(efficacy, below) > 0.4
    # R_v < 1: the outbreak is a small multiple of the first cases, i0 / (1 - R_v)
    small = final_size_in_a_well_mixed_town(efficacy, above)
    assert small < 0.1
    assert small == pytest.approx(
        FINAL_SIZE_CASES / FINAL_SIZE_POPULATION / (1 - r_above), abs=FINAL_SIZE_TOLERANCE
    )


def test_a_vaccine_that_is_too_weak_cannot_stop_the_epidemic_even_when_most_have_it():
    assert (1 - 1 / R0) / 0.5 > 1, "no coverage below 100 % reaches the critical one"
    assert final_size_in_a_well_mixed_town(0.5, 0.8) > 0.4


def test_a_campaign_that_starts_late_changes_less_and_follows_the_equations():
    population, cases, days = 8000, 80, 55
    nu, efficacy = 0.05, 0.9
    attack = {}
    for start in (0.0, 6.0, 30.0):
        runs = [
            run_well_mixed(
                SvirModel,
                population=population,
                cases=cases,
                days=days,
                seed=seed,
                **WELL_MIXED,
                vaccine_efficacy=efficacy,
                vaccination_rate=nu,
                vaccination_start_day=start,
                initial_coverage=0.0,
            )[1][-1]
            for seed in range(3)
        ]
        observed = np.mean([row[[2, 3]].sum() for row in runs]) / population
        infected = cases / population
        ode = integrate(
            svir_equations(nu, start, efficacy, KEPT_FOR_LIFE["vaccine_days"]),
            np.array([1 - infected, 0.0, infected, 0.0, 0.0]),
            days,
        )
        # the mean of 3 runs deviates from the equations by at most 4 x 0.012 / sqrt(3) = 0.028
        assert observed == pytest.approx(ode[-1, 2] + ode[-1, 3], abs=0.028), start
        attack[start] = observed
    assert attack[0.0] < attack[6.0] < attack[30.0]
    assert attack[30.0] > 0.9 * leaky_final_size(R0, cases / population, 0.0, efficacy)


def test_the_shorter_the_protection_the_larger_the_epidemic():
    sizes = [
        ever_infected(
            2,
            6000,
            120,
            80,
            **{**WELL_MIXED, "vaccine_days": protection_days},
            vaccine_efficacy=0.9,
            initial_coverage=0.6,
            vaccination_rate=0.0,
        )
        for protection_days in (5.0, 30.0, float(LIFELONG_DAYS))
    ]
    assert sizes[0] > sizes[1] + 0.1 > sizes[2] + 0.1
    assert sizes[0] > 0.85, "with a few days of protection nearly everybody is infected"


# --- in the town -------------------------------------------------------------------------------


def town_infections(coverage: float, efficacy: float) -> int:
    parameters = {
        "transmission": 0.9,
        "contact_distance": 4.0,
        "vaccine_efficacy": efficacy,
        "initial_coverage": coverage,
        "vaccination_rate": 0.0,
        "vaccine_days": LIFELONG_DAYS,
    }
    model = make_model(
        {"a": disease_definition("svir", 5, **parameters)}, population=300, arena_size=500
    )
    for _ in range(24 * 5):
        model.step()
    return model.disease_states["a"].cumulative_infections


def test_in_the_town_people_with_a_perfect_vaccine_are_never_infected_and_without_it_they_are():
    assert town_infections(coverage=1.0, efficacy=1.0) == 5, "only the first cases"
    assert town_infections(coverage=0.0, efficacy=1.0) > 50


def test_a_vaccinated_person_who_catches_it_goes_through_infectious_to_recovered():
    model = SvirModel(
        ONE_DAY, **SURE, vaccine_efficacy=0.5, infectious_days=0.5, duration_shape=50.0
    )
    disease_state, rng = new_state(model)
    place(model, disease_state, rng, [0], VACCINATED)
    advance(model, disease_state, rng, exposure=1)
    visited = [state_names(model, disease_state)[0]]
    for _ in range(10):
        advance(model, disease_state, rng)
        if (current := state_names(model, disease_state)[0]) != visited[-1]:
            visited.append(current)
    assert visited == [INFECTIOUS, RECOVERED]


# (values both the demo disease and the model default have, demo-only values, age effects)
SOURCED = (
    {  # Carrat et al. 2008, Belongia et al. 2016, CDC 2010
        "infectious_days": 4.8,
        "vaccine_efficacy": 0.61,
        "vaccination_rate": 0.0025,
        "vaccine_days": 110,
    },
    {"initial_coverage": 0.3, "vaccination_start_day": 0},
    {"susceptibility": [[0, 65, 1.0], [65, 120, 0.66]]},  # Hancock et al. 2009
)


def test_the_demo_disease_has_the_numbers_its_comment_cites():
    check_demo_numbers("svir", *SOURCED)


def test_the_model_defaults_are_the_cited_numbers():
    check_default_numbers("svir", SOURCED[0])
