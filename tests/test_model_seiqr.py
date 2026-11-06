"""The SEIQR model against its definition: isolation of detected cases in an SEIR course.

Structure (who spreads, who is confined, who can die), the shares and times of the course against
the declaration, the well-mixed dynamics against the SEIQR equations and the final size relation,
the effect of every parameter in the direction theory gives, and the model in a small town.
"""

import math

import numpy as np
import pytest
from cited_numbers import check_default_numbers, check_disease_numbers
from config_factory import LIBRARY, disease_of, make_model
from disease_helpers import ONE_DAY, SURE, advance, count_in, new_state, place
from mean_field import final_size, integrate, run_well_mixed
from synthetic_models import LatentModel
from test_epidemic_model import arranged_model, put_into_state, run

from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    DEAD,
    DYING,
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.seiqr import QUARANTINED, UNDETECTED, SeiqrModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
SHARP = {"duration_shape": 50.0}
STATES = list(SeiqrModel.STATE_COLORS)
ODE_STATES = [
    STATES.index(state)
    for state in (SUSCEPTIBLE, EXPOSED, INFECTIOUS, UNDETECTED, QUARANTINED, RECOVERED)
]


def seiqr(step_days=ONE_DAY, **parameters) -> SeiqrModel:
    return SeiqrModel(step_days, **parameters)


def share_in(model, disease_state, state: str) -> float:
    return count_in(model, disease_state, state) / len(disease_state.state)


# --- structure -----------------------------------------------------------------------------------


def test_the_states_and_their_order_are_the_seiqr_course_with_a_state_for_the_undetected():
    assert STATES == [
        SUSCEPTIBLE,
        EXPOSED,
        INFECTIOUS,
        UNDETECTED,
        QUARANTINED,
        ASYMPTOMATIC,
        RECOVERED,
        DYING,
        DEAD,
    ]
    assert SeiqrModel.course_states() == {
        EXPOSED,
        INFECTIOUS,
        UNDETECTED,
        QUARANTINED,
        ASYMPTOMATIC,
        RECOVERED,
    }


def test_everybody_who_is_not_susceptible_spreads_only_while_infectious_or_isolated():
    model = seiqr(quarantine_infectiousness=0.25)
    disease_state, _ = new_state(model, population=len(STATES))
    disease_state.state[:] = np.arange(len(STATES))
    weights = dict(zip(STATES, model.infectious_weight(disease_state).tolist(), strict=True))
    assert weights.pop(ASYMPTOMATIC) > 0
    assert weights == {
        SUSCEPTIBLE: 0.0,
        EXPOSED: 0.0,
        INFECTIOUS: 1.0,
        UNDETECTED: 1.0,
        QUARANTINED: 0.25,
        RECOVERED: 0.0,
        DYING: 0.0,
        DEAD: 0.0,
    }


def test_only_the_detected_are_confined_and_only_the_free_react_to_being_ill():
    assert frozenset({QUARANTINED}) == SeiqrModel.CONFINED_STATES
    assert frozenset({INFECTIOUS, UNDETECTED}) == SeiqrModel.SYMPTOMATIC_STATES
    model = seiqr(stay_home=0.0, limit_outings=0.0)
    disease_state, rng = new_state(model, population=4)
    place(model, disease_state, rng, [0], INFECTIOUS)
    place(model, disease_state, rng, [1], UNDETECTED)
    place(model, disease_state, rng, [2], QUARANTINED)
    assert disease_state.stays_home[:4].tolist() == [False, False, True, False]


def test_a_person_detected_is_confined_whatever_the_stay_at_home_setting():
    model = seiqr(stay_home=0.0, limit_outings=0.0, detection_coverage=1.0, **SHARP)
    disease_state, rng = new_state(model, population=200)
    place(model, disease_state, rng, np.arange(200), INFECTIOUS)
    assert not disease_state.stays_home.any()
    for _ in range(6):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, QUARANTINED) == 200
    assert disease_state.stays_home.all()


def test_a_person_who_is_not_detected_keeps_the_decision_to_stay_home_they_made_when_infectious():
    model = seiqr(stay_home=0.5, detection_coverage=0.0, detection_delay_days=0.5, **SHARP)
    disease_state, rng = new_state(model, population=4000)
    place(model, disease_state, rng, np.arange(4000), INFECTIOUS)
    decided = disease_state.stays_home.copy()
    assert decided.mean() == pytest.approx(0.5, abs=0.04)
    advance(model, disease_state, rng)
    assert count_in(model, disease_state, UNDETECTED) == 4000
    assert (disease_state.stays_home == decided).all()


def test_the_first_cases_start_infectious_and_not_yet_detected():
    model = seiqr()
    disease_state, _ = new_state(model, population=10, cases=[3, 7])
    assert disease_state.state.tolist().count(STATES.index(INFECTIOUS)) == 2
    assert disease_state.ever_infected.sum() == 2


def test_an_infection_enters_the_exposed_state_and_recovered_people_are_immune():
    model = seiqr(**SURE, immunity_days=36500, immunity_protection=1.0)
    disease_state, rng = new_state(model, population=2)
    place(model, disease_state, rng, [1], RECOVERED)
    advance(model, disease_state, rng, exposure=1)
    assert disease_state.state.tolist() == [STATES.index(EXPOSED), STATES.index(RECOVERED)]


def test_nobody_dies_in_the_states_before_the_end_of_the_infectious_period():
    model = seiqr(fatality=1.0, detection_coverage=0.5, **SHARP)
    for state in (EXPOSED, INFECTIOUS):
        disease_state, rng = new_state(model, population=500)
        place(model, disease_state, rng, np.arange(500), state)
        for _ in range(int(model.parameters["detection_delay_days"])):
            advance(model, disease_state, rng)
        assert disease_state.deaths == 0, state


# --- shares and times of the course --------------------------------------------------------------


@pytest.mark.parametrize("coverage", [0.0, 0.3, 0.8, 1.0])
def test_the_share_detected_is_the_detection_coverage(coverage):
    model = seiqr(detection_coverage=coverage, **SHARP)
    population = 40_000
    disease_state, rng = new_state(model, population)
    place(model, disease_state, rng, np.arange(population), INFECTIOUS)
    for _ in range(5):  # the delay is 3 days, sharp: nobody is still waiting after 5 days
        advance(model, disease_state, rng)
    assert share_in(model, disease_state, QUARANTINED) == pytest.approx(
        coverage, abs=4 * math.sqrt(max(coverage * (1 - coverage), 1e-4) / population) + 1e-3
    )
    assert share_in(model, disease_state, UNDETECTED) + share_in(
        model, disease_state, QUARANTINED
    ) == pytest.approx(1.0, abs=1e-3)  # a few recover in 5 days: delay plus stay is 8 +- 0.8


def test_the_share_detected_depends_on_age_through_the_branching_effect_and_is_capped_at_one():
    effects = {"branching": [[0, 15, 0.5], [65, 120, 2.5]]}
    model = seiqr(detection_coverage=0.4, age_effects=effects, **SHARP)
    population = 90_000
    ages = np.repeat([5, 40, 80], population // 3)
    disease_state, rng = new_state(model, population, age=ages)
    place(model, disease_state, rng, np.arange(population), INFECTIOUS)
    for _ in range(6):
        advance(model, disease_state, rng)
    detected = (disease_state.state == STATES.index(QUARANTINED)).reshape(3, -1).mean(axis=1)
    sigma = np.sqrt(np.array([0.2 * 0.8, 0.4 * 0.6, 1e-4]) / (population // 3))
    assert detected == pytest.approx([0.2, 0.4, 1.0], abs=4 * sigma.max() + 1e-3)


def test_the_age_effect_on_fatality_applies_to_the_detected_and_the_undetected_alike():
    effects = {"fatality": [[0, 15, 0.0], [65, 120, 2.0]]}
    model = seiqr(detection_coverage=0.5, fatality=0.2, age_effects=effects, **SHARP)
    population = 60_000
    ages = np.repeat([5, 40, 80], population // 3)
    disease_state, rng = new_state(model, population, age=ages)
    place(model, disease_state, rng, np.arange(population), INFECTIOUS)
    for _ in range(40):
        advance(model, disease_state, rng)
    assert disease_state.state.min() >= STATES.index(RECOVERED)
    dead = (disease_state.state == model.dead_index).reshape(3, -1).mean(axis=1)
    sigma = math.sqrt(0.4 * 0.6 / (population // 3))
    assert dead == pytest.approx([0.0, 0.2, 0.4], abs=4 * sigma)


def cohort_stays(model, population, days, seed=0):
    """Everybody infectious at once: visits, steps spent in each state and deaths by route."""
    disease_state, rng = new_state(model, population, seed=seed)
    place(model, disease_state, rng, np.arange(population), INFECTIOUS)
    visited = np.zeros((len(STATES), population), np.bool_)
    steps_in = np.zeros((len(STATES), population))
    for _ in range(round(days / model.step_days)):
        for index in range(len(STATES)):
            here = disease_state.state == index
            visited[index] |= here
            steps_in[index] += here
        advance(model, disease_state, rng)
    return visited, steps_in, disease_state


def test_the_time_spent_free_and_isolated_is_what_the_parameters_say_and_death_follows_either():
    step_days, population, coverage, fatality = 1 / 12, 20_000, 0.6, 0.3
    model = seiqr(
        step_days,
        detection_delay_days=2.0,
        undetected_days=5.0,
        quarantine_days=4.0,
        detection_coverage=coverage,
        fatality=fatality,
    )
    visited, steps_in, disease_state = cohort_stays(model, population, days=60.0)
    days_in = steps_in * step_days
    half = step_days / 2  # a stay is the continuous time rounded up to whole steps
    free = days_in[STATES.index(INFECTIOUS)] + days_in[STATES.index(UNDETECTED)]
    isolated = days_in[STATES.index(QUARANTINED)]
    expected_free = 2.0 + (1 - coverage) * 5.0 + half * (2 - coverage)
    expected_isolated = coverage * (4.0 + half)
    assert free.mean() == pytest.approx(expected_free, abs=5 * free.std() / math.sqrt(population))
    assert isolated.mean() == pytest.approx(
        expected_isolated, abs=5 * isolated.std() / math.sqrt(population)
    )
    assert visited[STATES.index(QUARANTINED)].mean() == pytest.approx(coverage, abs=0.01)
    assert disease_state.deaths / population == pytest.approx(fatality, abs=0.015)
    for route in (QUARANTINED, UNDETECTED):
        took = visited[STATES.index(route)]
        died = (disease_state.state == model.dead_index) & took
        assert died.sum() / took.sum() == pytest.approx(fatality, abs=0.025), route


def test_the_natural_infectious_period_is_the_delay_plus_the_undetected_period():
    model = seiqr(detection_coverage=0.0, detection_delay_days=2.0, undetected_days=5.0)
    assert model.infectious_days() == pytest.approx(
        LatentModel(ONE_DAY, infectious_days=7.0).infectious_days()
    )
    assert model.pair_transmission_probability() > 0


# --- preview numbers -----------------------------------------------------------------------------


def test_infectious_days_are_the_free_time_plus_the_isolated_time_weighted_by_infectiousness():
    model = seiqr(
        detection_delay_days=3.0,
        undetected_days=6.0,
        quarantine_days=5.0,
        detection_coverage=0.7,
        quarantine_infectiousness=0.4,
    )
    assert model.infectious_days() == pytest.approx(3.0 + 0.3 * 6.0 + 0.7 * 0.4 * 5.0)


def test_mean_field_reproduction_number_follows_the_weighted_infectious_days():
    model = seiqr(transmission=0.5, detection_coverage=0.5, quarantine_infectiousness=0.2)
    assert model.mean_field_r0(2.0) == pytest.approx(-math.log(0.5) * 2.0 * model.infectious_days())
    assert seiqr(**SURE).mean_field_r0(2.0) is None


def test_the_reproduction_number_falls_with_coverage_and_rises_with_delay_and_leakage():
    def days(**parameters):
        return seiqr(**parameters).infectious_days()

    base = {"detection_coverage": 0.5, "quarantine_infectiousness": 0.3}
    assert days(detection_coverage=0.0) > days(**base) > days(detection_coverage=1.0)
    assert days(**base, detection_delay_days=1.0) < days(**base, detection_delay_days=4.0)
    assert days(**{**base, "quarantine_infectiousness": 0.0}) < days(**base)
    assert days(**base, quarantine_days=2.0) < days(**base, quarantine_days=9.0)
    delay = seiqr().parameters["detection_delay_days"]
    assert days(detection_coverage=1.0, quarantine_infectiousness=0.0) == pytest.approx(delay)


@pytest.mark.parametrize("shape", [1.0, 4.0, 30.0])
def test_pair_transmission_probability_matches_sampled_courses(shape):
    coverage, weight, delay, free, isolated = 0.6, 0.35, 2.0, 5.0, 4.0
    model = seiqr(
        transmission=0.2,
        duration_shape=shape,
        detection_coverage=coverage,
        quarantine_infectiousness=weight,
        detection_delay_days=delay,
        undetected_days=free,
        quarantine_days=isolated,
    )
    rng = np.random.default_rng(0)
    count = 600_000
    detected = rng.random(count) < coverage
    spread = rng.gamma(shape, delay / shape, count) + np.where(
        detected,
        weight * rng.gamma(shape, isolated / shape, count),
        rng.gamma(shape, free / shape, count),
    )
    risk = 1 - np.exp(math.log(0.8) * spread)
    assert model.pair_transmission_probability() == pytest.approx(
        risk.mean(), abs=5 * risk.std() / math.sqrt(count)
    )


# --- well-mixed dynamics against the equations ---------------------------------------------------


def hazard_per_contact(transmission: float) -> float:
    return -math.log(1 - transmission)


def test_with_exponential_stays_the_mean_course_follows_the_seiqr_equations():
    transmission, contacts = 0.3, 3.0
    beta = hazard_per_contact(transmission) * contacts
    p = {
        "incubation_days": 3.0,
        "detection_delay_days": 2.0,
        "undetected_days": 4.0,
        "quarantine_days": 4.0,
        "detection_coverage": 0.6,
        "quarantine_infectiousness": 0.3,
    }
    q, w = p["detection_coverage"], p["quarantine_infectiousness"]

    def derivative(y):
        s, e, a, b, quarantined, r = y
        force = beta * (a + b + w * quarantined)
        return np.array(
            [
                -force * s,
                force * s - e / p["incubation_days"],
                e / p["incubation_days"] - a / p["detection_delay_days"],
                (1 - q) * a / p["detection_delay_days"] - b / p["undetected_days"],
                q * a / p["detection_delay_days"] - quarantined / p["quarantine_days"],
                b / p["undetected_days"] + quarantined / p["quarantine_days"],
            ]
        )

    population, cases, days, runs = 12_000, 100, 30, 4
    ode = integrate(
        derivative, np.array([1 - cases / population, 0, cases / population, 0, 0, 0]), days
    )
    counts = np.array(
        [
            run_well_mixed(
                SeiqrModel,
                population=population,
                cases=cases,
                days=days,
                contacts=contacts,
                seed=seed,
                transmission=transmission,
                duration_shape=1.0,
                **p,
                **IDENTICAL,
            )[1][:, ODE_STATES]
            / population
            for seed in range(runs)
        ]
    )
    mean, spread = counts.mean(axis=0), counts.std(axis=0, ddof=1)
    # Noise: 4 standard errors of the mean of the runs. Bias: every stay is rounded up to whole
    # steps, which delays the course by well under 0.1 day, so the curve may be off by that much
    # of its slope.
    tolerance = 4 * spread / math.sqrt(runs) + 0.1 * np.abs(np.gradient(ode, 1 / 24, axis=0))
    days_checked = np.array([5, 10, 15, 20, 30]) * 24
    assert (np.abs(mean - ode)[days_checked] <= tolerance[days_checked] + 1e-3).all()


def relative_bias(model, step_days):
    """Days of spreading that rounding the stays up to whole steps adds, in the courses' mean."""
    return sum(
        course.probability
        * sum(weight * step_days / 2 for _, _, weight in course.stays if weight > 0)
        for course in model.courses()
    )


FINAL_SIZE_CASES = {
    "no detection": {"detection_coverage": 0.0},
    "half detected, leaky": {"detection_coverage": 0.5, "quarantine_infectiousness": 0.5},
    "all detected, tight": {
        "detection_coverage": 1.0,
        "quarantine_infectiousness": 0.0,
        "detection_delay_days": 3.5,
    },
    "late detection of most": {
        "detection_coverage": 0.9,
        "quarantine_infectiousness": 0.1,
        "detection_delay_days": 4.5,
        "undetected_days": 1.5,
        "quarantine_days": 1.5,
    },
}


@pytest.mark.parametrize("name", FINAL_SIZE_CASES)
def test_the_share_ever_infected_follows_the_final_size_relation_of_the_weighted_infectious_time(
    name,
):
    contacts, transmission, step_days = 0.6, 0.5, 1 / 12
    parameters = {
        "incubation_days": 2.0,
        "detection_delay_days": 2.0,
        "undetected_days": 4.0,
        "quarantine_days": 4.0,
        **FINAL_SIZE_CASES[name],
    }
    model = SeiqrModel(step_days, transmission=transmission, **parameters)
    r = (
        hazard_per_contact(transmission)
        * contacts
        * (model.infectious_days() + relative_bias(model, step_days))
    )
    assert 1.3 < r < 2.6, "the case is chosen to be sensitive to the reproduction number"
    population, cases, runs = 10_000, 100, 4
    shares = []
    for seed in range(runs):
        _, counts = run_well_mixed(
            SeiqrModel,
            population=population,
            cases=cases,
            days=150,
            contacts=contacts,
            seed=seed,
            step_days=step_days,
            transmission=transmission,
            **parameters,
            **IDENTICAL,
        )
        shares.append(1 - counts[-1, 0] / population)
    expected = final_size(r, cases / population)
    # the spread between runs, never less than the binomial noise of one run
    sigma = max(np.std(shares, ddof=1), math.sqrt(expected * (1 - expected) / population))
    assert np.mean(shares) == pytest.approx(expected, abs=4 * sigma / math.sqrt(runs))


def test_without_detection_the_course_has_the_final_size_of_an_seir_epidemic():
    contacts, transmission, population, cases, runs = 1.2, 0.5, 10_000, 100, 4
    common = {
        "incubation_days": 3.0,
        "transmission": transmission,
        "population": population,
        "cases": cases,
        "days": 100,
        "contacts": contacts,
        "step_days": 1 / 12,
        **IDENTICAL,
    }

    def ever_infected(model_class, **parameters):
        return np.array(
            [
                1
                - run_well_mixed(model_class, seed=seed, **common, **parameters)[1][-1, 0]
                / population
                for seed in range(runs)
            ]
        )

    seir = ever_infected(LatentModel, infectious_days=3.0, immunity_days=36500)
    nobody = ever_infected(
        SeiqrModel,
        detection_coverage=0.0,
        detection_delay_days=1.0,
        undetected_days=2.0,
        quarantine_days=2.0,
    )
    standard_error = math.sqrt((seir.var(ddof=1) + nobody.var(ddof=1)) / runs)
    assert nobody.mean() == pytest.approx(seir.mean(), abs=4 * standard_error + 1e-3)
    r0 = hazard_per_contact(transmission) * contacts * 3.0
    assert seir.mean() == pytest.approx(final_size(r0, cases / population), abs=0.015)


def test_perfect_prompt_isolation_stops_the_epidemic_that_spreads_freely():
    contacts, transmission, population = 2.0, 0.5, 10_000
    free = {"detection_coverage": 0.0}
    prompt = {
        "detection_coverage": 1.0,
        "detection_delay_days": 0.1,
        "quarantine_infectiousness": 0.0,
    }
    ever = {}
    for name, parameters in (("free", free), ("prompt", prompt)):
        _, counts = run_well_mixed(
            SeiqrModel,
            population=population,
            cases=100,
            days=60,
            contacts=contacts,
            seed=1,
            step_days=1 / 12,
            transmission=transmission,
            **parameters,
            **IDENTICAL,
        )
        ever[name] = 1 - counts[-1, 0] / population
    step_days = 1 / 12
    model = seiqr(step_days, **prompt)
    r = (
        hazard_per_contact(transmission)
        * contacts
        * (model.infectious_days() + relative_bias(model, step_days))
    )
    assert r < 0.2, "R is far below 1"
    assert ever["free"] > 0.9
    # a subcritical branching process: the 100 cases found 100 / (1 - R) infections in total, with a
    # variance of 100 R / (1 - R)^3 (Poisson offspring)
    total = 100 / (1 - r)
    assert ever["prompt"] * population == pytest.approx(
        total, abs=4 * math.sqrt(100 * r / (1 - r) ** 3)
    )


def test_a_longer_delay_raises_the_reproduction_number_past_one():
    contacts, transmission = 1.0, 0.5
    hazard = hazard_per_contact(transmission) * contacts
    for delay, spreads in ((1.0, False), (2.5, True)):
        model = seiqr(
            detection_coverage=1.0, quarantine_infectiousness=0.0, detection_delay_days=delay
        )
        assert (hazard * model.infectious_days() > 1) is spreads


def test_a_longer_latent_period_postpones_the_peak_but_keeps_the_final_size():
    peaks, finals = [], []
    for incubation in (2.0, 6.0):
        _, counts = run_well_mixed(
            SeiqrModel,
            population=10_000,
            cases=100,
            days=100,
            contacts=2.0,
            seed=3,
            step_days=1 / 12,
            transmission=0.4,
            incubation_days=incubation,
            **IDENTICAL,
        )
        spreading = counts[:, 2:5].sum(axis=1)
        peaks.append(spreading.argmax() / 12)
        finals.append(1 - counts[-1, 0] / 10_000)
    assert peaks[1] - peaks[0] > 3.0
    assert finals[1] == pytest.approx(finals[0], abs=0.02)


# --- the town ------------------------------------------------------------------------------------


def seiqr_town(**parameters):
    diseases = {"a": {"model": "seiqr", "initial_infected": 0, "parameters": parameters}}
    return arranged_model(diseases, {})


def test_detected_people_are_at_home_even_when_their_day_takes_them_out():
    model = seiqr_town(stay_home=0.0, limit_outings=0.0)
    model.disease_states["a"].timer[:] = 1e6
    model.diseases["a"]._enter(
        model.disease_states["a"],
        np.arange(model.population),
        QUARANTINED,
        np.random.default_rng(0),
    )
    run(model, 24)
    away = np.hypot(model.x - model.town.home_x, model.y - model.town.home_y)
    assert away.max() < 1e-6


def test_only_the_detected_stay_at_home_when_nobody_chooses_to():
    model = seiqr_town(stay_home=0.0, limit_outings=0.0)
    for person, state in ((0, INFECTIOUS), (1, UNDETECTED), (2, QUARANTINED), (3, EXPOSED)):
        model.diseases["a"]._enter(
            model.disease_states["a"], np.array([person]), state, np.random.default_rng(person)
        )
    assert model._goes_home()[:4].tolist() == [False, False, True, False]


def test_an_isolated_person_spreads_with_the_reduced_infectiousness_and_the_free_with_the_full():
    model = seiqr_town(quarantine_infectiousness=0.3)
    put_into_state(model, 0, "a", INFECTIOUS)
    put_into_state(model, 1, "a", UNDETECTED)
    put_into_state(model, 2, "a", QUARANTINED)
    put_into_state(model, 3, "a", EXPOSED)
    assert model._infectiousness("a")[:5].tolist() == pytest.approx([1.0, 1.0, 0.3, 0.0, 0.0])


def test_the_hover_card_of_an_isolated_person_shows_the_reduced_infectiousness():
    from epidemic_sandbox.person_info import describe_person

    model = seiqr_town(quarantine_infectiousness=0.3)
    put_into_state(model, 2, "a", QUARANTINED)
    health = describe_person(model, 2)["health"]["a"]
    assert health["state"] == QUARANTINED and health["infectious"]
    assert health["infectiousness"] == pytest.approx(0.3)


def test_the_disease_is_a_seiqr_with_every_case_admitted_and_age_dependent_fatality():
    definition = LIBRARY.catalog[disease_of("seiqr")]
    assert definition.model == "seiqr"
    model = SeiqrModel(1 / 48, age_effects=definition.age_effects, **definition.parameters)
    parameters = model.parameters
    # every case is recognised and taken to hospital, where it infects nobody in the town
    assert parameters["detection_coverage"] == 1.0
    assert parameters["quarantine_infectiousness"] == 0.0
    assert parameters["detection_delay_days"] < parameters["undetected_days"]
    assert model.mean_field_r0(10.0) > 1.0
    ages = np.array([10, 30, 55, 80])
    risk = model._age_multiplier("fatality", ages) * parameters["fatality"]
    assert (np.diff(risk) > 0).all() and risk[0] < 0.01


def small_town_attack_rate(seed, **parameters) -> float:
    diseases = {"a": {"model": "seiqr", "initial_infected": 8, "parameters": parameters}}
    model = make_model(diseases, population=400, arena_size=495, seed=seed, steps_per_day=24)
    for _ in range(24 * 30):
        model.step()
    return float(model.disease_states["a"].ever_infected.mean())


def test_prompt_and_tight_isolation_reduces_the_spread_in_the_town():
    shared = {
        "transmission": 0.7,
        "contact_distance": 4.0,
        "detection_delay_days": 1.0,
        "undetected_days": 5.0,
        "quarantine_days": 5.0,
        "incubation_days": 3.0,
        "stay_home": 0.0,
        "limit_outings": 0.0,
    }
    free = [small_town_attack_rate(s, **shared, detection_coverage=0.0) for s in range(3)]
    isolated = [
        small_town_attack_rate(s, **shared, detection_coverage=1.0, quarantine_infectiousness=0.0)
        for s in range(3)
    ]
    # measured: the free epidemics reach 93 to 96 % of the people, the isolated ones 37 to 61 %
    # (the household is still infected before the case is found); a gap of 0.1 is far inside it
    assert min(free) > 0.8, "the free epidemic takes off in every run"
    assert max(isolated) < min(free) - 0.1


# (values both the disease and the model default have, the disease's own values, age effects)
SOURCED = (
    {"incubation_days": 4.6},  # Leung et al. 2004
    {
        "detection_delay_days": 3.67,  # Donnelly et al. 2003
        "undetected_days": 7.1,
        "quarantine_days": 7.1,
        "detection_coverage": 1.0,
        "quarantine_infectiousness": 0.0,
        "fatality": 0.06,  # WHO 2003
        "death_delay_days": 12.9,  # Leung et al. 2004
        "transmission": 0.049,
        "community_factor": 44.0,
        "duration_shape": 4.9,  # Lipsitch et al. 2003
        "infectiousness_dispersion": 0.25,
        "stay_home": 0.40,
    },
    {
        "fatality": [[0, 25, 0.0833], [25, 45, 1.0], [45, 65, 2.5], [65, 120, 8.333]],  # WHO 2003
        "susceptibility": [[0, 18, 0.15]],
    },
)
# the model's own defaults: the home isolation of Chowell et al. 2003 (Table 1 and the model's own
# constraint), not the admission to hospital of the disease of the catalog
DEFAULTS = {
    "incubation_days": 4.6,
    "detection_delay_days": 3.0,
    "undetected_days": 7.8,
    "quarantine_days": 7.8,
    "detection_coverage": 0.77,
    "quarantine_infectiousness": 0.38,
}


def test_the_disease_has_the_numbers_its_comment_cites():
    check_disease_numbers("seiqr", *SOURCED)


def test_the_model_defaults_are_the_cited_numbers():
    check_default_numbers("seiqr", DEFAULTS)
