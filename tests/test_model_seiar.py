"""The SEIAR model against its definition: the fork after the latent period, who spreads, who stays
at home, who can die, the textbook equations and the numbers of its demo disease."""

import math

import numpy as np
import pytest
from config_factory import LIBRARY, disease_definition, make_model
from disease_helpers import ONE_DAY, UNIFORM, advance, count_in, new_state, place
from mean_field import final_size, integrate, mean_course, run_well_mixed

from epidemic_sandbox.disease_models.disease_model_abstract import (
    DEAD,
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.seiar import ASYMPTOMATIC, SeiarModel

SHARP = {"duration_shape": 50.0}
IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
STATES = list(SeiarModel.STATE_COLORS)
FREE, LATENT, SILENT, ILL, CURED, GONE = range(len(STATES))


def short_model(**parameters) -> SeiarModel:
    """Model whose stays all end within one day step, so one `advance` moves everybody on."""
    return SeiarModel(
        ONE_DAY,
        incubation_days=0.5,
        asymptomatic_days=0.5,
        infectious_days=0.5,
        **SHARP,
        **parameters,
    )


def four_sigma(chance: float, people: int) -> float:
    """Four standard deviations of a share measured on `people` independent people."""
    return 4 * math.sqrt(max(chance * (1 - chance), 1e-4) / people)


def fork_all(model, population, ages=None, seed=0):
    """Everybody exposed at once and moved through the fork by one step; returns the state."""
    disease_state, rng = new_state(model, population, age=40 if ages is None else ages, seed=seed)
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    advance(model, disease_state, rng)
    return disease_state, rng


# --- the declaration ----------------------------------------------------------------------------


def test_the_model_is_susceptible_exposed_asymptomatic_or_symptomatic_then_recovered_or_dead():
    assert STATES == [SUSCEPTIBLE, EXPOSED, ASYMPTOMATIC, INFECTIOUS, RECOVERED, DEAD]
    assert SeiarModel.INFECTION_ENTRY_STATE == EXPOSED
    assert set(SeiarModel.INFECTIOUS_STATES) == {ASYMPTOMATIC, INFECTIOUS}
    assert SeiarModel.course_states() == {EXPOSED, ASYMPTOMATIC, INFECTIOUS, RECOVERED}
    phases = SeiarModel.PHASES
    assert (phases[EXPOSED].next_state, phases[EXPOSED].branch_state) == (INFECTIOUS, ASYMPTOMATIC)
    assert phases[ASYMPTOMATIC].next_state == phases[INFECTIOUS].next_state == RECOVERED
    assert [state for state, phase in phases.items() if phase.fatal] == [INFECTIOUS]


def test_an_infected_person_takes_one_of_two_ways_and_only_the_symptomatic_can_die():
    model = short_model(fatality=0.5, asymptomatic_share=0.5)
    population = 20_000
    disease_state, rng = new_state(model, population)
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    visited = np.zeros((len(STATES), population), np.bool_)
    for _ in range(4):
        advance(model, disease_state, rng)
        for index in range(len(STATES)):
            visited[index] |= disease_state.state == index
    assert not (visited[SILENT] & visited[ILL]).any(), "the two ways exclude each other"
    assert (visited[SILENT] | visited[ILL]).all()
    dead = disease_state.state == GONE
    assert not (dead & visited[SILENT]).any(), "nobody dies of a silent infection"
    assert dead.sum() == disease_state.deaths
    assert dead[visited[ILL]].mean() == pytest.approx(0.5, abs=four_sigma(0.5, visited[ILL].sum()))
    assert (disease_state.state[visited[SILENT]] == CURED).all()


# --- the fork -----------------------------------------------------------------------------------


def test_the_asymptomatic_share_of_each_age_group_is_the_share_times_its_multiplier():
    effects = {"branching": [[0, 20, 1.2], [20, 60, 1.0], [60, 120, 0.5]]}
    model = short_model(asymptomatic_share=0.6, age_effects=effects)
    per_group = 30_000
    ages = np.repeat([10, 40, 80], per_group)
    disease_state, _ = fork_all(model, 3 * per_group, ages)
    silent = (disease_state.state == SILENT).reshape(3, per_group).mean(axis=1)
    expected = [0.72, 0.6, 0.3]
    assert (disease_state.state == ILL).sum() + (
        disease_state.state == SILENT
    ).sum() == 3 * per_group
    for found, chance in zip(silent, expected, strict=True):
        assert found == pytest.approx(chance, abs=four_sigma(chance, per_group))


@pytest.mark.parametrize(("share", "everybody_silent"), [(0.0, False), (1.0, True)])
def test_a_share_of_zero_or_one_sends_everybody_the_same_way(share, everybody_silent):
    disease_state, _ = fork_all(short_model(asymptomatic_share=share), 500)
    way = ASYMPTOMATIC if everybody_silent else INFECTIOUS
    assert count_in(short_model(), disease_state, way) == 500


def test_a_multiplier_that_pushes_the_chance_above_one_means_certainty():
    effects = {"branching": [[0, 120, 5.0]]}
    disease_state, _ = fork_all(short_model(asymptomatic_share=0.5, age_effects=effects), 2000)
    assert (disease_state.state == SILENT).all()


def test_the_branching_age_effect_works_on_the_symptomatic_side_too():
    effects = {"branching": [[0, 120, 0.0]]}
    disease_state, _ = fork_all(short_model(asymptomatic_share=0.9, age_effects=effects), 2000)
    assert (disease_state.state == ILL).all()


# --- who spreads, who stays at home, who dies ----------------------------------------------------


def test_the_asymptomatic_spread_with_their_relative_infectiousness_and_the_exposed_not_at_all():
    model = SeiarModel(ONE_DAY, asymptomatic_infectiousness=0.3)
    disease_state, _ = new_state(model, population=len(STATES))
    disease_state.state[:] = np.arange(len(STATES))
    weights = dict(zip(STATES, model.infectious_weight(disease_state).tolist(), strict=True))
    assert weights == {
        SUSCEPTIBLE: 0.0,
        EXPOSED: 0.0,
        ASYMPTOMATIC: pytest.approx(0.3),
        INFECTIOUS: 1.0,
        RECOVERED: 0.0,
        DEAD: 0.0,
    }


def test_only_people_who_fall_ill_stay_home_or_skip_outings():
    model = SeiarModel(ONE_DAY, stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, population=4)
    for person, state in enumerate((INFECTIOUS, ASYMPTOMATIC, EXPOSED, RECOVERED)):
        place(model, disease_state, rng, [person], state)
    assert disease_state.stays_home.tolist() == [True, False, False, False]
    assert disease_state.limits_outings.tolist() == [True, False, False, False]


def test_a_person_who_goes_through_the_fork_behaves_by_the_way_they_took():
    model = short_model(stay_home=1.0, limit_outings=1.0, asymptomatic_share=0.5)
    disease_state, _ = fork_all(model, 2000)
    silent, ill = disease_state.state == SILENT, disease_state.state == ILL
    assert silent.any() and ill.any()
    assert disease_state.stays_home[ill].all() and not disease_state.stays_home[silent].any()
    assert disease_state.limits_outings[ill].all()
    assert not disease_state.limits_outings[silent].any()


def test_the_fatality_is_per_symptomatic_case_and_scaled_by_age():
    effects = {
        "branching": [[0, 50, 1.0], [50, 120, 0.5]],
        "fatality": [[0, 50, 0.5], [50, 120, 2.0]],
    }
    model = short_model(asymptomatic_share=0.6, fatality=0.1, age_effects=effects)
    per_group = 40_000
    ages = np.repeat([20, 70], per_group)
    disease_state, rng = new_state(model, 2 * per_group, age=ages)
    place(model, disease_state, rng, np.arange(2 * per_group), EXPOSED)
    advance(model, disease_state, rng)
    ill_now = (disease_state.state == ILL).reshape(2, per_group)
    assert ill_now.mean(axis=1) == pytest.approx([0.4, 0.7], abs=0.01)
    advance(model, disease_state, rng)
    dead = (disease_state.state == GONE).reshape(2, per_group).mean(axis=1)
    # a person dies with the fatality of their age if they fell ill: 0.4 x 0.05 and 0.7 x 0.2
    for found, chance in zip(dead, [0.4 * 0.05, 0.7 * 0.2], strict=True):
        assert found == pytest.approx(chance, abs=four_sigma(chance, per_group))


def test_the_susceptibility_age_effect_scales_the_chance_of_catching_it():
    effects = {"susceptibility": [[0, 20, 0.5]]}
    model = SeiarModel(ONE_DAY, age_effects=effects, **UNIFORM)
    disease_state, _ = new_state(model, population=2, age=[10, 40])
    assert disease_state.susceptibility.tolist() == [0.5, 1.0]


def test_an_age_effect_on_branching_and_fatality_is_accepted():
    SeiarModel(ONE_DAY, age_effects={"branching": [[0, 120, 1.0]], "fatality": [[0, 120, 1.0]]})


# --- the numbers shown before a run -------------------------------------------------------------


def expected_spreading_days(share, asymptomatic_days, symptomatic_days, relative):
    return share * relative * asymptomatic_days + (1 - share) * symptomatic_days


def test_infectious_days_and_the_mean_field_r0_are_the_weighted_mean_of_the_two_ways():
    model = SeiarModel(
        ONE_DAY,
        transmission=0.4,
        asymptomatic_share=0.7,
        asymptomatic_infectiousness=0.5,
        asymptomatic_days=4.0,
        infectious_days=6.0,
    )
    days = 0.7 * 0.5 * 4.0 + 0.3 * 6.0
    assert days == pytest.approx(expected_spreading_days(0.7, 4.0, 6.0, 0.5))
    assert model.infectious_days() == pytest.approx(days)
    contacts = 2.5
    assert model.mean_field_r0(contacts) == pytest.approx(-math.log(0.6) * contacts * days)


def test_the_incubation_period_does_not_change_the_reproduction_number():
    short, long = (SeiarModel(ONE_DAY, incubation_days=days) for days in (0.5, 20.0))
    assert short.mean_field_r0(3.0) == long.mean_field_r0(3.0)
    assert short.pair_transmission_probability() == long.pair_transmission_probability()


@pytest.mark.parametrize("shape", [1.0, 4.0, 30.0])
def test_the_pair_transmission_probability_matches_sampled_courses_with_the_fork(shape):
    model = SeiarModel(
        ONE_DAY,
        transmission=0.3,
        asymptomatic_share=0.6,
        asymptomatic_infectiousness=0.4,
        asymptomatic_days=3.0,
        infectious_days=5.0,
        duration_shape=shape,
    )
    rng = np.random.default_rng(1)
    samples = 400_000
    silent = rng.random(samples) < 0.6
    weighted_days = np.where(
        silent,
        0.4 * rng.gamma(shape, 3.0 / shape, samples),
        rng.gamma(shape, 5.0 / shape, samples),
    )
    hazard = -math.log(0.7)
    assert model.pair_transmission_probability() == pytest.approx(
        np.mean(1 - np.exp(-hazard * weighted_days)), abs=0.003
    )


def test_the_preview_ignores_the_age_effects_and_uses_the_chances_of_an_average_person():
    effects = {"branching": [[0, 120, 0.1]]}
    model = SeiarModel(ONE_DAY, asymptomatic_share=0.5, age_effects=effects)
    plain = SeiarModel(ONE_DAY, asymptomatic_share=0.5)
    assert model.infectious_days() == plain.infectious_days()


# --- the dynamics against the equations ---------------------------------------------------------

ODE = {
    "transmission": 0.3,
    "incubation_days": 2.0,
    "asymptomatic_days": 4.0,
    "infectious_days": 5.0,
    "asymptomatic_share": 0.6,
    "asymptomatic_infectiousness": 0.5,
    "fatality": 0.2,
    "immunity_days": 36500,
}
CONTACTS = 3.0
POPULATION = 20_000
CASES = 100
DAYS = 60
STEP = 1 / 24


def seiar_equations(contacts, transmission, incubation, days_a, days_i, share, relative, fatality):
    beta = -math.log(1 - transmission) * contacts

    def derivative(y):
        s, e, a, i, r, d = y
        force = beta * (i + relative * a)
        return np.array(
            [
                -force * s,
                force * s - e / incubation,
                share * e / incubation - a / days_a,
                (1 - share) * e / incubation - i / days_i,
                a / days_a + (1 - fatality) * i / days_i,
                fatality * i / days_i,
            ]
        )

    return derivative, beta * (share * relative * days_a + (1 - share) * days_i)


@pytest.fixture(scope="module")
def well_mixed():
    """Three runs with exponential stays and the textbook solution of the same SEIAR equations."""
    derivative, r0 = seiar_equations(CONTACTS, 0.3, 2.0, 4.0, 5.0, 0.6, 0.5, 0.2)
    solution = integrate(
        derivative, np.array([1 - CASES / POPULATION, 0, 0, CASES / POPULATION, 0, 0]), DAYS
    )
    _, counts = mean_course(
        SeiarModel,
        runs=3,
        population=POPULATION,
        cases=CASES,
        days=DAYS,
        contacts=CONTACTS,
        duration_shape=1.0,
        **ODE,
        **IDENTICAL,
    )
    return solution, counts / POPULATION, r0


def test_the_reproduction_number_is_the_weighted_sum_over_the_two_ways(well_mixed):
    *_, r0 = well_mixed
    model = SeiarModel(ONE_DAY, **ODE)
    assert r0 == pytest.approx(model.mean_field_r0(CONTACTS))
    assert r0 > 1


def test_the_course_of_every_state_follows_the_seiar_equations(well_mixed):
    solution, found, _ = well_mixed
    for index in (LATENT, SILENT, ILL):
        # Monte Carlo noise of the mean of 3 runs is about 1.5 % of the peak (the spread between
        # seeds was 1.4 % for A and 1.7 % for I over six runs); the stays are lengthened by half a
        # step, which delays everything by a fraction of a day; 5 % and 3 days leave the rest
        assert found[:, index].max() == pytest.approx(solution[:, index].max(), rel=0.05)
        assert found[:, index].argmax() * STEP == pytest.approx(
            solution[:, index].argmax() * STEP, abs=3.0
        )
    assert found[:, FREE].min() == pytest.approx(solution[:, FREE].min(), abs=0.01)


def test_the_final_size_and_the_deaths_follow_the_equations(well_mixed):
    solution, found, r0 = well_mixed
    ever = 1 - found[-1, FREE]
    assert ever == pytest.approx(final_size(r0, CASES / POPULATION), abs=0.015)
    assert ever == pytest.approx(1 - solution[-1, FREE], abs=0.01)
    assert found[-1, GONE] == pytest.approx(solution[-1, GONE], abs=0.005)
    # deaths come from the symptomatic only: fatality x symptomatic share of the infected, where the
    # first cases (all symptomatic) are counted separately
    symptomatic = (1 - 0.6) * (ever - CASES / POPULATION) + CASES / POPULATION
    assert found[-1, GONE] == pytest.approx(0.2 * symptomatic, abs=0.005)
    assert found[-1, LATENT : ILL + 1].sum() < 0.005, "the epidemic is over"


def test_the_asymptomatic_fraction_of_the_infections_is_the_share(well_mixed):
    _, found, _ = well_mixed
    # people-days in a state / mean stay (plus half a step) = people who went through it
    went_silent = found[:, SILENT].sum() * STEP / (4.0 + STEP / 2)
    fell_ill = found[:, ILL].sum() * STEP / (5.0 + STEP / 2) - CASES / POPULATION
    infected = went_silent + fell_ill
    assert went_silent / infected == pytest.approx(0.6, abs=four_sigma(0.6, POPULATION * 0.8))


def test_the_asymptomatic_cause_the_share_of_the_spreading_the_weights_and_durations_say(
    well_mixed,
):
    _, found, _ = well_mixed
    silent_spread = 0.5 * found[:, SILENT].sum() * STEP
    ill_spread = found[:, ILL].sum() * STEP - CASES / POPULATION * 5.0
    own_share = 0.6 * 0.5 * 4.0 / (0.6 * 0.5 * 4.0 + 0.4 * 5.0)
    assert silent_spread / (silent_spread + ill_spread) == pytest.approx(own_share, abs=0.02)
    assert 0.35 < own_share < 0.45


def final_infected(**parameters) -> float:
    given = {**ODE, "duration_shape": 1.0, **IDENTICAL, **parameters}
    _, counts = run_well_mixed(
        SeiarModel,
        population=8000,
        cases=100,
        days=80,
        contacts=CONTACTS,
        seed=7,
        **given,
    )
    return 1 - counts[-1, FREE] / 8000


@pytest.mark.parametrize(
    ("parameter", "low", "high", "share", "grows"),
    [
        # R0 = beta (share x relative x days_a + (1 - share) x days_i): it grows with everything
        # that makes the silent or the ill spread more and falls with the share of the silent,
        # who spread less per day than the ill
        ("asymptomatic_share", 0.2, 1.0, None, False),
        ("asymptomatic_infectiousness", 0.1, 1.0, 0.8, True),
        ("asymptomatic_days", 1.0, 6.0, 0.8, True),
        ("infectious_days", 1.5, 5.0, 0.5, True),
    ],
)
def test_every_parameter_of_the_spreading_moves_the_epidemic_the_way_the_theory_says(
    parameter, low, high, share, grows
):
    base = {} if share is None else {"asymptomatic_share": share}
    first = final_infected(**{**base, parameter: low})
    second = final_infected(**{**base, parameter: high})
    gain = second - first if grows else first - second
    assert gain > 0.1, (first, second)


def test_the_latent_period_delays_the_epidemic_without_changing_its_size():
    def peak_day_and_size(incubation):
        _, counts = run_well_mixed(
            SeiarModel,
            population=8000,
            cases=100,
            days=80,
            contacts=CONTACTS,
            seed=3,
            **{**ODE, "incubation_days": incubation, "duration_shape": 1.0, **IDENTICAL},
        )
        return counts[:, ILL].argmax() * STEP, 1 - counts[-1, FREE] / 8000

    (early, early_size), (late, late_size) = peak_day_and_size(1.0), peak_day_and_size(6.0)
    assert late - early > 5.0
    assert late_size == pytest.approx(early_size, abs=0.02)


def test_when_the_silent_do_not_spread_only_the_first_cases_ever_fall_ill():
    _, counts = run_well_mixed(
        SeiarModel,
        population=8000,
        cases=100,
        days=60,
        contacts=CONTACTS,
        seed=5,
        **{**ODE, "asymptomatic_share": 1.0, "asymptomatic_infectiousness": 0.0, **IDENTICAL},
    )
    assert counts[:, ILL].max() <= 100
    assert counts[:, LATENT].max() > 0, "the first cases do infect people"
    assert counts[-1, FREE] > 8000 - 100 * 7, "and these infect nobody further"


# --- inside the town ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def town_through_the_fork(population=1500):
    """A town in which everybody has gone through the fork, half of them ill and at home."""
    diseases = {
        "a": disease_definition(
            "seiar",
            0,
            stay_home=1.0,
            asymptomatic_share=0.5,
            asymptomatic_infectiousness=0.5,
            incubation_days=0.5,
            infectious_days=60.0,
            asymptomatic_days=60.0,
            transmission=0.0,
        )
    }
    model = make_model(diseases, population=population, arena_size=2000)
    disease_state = model.disease_states["a"]
    model.diseases["a"]._enter(
        disease_state, np.arange(population), EXPOSED, np.random.default_rng(0)
    )
    for _ in range(2 * 24 + 12):  # the latent period, then the noon of a weekday
        model.step()
    return model


def test_in_the_town_the_ill_go_home_and_the_silent_follow_their_schedule_unchanged(
    town_through_the_fork,
):
    model = town_through_the_fork
    disease_state = model.disease_states["a"]
    silent, ill = disease_state.state == SILENT, disease_state.state == ILL
    assert silent.sum() + ill.sum() == model.population
    assert abs(silent.mean() - 0.5) < four_sigma(0.5, model.population)
    home = np.column_stack([model.town.home_x, model.town.home_y])
    for person in np.flatnonzero(ill)[:50]:
        assert model.destination(person) == (home[person, 0], home[person, 1])
    # the silent are not held back: their destination is the one of their schedule
    for person in np.flatnonzero(silent)[:50]:
        assert model.destination(person) == (
            float(model._target_x[person]),
            float(model._target_y[person]),
        )


def test_home_isolation_removes_the_spreading_of_the_ill_and_nothing_else(town_through_the_fork):
    model = town_through_the_fork
    disease_state = model.disease_states["a"]
    weight = model.diseases["a"].infectious_weight(disease_state)
    ill = disease_state.state == ILL
    home = np.column_stack([model.town.home_x, model.town.home_y])

    def away(points):
        return np.hypot(*(points - home).T) > 1.0

    scheduled = away(np.column_stack([model._target_x, model._target_y]))
    kept = away(np.array([model.destination(person) for person in range(model.population)]))
    assert not kept[ill].any(), "nobody who is ill goes out"
    assert (kept[~ill] == scheduled[~ill]).all(), "everybody else goes where the schedule says"
    cut = 1 - (weight * kept).sum() / (weight * scheduled).sum()
    # the ill carry weight 1 and the silent 0.5, half of the people each: the cut in the spreading
    # outside the home is the ill's share of it, 0.5 / (0.5 + 0.5 x 0.5) = 2/3, up to who happens
    # to be out at noon; the 1500 people give a standard error of about 0.02
    assert cut == pytest.approx(2 / 3, abs=0.08)


# --- the demo disease ----------------------------------------------------------------------------

POLETTI_SYMPTOMATIC = {  # Poletti et al. 2021, symptomatic share of the infected by age group
    (0, 20): 0.1809,
    (20, 40): 0.2241,
    (40, 60): 0.3054,
    (60, 80): 0.3546,
    (80, 120): 0.6456,
}
LEVIN_MIDDLE_AGE = {(0, 20): 10, (20, 40): 30, (40, 60): 50, (60, 80): 70, (80, 120): 85}


def multiplier_from(rows, start):
    return next(value for low, _, value in rows if low == start)


def test_the_demo_asymptomatic_shares_by_age_are_those_measured_by_poletti():
    demo = LIBRARY.catalog["demo_seiar"]
    base = demo.parameters["asymptomatic_share"]
    for (low, high), symptomatic in POLETTI_SYMPTOMATIC.items():
        chance = base * multiplier_from(demo.age_effects["branching"], low)
        assert chance <= 1.0
        assert chance == pytest.approx(1 - symptomatic, abs=0.006), (low, high)


def test_the_demo_fatality_per_symptomatic_case_follows_the_levin_infection_fatality_rate():
    demo = LIBRARY.catalog["demo_seiar"]
    for (low, high), symptomatic in POLETTI_SYMPTOMATIC.items():
        ifr = 10 ** (-3.27 + 0.0524 * LEVIN_MIDDLE_AGE[low, high]) / 100
        per_case = demo.parameters["fatality"] * multiplier_from(demo.age_effects["fatality"], low)
        assert per_case == pytest.approx(ifr / symptomatic, rel=0.05), low


def test_the_demo_spreads_like_covid_19_and_can_take_off():
    demo = LIBRARY.catalog["demo_seiar"]
    model = SeiarModel(1 / 48, **demo.parameters)
    assert demo.model == "seiar"
    assert model.infectious_days() == pytest.approx(0.7 * 0.5 * 5.0 + 0.3 * 5.0)
    assert model.mean_field_r0(2.0) > 1
