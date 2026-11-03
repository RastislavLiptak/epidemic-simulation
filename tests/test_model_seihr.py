"""The SEIHR model against its definition and the literature it is built on.

Structure (who can reach what, who spreads, who is confined), the age profile of hospitalisation
and death against the Imperial College severity table, the mean course against the textbook
equations in a well-mixed population, the invariance of R0 to the hospital branch, the effect of
every parameter, the numbers shown in the preview and the behaviour in the town.
"""

import csv
import math

import numpy as np
import pytest
from config_factory import CONFIG_DIRECTORY, LIBRARY, disease_definition
from disease_helpers import ADULT, ONE_DAY, SURE, UNIFORM, advance, new_state, place, state_names
from mean_field import final_size, integrate, mean_course
from test_epidemic_model import arranged_model, put_into_state, run

from epidemic_sandbox.disease_models.disease_model_abstract import (
    DEAD,
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.seihr import HOSPITALISED, SeihrModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
DEMO = LIBRARY.catalog["demo_seihr"]
NAMES = list(SeihrModel.STATE_COLORS)
# Ferguson et al. 2020, Imperial College Report 9, Table 1 (percent): first age of the ten-year
# group, share of the infected requiring hospitalisation, infection fatality ratio.
FERGUSON = [
    (0, 0.1, 0.002),
    (10, 0.3, 0.006),
    (20, 1.2, 0.03),
    (30, 3.2, 0.08),
    (40, 4.9, 0.15),
    (50, 10.2, 0.60),
    (60, 16.6, 2.2),
    (70, 24.3, 5.1),
    (80, 27.3, 9.3),
]
GROUP_AGES = [start + 5 for start, _, _ in FERGUSON]  # a typical age of each group
# The demo's multipliers are Ferguson's values divided by a base rounded to two digits and
# rounded to three: the figures are reproduced to within this relative error.
ROUNDING = 0.01


def visits(model, population, age, horizon_days, seed, start_state=EXPOSED):
    """Everybody infected at once, no further exposure: the states visited and the final state."""
    ages = np.full(population, age) if np.isscalar(age) else np.asarray(age)
    disease_state, rng = new_state(model, population=population, age=ages, seed=seed)
    place(model, disease_state, rng, np.arange(population), start_state)
    visited = np.zeros((len(NAMES), population), np.bool_)
    for _ in range(round(horizon_days / model.step_days)):
        for index in range(len(NAMES)):
            visited[index] |= disease_state.state == index
        model.advance(disease_state, np.zeros(population, np.float32), rng)
    return visited, disease_state


def demo_model(step_days=0.25, **parameters):
    return SeihrModel(
        step_days, age_effects=dict(DEMO.age_effects), **{**DEMO.parameters, **parameters}
    )


def czech_age_shares() -> np.ndarray:
    """Share of the Czech population (the town's age structure) in each of Ferguson's groups."""
    path = CONFIG_DIRECTORY.parent / "data" / "reference" / "czechia_population_2019.csv"
    with path.open() as file:
        counts = np.array([int(row["male"]) + int(row["female"]) for row in csv.DictReader(file)])
    starts = [start for start, _, _ in FERGUSON] + [len(counts)]
    return (
        np.array([counts[a:b].sum() for a, b in zip(starts, starts[1:], strict=False)])
        / counts.sum()
    )


# --- structure -----------------------------------------------------------------------------------


def test_the_states_and_what_each_of_them_does():
    assert NAMES == [SUSCEPTIBLE, EXPOSED, INFECTIOUS, HOSPITALISED, RECOVERED, DEAD]
    model = SeihrModel(ONE_DAY)
    assert {name: model.state_infectiousness(name) for name in NAMES} == {
        SUSCEPTIBLE: 0.0,
        EXPOSED: 0.0,
        INFECTIOUS: 1.0,
        HOSPITALISED: 0.0,
        RECOVERED: 0.0,
        DEAD: 0.0,
    }
    assert {HOSPITALISED} == SeihrModel.CONFINED_STATES
    assert (
        SeihrModel.INFECTION_ENTRY_STATE == EXPOSED and SeihrModel.INITIAL_CASE_STATE == INFECTIOUS
    )
    assert SeihrModel.course_states() == {EXPOSED, INFECTIOUS, HOSPITALISED, RECOVERED}


def test_only_the_declared_moves_happen_and_nobody_dies_outside_hospital():
    model = SeihrModel(0.25, hospitalised=0.5, fatality=0.5, immunity_days=20.0)
    names = np.array(NAMES)
    disease_state, rng = new_state(model, population=2000, cases=range(2000))
    seen = set()
    for _ in range(400):
        before = disease_state.state.copy()
        model.advance(disease_state, np.zeros(2000, np.float32), rng)
        moved = before != disease_state.state
        seen |= set(zip(names[before[moved]], names[disease_state.state[moved]], strict=True))
    assert seen == {
        (INFECTIOUS, HOSPITALISED),
        (INFECTIOUS, RECOVERED),
        (HOSPITALISED, RECOVERED),
        (HOSPITALISED, DEAD),
        (RECOVERED, SUSCEPTIBLE),
    }


def test_infection_enters_the_exposed_state_and_the_first_cases_are_infectious():
    model = SeihrModel(ONE_DAY, **SURE)
    disease_state, rng = new_state(model, population=3, cases=[0])
    advance(model, disease_state, rng, exposure=1)
    assert state_names(model, disease_state) == [INFECTIOUS, EXPOSED, EXPOSED]


def test_nobody_dies_without_a_hospital_stay_whatever_the_fatality():
    model = SeihrModel(0.25, hospitalised=0.0, fatality=1.0)
    _, disease_state = visits(model, 3000, ADULT, 60, seed=1)
    assert disease_state.deaths == 0 and (disease_state.state == NAMES.index(RECOVERED)).all()


def test_everybody_in_hospital_dies_at_the_end_of_the_stay_with_certain_fatality():
    model = SeihrModel(0.25, hospitalised=1.0, fatality=1.0, duration_shape=50.0)
    population = 5000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    death_day = np.full(population, np.nan)
    for step in range(1, 241):
        model.advance(disease_state, np.zeros(population, np.float32), rng)
        newly = (disease_state.state == model.dead_index) & np.isnan(death_day)
        death_day[newly] = step * model.step_days
    assert disease_state.deaths == population
    expected = model.parameters["incubation_days"] + model.parameters["infectious_days"]
    expected += model.parameters["hospital_days"]
    # three stays rounded up to whole steps add 1.5 steps on average
    error = 5 * np.nanstd(death_day) / math.sqrt(population)
    assert np.nanmean(death_day) == pytest.approx(expected + 1.5 * model.step_days, abs=error)


# --- who is confined and who spreads -------------------------------------------------------------


def test_the_hospitalised_are_always_at_home_and_are_free_again_after_the_stay():
    model = SeihrModel(0.25, hospitalised=1.0, fatality=0.0, stay_home=0.0, duration_shape=50.0)
    disease_state, rng = new_state(model, population=1, cases=[0])
    history = []
    for _ in range(200):
        history.append((state_names(model, disease_state)[0], bool(disease_state.stays_home[0])))
        advance(model, disease_state, rng)
    states = {state: {home for name, home in history if name == state} for state, _ in history}
    assert states == {INFECTIOUS: {False}, HOSPITALISED: {True}, RECOVERED: {False}}


def test_everybody_ill_stays_at_home_with_certain_stay_home_and_is_free_when_recovered():
    model = SeihrModel(0.25, hospitalised=0.5, stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, population=1000, cases=range(1000))
    ill = [NAMES.index(INFECTIOUS), NAMES.index(HOSPITALISED)]
    for _ in range(300):
        assert (disease_state.stays_home == np.isin(disease_state.state, ill)).all()
        assert (
            disease_state.limits_outings == (disease_state.state == NAMES.index(INFECTIOUS))
        ).all()
        advance(model, disease_state, rng)
    assert (disease_state.state == NAMES.index(RECOVERED)).sum() > 400


def test_the_hospitalised_do_not_spread_and_cannot_be_infected_again():
    model = SeihrModel(ONE_DAY, **SURE)
    disease_state, rng = new_state(model, population=6, cases=[0])
    for person, state in enumerate((INFECTIOUS, HOSPITALISED, EXPOSED, SUSCEPTIBLE, RECOVERED), 0):
        place(model, disease_state, rng, [person], state)
    spreading = model.infectious_weight(disease_state)
    assert spreading.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert model.exposable_mask(disease_state).tolist() == [False, False, False, True, False, True]


# --- the age profile of severity -----------------------------------------------------------------


def test_the_hospitalised_share_and_the_fatality_by_age_are_those_of_the_severity_table():
    model = demo_model()
    hospital = model.parameters["hospitalised"]
    for age, (_, hospitalised, ifr) in zip(GROUP_AGES, FERGUSON, strict=True):
        disease_state, _ = new_state(model, population=1, age=age)
        admitted = hospital * float(disease_state.branching[0])
        dies = float(disease_state.fatality_risk[0])
        assert admitted == pytest.approx(hospitalised / 100, rel=ROUNDING), age
        assert admitted * dies == pytest.approx(ifr / 100, rel=ROUNDING), age
        assert admitted <= 1 and dies <= 1


def test_the_infection_fatality_ratio_of_the_town_age_structure_follows_the_severity_table():
    model = demo_model()
    shares = czech_age_shares()
    admitted, deaths = [], []
    for age in GROUP_AGES:
        disease_state, _ = new_state(model, population=1, age=age)
        chance = model.parameters["hospitalised"] * float(disease_state.branching[0])
        admitted.append(chance)
        deaths.append(chance * float(disease_state.fatality_risk[0]))
    table_admitted = sum(
        share * hospitalised / 100
        for share, (_, hospitalised, _) in zip(shares, FERGUSON, strict=True)
    )
    table_ifr = sum(share * ifr / 100 for share, (_, _, ifr) in zip(shares, FERGUSON, strict=True))
    assert float(shares @ admitted) == pytest.approx(table_admitted, rel=ROUNDING)
    assert float(shares @ deaths) == pytest.approx(table_ifr, rel=ROUNDING)
    # the base values are the means of the table over the town's age structure
    assert float(shares @ admitted) == pytest.approx(model.parameters["hospitalised"], rel=0.02)
    assert float(shares @ deaths) / float(shares @ admitted) == pytest.approx(
        model.parameters["fatality"], rel=0.02
    )


def test_the_simulated_shares_hospitalised_and_dying_by_age_match_the_table():
    model = demo_model()
    per_group = 30_000
    ages = np.repeat(GROUP_AGES, per_group)
    visited, disease_state = visits(model, len(ages), ages, 90.0, seed=11)
    in_hospital = visited[NAMES.index(HOSPITALISED)].reshape(len(GROUP_AGES), per_group)
    dead = (disease_state.state == model.dead_index).reshape(len(GROUP_AGES), per_group)
    for row, (_, hospitalised, ifr) in enumerate(FERGUSON):
        for observed, expected in (
            (in_hospital[row].mean(), hospitalised),
            (dead[row].mean(), ifr),
        ):
            expected /= 100
            sigma = math.sqrt(expected * (1 - expected) / per_group)
            assert observed == pytest.approx(
                expected, abs=4 * sigma + ROUNDING * expected + 1 / per_group
            )
    # where the counts are large enough to say something: the oldest group
    assert in_hospital[-1].mean() == pytest.approx(0.273, abs=0.01)
    assert dead[-1].mean() == pytest.approx(0.093, abs=0.007)


def test_the_branching_chance_is_capped_at_one():
    model = demo_model(hospitalised=0.5)  # 0.5 x 3.4125 in the oldest group
    visited, _ = visits(model, 2000, 85, 60.0, seed=2)
    assert visited[NAMES.index(HOSPITALISED)].all()


def test_children_are_half_as_susceptible_as_adults():
    model = demo_model(1.0, transmission=0.5, **UNIFORM)
    population = 40_000
    ages = np.repeat([8, ADULT], population // 2)
    disease_state, rng = new_state(model, population=population, age=ages, seed=3)
    exposure = np.full(population, 0.4, np.float32)
    infected = model.draw_infections(disease_state, exposure, rng)
    for age, factor in ((8, 0.5), (ADULT, 1.0)):
        chance = 1 - 0.5 ** (0.4 * factor)
        count = int(np.count_nonzero(ages[infected] == age))
        sigma = math.sqrt(population / 2 * chance * (1 - chance))
        assert count == pytest.approx(population / 2 * chance, abs=4 * sigma)


# --- every parameter does what the theory says ---------------------------------------------------


def test_the_hospitalised_share_sets_who_goes_to_hospital_and_the_fatality_who_dies():
    population = 20_000
    results = {}
    for hospitalised, fatality in ((0.1, 0.2), (0.4, 0.2), (0.4, 0.6)):
        model = SeihrModel(0.25, hospitalised=hospitalised, fatality=fatality)
        visited, disease_state = visits(model, population, ADULT, 70.0, seed=5)
        results[hospitalised, fatality] = (
            visited[NAMES.index(HOSPITALISED)].mean(),
            disease_state.deaths / population,
        )
    for (hospitalised, fatality), (admitted, died) in results.items():
        sigma = math.sqrt(hospitalised * (1 - hospitalised) / population)
        assert admitted == pytest.approx(hospitalised, abs=4 * sigma)
        chance = hospitalised * fatality
        assert died == pytest.approx(chance, abs=4 * math.sqrt(chance * (1 - chance) / population))


def test_the_stays_in_the_three_timed_states_follow_their_parameters():
    model = SeihrModel(
        1 / 12,
        incubation_days=3.0,
        infectious_days=2.0,
        hospital_days=7.0,
        hospitalised=1.0,
        fatality=0.5,
    )
    population = 20_000
    disease_state, rng = new_state(model, population=population, seed=7)
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    steps = np.zeros((len(NAMES), population))
    for _ in range(round(80 / model.step_days)):
        for index in range(len(NAMES)):
            steps[index] += disease_state.state == index
        model.advance(disease_state, np.zeros(population, np.float32), rng)
    days = steps * model.step_days
    died_in_hospital = disease_state.state == model.dead_index
    for state, mean in ((EXPOSED, 3.0), (INFECTIOUS, 2.0)):
        stay = days[NAMES.index(state)]
        assert stay.mean() == pytest.approx(
            mean + model.step_days / 2, abs=5 * stay.std() / math.sqrt(population)
        )
    stay = days[NAMES.index(HOSPITALISED)]
    expected = 7.0 + model.step_days / 2
    assert stay.mean() == pytest.approx(expected, abs=5 * stay.std() / math.sqrt(population))
    # the dead stay as long as the survivors: the model has one stay per state
    dead_stay, alive_stay = stay[died_in_hospital], stay[~died_in_hospital]
    error = 4 * stay.std() * math.sqrt(1 / len(dead_stay) + 1 / len(alive_stay))
    assert dead_stay.mean() == pytest.approx(alive_stay.mean(), abs=error)


def test_a_longer_latent_period_delays_the_epidemic_and_a_longer_stay_keeps_the_beds_full():
    def course(**parameters):
        _, counts = mean_course(
            SeihrModel, runs=2, population=10_000, cases=50, days=80, contacts=2.0,
            transmission=0.3, duration_shape=2.0, **IDENTICAL, **parameters,
        )  # fmt: skip
        return counts

    base, late = course(), course(incubation_days=9.2)
    assert (
        late[:, NAMES.index(INFECTIOUS)].argmax()
        > base[:, NAMES.index(INFECTIOUS)].argmax() + 24 * 5
    )
    short, long = course(hospital_days=5.0), course(hospital_days=20.0)
    column = NAMES.index(HOSPITALISED)
    assert long[:, column].max() > 2 * short[:, column].max()


def test_reinfection_after_recovery_happens_at_the_reduced_hazard():
    model = SeihrModel(ONE_DAY, transmission=0.5, immunity_protection=0.75, **UNIFORM)
    population = 40_000
    disease_state, rng = new_state(model, population=population, seed=9)
    place(model, disease_state, rng, np.arange(population), RECOVERED)
    infected = model.draw_infections(disease_state, np.ones(population, np.float32), rng)
    chance = 1 - 0.5**0.25  # exposure weakened to the unprotected quarter
    sigma = math.sqrt(population * chance * (1 - chance))
    assert len(infected) == pytest.approx(population * chance, abs=4 * sigma)


# --- the preview ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parameters",
    [{}, {"hospitalised": 0.9, "fatality": 0.7}, {"hospital_days": 40.0}, {"hospitalised": 0.0}],
)
def test_the_infectious_time_is_the_infectious_period_whatever_happens_in_hospital(parameters):
    model = SeihrModel(1 / 48, transmission=0.4, **parameters)
    days = model.parameters["infectious_days"]
    assert model.infectious_days() == pytest.approx(days)
    assert model.mean_field_r0(3.0) == pytest.approx(-math.log(0.6) * 3.0 * days)
    hazard, shape = -math.log(0.6), model.parameters["duration_shape"]
    assert model.pair_transmission_probability() == pytest.approx(
        1 - (1 + hazard * days / shape) ** -shape
    )


def test_the_courses_of_an_infection_have_the_chances_of_the_branches():
    model = SeihrModel(1 / 48, hospitalised=0.3, fatality=0.2)
    chances = {}
    for course in model.courses():
        key = (HOSPITALISED in [state for state, _, _ in course.stays], course.end)
        chances[key] = chances.get(key, 0.0) + course.probability
    assert chances == pytest.approx(
        {(True, DEAD): 0.06, (True, RECOVERED): 0.24, (False, RECOVERED): 0.7}
    )


# --- the textbook equations ----------------------------------------------------------------


def seihr_derivative(stages, contacts, hazard, rates, hospitalised, fatality):
    """Mean course: every timed state is a chain of `stages` exponential stages (Erlang)."""
    latent, infectious, hospital = (stages * rate for rate in rates)
    beta = contacts * hazard

    def derivative(y):
        s, e = y[0], y[1 : 1 + stages]
        i, h = y[1 + stages : 1 + 2 * stages], y[1 + 2 * stages : 1 + 3 * stages]
        force = beta * i.sum()
        d_e = np.concatenate([[force * s], latent * e[:-1]]) - latent * e
        d_i = np.concatenate([[latent * e[-1]], infectious * i[:-1]]) - infectious * i
        d_h = (
            np.concatenate([[hospitalised * infectious * i[-1]], hospital * h[:-1]]) - hospital * h
        )
        d_r = (1 - hospitalised) * infectious * i[-1] + (1 - fatality) * hospital * h[-1]
        d_d = fatality * hospital * h[-1]
        return np.concatenate([[-force * s], d_e, d_i, d_h, [d_r, d_d]])

    return derivative


@pytest.mark.parametrize("shape", [1, 3])
def test_the_mean_course_follows_the_seihr_equations(shape):
    transmission, contacts = 0.25, 2.5
    incubation, infectious, hospital = 4.6, 5.5, 10.4
    hospitalised, fatality = 0.3, 0.4
    population, cases, days = 20_000, 100, 90
    hazard = -math.log(1 - transmission)
    stages = shape
    y0 = np.zeros(3 * stages + 3)
    y0[0], y0[1 + stages] = 1 - cases / population, cases / population
    ode = integrate(
        seihr_derivative(
            stages,
            contacts,
            hazard,
            (1 / incubation, 1 / infectious, 1 / hospital),
            hospitalised,
            fatality,
        ),
        y0,
        days,
    )
    _, counts = mean_course(
        SeihrModel, runs=3, population=population, cases=cases, days=days, contacts=contacts,
        transmission=transmission, incubation_days=incubation, infectious_days=infectious,
        hospital_days=hospital, hospitalised=hospitalised, fatality=fatality,
        duration_shape=float(shape), immunity_days=36500, immunity_protection=1.0, **IDENTICAL,
    )  # fmt: skip
    share = counts / population
    infectious_ode = ode[:, 1 + stages : 1 + 2 * stages].sum(axis=1)
    hospital_ode = ode[:, 1 + 2 * stages : 1 + 3 * stages].sum(axis=1)
    # each stay is rounded up to whole steps: half a step longer per state
    assert share[:, NAMES.index(INFECTIOUS)].max() == pytest.approx(infectious_ode.max(), rel=0.08)
    assert share[:, NAMES.index(HOSPITALISED)].max() == pytest.approx(hospital_ode.max(), rel=0.08)
    assert share[:, NAMES.index(INFECTIOUS)].argmax() / 24 == pytest.approx(
        infectious_ode.argmax() / 24, abs=2.0
    )
    assert share[:, NAMES.index(HOSPITALISED)].argmax() / 24 == pytest.approx(
        hospital_ode.argmax() / 24, abs=2.0
    )
    assert share[-1, NAMES.index(DEAD)] == pytest.approx(ode[-1, -1], rel=0.05)


@pytest.mark.parametrize("hospitalised", [0.0, 0.9])
def test_the_hospital_branch_does_not_change_the_final_size_or_r0(hospitalised):
    transmission, contacts = 0.2, 2.0  # R0 = 2.45: a final size that depends on it
    model = SeihrModel(1 / 24, transmission=transmission, hospitalised=hospitalised)
    r0 = model.mean_field_r0(contacts)
    _, counts = mean_course(
        SeihrModel, runs=3, population=20_000, cases=300, days=130, contacts=contacts,
        transmission=transmission, hospitalised=hospitalised, fatality=0.2,
        immunity_days=36500, immunity_protection=1.0, **IDENTICAL,
    )  # fmt: skip
    ever_infected = 1 - counts[-1, 0] / 20_000
    assert ever_infected == pytest.approx(final_size(r0, 300 / 20_000), abs=0.015)


def test_the_epidemic_without_the_hospital_branch_is_the_same_as_with_it():
    """Same R0, so the same infectious curve: hospital only takes people after their time in I."""
    common = {"runs": 3, "population": 12_000, "cases": 100, "days": 45, "contacts": 3.0}
    kwargs = {"transmission": 0.4, "immunity_days": 36500, "immunity_protection": 1.0, **IDENTICAL}
    _, none = mean_course(SeihrModel, hospitalised=0.0, **common, **kwargs)
    _, many = mean_course(SeihrModel, hospitalised=0.9, fatality=0.5, **common, **kwargs)
    column = NAMES.index(INFECTIOUS)
    assert many[:, column].max() == pytest.approx(none[:, column].max(), rel=0.08)
    assert many[:, column].argmax() / 24 == pytest.approx(none[:, column].argmax() / 24, abs=2.0)


# --- in the town ----------------------------------------------------------------------------------


def seihr_model(**parameters):
    return arranged_model({"a": disease_definition("seihr", 0, **parameters)}, {})


def test_a_hospitalised_person_is_at_home_all_day_even_if_the_disease_lets_people_go_out():
    model = seihr_model(stay_home=0.0, limit_outings=0.0)
    model.disease_states["a"].timer[:] = 1e6
    model.diseases["a"]._enter(
        model.disease_states["a"],
        np.arange(model.population),
        HOSPITALISED,
        np.random.default_rng(0),
    )
    run(model, 24)
    away = np.hypot(model.x - model.town.home_x, model.y - model.town.home_y)
    assert away.max() < 1e-6


def test_a_recovered_person_is_not_confined():
    model = seihr_model(stay_home=1.0, limit_outings=1.0)
    for person, state in enumerate((INFECTIOUS, HOSPITALISED, RECOVERED, EXPOSED)):
        model.diseases["a"]._enter(
            model.disease_states["a"], np.array([person]), state, np.random.default_rng(person)
        )
    assert model._goes_home()[:4].tolist() == [True, True, False, False]


def test_an_infectious_neighbour_infects_and_a_hospitalised_one_does_not():
    positions = {0: (50, 50), 1: (51.0, 50), 2: (200, 200), 3: (201.0, 200)}
    model = arranged_model(
        {"a": disease_definition("seihr", 0, transmission=1.0, incubation_days=30.0, **UNIFORM)},
        positions,
    )
    put_into_state(model, 0, "a", INFECTIOUS)
    put_into_state(model, 2, "a", HOSPITALISED)
    model.town.home_x[2], model.town.home_y[2] = 200, 200
    run(model, 24)
    states = model.disease_states["a"].state
    assert states[1] == NAMES.index(EXPOSED)
    assert states[3] == NAMES.index(SUSCEPTIBLE)
