"""The SEIS model against its definition: susceptible -> exposed -> infectious -> susceptible.

The textbook equations (Hethcote 2000) are dS = -beta S I + gamma I, dE = beta S I - sigma E,
dI = sigma E - gamma I without demography. The infection persists for R0 = beta / gamma > 1 at
S* = 1/R0, E* + I* = 1 - 1/R0 and E*/I* = gamma / sigma, and dies out otherwise. The model is
compared with them in a well-mixed population (`mean_field.py`) with exponential stays. A stay
measured in steps is the continuous one rounded up, so the stays are longer by half a step: the
equations are integrated with the lengths `stay + STEP / 2`, and the equilibrium has a further
discretisation bias of at most 0.002 in a share (measured over 6 seeds at R0 = 2 and 4).
"""

import math

import numpy as np
import pytest
from cited_numbers import check_disease_numbers
from config_factory import disease_definition
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, new_state, place, state_names
from mean_field import integrate, mean_course
from synthetic_models import RepeatModel
from test_epidemic_model import arranged_model, make_infectious, state_of

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    COMMON_PARAMETERS,
    DEAD,
    DYING,
    EXPOSED,
    INFECTIOUS,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.seis import SeisModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
POPULATION = 20_000
CONTACTS = 3.0
STEP = 1 / 24
# Noise of a share averaged over 4 runs of 50 days is at most 0.0028 / 2 (measured over 6 seeds);
# 4 sigma plus the 0.002 discretisation bias.
SHARE_TOLERANCE = 0.008
BASE = {"transmission": 0.2, "incubation_days": 3.0, "infectious_days": 4.0, "duration_shape": 1.0}


def rates(transmission, incubation_days, infectious_days):
    """beta, sigma and gamma of the equations for stays that are half a step longer."""
    beta = -math.log(1 - transmission) * CONTACTS
    return beta, 1 / (incubation_days + STEP / 2), 1 / (infectious_days + STEP / 2)


def endemic_shares(runs=4, cases=2000, days=100, **overrides):
    """Shares of (susceptible, exposed, infectious) averaged over the last 50 days and the runs."""
    _, counts = mean_course(
        SeisModel,
        runs=runs,
        population=POPULATION,
        cases=cases,
        days=days,
        contacts=CONTACTS,
        **{**BASE, **overrides},
        **IDENTICAL,
    )
    return counts[round((days - 50) / STEP) :, :3].mean(axis=0) / POPULATION


# --- the declaration ----------------------------------------------------------------------------


def test_the_states_flows_and_roles_follow_the_definition():
    states = [SUSCEPTIBLE, EXPOSED, INFECTIOUS, ASYMPTOMATIC, DYING, DEAD]
    assert list(SeisModel.STATE_COLORS) == states
    assert set(SeisModel.INFECTIOUS_STATES) == {INFECTIOUS, ASYMPTOMATIC}
    assert set(SeisModel.SYMPTOMATIC_STATES) == {INFECTIOUS}
    assert SeisModel.INFECTION_ENTRY_STATE == EXPOSED and SeisModel.INITIAL_CASE_STATE == INFECTIOUS
    phases = SeisModel.PHASES
    assert set(phases) == {EXPOSED, INFECTIOUS, ASYMPTOMATIC}, "the susceptible and dead have none"
    assert phases[EXPOSED].next_state == INFECTIOUS and not phases[EXPOSED].fatal
    assert phases[EXPOSED].branch_state == ASYMPTOMATIC
    assert phases[INFECTIOUS].next_state == SUSCEPTIBLE and phases[INFECTIOUS].fatal
    assert phases[ASYMPTOMATIC].next_state == SUSCEPTIBLE and not phases[ASYMPTOMATIC].fatal
    assert phases[INFECTIOUS].branch_state is None
    assert not SeisModel.PROTECTION and not SeisModel.RATES and not SeisModel.CONFINED_STATES
    assert SeisModel.course_states() == {EXPOSED, INFECTIOUS, ASYMPTOMATIC}


def test_the_parameters_are_the_common_ones_and_the_two_durations_without_immunity():
    assert SeisModel.PARAMETERS.keys() == COMMON_PARAMETERS.keys() | {
        "incubation_days",
        "infectious_days",
        "death_delay_days",
        "severe_protection",
        "asymptomatic_share",
        "asymptomatic_days",
        "asymptomatic_infectiousness",
    }


def test_only_infectious_people_spread_and_only_susceptible_people_can_be_infected():
    model = SeisModel(ONE_DAY)
    assert model.state_infectiousness(INFECTIOUS) == 1.0
    assert model.state_infectiousness(EXPOSED) == model.state_infectiousness(SUSCEPTIBLE) == 0.0
    disease_state, rng = new_state(model, population=4)
    for person, state in enumerate((SUSCEPTIBLE, EXPOSED, INFECTIOUS, DEAD)):
        place(model, disease_state, rng, [person], state)
    assert model.exposable_mask(disease_state).tolist() == [True, False, False, False]
    assert model.infectious_weight(disease_state).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_the_first_cases_are_infectious_and_infected_people_are_exposed_first():
    model = SeisModel(ONE_DAY, **SURE)
    disease_state, rng = new_state(model, population=10, cases=[2, 5])
    assert state_names(model, disease_state).count(INFECTIOUS) == 2
    assert disease_state.ever_infected.sum() == 2 and disease_state.cumulative_infections == 2
    exposure = np.zeros(10, np.float32)
    exposure[7] = 1.0
    model.advance(disease_state, exposure, rng)
    assert state_names(model, disease_state)[7] == EXPOSED


# --- the course of one person -------------------------------------------------------------------


def test_a_person_goes_through_the_states_in_order_and_can_be_infected_again():
    model = SeisModel(
        ONE_DAY, **SURE, incubation_days=2.0, infectious_days=2.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model)
    advance(model, disease_state, rng, exposure=1)
    visited = [EXPOSED]
    for _ in range(40):
        advance(model, disease_state, rng)
        current = state_names(model, disease_state)[0]
        if current != visited[-1]:
            visited.append(current)
    assert visited == [EXPOSED, INFECTIOUS, SUSCEPTIBLE]
    advance(model, disease_state, rng, exposure=1)
    assert state_names(model, disease_state) == [EXPOSED]
    assert disease_state.cumulative_infections == 2 and disease_state.ever_infected.sum() == 1


def test_a_random_run_makes_only_the_moves_of_the_definition():
    model = SeisModel(
        0.25, transmission=0.6, fatality=0.3, incubation_days=1.0, infectious_days=1.5
    )
    disease_state, rng = new_state(model, population=3000, cases=np.arange(100))
    names = list(model.STATE_COLORS)
    moves = set()
    for _ in range(80):
        before = disease_state.state.copy()
        model.advance(disease_state, np.full(3000, 0.4, np.float32), rng)
        moves |= {
            (names[a], names[b]) for a, b in zip(before, disease_state.state, strict=True) if a != b
        }
    assert moves == {
        (SUSCEPTIBLE, EXPOSED),
        (EXPOSED, INFECTIOUS),
        (INFECTIOUS, SUSCEPTIBLE),
        (INFECTIOUS, DEAD),
    }


def test_infected_people_are_not_infected_again_until_they_have_recovered():
    model = SeisModel(
        ONE_DAY, **SURE, incubation_days=5.0, infectious_days=5.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model, population=300)
    place(model, disease_state, rng, np.arange(100), EXPOSED)
    place(model, disease_state, rng, np.arange(100, 200), INFECTIOUS)
    advance(model, disease_state, rng, exposure=50)
    assert disease_state.cumulative_infections == 100, "only the 100 susceptible were infected"
    names = list(model.STATE_COLORS)
    assert (disease_state.state[:100] == names.index(EXPOSED)).all()
    assert (disease_state.state[100:200] == names.index(INFECTIOUS)).all()
    assert (disease_state.state[200:] == names.index(EXPOSED)).all()


def test_reinfections_in_an_endemic_run_are_counted_as_infections_not_as_people():
    model = SeisModel(1 / 12, transmission=0.4, incubation_days=2.0, infectious_days=3.0)
    population = 3000
    disease_state, rng = new_state(model, population=population, cases=np.arange(300))
    for _ in range(12 * 60):
        spreading = model.infectious_weight(disease_state).sum() * CONTACTS / population
        model.advance(disease_state, np.full(population, spreading, np.float32), rng)
    assert disease_state.ever_infected.sum() <= population
    assert disease_state.cumulative_infections > 2 * disease_state.ever_infected.sum()


def test_the_exposed_do_not_stay_at_home_but_the_infectious_do_until_they_recover():
    model = SeisModel(ONE_DAY, stay_home=1.0, limit_outings=1.0, duration_shape=50.0)
    disease_state, rng = new_state(model, population=2)
    place(model, disease_state, rng, [0], EXPOSED)
    place(model, disease_state, rng, [1], INFECTIOUS)
    assert not disease_state.stays_home[0] and not disease_state.limits_outings[0]
    assert disease_state.stays_home[1] and disease_state.limits_outings[1]
    for _ in range(int(model.parameters["incubation_days"]) + 2):
        advance(model, disease_state, rng)
    assert state_names(model, disease_state)[0] == INFECTIOUS
    assert disease_state.stays_home[0] and disease_state.limits_outings[0]
    for _ in range(int(model.parameters["infectious_days"]) + 2):
        advance(model, disease_state, rng)
    assert state_names(model, disease_state) == [SUSCEPTIBLE, SUSCEPTIBLE]
    assert not disease_state.stays_home.any()


# --- deaths and the age effects -----------------------------------------------------------------


def test_people_die_only_at_the_end_of_the_infectious_period():
    model = SeisModel(
        ONE_DAY, fatality=1.0, incubation_days=3.0, infectious_days=4.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model, population=200)
    place(model, disease_state, rng, np.arange(200), EXPOSED)
    infectious = list(model.STATE_COLORS).index(INFECTIOUS)
    dead_by_day = []
    for _ in range(12):
        before = disease_state.state.copy()
        advance(model, disease_state, rng)
        died = (disease_state.state == model.dead_index) & (before != model.dead_index)
        assert (before[died] == infectious).all(), "the dead were infectious the step before"
        dead_by_day.append(int((disease_state.state == model.dead_index).sum()))
    assert dead_by_day[:4] == [0] * 4, "nobody dies while exposed or early in the illness"
    assert dead_by_day[-1] == 200 and disease_state.deaths == 200
    assert (disease_state.state == model.dead_index).all(), "nobody returns to susceptible"


def test_a_share_of_infections_equal_to_the_fatality_ends_in_death_and_the_rest_in_susceptible():
    model = SeisModel(ONE_DAY, fatality=0.2, incubation_days=1.0, infectious_days=2.0)
    disease_state, rng = new_state(model, population=50_000)
    place(model, disease_state, rng, np.arange(50_000), EXPOSED)
    for _ in range(40):
        advance(model, disease_state, rng)
    dead = disease_state.deaths / 50_000
    assert dead == pytest.approx(0.2, abs=4 * math.sqrt(0.2 * 0.8 / 50_000) + 1e-3)
    assert (disease_state.state == 0).sum() == 50_000 - disease_state.deaths


def test_fatality_depends_on_age():
    effects = {"fatality": [[0, 15, 0.0], [65, 120, 5.0]]}
    model = SeisModel(
        ONE_DAY, fatality=0.1, incubation_days=1.0, infectious_days=2.0, age_effects=effects
    )
    ages = np.repeat([5, 40, 80], 20_000)
    disease_state, rng = new_state(model, population=60_000, age=ages)
    place(model, disease_state, rng, np.arange(60_000), EXPOSED)
    for _ in range(40):
        advance(model, disease_state, rng)
    dead = (disease_state.state == model.dead_index).reshape(3, 20_000).mean(axis=1)
    assert dead[0] == 0.0
    assert dead[1] == pytest.approx(0.1, abs=4 * math.sqrt(0.09 / 20_000))
    assert dead[2] == pytest.approx(0.5, abs=4 * math.sqrt(0.25 / 20_000))


def test_susceptibility_depends_on_age_and_there_is_no_branching_to_scale():
    effects = {"susceptibility": [[0, 15, 0.5], [65, 120, 2.0]]}
    model = SeisModel(ONE_DAY, transmission=0.3, age_effects=effects, **UNIFORM)
    ages = np.repeat([5, 40, 80], 50_000)
    disease_state, rng = new_state(model, population=150_000, age=ages)
    advance(model, disease_state, rng, exposure=1.0)
    infected = (disease_state.state == 1).reshape(3, 50_000).mean(axis=1)
    expected = [1 - 0.7**factor for factor in (0.5, 1.0, 2.0)]
    assert infected == pytest.approx(expected, abs=4 * math.sqrt(0.25 / 50_000))
    with pytest.raises(ValueError, match="branching"):
        SeisModel(ONE_DAY, age_effects={"branching": [[0, 15, 0.5]]})


# --- the numbers shown before a run -------------------------------------------------------------


def test_the_latent_period_does_not_count_towards_the_time_spent_spreading():
    short = SeisModel(ONE_DAY, transmission=0.3, incubation_days=1.0, infectious_days=6.0)
    long = SeisModel(ONE_DAY, transmission=0.3, incubation_days=20.0, infectious_days=6.0)
    for model in (short, long):
        assert model.infectious_days() == 6.0
        assert model.mean_field_r0(2.0) == pytest.approx(-math.log(0.7) * 2.0 * 6.0)
    assert short.pair_transmission_probability() == long.pair_transmission_probability()


@pytest.mark.parametrize("shape", [1.0, 4.0])
def test_a_close_contact_is_infected_with_the_chance_of_the_gamma_infectious_period(shape):
    model = SeisModel(
        ONE_DAY, transmission=0.3, incubation_days=3.0, infectious_days=5.0, duration_shape=shape
    )
    hazard = -math.log(0.7)
    assert model.pair_transmission_probability() == pytest.approx(
        1 - (1 + hazard * 5.0 / shape) ** -shape
    )
    periods = np.random.default_rng(0).gamma(shape, 5.0 / shape, 400_000)
    assert model.pair_transmission_probability() == pytest.approx(
        np.mean(1 - np.exp(-hazard * periods)), abs=0.003
    )


def test_the_courses_are_the_two_stays_followed_by_death_or_susceptibility():
    model = SeisModel(ONE_DAY, fatality=0.25, incubation_days=2.0, infectious_days=5.0)
    survive, die = sorted(model.courses(), key=lambda course: course.end != SUSCEPTIBLE)
    assert survive.end == SUSCEPTIBLE and survive.probability == pytest.approx(0.75)
    assert die.end == DEAD and die.probability == pytest.approx(0.25)
    assert survive.stays == ((EXPOSED, 2.0, 0.0), (INFECTIOUS, 5.0, 1.0))
    assert die.stays == ((EXPOSED, 2.0, 0.0), (INFECTIOUS, 5.0, 1.0), (DYING, 0.0, 0.0))
    assert [c.end for c in SeisModel(ONE_DAY).courses()] == [SUSCEPTIBLE]


# --- the dynamics against the equations ---------------------------------------------------------


def test_the_mean_epidemic_follows_the_seis_equations_in_time():
    transmission, cases, days = 0.2, 400, 60
    beta, sigma, gamma = rates(transmission, 3.0, 4.0)

    def derivative(y):
        s, e, i = y
        return np.array(
            [-beta * s * i + gamma * i, beta * s * i - sigma * e, sigma * e - gamma * i]
        )

    ode = integrate(derivative, np.array([1 - cases / POPULATION, 0.0, cases / POPULATION]), days)
    _, counts = mean_course(
        SeisModel,
        runs=4,
        population=POPULATION,
        cases=cases,
        days=days,
        contacts=CONTACTS,
        **BASE,
        **IDENTICAL,
    )
    # a run's share is off by at most 0.0064 (8 seeds), the mean of 4 runs by 4 x 0.0032 and the
    # bias is 0.002
    assert np.abs(counts[:, :3] / POPULATION - ode).max() < 0.015
    assert counts[-1, 1] > 0.2 * POPULATION, "by day 60 it has reached the endemic level"


@pytest.mark.parametrize("transmission", [0.15, 0.3])
def test_the_endemic_equilibrium_is_the_one_of_the_equations(transmission):
    beta, sigma, gamma = rates(transmission, 3.0, 4.0)
    r0 = beta / gamma
    infectious = (1 - 1 / r0) / (1 + gamma / sigma)
    expected = [1 / r0, infectious * gamma / sigma, infectious]
    shares = endemic_shares(transmission=transmission)
    assert shares == pytest.approx(expected, abs=SHARE_TOLERANCE)
    assert shares[1] + shares[2] == pytest.approx(1 - 1 / r0, abs=SHARE_TOLERANCE)


def test_the_early_growth_rate_is_the_one_of_the_linearised_equations():
    transmission, cases = 0.2, 50
    beta, sigma, gamma = rates(transmission, 3.0, 4.0)
    growth = (-(sigma + gamma) + math.sqrt((sigma - gamma) ** 2 + 4 * sigma * beta)) / 2
    assert growth < beta - gamma, "the latent period slows the growth down"
    _, counts = mean_course(
        SeisModel,
        runs=6,
        population=POPULATION,
        cases=cases,
        days=30,
        contacts=CONTACTS,
        **BASE,
        **IDENTICAL,
    )
    infected = counts[:, 1] + counts[:, 2]
    # past the start-up, below 3% infected; fits over other windows of days 4-14 differ by 2%
    window = slice(4 * 24, 10 * 24)
    assert infected[window].max() < 0.03 * POPULATION
    fitted = np.polyfit(np.arange(len(infected))[window] * STEP, np.log(infected[window]), 1)[0]
    assert fitted == pytest.approx(growth, rel=0.05)


@pytest.mark.parametrize(
    ("transmission", "survives"),
    [(0.06, False), (0.15, True)],
    ids=["below threshold", "above threshold"],
)
def test_the_infection_dies_out_below_the_threshold_and_persists_above_it(transmission, survives):
    beta, _, gamma = rates(transmission, 3.0, 4.0)
    assert (beta / gamma > 1) == survives
    _, counts = mean_course(
        SeisModel,
        runs=2,
        population=POPULATION,
        cases=2000,
        days=120,
        contacts=CONTACTS,
        **{**BASE, "transmission": transmission},
        **IDENTICAL,
    )
    infected = (counts[:, 1] + counts[:, 2]) / POPULATION
    if survives:
        assert infected[-1] > 0.2
    else:
        assert infected[24 * 20] < infected[0], "declining from the start"
        # the decay rate is -0.039 per day, so 120 days leave exp(-4.7) = 0.9% of the 10% at start
        assert infected[-1] < 0.002


def test_a_latent_period_slows_the_start_but_not_the_endemic_level_of_the_infected():
    transmission = 0.2
    beta, sigma, gamma = rates(transmission, 3.0, 4.0)
    kwargs = {"population": POPULATION, "contacts": CONTACTS, "transmission": transmission}
    _, seis = mean_course(
        SeisModel,
        runs=4,
        cases=400,
        days=100,
        incubation_days=3.0,
        infectious_days=4.0,
        duration_shape=1.0,
        **kwargs,
        **IDENTICAL,
    )
    _, sis = mean_course(
        RepeatModel,
        runs=4,
        cases=400,
        days=100,
        infectious_days=4.0,
        duration_shape=1.0,
        **kwargs,
        **IDENTICAL,
    )
    seis_infected = (seis[:, 1] + seis[:, 2]) / POPULATION
    sis_infected = sis[:, 1] / POPULATION
    last = slice(-50 * 24, None)
    assert seis_infected[last].mean() == pytest.approx(sis_infected[last].mean(), abs=0.012)
    assert sis_infected[last].mean() == pytest.approx(1 - gamma / beta, abs=0.012)

    def seis_derivative(y):
        s, e, i = y
        return np.array(
            [-beta * s * i + gamma * i, beta * s * i - sigma * e, sigma * e - gamma * i]
        )

    def sis_derivative(y):
        return np.array([-beta * y[0] * y[1] + gamma * y[1], beta * y[0] * y[1] - gamma * y[1]])

    start = 400 / POPULATION
    ode_seis = integrate(seis_derivative, np.array([1 - start, 0.0, start]), 40)
    ode_sis = integrate(sis_derivative, np.array([1 - start, start]), 40)

    def first_day_above(share, level=0.3):
        return np.argmax(share > level) * STEP

    expected = first_day_above(ode_seis[:, 1] + ode_seis[:, 2]) - first_day_above(ode_sis[:, 1])
    assert expected > 5, "the latent period delays the epidemic by days"
    # a run's crossing time varies by 0.12 days (6 seeds), the difference of two means of 4 by
    # 0.09 days; the rest of the tolerance is the discretisation
    delay = first_day_above(seis_infected) - first_day_above(sis_infected)
    assert delay == pytest.approx(expected, abs=0.5)


def test_every_parameter_acts_in_the_direction_the_theory_says():
    base = endemic_shares(runs=1, days=100)
    more_transmission = endemic_shares(runs=1, days=100, transmission=0.3)
    longer_illness = endemic_shares(runs=1, days=100, infectious_days=6.0)
    longer_latency = endemic_shares(runs=1, days=100, incubation_days=6.0)
    infected = base[1] + base[2]
    assert more_transmission[1] + more_transmission[2] > infected + 0.1
    assert longer_illness[1] + longer_illness[2] > infected + 0.05
    # E*/I* = latent period / infectious period: the same infected, more of them still exposed
    # the infected share of a run averaged over 50 days varies by at most 0.0028 (6 seeds), the
    # difference of two runs by 0.004, so 4 sigma is 0.016
    assert longer_latency[1] + longer_latency[2] == pytest.approx(infected, abs=0.016)
    assert longer_latency[1] / longer_latency[2] > 1.6 * base[1] / base[2]


# --- in a town ----------------------------------------------------------------------------------


def test_an_exposed_neighbour_does_not_infect_but_an_infectious_one_does():
    diseases = {"a": disease_definition("seis", 0, transmission=1.0)}
    model = arranged_model(diseases, {1: (5, 6.5)})
    place(
        model.diseases["a"],
        model.disease_states["a"],
        np.random.default_rng(0),
        [0],
        EXPOSED,
    )
    model.disease_states["a"].timer[0] = 1e6
    model.step()
    assert state_of(model, 1, "a") == SUSCEPTIBLE
    make_infectious(model, 0, "a")
    model.step()
    assert state_of(model, 1, "a") == EXPOSED, "infected people are exposed, not yet infectious"


def test_the_model_is_found_by_the_registry_under_its_key():
    assert DISEASE_MODELS["seis"] is SeisModel


# --- the strep throat disease ---------------------------------------------------------------------

# (values both the disease and the model default have, the disease's own values, age effects)
SOURCED = (
    {},
    {  # CDC clinical guidance, Danchin et al. 2007, Fragaszy et al. 2018, Van Kerckhove et al. 2013
        "transmission": 0.042,
        "contact_distance": 2.0,
        "community_factor": 1.0,
        "incubation_days": 3.5,
        "infectious_days": 4.0,
        "duration_shape": 5.4,
        "fatality": 0.0,
        "stay_home": 0.40,
        "limit_outings": 0.33,
        "outside_infections": 0.076,
        "susceptibility_cv": 0.0,
        "infectiousness_dispersion": 0.0,
    },
    {"outside_infections": [[0, 3, 0.37], [3, 16, 1.0], [16, 120, 0.37]]},
)


def test_the_disease_has_the_numbers_its_comment_cites():
    check_disease_numbers("seis", *SOURCED)
