"""The SVEPIR model against its definition: structure, mechanisms, textbook results.

With identical people in a well-mixed population (tests/mean_field.py), the share `a` ever
infected solves the final size relation of a leaky vaccine whatever the latent and presymptomatic
periods are, because the force of infection integrates to R0 times `a`, with R0 = hazard x
contacts x (w d_P + d_I + v d_W) (w the presymptomatic infectiousness, v that of the waning
period W):
    a = i0 + s0 (1 - exp(-R0 a)) + v0 (1 - exp(-(1 - VE) R0 a)).
"""

import math

import numpy as np
import pytest
from cited_numbers import check_default_numbers, check_disease_numbers
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, count_in, new_state, place
from mean_field import run_well_mixed

from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    DEAD,
    DYING,
    EXPOSED,
    INFECTIOUS,
    LIFELONG_DAYS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.svepir import (
    PRESYMPTOMATIC,
    VACCINATED,
    WANING,
    SvepirModel,
)

ORDER = [
    SUSCEPTIBLE,
    VACCINATED,
    EXPOSED,
    PRESYMPTOMATIC,
    INFECTIOUS,
    WANING,
    ASYMPTOMATIC,
    RECOVERED,
    DYING,
    DEAD,
]
STEP = 1 / 24
TRANSMISSION, CONTACTS = 0.3, 2.0
D_E, D_P, D_I, W = 1.0, 0.5, 2.0, 0.6
D_W, V = 2.0, 0.5
WELL_MIXED = {
    "transmission": TRANSMISSION,
    "latent_days": D_E,
    "presymptomatic_days": D_P,
    "infectious_days": D_I,
    "waning_days": D_W,
    "waning_infectiousness": V,
    "presymptomatic_infectiousness": W,
    "duration_shape": 1.0,
    "contacts": CONTACTS,
    "susceptibility_cv": 0.0,
    "infectiousness_dispersion": 0.0,
    "vaccine_days": float(LIFELONG_DAYS),
    "immunity_days": float(LIFELONG_DAYS),
    "vaccination_rate": 0.0,
}
# every stay is rounded up to whole steps: half a step longer on average
R0 = (
    -math.log(1 - TRANSMISSION)
    * CONTACTS
    * (W * (D_P + STEP / 2) + D_I + STEP / 2 + V * (D_W + STEP / 2))
)


def leaky_final_size(r0: float, infected: float, vaccinated: float, efficacy: float) -> float:
    susceptible = 1 - infected - vaccinated
    size = 0.5
    for _ in range(2000):
        size = (
            infected
            + susceptible * (1 - math.exp(-r0 * size))
            + vaccinated * (1 - math.exp(-(1 - efficacy) * r0 * size))
        )
    return size


# --- structure -------------------------------------------------------------------------------


def test_the_states_are_in_the_order_of_the_course_and_only_p_i_and_w_spread():
    assert list(SvepirModel.STATE_COLORS) == ORDER
    model = SvepirModel(ONE_DAY, presymptomatic_infectiousness=0.4, waning_infectiousness=0.3)
    without_symptoms = model.state_infectiousness(ASYMPTOMATIC)
    assert [model.state_infectiousness(state) for state in ORDER] == pytest.approx(
        [0, 0, 0, 0.4, 1, 0.3, without_symptoms, 0, 0, 0]
    )
    assert without_symptoms > 0
    assert SvepirModel.course_states() == {
        EXPOSED,
        PRESYMPTOMATIC,
        INFECTIOUS,
        WANING,
        ASYMPTOMATIC,
        RECOVERED,
    }
    assert {INFECTIOUS, WANING} == SvepirModel.SYMPTOMATIC_STATES


def test_an_infection_runs_through_latent_presymptomatic_symptomatic_to_recovered():
    model = SvepirModel(ONE_DAY, **SURE, duration_shape=50.0, initial_coverage=0.0)
    disease_state, rng = new_state(model)
    advance(model, disease_state, rng, exposure=1)
    visited = [ORDER[disease_state.state[0]]]
    for _ in range(40):
        advance(model, disease_state, rng)
        if (current := ORDER[disease_state.state[0]]) != visited[-1]:
            visited.append(current)
    assert visited == [EXPOSED, PRESYMPTOMATIC, INFECTIOUS, WANING, RECOVERED]


def test_the_incubation_period_is_the_latent_plus_the_presymptomatic_period():
    model = SvepirModel(1 / 24, latent_days=1.0, presymptomatic_days=0.4, duration_shape=8.0)
    disease_state, rng = new_state(model, population=20_000)
    model.apply_infections(disease_state, np.arange(20_000), rng)
    onset = np.full(20_000, np.nan)
    for step in range(1, 24 * 6):
        advance(model, disease_state, rng)
        ill = (disease_state.state == ORDER.index(INFECTIOUS)) & np.isnan(onset)
        onset[ill] = step / 24
    assert np.nanmean(onset) == pytest.approx(1.4 + 1 / 48, abs=0.03)


# --- behaviour -------------------------------------------------------------------------------


def test_only_the_symptomatic_react_and_the_presymptomatic_carry_on():
    model = SvepirModel(ONE_DAY, stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, population=4)
    for person, state in enumerate((EXPOSED, PRESYMPTOMATIC, INFECTIOUS, VACCINATED)):
        place(model, disease_state, rng, [person], state)
    assert disease_state.stays_home.tolist() == [False, False, True, False]
    assert disease_state.limits_outings.tolist() == [False, False, True, False]


def test_the_decision_to_stay_at_home_is_drawn_when_the_symptoms_begin():
    model = SvepirModel(1 / 24, stay_home=0.3, presymptomatic_days=0.5, duration_shape=50.0)
    disease_state, rng = new_state(model, population=40_000, cases=np.arange(40_000))
    assert not disease_state.stays_home.any(), "the first cases start without symptoms"
    for _ in range(24):
        advance(model, disease_state, rng)
    ill = disease_state.state == ORDER.index(INFECTIOUS)
    assert ill.mean() > 0.99
    assert disease_state.stays_home[ill].mean() == pytest.approx(0.3, abs=0.01)


# --- vaccination -----------------------------------------------------------------------------


@pytest.mark.parametrize(("state", "protection"), [(VACCINATED, 0.7), (RECOVERED, 0.4)])
def test_each_state_catches_the_infection_at_the_hazard_its_protection_leaves(state, protection):
    model = SvepirModel(
        ONE_DAY, transmission=0.5, vaccine_efficacy=0.7, immunity_protection=0.4, **UNIFORM
    )
    disease_state, rng = new_state(model, population=100_000)
    place(model, disease_state, rng, np.arange(100_000), state)
    infected = model.draw_infections(disease_state, np.ones(100_000, np.float32), rng)
    assert len(infected) / 100_000 == pytest.approx(1 - 0.5 ** (1 - protection), abs=0.006)


def test_a_share_of_the_people_but_not_the_first_cases_starts_vaccinated_scaled_by_age():
    model = SvepirModel(
        ONE_DAY,
        initial_coverage=0.1,
        age_effects={"starting_share": [[0, 65, 0.5], [65, 120, 3.0]]},
    )
    ages = np.repeat([30, 70], 50_000)
    disease_state, _ = new_state(model, population=100_000, cases=np.arange(100), age=ages)
    vaccinated = (disease_state.state == ORDER.index(VACCINATED)).reshape(2, 50_000)
    assert vaccinated[0].mean() == pytest.approx(0.05, abs=0.005)
    assert vaccinated[1].mean() == pytest.approx(0.30, abs=0.01)
    assert (disease_state.state[:100] == ORDER.index(PRESYMPTOMATIC)).all()


def test_the_vaccinated_lose_their_protection_after_the_given_time_and_a_campaign_runs():
    model = SvepirModel(
        ONE_DAY,
        vaccine_days=30.0,
        duration_shape=50.0,
        vaccination_rate=0.1,
        vaccination_start_day=10,
    )
    disease_state, rng = new_state(model, population=20_000)
    place(model, disease_state, rng, np.arange(10_000), VACCINATED)
    for _ in range(9):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) == 10_000, "no campaign before day 10"
    for _ in range(26):
        advance(model, disease_state, rng)
    # the first 10,000 lost their protection around day 30; the campaign vaccinates 1 - e^-0.1 a day
    first = disease_state.state[:10_000]
    assert (first == ORDER.index(VACCINATED)).mean() < 0.9


# --- textbook result -------------------------------------------------------------------------


@pytest.mark.parametrize(("efficacy", "coverage"), [(0.8, 0.0), (0.6, 0.4), (0.0, 0.5)])
def test_the_final_size_follows_the_relation_of_a_leaky_vaccine(efficacy, coverage):
    population, cases, days, runs = 6000, 120, 90, 4
    shares = []
    for seed in range(runs):
        _, counts = run_well_mixed(
            SvepirModel,
            population=population,
            cases=cases,
            days=days,
            seed=seed,
            vaccine_efficacy=efficacy,
            initial_coverage=coverage,
            **WELL_MIXED,
        )
        course = (EXPOSED, PRESYMPTOMATIC, INFECTIOUS, WANING)
        spreading = counts[-1, [ORDER.index(s) for s in course]]
        assert spreading.sum() < 0.01 * population, "the epidemic is over at the end of the run"
        ever = counts[-1, [ORDER.index(s) for s in (RECOVERED, DYING, DEAD)]].sum()
        shares.append(ever / population)
    infected = cases / population
    expected = leaky_final_size(R0, infected, coverage * (1 - infected), efficacy)
    assert np.mean(shares) == pytest.approx(expected, abs=0.03)


# --- the disease of the catalog ----------------------------------------------------------------

# (values both the disease and the model default have, the disease's own values, age effects)
SOURCED = (
    {  # Carrat et al. 2008, Lessler et al. 2009, Lau et al. 2010, Tsang et al. 2015,
        # Belongia et al. 2016, Ferdinands et al. 2017
        "latent_days": 1.0,
        "presymptomatic_days": 0.5,
        "infectious_days": 2.5,
        "waning_days": 1.8,
        "waning_infectiousness": 0.5,
        "presymptomatic_infectiousness": 0.4,
        "vaccine_efficacy": 0.61,
        "vaccine_days": 110,
    },
    {
        "transmission": 0.18,
        "community_factor": 2.6,
        "duration_shape": 5.4,
        "fatality": 0.00014,  # CDC burden estimates 2010-2020, Tsang et al. 2023
        "death_delay_days": 7.7,  # Louie et al. 2009
        "stay_home": 0.40,  # Fragaszy et al. 2018
        "limit_outings": 0.33,
        "infectiousness_dispersion": 0.94,  # Fraser et al. 2011
        "initial_coverage": 0.074,
        "asymptomatic_share": 0.36,  # Tsang et al. 2023
        "asymptomatic_infectiousness": 0.57,  # Tsang et al. 2023
        "asymptomatic_days": 4.8,  # Carrat et al. 2008
    },
    {
        "fatality": [[0, 5, 0.3345], [5, 18, 0.2108], [18, 50, 1.0], [50, 65, 2.969]],
        "susceptibility": [[0, 19, 2.0]],  # Cauchemez et al. 2009
    },
)


def test_the_disease_has_the_numbers_its_comment_cites():
    check_disease_numbers("svepir", *SOURCED)


def test_the_model_defaults_are_the_cited_numbers():
    check_default_numbers("svepir", SOURCED[0])
