"""The SICR model against its definition: the structure, the carrier flow and the textbook ODE.

The model is susceptible -> infectious (acutely ill) -> recovered, or carrier -> recovered, where
the carriers spread less than the ill but feel well and carry on as usual. The dynamics are
compared with the well-mixed equations (with exponential stays, `duration_shape=1`)

    dS = -lambda S,  dI = lambda S - I / d_I,  dC = (1 - f) q I / d_I - C / d_C,
    dR = (1 - f)(1 - q) I / d_I + C / d_C,  dD = f I / d_I,  lambda = beta (I + w C),

whose reproduction number is R0 = beta (d_I + (1 - f) q w d_C).
"""

import math

import numpy as np
import pytest
from config_factory import disease_definition
from disease_helpers import ADULT, advance, count_in, new_state, place
from mean_field import final_size, integrate, mean_course
from test_epidemic_model import arranged_model

from epidemic_sandbox.disease_models.disease_model_abstract import (
    DEAD,
    INFECTIOUS,
    LIFELONG_DAYS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.sicr import CARRIER, SicrModel
from epidemic_sandbox.disease_models.sir import SirModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
FOREVER = float(LIFELONG_DAYS)
STEP_DAYS = 1 / 24


def hazard(transmission: float) -> float:
    return -math.log(1.0 - transmission)


def sicr(step_days=STEP_DAYS, **parameters) -> SicrModel:
    """A model of identical people in which nobody dies unless the test says so."""
    return SicrModel(step_days, **{"fatality": 0.0, **IDENTICAL, **parameters})


def infect_everybody(model, population, ages=None, seed=0):
    """A cohort that has just caught the infection, with no further exposure."""
    disease_state, rng = new_state(
        model, population=population, age=ADULT if ages is None else ages, seed=seed
    )
    place(model, disease_state, rng, np.arange(population), model.INFECTION_ENTRY_STATE)
    return disease_state, rng


def run_cohort(model, disease_state, rng, days):
    for _ in range(round(days / model.step_days)):
        advance(model, disease_state, rng)


# --- structure -----------------------------------------------------------------------------------


def test_the_states_and_who_can_reach_what():
    model = sicr()
    assert list(model.STATE_COLORS) == [SUSCEPTIBLE, INFECTIOUS, CARRIER, RECOVERED, DEAD]
    assert model.INFECTION_ENTRY_STATE == model.INITIAL_CASE_STATE == INFECTIOUS
    assert model.course_states() == {INFECTIOUS, CARRIER, RECOVERED}
    acute, carrier = model.PHASES[INFECTIOUS], model.PHASES[CARRIER]
    assert (acute.next_state, acute.branch_state, acute.fatal) == (RECOVERED, CARRIER, True)
    assert (carrier.next_state, carrier.branch_state, carrier.fatal) == (RECOVERED, None, False)
    assert model.PHASES[RECOVERED].next_state == SUSCEPTIBLE


def test_both_the_ill_and_the_carriers_spread_but_only_the_ill_react_to_illness():
    model = sicr(carrier_infectiousness=0.2)
    assert {INFECTIOUS, CARRIER} == model.INFECTIOUS_STATES
    assert {INFECTIOUS} == model.SYMPTOMATIC_STATES
    assert not model.CONFINED_STATES
    assert [model.state_infectiousness(state) for state in model.STATE_COLORS] == pytest.approx(
        [0.0, 1.0, 0.2, 0.0, 0.0]
    )


def test_recovered_people_are_protected_and_carriers_cannot_be_infected_again():
    model = sicr(immunity_protection=0.7)
    assert model.PROTECTION == {RECOVERED: "immunity_protection"}
    disease_state, rng = new_state(model, population=1000)
    place(model, disease_state, rng, np.arange(1000), CARRIER)
    assert not model.exposable_mask(disease_state).any()
    model.advance(disease_state, np.full(1000, 50.0, np.float32), rng)
    assert count_in(model, disease_state, CARRIER) == 1000


def test_the_parameters_cover_the_model_and_all_have_a_range_for_months_to_a_lifetime():
    parameters = SicrModel.PARAMETERS
    assert {"infectious_days", "carrier_share", "carrier_days", "carrier_infectiousness"} <= (
        parameters.keys()
    )
    carrier_days = parameters["carrier_days"]
    assert carrier_days.minimum <= 30 and carrier_days.maximum == LIFELONG_DAYS
    assert parameters["carrier_infectiousness"].maximum == 1.0


# --- who becomes a carrier -----------------------------------------------------------------------


def shares_after_the_acute_illness(model, ages, seed):
    """Share of each age group that is a carrier, recovered or dead once the acute illness ends."""
    population = len(ages)
    disease_state, rng = infect_everybody(model, population, ages, seed)
    run_cohort(model, disease_state, rng, days=60)
    assert count_in(model, disease_state, INFECTIOUS) == 0
    groups = np.unique(ages)
    return {
        state: np.array([np.mean(disease_state.state[ages == age] == index) for age in groups])
        for state, index in ((CARRIER, 2), (RECOVERED, 3), (DEAD, 4))
    }


def test_the_carrier_share_is_the_parameter_times_the_branching_age_effect_per_age_group():
    effects = {"branching": [[0, 15, 0.25], [15, 50, 1.0], [50, 120, 2.5]]}
    model = sicr(
        step_days=0.25, infectious_days=4.0, carrier_share=0.2, carrier_days=FOREVER,
        age_effects=effects,
    )  # fmt: skip
    group_size = 20_000
    ages = np.repeat([5, 30, 70], group_size)
    shares = shares_after_the_acute_illness(model, ages, seed=1)
    for group, share in enumerate((0.05, 0.2, 0.5)):
        tolerance = 4 * math.sqrt(share * (1 - share) / group_size)
        assert shares["carrier"][group] == pytest.approx(share, abs=tolerance), group
        assert shares["recovered"][group] == pytest.approx(1 - share, abs=tolerance), group
    assert shares["dead"].tolist() == [0.0, 0.0, 0.0]


def test_the_chance_of_becoming_a_carrier_is_capped_at_one():
    model = sicr(
        step_days=0.25, infectious_days=2.0, carrier_share=0.4, carrier_days=FOREVER,
        age_effects={"branching": [[0, 10, 5.0]]},
    )  # fmt: skip
    shares = shares_after_the_acute_illness(model, np.repeat([5], 2000), seed=2)
    assert shares["carrier"][0] == 1.0


def test_only_the_survivors_of_the_acute_illness_can_become_carriers():
    model = sicr(
        step_days=0.25, infectious_days=3.0, fatality=0.3, carrier_share=0.4, carrier_days=FOREVER
    )
    population = 40_000
    shares = shares_after_the_acute_illness(model, np.full(population, ADULT), seed=3)
    for state, expected in (("dead", 0.3), ("carrier", 0.7 * 0.4), ("recovered", 0.7 * 0.6)):
        sigma = math.sqrt(expected * (1 - expected) / population)
        assert shares[state][0] == pytest.approx(expected, abs=4 * sigma), state


def test_fatality_and_the_branching_effect_come_with_the_age_of_the_person():
    effects = {"fatality": [[0, 15, 0.5], [65, 120, 4.0]], "branching": [[65, 120, 2.0]]}
    model = sicr(fatality=0.1, carrier_share=0.2, age_effects=effects)
    disease_state, _ = new_state(model, population=3, age=[10, 40, 80])
    assert disease_state.fatality_risk.tolist() == pytest.approx([0.05, 0.1, 0.4])
    assert disease_state.branching.tolist() == [1.0, 1.0, 2.0]


def test_susceptibility_depends_on_age():
    model = sicr(age_effects={"susceptibility": [[0, 15, 0.5], [65, 120, 1.5]]})
    disease_state, _ = new_state(model, population=3, age=[10, 40, 80])
    assert disease_state.susceptibility.tolist() == [0.5, 1.0, 1.5]


def test_a_carrier_never_dies_of_the_infection():
    model = sicr(step_days=0.25, fatality=1.0, carrier_days=6.0)
    disease_state, rng = new_state(model, population=5000)
    place(model, disease_state, rng, np.arange(5000), CARRIER)
    run_cohort(model, disease_state, rng, days=80)
    assert count_in(model, disease_state, RECOVERED) == 5000
    assert disease_state.deaths == 0


def test_with_certain_death_nobody_survives_to_become_a_carrier():
    model = sicr(step_days=0.25, infectious_days=2.0, fatality=1.0, carrier_share=1.0)
    disease_state, rng = infect_everybody(model, 2000)
    run_cohort(model, disease_state, rng, days=40)
    assert count_in(model, disease_state, DEAD) == 2000


# --- how long ------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [1.0, 4.0])
def test_a_carrier_stays_for_the_declared_time_and_then_is_immune(shape):
    model = sicr(carrier_days=12.0, duration_shape=shape, immunity_protection=1.0)
    population = 20_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), CARRIER)
    steps = np.zeros(population)
    for _ in range(round(150 / model.step_days)):
        steps += disease_state.state == 2
        advance(model, disease_state, rng)
    days = steps * model.step_days
    assert count_in(model, disease_state, CARRIER) == 0
    assert count_in(model, disease_state, RECOVERED) > 0.99 * population
    # a stay is the continuous gamma stay rounded up to whole steps
    error = 5 * days.std() / math.sqrt(population)
    assert days.mean() == pytest.approx(12.0 + model.step_days / 2, abs=error + 0.005)
    assert days.std() == pytest.approx(12.0 / math.sqrt(shape), rel=0.03)


def test_carriage_for_life_never_ends_and_the_carrier_keeps_spreading():
    model = sicr(carrier_days=FOREVER, step_days=1.0)
    disease_state, rng = new_state(model, population=300)
    place(model, disease_state, rng, np.arange(300), CARRIER)
    for _ in range(3000):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, CARRIER) == 300
    assert not model.is_quiescent(disease_state)


def test_the_run_is_over_when_nobody_is_ill_or_a_carrier_any_more():
    model = sicr(carrier_days=30.0, immunity_days=FOREVER)
    disease_state, rng = new_state(model, population=10)
    place(model, disease_state, rng, [0], CARRIER)
    assert not model.is_quiescent(disease_state)
    place(model, disease_state, rng, [0], RECOVERED)
    assert model.is_quiescent(disease_state)


# --- behaviour -----------------------------------------------------------------------------------


def test_the_ill_stay_at_home_and_skip_outings_but_carriers_carry_on_as_usual():
    model = sicr(stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, population=4)
    place(model, disease_state, rng, [0], INFECTIOUS)
    place(model, disease_state, rng, [1], CARRIER)
    assert disease_state.stays_home.tolist() == [True, False, False, False]
    assert disease_state.limits_outings.tolist() == [True, False, False, False]


def test_a_person_who_becomes_a_carrier_stops_staying_at_home():
    model = sicr(
        step_days=0.25, stay_home=1.0, limit_outings=1.0, infectious_days=2.0, carrier_share=1.0,
        carrier_days=FOREVER,
    )  # fmt: skip
    disease_state, rng = infect_everybody(model, 200)
    assert disease_state.stays_home.all() and disease_state.limits_outings.all()
    run_cohort(model, disease_state, rng, days=40)
    assert count_in(model, disease_state, CARRIER) == 200
    assert not disease_state.stays_home.any() and not disease_state.limits_outings.any()


def test_in_the_town_the_ill_stay_in_and_carriers_go_about_their_day():
    diseases = {
        "a": disease_definition(
            "sicr", 0, stay_home=1.0, transmission=0.0, carrier_days=FOREVER, infectious_days=60.0
        )
    }
    model = arranged_model(diseases, {})
    ill, carriers = np.arange(0, 100, 2), np.arange(1, 100, 2)
    disease_state, disease = model.disease_states["a"], model.diseases["a"]
    rng = np.random.default_rng(0)
    disease._enter(disease_state, ill, INFECTIOUS, rng)
    disease._enter(disease_state, carriers, CARRIER, rng)
    disease_state.timer[:] = 1e6
    assert model._goes_home().tolist() == [True, False] * 50
    # the arranged town keeps everybody at home on a sparse grid, so move the schedule: Monday
    # morning sends the carriers to work and school, the ill stay where they are
    town = model.town
    model.x[:], model.y[:] = town.home_x, town.home_y
    away = np.zeros(model.population)
    for _ in range(24):
        model.step()
        away = np.maximum(away, np.hypot(model.x - town.home_x, model.y - town.home_y))
    assert away[ill].max() < 1e-6
    assert away[carriers].max() > 50.0


# --- the dynamics against the equations ----------------------------------------------------------


def sicr_ode(contacts, transmission, infectious_days, fatality, share, weight, carrier_days):
    beta = hazard(transmission) * contacts

    def derivative(y):
        s, i, c, r, d = y
        force = beta * (i + weight * c)
        leaving = i / infectious_days
        return np.array(
            [
                -force * s,
                force * s - leaving,
                (1 - fatality) * share * leaving - c / carrier_days,
                (1 - fatality) * (1 - share) * leaving + c / carrier_days,
                fatality * leaving,
            ]
        )

    return derivative


def test_the_well_mixed_course_follows_the_sicr_equations():
    contacts, transmission, days_ill, days_carrier = 3.0, 0.3, 4.0, 40.0
    fatality, share, weight = 0.1, 0.3, 0.5
    population, cases, days = 20_000, 100, 110
    ode = integrate(
        sicr_ode(contacts, transmission, days_ill, fatality, share, weight, days_carrier),
        np.array([1 - cases / population, cases / population, 0.0, 0.0, 0.0]),
        days,
    )
    _, counts = mean_course(
        SicrModel,
        runs=3,
        population=population,
        cases=cases,
        days=days,
        contacts=contacts,
        transmission=transmission,
        infectious_days=days_ill,
        fatality=fatality,
        carrier_share=share,
        carrier_infectiousness=weight,
        carrier_days=days_carrier,
        duration_shape=1.0,
        immunity_days=FOREVER,
        **IDENTICAL,
    )
    shares = counts / population
    ill, carriers = shares[:, 1], shares[:, 2]
    # the half-step offset of the stays and the noise of three runs fit within these margins
    assert ill.max() == pytest.approx(ode[:, 1].max(), rel=0.08)
    assert ill.argmax() / 24 == pytest.approx(ode[:, 1].argmax() / 24, abs=2.0)
    assert carriers.max() == pytest.approx(ode[:, 2].max(), rel=0.08)
    assert carriers.argmax() / 24 == pytest.approx(ode[:, 2].argmax() / 24, abs=6.0)
    for day in (30, 50, 110):
        assert carriers[day * 24] == pytest.approx(ode[day * 24, 2], rel=0.1, abs=0.003), day
    assert shares[-1, 4] == pytest.approx(ode[-1, 4], rel=0.05)
    assert shares[-1, 3] == pytest.approx(ode[-1, 3], rel=0.05)


def reproduction_number(contacts, transmission, days_ill, fatality, share, weight, days_carrier):
    return (
        hazard(transmission)
        * contacts
        * (days_ill + (1 - fatality) * share * weight * days_carrier)
    )


def test_carriers_make_the_difference_between_a_fizzle_and_an_epidemic():
    contacts, transmission, days_ill, days_carrier = 3.0, 0.2, 1.0, 20.0
    population, cases = 10_000, 100
    common = {
        "population": population, "cases": cases, "days": 200, "contacts": contacts,
        "transmission": transmission, "infectious_days": days_ill, "fatality": 0.0,
        "carrier_share": 0.5, "carrier_days": days_carrier, "immunity_days": FOREVER,
        **IDENTICAL,
    }  # fmt: skip
    acute_only = reproduction_number(contacts, transmission, days_ill, 0.0, 0.5, 0.0, days_carrier)
    with_carriers = reproduction_number(
        contacts, transmission, days_ill, 0.0, 0.5, 0.5, days_carrier
    )
    assert acute_only < 1 < with_carriers
    _, silent = mean_course(SicrModel, runs=2, carrier_infectiousness=0.0, **common)
    _, spreading = mean_course(SicrModel, runs=2, carrier_infectiousness=0.5, **common)
    ever_infected = 1 - spreading[-1, 0] / population
    assert ever_infected == pytest.approx(final_size(with_carriers, cases / population), abs=0.015)
    # without carriers each case causes 1 / (1 - R) cases in all, so the outbreak stays small
    small = cases / population / (1 - acute_only)
    assert 1 - silent[-1, 0] / population == pytest.approx(small, rel=0.2)


def test_the_preview_numbers_follow_the_reproduction_number_of_the_equations():
    model = sicr(
        transmission=0.2, infectious_days=6.0, fatality=0.1, carrier_share=0.25,
        carrier_infectiousness=0.4, carrier_days=100.0,
    )  # fmt: skip
    expected_days = 6.0 + 0.9 * 0.25 * 0.4 * 100.0
    assert model.infectious_days() == pytest.approx(expected_days)
    assert model.mean_field_r0(3.0) == pytest.approx(hazard(0.2) * 3.0 * expected_days)
    assert model.infectious_days() - 6.0 == pytest.approx(0.9 * 0.25 * 0.4 * 100.0)


def test_with_carriage_for_life_the_reproduction_number_is_unbounded_not_a_number():
    model = sicr(
        carrier_days=FOREVER, carrier_share=0.03, carrier_infectiousness=0.1, fatality=0.02
    )
    assert math.isinf(model.infectious_days())
    assert model.mean_field_r0(10.0) is None
    carriers = [course for course in model.courses() if course.end == CARRIER]
    assert [course.probability for course in carriers] == pytest.approx([0.98 * 0.03])
    assert math.isinf(carriers[0].stays[-1][1])
    household_risk = model.pair_transmission_probability()
    assert 0.0 <= household_risk <= 1.0 and math.isfinite(household_risk)


def test_carriers_that_do_not_spread_leave_the_preview_finite_whatever_their_stay():
    model = sicr(carrier_days=FOREVER, carrier_infectiousness=0.0, infectious_days=5.0)
    assert model.infectious_days() == pytest.approx(5.0)
    assert math.isfinite(model.mean_field_r0(10.0))


def test_without_carriers_the_model_is_an_sir_model_and_the_preview_agrees():
    model = sicr(carrier_share=0.0, carrier_days=FOREVER, transmission=0.25, infectious_days=9.0)
    sir = SirModel(STEP_DAYS, transmission=0.25, infectious_days=9.0, **IDENTICAL)
    assert model.infectious_days() == sir.infectious_days()
    assert model.mean_field_r0(4.0) == pytest.approx(sir.mean_field_r0(4.0))
    assert model.pair_transmission_probability() == pytest.approx(
        sir.pair_transmission_probability()
    )


def test_pair_transmission_probability_is_the_closed_form_of_the_gamma_stays():
    shape, tau = 3.0, 0.15
    model = sicr(
        transmission=tau, infectious_days=5.0, fatality=0.2, carrier_share=0.3,
        carrier_infectiousness=0.25, carrier_days=40.0, duration_shape=shape,
    )  # fmt: skip

    def laplace(mean, weight):  # E[exp(-hazard * weight * T)] for T ~ gamma(shape, mean)
        return (1 + hazard(tau) * weight * mean / shape) ** -shape

    survival = laplace(5.0, 1.0) * (1 - 0.8 * 0.3 * (1 - laplace(40.0, 0.25)))
    assert model.pair_transmission_probability() == pytest.approx(1 - survival, abs=1e-6)


def test_the_simulated_infectious_time_splits_between_the_ill_and_the_carriers():
    model = sicr(
        step_days=1 / 24, infectious_days=5.0, fatality=0.1, carrier_share=0.3,
        carrier_infectiousness=0.25, carrier_days=15.0, transmission=0.3,
    )  # fmt: skip
    population = 20_000
    disease_state, rng = infect_everybody(model, population, seed=5)
    spread, as_carrier = np.zeros(population), np.zeros(population)
    for _ in range(round(100 / model.step_days)):
        weights = model.infectious_weight(disease_state)
        spread += weights * model.step_days
        as_carrier += (disease_state.state == 2) * weights * model.step_days
        advance(model, disease_state, rng)
    # every stay is rounded up to whole steps, half a step on the average
    bias = model.step_days / 2 * (1 + 0.9 * 0.3 * 0.25)
    assert spread.mean() == pytest.approx(
        model.infectious_days() + bias, abs=5 * spread.std() / math.sqrt(population)
    )
    carrier_part = 0.9 * 0.3 * 0.25 * 15.0
    assert as_carrier.mean() == pytest.approx(
        carrier_part + 0.9 * 0.3 * 0.25 * model.step_days / 2,
        abs=5 * as_carrier.std() / math.sqrt(population),
    )
    risk = 1 - np.exp(-hazard(0.3) * spread)
    assert risk.mean() == pytest.approx(
        model.pair_transmission_probability(),
        abs=5 * risk.std() / math.sqrt(population) + hazard(0.3) * bias,
    )


# --- the infection that carriers keep alive ------------------------------------------------------

OUTBREAK = {
    "transmission": 0.2, "infectious_days": 5.0, "fatality": 0.0, "carrier_share": 0.1,
    "carrier_infectiousness": 0.3, "carrier_days": FOREVER, "immunity_days": FOREVER,
    "duration_shape": 1.0, **IDENTICAL,
}  # fmt: skip
CONTACTS = 3.0


def infections_by_source(population, cases, days, seed, **parameters):
    """Well-mixed run that also counts how many new infections the carriers caused.

    A step's new infections are split in proportion to the carriers' share of the exposure,
    which is what the infection chance of every person is made of. Returns the infections
    caused by carriers, all infections, and the numbers susceptible and ill at every step.
    """
    model = SicrModel(STEP_DAYS, **{**OUTBREAK, **parameters})
    rng = np.random.default_rng(seed)
    ages = np.full(population, ADULT)
    state = model.create_state(population, rng.choice(population, cases, replace=False), rng, ages)
    from_carriers = 0.0
    curve = np.zeros((round(days / STEP_DAYS) + 1, 2))
    curve[0] = count_in(model, state, SUSCEPTIBLE), count_in(model, state, INFECTIOUS)
    for step in range(round(days / STEP_DAYS)):
        weights = model.infectious_weight(state) * state.infectiousness
        carriers = (weights * (state.state == 2)).sum()
        before = state.cumulative_infections
        exposure = np.full(population, CONTACTS * weights.sum() / population, np.float32)
        model.advance(state, exposure, rng)
        if weights.sum() > 0:
            from_carriers += (state.cumulative_infections - before) * carriers / weights.sum()
        curve[step + 1] = count_in(model, state, SUSCEPTIBLE), count_in(model, state, INFECTIOUS)
    return from_carriers, state.cumulative_infections, curve


def test_carriers_keep_infecting_after_the_acute_wave_until_nobody_is_left_to_infect():
    population, cases, days = 20_000, 100, 240
    ode = integrate(
        sicr_ode(CONTACTS, 0.2, 5.0, 0.0, 0.1, 0.3, FOREVER),
        np.array([1 - cases / population, cases / population, 0, 0, 0]),
        days,
    )
    by_carriers, total, curve = infections_by_source(population, cases, days, seed=7)
    # the acute wave is over for good, yet the infection goes on: the carriers use up the
    # susceptible (as the ODE says), while without carriers a share would stay untouched
    assert curve[:, 1].argmax() * STEP_DAYS < 60 and curve[-1, 1] < 0.001 * population
    assert curve[-1, 0] / population == pytest.approx(ode[-1, 0], abs=0.01)
    r0_acute = reproduction_number(CONTACTS, 0.2, 5.0, 0.0, 0.1, 0.0, FOREVER)
    untouched_without_carriers = 1 - final_size(r0_acute, cases / population)
    assert untouched_without_carriers > 10 * curve[-1, 0] / population
    # the carriers' share of all new infections, bookkept the same way on the ODE
    ode_from_carriers = hazard(0.2) * CONTACTS * 0.3 * (ode[:, 2] * ode[:, 0]).sum() * STEP_DAYS
    assert by_carriers / population == pytest.approx(ode_from_carriers, rel=0.15)
    assert total / population == pytest.approx(1 - ode[-1, 0], abs=0.01)


def test_without_carriers_the_infection_ends_with_the_acute_wave():
    population, cases = 20_000, 100
    by_carriers, total, curve = infections_by_source(
        population, cases, 200, seed=7, carrier_share=0.0
    )
    r0 = reproduction_number(CONTACTS, 0.2, 5.0, 0.0, 0.0, 0.0, FOREVER)
    expected = final_size(r0, cases / population)
    assert by_carriers == 0.0 and curve[-1, 1] == 0
    assert 1 - curve[-1, 0] / population == pytest.approx(expected, abs=0.015)
    assert total / population == pytest.approx(expected, abs=0.015)


def test_without_carriers_the_course_is_that_of_the_sir_model():
    common = {
        "population": 10_000, "cases": 250, "days": 60, "contacts": 3.0, "transmission": 0.15,
        "infectious_days": 6.0, "immunity_days": FOREVER, **IDENTICAL,
    }  # fmt: skip
    _, from_sicr = mean_course(SicrModel, runs=4, carrier_share=0.0, fatality=0.0, **common)
    _, from_sir = mean_course(SirModel, runs=4, seed=100, **common)
    assert from_sicr[:, 2].max() == 0
    ill = np.stack([from_sicr[:, 1], from_sir[:, 1]]) / 10_000
    assert ill[0].max() == pytest.approx(ill[1].max(), rel=0.05)
    assert ill[0].argmax() / 24 == pytest.approx(ill[1].argmax() / 24, abs=1.5)
    assert from_sicr[-1, 0] == pytest.approx(from_sir[-1, 0], abs=0.01 * 10_000)


# --- the effect of the parameters ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("weaker", "stronger"),
    [
        ({"carrier_share": 0.03}, {"carrier_share": 0.2}),
        ({"carrier_infectiousness": 0.1}, {"carrier_infectiousness": 0.5}),
        ({"carrier_days": 20.0}, {"carrier_days": FOREVER}),
    ],
)
def test_more_longer_and_stronger_carriers_cause_more_infections(weaker, stronger):
    def caused(parameters):
        return np.mean(
            [infections_by_source(5_000, 50, 90, seed, **parameters)[0] for seed in range(3)]
        )

    assert caused(stronger) > 1.5 * caused(weaker)


def test_carriers_that_do_not_spread_cause_no_infections():
    by_carriers, _, _ = infections_by_source(10_000, 100, 60, seed=0, carrier_infectiousness=0.0)
    assert by_carriers == 0.0
