"""SEPIR against its definition: the compartments, the textbook ODE, presymptomatic spread.

The model is S -> E -> P -> I -> R (or dead) with the relative infectiousness w of P. In a
well-mixed population with exponential stays its mean course is the ODE
    S' = -b S (w P + I) / N,  E' = -S' - E / d_E,  P' = E / d_E - P / d_P,  I' = P / d_P - I / d_I
with b = -ln(1 - transmission) * contacts, so that R0 = b (w d_P + d_I) and the share of
transmission before symptoms is theta = w d_P / (w d_P + d_I) (Fraser et al. 2004, PNAS).
A stay measured in steps is the continuous stay rounded up, which adds half a step to its mean;
the predictions below use these means (a bias the simulation has by construction).
"""

import math
from functools import cache

import numpy as np
import pytest
from config_factory import LIBRARY
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, new_state, place, state_names
from mean_field import final_size, integrate, run_well_mixed
from test_epidemic_model import arranged_model

from epidemic_sandbox.disease_models.disease_model_abstract import (
    DEAD,
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.sepir import PRESYMPTOMATIC, SepirModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0, "immunity_days": 36500}
ORDER = [SUSCEPTIBLE, EXPOSED, PRESYMPTOMATIC, INFECTIOUS, RECOVERED, DEAD]
DAYS = {"latent_days": 3.0, "presymptomatic_days": 2.5, "infectious_days": 3.0}


class IsolatingSepirModel(SepirModel):
    """SEPIR in a well-mixed population where people who stay at home meet nobody there.

    The town keeps them in touch with their household; a well-mixed population has no households,
    so this is the perfectly isolating case of Fraser et al. (2004).
    """

    def infectious_weight(self, disease_state):
        return super().infectious_weight(disease_state) * ~disease_state.stays_home


def hazard(transmission: float) -> float:
    return -math.log(1.0 - transmission)


def stays(step_days: float, latent: float, presymptomatic: float, symptomatic: float) -> tuple:
    """Mean stays of the simulation: the continuous mean plus half a step."""
    return tuple(days + step_days / 2 for days in (latent, presymptomatic, symptomatic))


def r0(b: float, w: float, d_presymptomatic: float, d_symptomatic: float) -> float:
    return b * (w * d_presymptomatic + d_symptomatic)


def theta(w: float, d_presymptomatic: float, d_symptomatic: float) -> float:
    return w * d_presymptomatic / (w * d_presymptomatic + d_symptomatic)


def growth_rate(b, w, d_latent, d_presymptomatic, d_symptomatic, shape) -> float:
    """The root r > 0 of the Euler-Lotka equation for gamma distributed stays of this shape.

    1 = b phi_E(r) [w (1 - phi_P(r)) + phi_P(r) (1 - phi_I(r))] / r,  phi(r) = (1 + r d / k)^-k
    is the Laplace transform of the infectiousness of an infected person against its age.
    """

    def phi(r, days):
        return (1 + r * days / shape) ** -shape

    def excess(r):
        e, p, i = (phi(r, d) for d in (d_latent, d_presymptomatic, d_symptomatic))
        return b * e * (w * (1 - p) + p * (1 - i)) / r - 1

    low, high = 1e-6, 10.0
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if excess(middle) > 0 else (low, middle)
    return (low + high) / 2


# --- the declaration -----------------------------------------------------------------------------


def test_the_states_are_in_the_order_of_the_course_and_only_p_and_i_spread():
    assert list(SepirModel.STATE_COLORS) == ORDER
    assert {PRESYMPTOMATIC, INFECTIOUS} == SepirModel.INFECTIOUS_STATES
    assert SepirModel.course_states() == {EXPOSED, PRESYMPTOMATIC, INFECTIOUS, RECOVERED}
    assert SepirModel.INFECTION_ENTRY_STATE == EXPOSED
    assert SepirModel.INITIAL_CASE_STATE == PRESYMPTOMATIC, "the seeds spread for the whole course"
    assert not SepirModel.CONFINED_STATES and not SepirModel.RATES


def test_the_incubation_period_is_the_latent_plus_the_presymptomatic_period():
    step_days, population = 1 / 24, 30_000
    model = SepirModel(step_days, **DAYS, duration_shape=4.0)
    disease_state, rng = new_state(model, population)
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    onset = np.full(population, np.nan)
    for step in range(1, round(40 / step_days)):
        advance(model, disease_state, rng)
        symptomatic = (disease_state.state == ORDER.index(INFECTIOUS)) & np.isnan(onset)
        onset[symptomatic] = step * step_days
    assert not np.isnan(onset).any()
    # two stays, each rounded up to whole steps (mean + step / 2); the onset is seen one step
    # after the entry, which the half steps do not cover, hence the third half step
    mean = DAYS["latent_days"] + DAYS["presymptomatic_days"] + 3 * step_days / 2
    assert onset.mean() == pytest.approx(mean, abs=5 * onset.std() / math.sqrt(population))
    variance = (DAYS["latent_days"] ** 2 + DAYS["presymptomatic_days"] ** 2) / 4.0
    assert onset.std() == pytest.approx(math.sqrt(variance), rel=0.03)


def test_a_person_goes_through_every_state_in_order_and_recovers():
    model = SepirModel(ONE_DAY, **SURE, **DAYS, duration_shape=50.0)
    disease_state, rng = new_state(model)
    advance(model, disease_state, rng, exposure=1)
    visited = [state_names(model, disease_state)[0]]
    for _ in range(40):
        advance(model, disease_state, rng)
        current = state_names(model, disease_state)[0]
        if current != visited[-1]:
            visited.append(current)
    assert visited == [EXPOSED, PRESYMPTOMATIC, INFECTIOUS, RECOVERED]


def test_a_reinfection_starts_again_in_the_latent_state():
    model = SepirModel(ONE_DAY, **SURE, immunity_protection=0.0, **UNIFORM)
    disease_state, rng = new_state(model, population=20)
    disease_state.state[:] = ORDER.index(RECOVERED)
    advance(model, disease_state, rng, exposure=5)
    assert set(state_names(model, disease_state)) == {EXPOSED}


def test_the_relative_infectiousness_of_each_state():
    model = SepirModel(ONE_DAY, presymptomatic_infectiousness=0.4)
    assert [model.state_infectiousness(state) for state in ORDER] == pytest.approx(
        [0, 0, 0.4, 1, 0, 0]
    )
    assert SepirModel(ONE_DAY, presymptomatic_infectiousness=2.0).state_infectiousness(
        PRESYMPTOMATIC
    ) == pytest.approx(2.0)


def test_only_the_symptomatic_react_to_their_illness_and_the_presymptomatic_carry_on():
    model = SepirModel(ONE_DAY, stay_home=1.0, limit_outings=1.0, duration_shape=50.0, **DAYS)
    disease_state, rng = new_state(model, population=200, cases=np.arange(200))
    seen = set()
    for _ in range(30):
        names = np.array(state_names(model, disease_state))
        seen |= set(names)
        for state in (EXPOSED, PRESYMPTOMATIC, INFECTIOUS, RECOVERED):
            here = names == state
            assert (disease_state.stays_home[here] == (state == INFECTIOUS)).all(), state
            assert (disease_state.limits_outings[here] == (state == INFECTIOUS)).all(), state
        advance(model, disease_state, rng)
    assert {PRESYMPTOMATIC, INFECTIOUS, RECOVERED} <= seen, "every state was checked"


def test_the_decision_to_stay_at_home_is_drawn_when_the_symptoms_begin():
    model = SepirModel(ONE_DAY, stay_home=0.3, **DAYS)
    disease_state, rng = new_state(model, population=40_000)
    place(model, disease_state, rng, np.arange(20_000), PRESYMPTOMATIC)
    place(model, disease_state, rng, np.arange(20_000, 40_000), INFECTIOUS)
    assert not disease_state.stays_home[:20_000].any()
    assert disease_state.stays_home[20_000:].mean() == pytest.approx(0.3, abs=0.01)


def test_in_a_town_presymptomatic_people_spread_normally_and_stay_in_the_open():
    model = arranged_model(
        {
            "a": {
                "model": "sepir",
                "initial_infected": 0,
                "parameters": {
                    "stay_home": 1.0,
                    "limit_outings": 1.0,
                    "presymptomatic_infectiousness": 0.6,
                },
            }
        },
        {},
    )
    disease = model.diseases["a"]
    for person, state in enumerate((EXPOSED, PRESYMPTOMATIC, INFECTIOUS, RECOVERED)):
        disease._enter(
            model.disease_states["a"], np.array([person]), state, np.random.default_rng(1)
        )
    assert model._goes_home()[:4].tolist() == [False, False, True, False]
    assert model._infectiousness("a")[:4].tolist() == pytest.approx([0.0, 0.6, 1.0, 0.0])


# --- death, age and immunity ---------------------------------------------------------------------


def cohort_visits(model, population, days, seed=0, ages=None):
    """Everybody infected at once: which states each person visited by the end."""
    disease_state, rng = new_state(
        model, population=population, age=40 if ages is None else ages, seed=seed
    )
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    visited = np.zeros((len(ORDER), population), np.bool_)
    for _ in range(round(days / model.step_days)):
        for index in range(len(ORDER)):
            visited[index] |= disease_state.state == index
        advance(model, disease_state, rng)
    return visited, disease_state


def test_people_die_only_at_the_end_of_the_symptomatic_period_of_a_whole_illness():
    model = SepirModel(1 / 12, fatality=0.25, **DAYS)
    visited, disease_state = cohort_visits(model, 20_000, 120.0)
    dead = disease_state.state == ORDER.index(DEAD)
    assert dead.mean() == pytest.approx(0.25, abs=4 * math.sqrt(0.25 * 0.75 / 20_000))
    for state in (EXPOSED, PRESYMPTOMATIC, INFECTIOUS):
        assert visited[ORDER.index(state)][dead].all(), f"the dead had been {state}"
    assert not visited[ORDER.index(RECOVERED)][dead].any()
    assert disease_state.deaths == dead.sum()


def test_nobody_dies_before_the_symptomatic_stay_is_over():
    model = SepirModel(ONE_DAY, fatality=1.0, duration_shape=50.0, **DAYS)
    disease_state, rng = new_state(model, population=300)
    place(model, disease_state, rng, np.arange(300), INFECTIOUS)
    advance(model, disease_state, rng)
    assert state_names(model, disease_state).count(INFECTIOUS) == 300, "alive on the first day"
    for _ in range(6):
        advance(model, disease_state, rng)
    assert state_names(model, disease_state).count(DEAD) == 300


def test_fatality_and_susceptibility_depend_on_age():
    effects = {"fatality": [[0, 15, 0.0], [65, 120, 4.0]], "susceptibility": [[0, 15, 0.5]]}
    ages = np.repeat([5, 40, 80], 20_000)
    model = SepirModel(1 / 6, fatality=0.1, age_effects=effects, **UNIFORM)
    _, disease_state = cohort_visits(model, 60_000, 90.0, ages=ages)
    dead = (disease_state.state == ORDER.index(DEAD)).reshape(3, 20_000).mean(axis=1)
    assert dead[0] == 0.0
    assert dead[1] == pytest.approx(0.1, abs=4 * math.sqrt(0.09 / 20_000))
    assert dead[2] == pytest.approx(0.4, abs=4 * math.sqrt(0.24 / 20_000))
    fresh, rng = new_state(model, population=60_000, age=ages)
    model.advance(fresh, np.full(60_000, 3.0, np.float32), rng)
    infected = (fresh.state == ORDER.index(EXPOSED)).reshape(3, 20_000).mean(axis=1)
    chance = 1 - (1 - 0.3) ** (3.0 * np.array([0.5, 1.0, 1.0]) * model.step_days)
    assert infected == pytest.approx(chance, abs=4 * math.sqrt(0.14 / 20_000))


def test_an_age_effect_on_branching_is_rejected_because_nobody_branches():
    with pytest.raises(ValueError, match="does nothing"):
        SepirModel(ONE_DAY, age_effects={"branching": [[0, 15, 2.0]]})


# --- the numbers shown before a run --------------------------------------------------------------


@pytest.mark.parametrize("w", [0.0, 0.5, 1.0, 2.0])
def test_the_infectious_time_weighs_the_presymptomatic_period_by_its_infectiousness(w):
    model = SepirModel(ONE_DAY, presymptomatic_infectiousness=w, transmission=0.4, **DAYS)
    assert model.infectious_days() == pytest.approx(w * 2.5 + 3.0)
    assert model.mean_field_r0(2.0) == pytest.approx(hazard(0.4) * 2.0 * (w * 2.5 + 3.0))


@pytest.mark.parametrize("shape", [1.0, 4.0, 20.0])
def test_a_pair_next_to_each_other_all_along_infects_with_the_chance_the_gamma_stays_give(shape):
    w, transmission = 0.7, 0.35
    model = SepirModel(
        ONE_DAY, duration_shape=shape, presymptomatic_infectiousness=w, transmission=transmission
    )
    h = hazard(transmission)
    # E[exp(-h (w X_P + X_I))] for independent gamma stays of the given means and shape
    survival = (1 + h * w * 2.5 / shape) ** -shape * (1 + h * 3.0 / shape) ** -shape
    # the weight w is stored as a float32
    assert model.pair_transmission_probability() == pytest.approx(1 - survival, abs=1e-6)
    rng = np.random.default_rng(0)
    spread = w * rng.gamma(shape, 2.5 / shape, 400_000) + rng.gamma(shape, 3.0 / shape, 400_000)
    assert model.pair_transmission_probability() == pytest.approx(
        np.mean(1 - np.exp(-h * spread)), abs=0.003
    )


def test_the_share_of_infectiousness_before_symptoms_is_theta_of_fraser_et_al():
    step_days, population = 1 / 24, 20_000
    w = 0.8
    model = SepirModel(step_days, presymptomatic_infectiousness=w, **DAYS)
    disease_state, rng = new_state(model, population)
    place(model, disease_state, rng, np.arange(population), EXPOSED)
    before = np.zeros(population)
    after = np.zeros(population)
    for _ in range(round(40 / step_days)):
        weight = model.infectious_weight(disease_state) * step_days
        symptomatic = disease_state.state == ORDER.index(INFECTIOUS)
        before += np.where(symptomatic, 0.0, weight)
        after += np.where(symptomatic, weight, 0.0)
        advance(model, disease_state, rng)
    expected = theta(w, 2.5 + step_days / 2, 3.0 + step_days / 2)
    # each stay is rounded up to whole steps, which the expected values include
    share = before.sum() / (before.sum() + after.sum())
    error = 5 * (before / (before + after)).std() / math.sqrt(population)
    assert share == pytest.approx(expected, abs=error + 0.002)
    assert model.infectious_days() == pytest.approx(w * 2.5 + 3.0)


def test_the_demo_disease_has_the_incubation_and_presymptomatic_share_its_comments_cite():
    definition = LIBRARY.catalog["demo_sepir"]
    parameters = definition.parameters
    assert definition.model == "sepir"
    assert parameters["latent_days"] + parameters["presymptomatic_days"] == 5.5
    assert theta(
        parameters["presymptomatic_infectiousness"],
        parameters["presymptomatic_days"],
        parameters["infectious_days"],
    ) == pytest.approx(0.45, abs=0.01)
    assert SepirModel(ONE_DAY, **parameters).mean_field_r0(2.0) > 1


# --- the well-mixed epidemic against the textbook ODE --------------------------------------------

STEP = 1 / 24
POPULATION, CASES = 20_000, 200
RUNS = 4
CONTACTS, TRANSMISSION, W = 1.5, 0.3, 1.5
B = hazard(TRANSMISSION) * CONTACTS
MEANS = {"latent_days": 2.0, "presymptomatic_days": 2.0, "infectious_days": 3.0}


def erlang_derivative(b, w, latent, presymptomatic, symptomatic, shape):
    """The SEPIR ODE with every stay split into `shape` equal stages: gamma(shape) stays.

    The state is (S, E stages, P stages, I stages, R, cumulative force from P, from I) as shares of
    the population. One stage per state (shape 1) is the textbook SEPIR system.
    """
    k = shape

    def derivative(y):
        s = y[0]
        e, p, i = (y[1 + j * k : 1 + (j + 1) * k] for j in range(3))
        force_p, force_i = b * s * w * p.sum(), b * s * i.sum()
        out = np.zeros_like(y)
        out[0] = -force_p - force_i
        inflow = force_p + force_i
        for j, (stages, days) in enumerate(((e, latent), (p, presymptomatic), (i, symptomatic))):
            rate = k / days
            out[1 + j * k : 1 + (j + 1) * k] = (
                np.concatenate([[inflow], stages[:-1] * rate]) - stages * rate
            )
            inflow = stages[-1] * rate
        out[1 + 3 * k] = inflow
        out[2 + 3 * k], out[3 + 3 * k] = force_p, force_i
        return out

    return derivative


def solve(b, w, means, shape, days, cases=CASES, population=POPULATION):
    """ODE solution as columns (S, E, P, I, R, cumulative force from P, cumulative force from I)."""
    start = np.zeros(4 + 3 * shape)
    start[0], start[1 + shape] = 1 - cases / population, cases / population
    raw = integrate(erlang_derivative(b, w, *means, shape), start, days, STEP)
    parts = [raw[:, 0]] + [
        raw[:, 1 + j * shape : 1 + (j + 1) * shape].sum(axis=1) for j in range(3)
    ]
    return np.column_stack([*parts, raw[:, 1 + 3 * shape :]])


def run_tallied(shape, seed, days=60.0, w=W, model_class=SepirModel, **parameters):
    """A well-mixed run (the driver of mean_field.py) that also tallies who infected whom.

    Returns the counts per step and the infections caused by presymptomatic and by symptomatic
    people: each step's new infections are shared between the two in proportion to the force each
    exerts, which is how the force of infection is made of its two parts.
    """
    model = model_class(
        STEP,
        duration_shape=float(shape),
        presymptomatic_infectiousness=w,
        transmission=TRANSMISSION,
        **MEANS,
        **IDENTICAL,
        **parameters,
    )
    rng = np.random.default_rng(seed)
    state = model.create_state(
        POPULATION, rng.choice(POPULATION, CASES, replace=False), rng, np.full(POPULATION, 40)
    )
    counts, by_p, by_i = [], [], []
    for _ in range(round(days / STEP)):
        counts.append(np.bincount(state.state, minlength=len(ORDER)))
        weight = model.infectious_weight(state) * state.infectiousness
        force_p = weight[state.state == ORDER.index(PRESYMPTOMATIC)].sum()
        force_i = weight[state.state == ORDER.index(INFECTIOUS)].sum()
        exposure = np.full(POPULATION, CONTACTS * (force_p + force_i) / POPULATION, np.float32)
        infected = model.draw_infections(state, exposure, rng)
        model.finish_phases(state, rng)
        model.apply_infections(state, infected, rng)
        share = force_p / (force_p + force_i) if force_p + force_i > 0 else 0.0
        by_p.append(len(infected) * share)
        by_i.append(len(infected) * (1 - share))
    return np.array(counts) / POPULATION, np.cumsum(by_p), np.cumsum(by_i)


def half_way(shares: np.ndarray) -> float:
    """Day on which half of the people have been infected, from the susceptible share."""
    return float(np.flatnonzero(shares <= 0.5)[0] * STEP)


@cache
def standard_runs(shape):
    return [run_tallied(shape, seed) for seed in range(RUNS)]


@pytest.mark.parametrize("shape", [1, 4])
def test_the_mean_course_follows_the_sepir_ode_with_the_same_gamma_stays(shape):
    ode = solve(B, W, stays(STEP, *MEANS.values()), shape, 60)
    runs = standard_runs(shape)
    mean = np.mean([counts for counts, _, _ in runs], axis=0)
    # The height of a peak varies by about 2 % between runs of this size (measured on 8 runs of
    # each shape), so the mean of 4 runs is within 4 sigma = 4 % of the ODE.
    for column, name in ((1, "exposed"), (2, "presymptomatic"), (3, "symptomatic")):
        assert mean[:, column].max() == pytest.approx(ode[:, column].max(), rel=0.04), name
    # the day the epidemic is half way varies by 0.2 to 0.25 days between runs (same 8 runs)
    halves = [half_way(counts[:, 0]) for counts, _, _ in runs]
    assert np.mean(halves) == pytest.approx(half_way(ode[:, 0]), abs=0.5)
    # the final size varies by 0.002 between runs
    ever = np.mean([1 - counts[-1, 0] for counts, _, _ in runs])
    assert ever == pytest.approx(1 - ode[-1, 0], abs=0.005)


@pytest.mark.parametrize("shape", [1, 4])
def test_the_share_of_infections_caused_before_symptoms_follows_the_force_of_infection(shape):
    ode = solve(B, W, stays(STEP, *MEANS.values()), shape, 60)
    expected = ode[-1, 5] / (ode[-1, 5] + ode[-1, 6])
    shares = [by_p[-1] / (by_p[-1] + by_i[-1]) for _, by_p, by_i in standard_runs(shape)]
    # the share varies by 0.0035 between runs (8 runs of each shape): 4 sigma of the mean of 4
    assert np.mean(shares) == pytest.approx(expected, abs=0.007)
    assert expected > theta(W, 2.0, 3.0), "a growing epidemic has more young than old infections"


@pytest.mark.parametrize("shape", [1, 3, 8])
def test_the_euler_lotka_growth_rate_is_the_eigenvalue_of_the_linearised_ode(shape):
    means = (2.0, 2.0, 3.0)
    size = 3 * shape
    rates = [shape / days for days in means for _ in range(shape)]
    jacobian = np.zeros((size, size))
    for index in range(size):
        jacobian[index, index] = -rates[index]
        if index > 0:  # a stage is fed by the one before, the first of P by the last of E
            jacobian[index, index - 1] = rates[index - 1]
    # new infections (S = 1) enter the first stage of E and come from all stages of P and I
    jacobian[0, shape : 2 * shape] += B * W
    jacobian[0, 2 * shape :] += B
    eigenvalue = max(np.linalg.eigvals(jacobian).real)
    assert growth_rate(B, W, *means, shape) == pytest.approx(eigenvalue, rel=1e-6)


def test_the_growth_rate_is_zero_at_the_threshold_and_grows_with_r0():
    means = (2.0, 2.0, 3.0)
    threshold_b = 1 / (W * means[1] + means[2])
    assert growth_rate(threshold_b * 1.0001, W, *means, 1) == pytest.approx(0.0, abs=1e-3)
    rates = [growth_rate(threshold_b * factor, W, *means, 4) for factor in (1.5, 2.0, 3.0)]
    assert rates == sorted(rates) and rates[0] > 0


def test_the_early_share_before_symptoms_is_theta_without_growth_and_larger_with_it():
    shape, means = 4, (2.0, 2.0, 3.0)

    def early_share(r):
        p, i = ((1 + r * days / shape) ** -shape for days in means[1:])
        return W * (1 - p) / (W * (1 - p) + p * (1 - i))

    assert early_share(1e-9) == pytest.approx(theta(W, means[1], means[2]), abs=1e-6)
    r = growth_rate(B, W, *means, shape)
    start = np.zeros(4 + 3 * shape)
    start[0], start[1 + shape] = 1.0, 1e-9
    ode = integrate(erlang_derivative(B, W, *means, shape), start, 25, STEP)
    # a seed of 1e-9 is still tiny at day 25: S = 1 and the growth is the pure exponential
    late_p = np.diff(ode[:, 2 + 3 * shape])[-24:].sum()
    late_i = np.diff(ode[:, 3 + 3 * shape])[-24:].sum()
    assert late_p / (late_p + late_i) == pytest.approx(early_share(r), rel=0.02)
    assert early_share(r) > theta(W, means[1], means[2])


# --- R0, the final size and what isolating the symptomatic can do --------------------------------

COARSE = 1 / 12  # a coarser step makes these runs fast; the stays below carry its half step
COARSE_RUNS = 6


def ever_infected(model_class=SepirModel, runs=COARSE_RUNS, **parameters) -> float:
    """Mean share ever infected (left the susceptible state) over well-mixed runs of 100 days."""
    parameters = {"latent_days": 2.0, "duration_shape": 4.0, **IDENTICAL, **parameters}
    results = []
    for seed in range(runs):
        _, counts = run_well_mixed(
            model_class,
            population=10_000,
            cases=100,
            days=100,
            contacts=CONTACTS,
            seed=seed,
            step_days=COARSE,
            transmission=TRANSMISSION,
            **parameters,
        )
        results.append(1 - counts[-1, 0] / 10_000)
    return float(np.mean(results))


def predicted_final_size(w, d_presymptomatic, d_symptomatic, isolated=0.0) -> float:
    """SIR final-size relation with R = R0 (1 - e + e theta), e the isolated share (Fraser 2004)."""
    _, d_p, d_i = stays(COARSE, 0.0, d_presymptomatic, d_symptomatic)
    r0_value = r0(B, w, d_p, d_i)
    return final_size(r0_value * (1 - isolated * (1 - theta(w, d_p, d_i))), 100 / 10_000)


@pytest.mark.parametrize(("isolated", "tolerance"), [(0.5, 0.013), (1.0, 0.017)])
def test_isolating_the_symptomatic_removes_only_their_part_of_r0(isolated, tolerance):
    """R_I = R0 (1 - e + e theta). The run-to-run sd of the final size is 0.002, 0.0075 and 0.010
    (16 runs each), so 4 sigma of the mean of 6 runs is the tolerance."""
    sim = ever_infected(
        IsolatingSepirModel,
        presymptomatic_days=2.0,
        infectious_days=3.0,
        presymptomatic_infectiousness=W,
        stay_home=isolated,
    )
    assert sim == pytest.approx(predicted_final_size(W, 2.0, 3.0, isolated), abs=tolerance)


def test_isolation_alone_controls_an_outbreak_only_if_theta_is_below_one_over_r0():
    """The same R0 (3.24 against 3.27 in the test above) with 17 % of the transmission before
    symptoms instead of 50 %: with perfect isolation R_I = R0 theta is 0.55 instead of 1.64, so the
    outbreak dies out. The sd of the final size is 0.0018 (16 runs): 4 sigma of 6 runs is 0.004."""
    few = {"presymptomatic_infectiousness": 0.5, "presymptomatic_days": 2.0, "infectious_days": 5.0}
    controlled = ever_infected(IsolatingSepirModel, stay_home=1.0, **few)
    assert controlled == pytest.approx(predicted_final_size(0.5, 2.0, 5.0, 1.0), abs=0.004)
    assert controlled < 0.03, "the 1 % seeds and a few generations of weak spread"
    assert r0(B, 0.5, 2.0, 5.0) * theta(0.5, 2.0, 5.0) < 1 < r0(B, W, 2.0, 3.0) * theta(W, 2.0, 3.0)


def test_without_presymptomatic_infectiousness_isolating_the_symptomatic_stops_everything():
    sim = ever_infected(
        IsolatingSepirModel,
        runs=2,
        presymptomatic_infectiousness=0.0,
        presymptomatic_days=2.0,
        infectious_days=3.0,
        stay_home=1.0,
    )
    assert sim == pytest.approx(0.01, abs=1e-12), "only the 100 seeds were ever infected"


@pytest.mark.parametrize(
    ("w", "d_presymptomatic", "d_symptomatic"), [(1.5, 2.0, 3.0), (0.5, 4.0, 4.0), (1.0, 1.0, 2.0)]
)
def test_the_final_size_follows_r0_whatever_the_split_between_the_periods(
    w, d_presymptomatic, d_symptomatic
):
    """The first two have the same R0 = b (w d_P + d_I) with theta 0.5 and 0.33."""
    sim = ever_infected(
        runs=4,
        presymptomatic_infectiousness=w,
        presymptomatic_days=d_presymptomatic,
        infectious_days=d_symptomatic,
    )
    assert sim == pytest.approx(predicted_final_size(w, d_presymptomatic, d_symptomatic), abs=0.005)


def test_a_longer_latent_period_slows_the_epidemic_but_not_its_size():
    # the sd of the final size of a free epidemic is 0.002 (16 runs): 4 sigma of 3 runs is 0.005
    sizes = {
        days: ever_infected(runs=3, latent_days=days, **MEANS_WITHOUT_LATENT) for days in (1, 8)
    }
    expected = predicted_final_size(W, 2.0, 3.0)
    assert sizes[1] == pytest.approx(expected, abs=0.005)
    assert sizes[8] == pytest.approx(expected, abs=0.005)
    slow, fast = (growth_rate(B, W, days, 2.0, 3.0, 4) for days in (8, 1))
    assert slow < fast


MEANS_WITHOUT_LATENT = {
    "presymptomatic_days": 2.0,
    "infectious_days": 3.0,
    "presymptomatic_infectiousness": W,
}
