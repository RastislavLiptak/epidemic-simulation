"""The SI model against its definition: logistic growth of a lifelong infection that nobody leaves.

dS/dt = -beta S I / N, dI/dt = beta S I / N with beta = -ln(1 - transmission) x contacts
(Chasnov, Mathematical Biology, section 4.1; Hethcote 2000). The solution is logistic, i(t) =
1 / (1 + ((1 - i0) / i0) exp(-beta t)), so everybody ends up infected.
"""

import json
import math
from statistics import NormalDist

import numpy as np
import pytest
from config_factory import LIBRARY, disease_definition, experiment_config, make_model
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, new_state, state_names
from mean_field import integrate, run_well_mixed

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    INFECTIOUS,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.si import SiModel
from epidemic_sandbox.schema import describe_preview
from epidemic_sandbox.town import mean_contacts

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}
STEP_DAYS = 1 / 24
POPULATION, CASES, RUNS = 4000, 100, 8


# --- the declaration ----------------------------------------------------------------------------


def test_the_model_is_registered_under_its_key():
    assert DISEASE_MODELS["si"] is SiModel


def test_there_are_two_states_that_matter_and_no_way_out_of_the_infection():
    assert list(SiModel.STATE_COLORS) == [SUSCEPTIBLE, INFECTIOUS, DEAD]
    assert INFECTIOUS in SiModel.INFECTIOUS_STATES and len(SiModel.INFECTIOUS_STATES) == 1
    assert SiModel.INFECTION_ENTRY_STATE == SiModel.INITIAL_CASE_STATE == INFECTIOUS
    assert SiModel.course_states() == {INFECTIOUS}
    for mechanism in (SiModel.PHASES, SiModel.RATES, SiModel.PROTECTION, SiModel.STARTING_SHARES):
        assert mechanism == {}
    assert not SiModel.CONFINED_STATES and not SiModel.STATE_INFECTIOUSNESS


def test_the_model_has_the_common_parameters_except_fatality_and_none_of_its_own():
    assert SiModel.PARAMETERS.keys() == COMMON_PARAMETERS.keys() - {"fatality"}


def test_nobody_dies_of_it_so_there_is_no_fatality_to_set_or_to_scale_by_age():
    with pytest.raises(ValueError, match="Unknown parameters"):
        SiModel(ONE_DAY, fatality=0.1)
    with pytest.raises(ValueError, match="fatality"):
        SiModel(ONE_DAY, age_effects={"fatality": [[0, 120, 2.0]]})
    with pytest.raises(ValueError, match="branching"):
        SiModel(ONE_DAY, age_effects={"branching": [[0, 120, 2.0]]})


def test_the_descriptions_say_what_the_behaviour_parameters_mean_for_a_lifelong_infection():
    for name in ("stay_home", "limit_outings"):
        spec = SiModel.PARAMETERS[name]
        assert spec.default == 0.0
        assert "rest of their lives" in spec.description, name
    assert "no effect" in SiModel.PARAMETERS["duration_shape"].description.lower()


# --- nobody leaves, nothing but infection happens -----------------------------------------------


def test_an_infected_person_stays_infectious_however_long_it_takes():
    model = SiModel(ONE_DAY)
    disease_state, rng = new_state(model, population=200, cases=np.arange(200))
    for _ in range(3000):
        advance(model, disease_state, rng)
    assert (disease_state.state == 1).all()
    assert not model.is_quiescent(disease_state)


def test_a_random_run_only_moves_people_from_susceptible_to_infectious():
    model = SiModel(1 / 12)
    population = 1000
    disease_state, rng = new_state(model, population, cases=np.arange(20))
    infectious, susceptible = 20, population - 20
    for _ in range(300):
        exposure = (rng.random(population) * rng.choice([0.0, 1.0, 5.0])).astype(np.float32)
        before = disease_state.state.copy()
        model.advance(disease_state, exposure, rng)
        after = disease_state.state
        assert not (after < before).any(), "nobody goes back"
        assert (after[before == 1] == 1).all()
        now_infectious = int((after == 1).sum())
        now_susceptible = int((after == 0).sum())
        assert now_infectious >= infectious and now_susceptible <= susceptible
        assert now_infectious + now_susceptible == population, "nobody is dead"
        infectious, susceptible = now_infectious, now_susceptible
        assert (
            disease_state.cumulative_infections == infectious == disease_state.ever_infected.sum()
        )
    assert infectious > 20 and disease_state.deaths == 0


def test_the_infected_cannot_be_infected_again_so_infections_are_counted_once():
    model = SiModel(ONE_DAY, **SURE)
    disease_state, rng = new_state(model, population=100, cases=np.arange(30))
    assert not model.exposable_mask(disease_state)[:30].any()
    assert model.exposable_mask(disease_state)[30:].all()
    advance(model, disease_state, rng, exposure=10)
    assert disease_state.cumulative_infections == 100
    advance(model, disease_state, rng, exposure=10)
    assert disease_state.cumulative_infections == 100


def test_without_transmission_the_first_cases_stay_the_only_ones():
    model = SiModel(ONE_DAY, transmission=0.0)
    disease_state, rng = new_state(model, population=500, cases=[0, 1, 2])
    for _ in range(20):
        advance(model, disease_state, rng, exposure=50)
    assert disease_state.cumulative_infections == 3


def test_the_first_cases_are_infectious_at_once():
    model = SiModel(ONE_DAY)
    disease_state, _ = new_state(model, population=10, cases=[2, 5])
    assert state_names(model, disease_state).count(INFECTIOUS) == 2
    assert model.infectious_weight(disease_state).tolist() == [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]


# --- the logistic curve ---------------------------------------------------------------------------


def logistic_time(share: float, first_share: float, beta: float) -> float:
    """Days until `share` of the people are infected, starting from `first_share`."""
    return math.log(share * (1 - first_share) / ((1 - share) * first_share)) / beta


def crossing_times(counts: np.ndarray, shares: tuple) -> list[float]:
    infectious = counts[:, 1] / counts[0].sum()
    return [float(np.argmax(infectious >= share)) * STEP_DAYS for share in shares]


@pytest.mark.parametrize(("transmission", "contacts"), [(0.1, 3.0), (0.3, 1.0)])
def test_a_well_mixed_town_follows_the_logistic_curve_and_everybody_is_infected(
    transmission, contacts
):
    beta = -math.log(1 - transmission) * contacts
    shares = (0.1, 0.5, 0.9)
    times = []
    for seed in range(RUNS):
        _, counts = run_well_mixed(
            SiModel,
            population=POPULATION,
            cases=CASES,
            days=50,
            contacts=contacts,
            seed=seed,
            transmission=transmission,
            **IDENTICAL,
        )
        assert counts[-1, 1] == POPULATION, "everybody is infected in the end"
        assert (np.diff(counts[:, 0]) <= 0).all() and (np.diff(counts[:, 1]) >= 0).all()
        assert counts[:, 2].sum() == 0
        times.append(crossing_times(counts, shares))
    # Early on the cases multiply as a pure birth process: the time shift of a run has a standard
    # deviation 1 / (beta sqrt(cases)). One step covers the rounding of the crossing and the
    # first-order scheme of the model.
    tolerance = 4 / (beta * math.sqrt(CASES)) / math.sqrt(RUNS) + STEP_DAYS
    for share, observed in zip(shares, np.mean(times, axis=0), strict=True):
        expected = logistic_time(share, CASES / POPULATION, beta)
        assert observed == pytest.approx(expected, abs=tolerance), share


def test_the_growth_rate_depends_on_the_hazard_and_the_contacts_only_through_their_product():
    """Half the hazard with twice the contacts is the same epidemic."""
    contacts, transmission = 2.0, 0.2
    halved = 1 - (1 - transmission) ** 0.5
    slow = run_well_mixed(
        SiModel, POPULATION, CASES, 30, contacts, seed=1, transmission=transmission, **IDENTICAL
    )[1]
    fast = run_well_mixed(
        SiModel, POPULATION, CASES, 30, 2 * contacts, seed=2, transmission=halved, **IDENTICAL
    )[1]
    beta = -math.log(1 - transmission) * contacts
    # two runs, each with a time shift of standard deviation 1 / (beta sqrt(cases))
    tolerance = 4 * math.sqrt(2) / (beta * math.sqrt(CASES)) + STEP_DAYS
    assert crossing_times(slow, (0.5,))[0] == pytest.approx(
        crossing_times(fast, (0.5,))[0], abs=tolerance
    )
    assert crossing_times(slow, (0.5,))[0] == pytest.approx(
        logistic_time(0.5, CASES / POPULATION, beta), abs=tolerance
    )


def lognormal_susceptibilities(cv: float, count: int) -> np.ndarray:
    """Equally likely susceptibilities of the model's lognormal distribution (mean 1)."""
    sigma = math.sqrt(math.log1p(cv**2))
    quantiles = NormalDist().inv_cdf
    values = np.array([math.exp(sigma * quantiles((k + 0.5) / count)) for k in range(count)])
    return values / values.mean()


def test_people_who_differ_in_susceptibility_slow_the_end_of_the_epidemic_as_the_theory_says():
    """The most susceptible are infected first: i = i0 + (1 - i0)(1 - E[exp(-a L)]), L' = beta i."""
    transmission, contacts, cv = 0.1, 3.0, 1.5
    beta = -math.log(1 - transmission) * contacts
    first_share = CASES / POPULATION
    susceptibility = lognormal_susceptibilities(cv, 4000)

    def infected_share(load):
        return first_share + (1 - first_share) * (1 - np.exp(-susceptibility * load).mean())

    load = integrate(lambda y: np.array([beta * infected_share(y[0])]), np.zeros(1), 100)[:, 0]
    theory = np.array([infected_share(value) for value in load])
    shares = (0.5, 0.9)
    expected = [float(np.argmax(theory >= share)) * STEP_DAYS for share in shares]
    times = []
    for seed in range(RUNS):
        _, counts = run_well_mixed(
            SiModel,
            population=POPULATION,
            cases=CASES,
            days=100,
            contacts=contacts,
            seed=seed,
            transmission=transmission,
            susceptibility_cv=cv,
            infectiousness_dispersion=0.0,
        )
        times.append(crossing_times(counts, shares))
    observed, spread = np.mean(times, axis=0), np.std(times, axis=0, ddof=1)
    for index, share in enumerate(shares):
        tolerance = 4 * spread[index] / math.sqrt(RUNS) + STEP_DAYS
        assert observed[index] == pytest.approx(expected[index], abs=tolerance), share
    uniform = logistic_time(0.9, first_share, beta)
    assert observed[1] > uniform + 10, "the last people to catch it are hard to infect"


def test_superspreaders_change_the_noise_of_the_epidemic_but_not_its_average_speed():
    transmission, contacts, dispersion = 0.1, 3.0, 0.5
    beta = -math.log(1 - transmission) * contacts
    times = []
    for seed in range(RUNS):
        _, counts = run_well_mixed(
            SiModel,
            population=POPULATION,
            cases=CASES,
            days=50,
            contacts=contacts,
            seed=seed,
            transmission=transmission,
            susceptibility_cv=0.0,
            infectiousness_dispersion=dispersion,
        )
        times.append(crossing_times(counts, (0.5,))[0])
    # the starting force of the first cases varies by 1 / sqrt(dispersion x cases), which shifts the
    # curve by 1 / (beta sqrt(dispersion x cases)), on top of the birth noise 1 / (beta sqrt(cases))
    spread = math.sqrt(1 + 1 / dispersion) / (beta * math.sqrt(CASES))
    expected = logistic_time(0.5, CASES / POPULATION, beta)
    tolerance = 4 * spread / math.sqrt(RUNS) + STEP_DAYS
    assert np.mean(times) == pytest.approx(expected, abs=tolerance)


def test_the_length_regularity_does_nothing_because_there_is_no_timed_phase():
    runs = [
        run_well_mixed(SiModel, 500, 10, 10, 3.0, seed=4, transmission=0.2, duration_shape=shape)[1]
        for shape in (1.0, 25.0)
    ]
    assert np.array_equal(runs[0], runs[1])


# --- who is susceptible ----------------------------------------------------------------------


def test_age_effect_on_susceptibility_scales_the_chance_to_catch_it():
    effects = {"susceptibility": [[0, 15, 0.5], [65, 120, 2.0]]}
    model = SiModel(1 / 24, transmission=0.5, age_effects=effects, **UNIFORM)
    ages = np.repeat([10, 40, 70], 100_000)
    disease_state, rng = new_state(model, population=len(ages), age=ages)
    advance(model, disease_state, rng, exposure=24.0)
    caught = (disease_state.state == 1).reshape(3, -1).mean(axis=1)
    for share, multiplier in zip(caught, (0.5, 1.0, 2.0), strict=True):
        expected = 1 - 0.5**multiplier
        assert share == pytest.approx(expected, abs=4 * math.sqrt(expected * (1 - expected) / 1e5))


# --- the numbers shown before a run -------------------------------------------------------------


def test_a_lifelong_infection_has_no_infectious_period_and_no_bounded_reproduction_number():
    for transmission in (0.0, 0.1, 1.0):
        model = SiModel(ONE_DAY, transmission=transmission)
        assert math.isinf(model.infectious_days())
        assert model.mean_field_r0(2.0) is None


def test_the_household_risk_is_certain_for_a_infection_that_lasts_and_zero_without_transmission():
    assert SiModel(ONE_DAY, transmission=0.01).pair_transmission_probability() == 1.0
    assert SiModel(ONE_DAY, **SURE).pair_transmission_probability() == 1.0
    assert SiModel(ONE_DAY, transmission=0.0).pair_transmission_probability() == 0.0


def test_the_course_of_an_infection_is_one_stay_without_end_in_the_infectious_state():
    (course,) = SiModel(ONE_DAY).courses()
    assert course.probability == 1.0 and course.end == INFECTIOUS
    assert course.stays == ((INFECTIOUS, math.inf, 1.0),)


@pytest.mark.parametrize("transmission", [0.0, 0.1, 1.0])
def test_the_preview_of_a_lifelong_infection_is_valid_json(transmission):
    config = experiment_config(
        {"a": disease_definition("si", 3, transmission=transmission)}, population=300
    )
    preview = describe_preview(config, LIBRARY.town)
    json.dumps(preview, allow_nan=False)
    estimate = preview["diseases"]["a"]
    assert estimate["mean_field_r0"] is None
    assert estimate["household_risk"] == (0.0 if transmission == 0 else 1.0)


# --- the days of the infected --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["stay_home", "limit_outings"])
def test_an_infected_person_keeps_the_decision_to_stay_in_for_life(name):
    share = 0.4
    model = SiModel(ONE_DAY, **{name: share})
    disease_state, rng = new_state(model, population=20_000, cases=np.arange(10_000))
    flag = disease_state.stays_home if name == "stay_home" else disease_state.limits_outings
    decided = flag.copy()
    assert decided[:10_000].mean() == pytest.approx(share, abs=4 * math.sqrt(0.24 / 10_000))
    assert not decided[10_000:].any()
    for _ in range(2000):
        advance(model, disease_state, rng)
    assert np.array_equal(flag, decided)


def test_somebody_who_catches_it_later_decides_when_infected():
    model = SiModel(ONE_DAY, stay_home=1.0, **SURE)
    disease_state, rng = new_state(model, population=10)
    assert not disease_state.stays_home.any()
    advance(model, disease_state, rng, exposure=1)
    assert disease_state.stays_home.all()


def spread_in_town(seed, days=40, **parameters):
    """Number of infected after `days` in a small town, and the households of the first cases."""
    model = make_model(
        {"a": disease_definition("si", 10, transmission=0.1, **parameters)},
        population=600,
        arena_size=894,
        seed=seed,
    )
    first = np.flatnonzero(model.disease_states["a"].state == 1)
    steps = model.world["steps_per_day"]
    counts = [model.state_counts()["a"][1]]
    for _ in range(days):
        for _ in range(steps):
            model.step()
        counts.append(model.state_counts()["a"][1])
    return np.array(counts), model, first


@pytest.mark.parametrize("seed", [1, 2])
def test_the_town_epidemic_is_slower_than_the_well_mixed_one_with_the_same_contacts(seed):
    """Most contacts are housemates, who are all infected soon and then add nothing."""
    counts, model, _ = spread_in_town(seed)
    contacts = mean_contacts(model.world, LIBRARY.town, model.diseases["a"].contact_distance)
    beta = -math.log(1 - 0.1) * contacts
    growth = math.log(counts[30] / counts[10]) / 20
    assert counts[40] > 10 * counts[0], "it takes off"
    assert 0.2 * beta < growth < 0.8 * beta


def test_isolated_carriers_only_walk_home_and_spread_it_where_they_live():
    """Somebody infected on a visit walks home and never goes out again; the first cases and
    their housemates are home all the time, so the epidemic stays small."""
    model = make_model(
        {"a": disease_definition("si", 10, transmission=0.1, stay_home=1.0)},
        population=600,
        arena_size=894,
        seed=1,
    )
    disease_state, town = model.disease_states["a"], model.town
    start = np.flatnonzero(disease_state.state == 1)
    for _ in range(25 * model.world["steps_per_day"]):
        away_before = np.hypot(model.x - town.home_x, model.y - town.home_y)
        carriers = disease_state.state == 1
        model.step()
        away = np.hypot(model.x - town.home_x, model.y - town.home_y)
        assert (away[carriers] <= away_before[carriers] + 1e-6).all(), "a carrier only goes home"
    assert away[start].max() < 1e-6
    free, _, _ = spread_in_town(1)
    assert (disease_state.state == 1).sum() < free[25] / 2
