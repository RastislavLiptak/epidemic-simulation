"""What every disease model in the registry has to satisfy, whatever it models.

The tests run for each model in DISEASE_MODELS, so a model added later is checked without
writing anything: its declaration, the transitions it may make, how long people stay in each
state and how many take each branch (against the declaration's own courses), and that its
numbers reach the browser as valid JSON. What a model is meant to be, as opposed to what it is
declared to be, is for the tests of the model itself.
"""

import json
import math
import re

import numpy as np
import pytest
from config_factory import (
    CONFIG_DIRECTORY,
    LIBRARY,
    disease_definition,
    disease_of,
    library_with,
    make_model,
)
from disease_helpers import new_state, place
from synthetic_models import ChronicModel, ForkModel, VaccinationModel

from epidemic_sandbox.app.person_info import describe_person
from epidemic_sandbox.app.schema import describe_preview, describe_run, describe_setup
from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import (
    DEAD,
    DYING,
    INFECTIOUS,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
)
from epidemic_sandbox.engine.epidemic_model import EpidemicModel

pytestmark = pytest.mark.usefixtures("synthetic_in_registry")


KEYS = sorted(DISEASE_MODELS)
# The engine-level checks also run on the test-only models, which use every mechanism, so that
# the checks themselves are exercised on branches, rates, confinement and open-ended infections.
ENGINE_MODELS = {
    **DISEASE_MODELS,
    "fork": ForkModel,
    "vaccination": VaccinationModel,
    "chronic": ChronicModel,
}
ENGINE_KEYS = sorted(ENGINE_MODELS)
COLOR = re.compile(r"#[0-9a-f]{6}")
CHECKED_STAYS_DAYS = 25.0  # longer stays do not end within the cohort's horizon with certainty


def parameters_of(model_class, rng) -> dict:
    """A random mix of default parameters and values in range."""
    given = {}
    for name, spec in model_class.PARAMETERS.items():
        if rng.random() < 0.5:
            continue
        high = min(spec.maximum, max(spec.default * 10, spec.minimum + 10 * spec.step))
        value = spec.minimum + rng.random() * (high - spec.minimum)
        if float(spec.step).is_integer():
            value = round(value)
        given[name] = float(np.clip(value, spec.minimum, spec.maximum))
    # the full-exposure distance cannot exceed the contact distance
    contact = given.get("contact_distance", model_class.PARAMETERS["contact_distance"].default)
    near = given.get("near_distance", model_class.PARAMETERS["near_distance"].default)
    if near > contact:
        given["near_distance"] = contact
    return given


def random_model(model_class, step_days, rng) -> DiseaseModelAbstract:
    """A model with random parameters (retrying the draws whose starting shares exceed 1)."""
    for _ in range(50):
        try:
            return model_class(step_days, **parameters_of(model_class, rng))
        except ValueError as error:
            assert "Starting shares" in str(error), error
    raise AssertionError("no valid random parameters")


def allowed_transitions(model_class) -> set[tuple[str, str]]:
    """Every move from one state to another that the declaration allows in a single step."""
    moves = set()
    for state, phase in model_class.PHASES.items():
        moves.add((state, phase.next_state))
        if phase.branch_state is not None:
            moves.add((state, phase.branch_state))
        if phase.fatal:
            moves.add((state, DEAD))  # without a time to death
            moves.add((state, DYING))
            moves.add((DYING, DEAD))
    for state, rate in model_class.RATES.items():
        moves.add((state, rate.next_state))
    for state in (SUSCEPTIBLE, *model_class.PROTECTION):
        moves.add((state, model_class.INFECTION_ENTRY_STATE))
        if model_class.INFECTION_BRANCH is not None:
            moves.add((state, model_class.INFECTION_BRANCH.state))
    return moves


# --- the declaration ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_a_registered_model_is_named_after_its_module(key):
    assert DISEASE_MODELS[key].__name__ == f"{key.capitalize()}Model"
    assert DISEASE_MODELS[key].__module__.endswith(f".disease_models.{key}")


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_a_model_is_declared_the_way_the_application_expects(key):
    model_class = ENGINE_MODELS[key]
    states = list(model_class.STATE_COLORS)
    assert model_class.__doc__ and model_class.__doc__.strip()
    assert states[0] == SUSCEPTIBLE and DEAD in states
    assert INFECTIOUS in states and INFECTIOUS in model_class.INFECTIOUS_STATES, (
        "the charts follow the state named infectious"
    )
    assert all(re.fullmatch(r"[a-z]+", state) for state in states)
    assert all(COLOR.fullmatch(color) for color in model_class.STATE_COLORS.values())
    assert len(set(model_class.STATE_COLORS.values())) == len(states), "colors tell states apart"
    assert model_class.INITIAL_CASE_STATE in model_class.INFECTIOUS_STATES


def brightness(color: str) -> float:
    red, green, blue = (int(color[start : start + 2], 16) for start in (1, 3, 5))
    return 0.299 * red + 0.587 * green + 0.114 * blue


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_state_colors_stand_out_from_both_themes_except_the_dead(key):
    """The charts replace a color too close to the background by the text color (charts.js)."""
    for state, color in ENGINE_MODELS[key].STATE_COLORS.items():
        if state != DEAD:
            for background in ("#ffffff", "#181c24"):
                assert abs(brightness(color) - brightness(background)) >= 40, (state, background)


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_every_parameter_is_explained_and_its_default_is_in_range(key):
    for name, spec in ENGINE_MODELS[key].PARAMETERS.items():
        assert re.fullmatch(r"[a-z][a-z_]*", name)
        assert spec.label.strip() and spec.description.strip(), name
        assert spec.minimum <= spec.default <= spec.maximum, name
        assert spec.step > 0 and spec.minimum < spec.maximum, name
        assert spec.validate(name, spec.default) == spec.default, name


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_a_model_runs_with_its_defaults_at_any_step_length(key):
    for step_days in (1.0, 1 / 24, 1 / 48):
        model = ENGINE_MODELS[key](step_days)
        disease_state, rng = new_state(model, population=300, cases=[0, 1, 2])
        for _ in range(5):
            model.advance(disease_state, np.full(300, 0.3, np.float32), rng)
        assert np.bincount(disease_state.state, minlength=len(model.STATE_COLORS)).sum() == 300
        assert disease_state.cumulative_infections >= 3


@pytest.mark.parametrize("key", KEYS)
def test_a_disease_with_comments_exists_in_the_catalog_for_every_model(key):
    name = disease_of(key)
    assert LIBRARY.catalog[name].model == key
    text = (CONFIG_DIRECTORY / "diseases" / f"{name}.toml").read_text()
    assert text.startswith("#"), "a disease file says what it is"


# --- the transitions a run makes ------------------------------------------------------------------


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_random_runs_keep_every_invariant_of_the_state_machine(key):
    model_class = ENGINE_MODELS[key]
    names = list(model_class.STATE_COLORS)
    allowed = allowed_transitions(model_class)
    rng = np.random.default_rng(sum(map(ord, key)))
    for trial in range(12):
        step_days = float(rng.choice([1.0, 0.25, 1 / 24, 1 / 48]))
        model = random_model(model_class, step_days, rng)
        population = 500
        ages = rng.integers(0, 95, population)
        cases = rng.choice(population, int(rng.integers(0, 60)), replace=False)
        disease_state, run_rng = new_state(model, population, cases, ages, seed=trial)
        dead_now = disease_state.state == model.dead_index
        for _ in range(250):
            before = disease_state.state.copy()
            exposure = (run_rng.random(population) * run_rng.choice([0.0, 0.5, 4.0])).astype(
                np.float32
            )
            model.advance(disease_state, exposure, run_rng)
            after = disease_state.state
            assert after.min() >= 0 and after.max() < len(names)
            for source, target in {
                (names[a], names[b]) for a, b in zip(before, after, strict=True) if a != b
            }:
                assert (source, target) in allowed, f"{key}: {source} -> {target} is not declared"
            assert (after[dead_now] == model.dead_index).all(), "the dead stay dead"
            dead_now = after == model.dead_index
            assert disease_state.deaths == dead_now.sum()
            assert disease_state.cumulative_infections >= disease_state.ever_infected.sum()
            for state in model_class.PHASES:
                assert (disease_state.timer[after == names.index(state)] > 0).all(), state
            if DYING in names:
                dying = after == names.index(DYING)
                assert (disease_state.timer[dying] > 0).all(), "the dying die when their time is up"
                assert (model.infectious_weight(disease_state)[dying] == 0).all(), "nor spread"
            symptomatic = model_class.SYMPTOMATIC_STATES
            if symptomatic is None:
                symptomatic = model_class.INFECTIOUS_STATES
            ill = np.isin(after, [names.index(s) for s in symptomatic])
            confined = np.isin(after, [names.index(s) for s in model_class.CONFINED_STATES])
            assert not (disease_state.limits_outings & ~ill).any()
            assert not (disease_state.stays_home & ~(ill | confined)).any()
            assert disease_state.stays_home[confined].all()
            # the admitted are in a hospital state, away from the town, spread nothing in it and
            # do not go out
            hospital = np.isin(after, [names.index(s) for s in model_class.HOSPITAL])
            admitted = model.admitted(disease_state)
            assert not (admitted & ~hospital).any()
            assert model.away_mask(disease_state)[admitted].all()
            assert not model.infectious_weight(disease_state)[admitted].any()
            assert disease_state.stays_home[admitted].all()
            weights = model.infectious_weight(disease_state)
            assert (weights >= 0).all() and not weights[
                ~np.isin(after, [names.index(s) for s in model_class.INFECTIOUS_STATES])
            ].any()
            assert not model.exposable_mask(disease_state)[dead_now].any()


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_the_dead_are_never_exposed_and_never_move(key):
    model = ENGINE_MODELS[key](1.0, transmission=1.0)
    disease_state, rng = new_state(model, population=50)
    disease_state.state[:] = model.dead_index
    for _ in range(5):
        model.advance(disease_state, np.ones(50, np.float32), rng)
    assert (disease_state.state == model.dead_index).all()
    assert disease_state.cumulative_infections == 0


# --- the engine against the declaration -----------------------------------------------------------


def cohort(model, population, horizon_days, seed):
    """Everybody infected at once, no further exposure: visits, steps per state, infectious time."""
    names = list(model.STATE_COLORS)
    disease_state, rng = new_state(model, population=population, seed=seed)
    place(model, disease_state, rng, np.arange(population), model.INFECTION_ENTRY_STATE)
    visited = np.zeros((len(names), population), np.bool_)
    steps_in = np.zeros((len(names), population))
    spread = np.zeros(population)
    for _ in range(round(horizon_days / model.step_days)):
        for index in range(len(names)):
            here = disease_state.state == index
            visited[index] |= here
            steps_in[index] += here
        spread += model.infectious_weight(disease_state) * model.step_days
        model.advance(disease_state, np.zeros(population, np.float32), rng)
    return visited, steps_in, spread, disease_state


def cumulative_before(courses):
    """For each state, the longest time any course takes to get there (finite parts only)."""
    longest = {}
    for course in courses:
        elapsed = 0.0
        for state, mean, _ in course.stays:
            longest[state] = max(longest.get(state, 0.0), elapsed)
            elapsed += mean
    return longest


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_people_visit_each_state_and_stay_in_it_as_the_declaration_says(key):
    step_days, population = 1 / 24, 30_000
    model = ENGINE_MODELS[key](step_days)
    courses = model.courses()
    assert sum(course.probability for course in courses) == pytest.approx(1.0)
    horizon = 120.0
    visited, steps_in, _, _ = cohort(model, population, horizon, seed=1)
    names = list(model.STATE_COLORS)
    before = cumulative_before(courses)
    means = {state: mean for course in courses for state, mean, _ in course.stays}
    for state, elapsed in before.items():
        if elapsed + 3 * min(means[state], 100) > horizon - 10 or math.isinf(elapsed):
            continue
        expected = sum(
            course.probability for course in courses if state in [s for s, _, _ in course.stays]
        )
        seen = visited[names.index(state)]
        tolerance = 4 * math.sqrt(max(expected * (1 - expected), 1e-4) / population) + 1e-3
        assert seen.mean() == pytest.approx(expected, abs=tolerance), f"{key}: visits to {state}"
        if math.isinf(means[state]) or means[state] > CHECKED_STAYS_DAYS or seen.sum() < 500:
            continue
        stay = steps_in[names.index(state)][seen] * step_days
        error = 5 * stay.std() / math.sqrt(len(stay))
        assert stay.mean() == pytest.approx(means[state] + step_days / 2, abs=error + 0.005), (
            f"{key}: stay in {state}"
        )


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_the_share_of_deaths_is_the_one_the_courses_predict(key):
    fatal = "fatality" in ENGINE_MODELS[key].PARAMETERS
    model = ENGINE_MODELS[key](1 / 12, **({"fatality": 0.3} if fatal else {}))
    population = 30_000
    courses = model.courses()
    expected = sum(course.probability for course in courses if course.end == DEAD)
    _, _, _, disease_state = cohort(model, population, 150.0, seed=2)
    tolerance = 4 * math.sqrt(max(expected * (1 - expected), 1e-4) / population) + 1e-3
    assert disease_state.deaths / population == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("key", ENGINE_KEYS)
def test_infectious_time_and_household_risk_match_the_simulated_courses(key):
    step_days = 1 / 48
    model = ENGINE_MODELS[key](step_days, transmission=0.3)
    courses = model.courses()
    stays = [(mean, weight) for course in courses for _, mean, weight in course.stays if weight > 0]
    if not stays or any(math.isinf(mean) or mean > CHECKED_STAYS_DAYS for mean, _ in stays):
        pytest.skip("the infectious time is not bounded by a cohort run")
    horizon = 12 + 2.5 * max(
        sum(mean for _, mean, _ in course.stays if math.isfinite(mean)) for course in courses
    )
    population = 12_000
    _, _, spread, _ = cohort(model, population, min(horizon, 150.0), seed=3)
    bias = sum(
        course.probability
        * sum(weight * step_days / 2 for _, _, weight in course.stays if weight > 0)
        for course in courses
    )
    error = 5 * spread.std() / math.sqrt(population)
    assert spread.mean() == pytest.approx(model.infectious_days() + bias, abs=error + 0.002)
    hazard = -math.log(1 - 0.3)
    risk = 1 - np.exp(-hazard * spread)
    # the rounding of stays up to whole steps adds at most `bias` days of spreading
    assert risk.mean() == pytest.approx(
        model.pair_transmission_probability(),
        abs=5 * risk.std() / math.sqrt(population) + hazard * bias,
    )


# --- the numbers the browser receives -------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_the_preview_and_the_descriptions_are_valid_json_for_the_disease(key):
    config = disease_config(key, population=300, seed=1)
    preview = describe_preview(config, LIBRARY.town)
    for description in (
        describe_setup(LIBRARY),
        describe_run(config, [config.world["seed"]]),
        preview,
    ):
        json.dumps(description, allow_nan=False)
    risk = preview["diseases"][disease_of(key)]["household_risk"]
    assert 0 <= risk <= 1


def disease_config(key, population, seed, steps_per_day=48):
    return library_with(steps_per_day).parse(
        {
            "world": {"population": population, "area_km2": 0.25, "seed": seed, "runs": 1},
            "diseases": {disease_of(key): {"initial_infected": 10}},
        }
    )


# --- inside the town -----------------------------------------------------------------------------


@pytest.mark.parametrize("key", KEYS)
def test_the_disease_runs_in_a_town_and_every_person_can_be_described(key):
    name = disease_of(key)
    model = EpidemicModel(
        disease_config(key, population=300, seed=3, steps_per_day=24), LIBRARY.town
    )
    for _ in range(24 * 6):
        model.step()
    assert sum(model.state_counts()[name]) == model.population
    for person in range(0, 300, 7):
        health = describe_person(model, person)["health"][name]
        json.dumps(health, allow_nan=False)
        assert health["infectiousness"] >= 0


@pytest.mark.parametrize("key", KEYS)
def test_the_model_works_together_with_another_disease_and_an_interaction(key):
    state = next(s for s in DISEASE_MODELS[key].STATE_COLORS if s not in (SUSCEPTIBLE, DEAD))
    deadly = {"fatality": 0.5, "transmission": 0.6}
    diseases = {
        "a": disease_definition(key, 5, **deadly),
        "b": disease_definition("plain", 5, **deadly),
    }
    interactions = [
        {
            "source": "a",
            "states": [state],
            "target": "b",
            "effect": "susceptibility",
            "factor": 0.5,
        },
        {
            "source": "b",
            "states": ["recovered"],
            "target": "a",
            "effect": "infectiousness",
            "factor": 0.5,
        },
    ]
    try:
        model = make_model(diseases, interactions, population=300, arena_size=500)
    except ValueError:  # the model has no fatality parameter
        diseases["a"] = disease_definition(key, 5, transmission=0.6)
        model = make_model(diseases, interactions, population=300, arena_size=500)
    dead_before = np.zeros(model.population, np.bool_)
    for _ in range(24 * 12):
        model.step()
        dead = model._dead()
        assert dead[dead_before].all(), "the dead stay dead"
        dead_before = dead
        for name, disease in model.diseases.items():
            disease_state = model.disease_states[name]
            assert (disease_state.state[dead] == disease.dead_index).all(), "dead for every disease"
            assert not disease_state.stays_home[dead].any()
    victims = sum(model.disease_states[name].deaths for name in model.diseases)
    assert victims == dead_before.sum(), "every death is counted once, under the disease"
    assert sum(model.state_counts()["a"]) <= model.population
