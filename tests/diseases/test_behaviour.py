"""How the ill behave and who is in hospital, for every model: the chances of each state, the age
effect, one draw per illness, and the checks of the declaration."""

import numpy as np
import pytest
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, new_state, place
from synthetic_models import PlainModel

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)


def test_the_stay_at_home_age_effect_scales_the_chance_and_is_capped():
    effects = {"stay_home": [[0, 16, 1.806]]}
    model = PlainModel(ONE_DAY, stay_home=0.31, age_effects=effects)
    ages = np.repeat([8, 40], 20_000)
    disease_state, _ = new_state(model, population=40_000, cases=np.arange(40_000), age=ages)
    children, adults = disease_state.stays_home[:20_000], disease_state.stays_home[20_000:]
    assert children.mean() == pytest.approx(0.56, abs=0.015)
    assert adults.mean() == pytest.approx(0.31, abs=0.015)
    capped = PlainModel(ONE_DAY, stay_home=0.8, age_effects=effects)
    disease_state, _ = new_state(capped, population=1000, cases=np.arange(1000), age=8)
    assert disease_state.stays_home.all()


def test_a_new_illness_brings_a_new_decision():
    model = PlainModel(
        ONE_DAY, **SURE, **UNIFORM, stay_home=0.5, infectious_days=1.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model, population=20_000, cases=np.arange(20_000))
    first = disease_state.stays_home.copy()
    for _ in range(3):
        advance(model, disease_state, rng)
    disease_state.state[:] = 0  # susceptible again
    advance(model, disease_state, rng, exposure=1.0)
    second = disease_state.stays_home
    assert second.mean() == pytest.approx(0.5, abs=0.015)
    assert np.mean(first == second) == pytest.approx(0.5, abs=0.015), "independent decisions"


def test_first_cases_are_seeded_with_the_traits_of_a_new_infection():
    model = PlainModel(ONE_DAY, stay_home=0.5, infectiousness_dispersion=0.5)
    disease_state, rng = new_state(model, population=10_000)
    model.seed_infections(disease_state, np.arange(5000), rng)
    assert disease_state.cumulative_infections == 5000
    assert disease_state.ever_infected[:5000].all() and not disease_state.ever_infected[5000:].any()
    assert disease_state.stays_home[:5000].mean() == pytest.approx(0.5, abs=0.03)


def test_a_behaviour_belongs_to_a_symptomatic_state_and_a_hospital_to_an_ill_one():
    states = {SUSCEPTIBLE: "#34a853", INFECTIOUS: "#ea4335", RECOVERED: "#9aa0a6", DEAD: "#202124"}
    common = {
        "PARAMETERS": dict(COMMON_PARAMETERS),
        "STATE_COLORS": states,
        "INFECTIOUS_STATES": frozenset({INFECTIOUS}),
        "INFECTION_ENTRY_STATE": INFECTIOUS,
        "INITIAL_CASE_STATE": INFECTIOUS,
        "PHASES": {INFECTIOUS: Phase(RECOVERED, "stay_home")},
    }
    behaviour = {RECOVERED: ("stay_home", "limit_outings")}
    with pytest.raises(TypeError, match="only a symptomatic state has a behaviour"):
        type("Bad", (DiseaseModelAbstract,), {**common, "BEHAVIOUR": behaviour})
    with pytest.raises(TypeError, match="hospital state"):
        type("Bad", (DiseaseModelAbstract,), {**common, "HOSPITAL": {RECOVERED: "stay_home"}})


@pytest.mark.parametrize("key", sorted(DISEASE_MODELS))
def test_every_model_keeps_the_decisions_of_one_illness_consistent(key):
    """Through a whole illness with random chances per state, a person stays at home in a state
    exactly when their draw is below the state's chance, so the shares are right in every state
    and nobody stays at home in a state with a lower chance than one in which they went out."""
    model_class = DISEASE_MODELS[key]
    rng = np.random.default_rng(3)
    parameters = {}
    for home, outings in model_class.BEHAVIOUR.values():
        parameters[home], parameters[outings] = rng.random(2)
    parameters["stay_home"], parameters["limit_outings"] = rng.random(2)
    model = model_class(ONE_DAY, **parameters)
    population = 5000
    disease_state, run_rng = new_state(model, population=population)
    place(model, disease_state, run_rng, np.arange(population), model.INFECTION_ENTRY_STATE)
    names = list(model.STATE_COLORS)
    for _ in range(60):
        symptomatic = np.isin(disease_state.state, model._symptomatic_indices)
        chance = model._home_chance[disease_state.state] * disease_state.home_scale
        expected = (disease_state.home_draw < chance) & symptomatic
        expected |= np.isin(disease_state.state, [names.index(s) for s in model.CONFINED_STATES])
        expected |= model.admitted(disease_state)
        assert (disease_state.stays_home == expected).all()
        advance(model, disease_state, run_rng)
