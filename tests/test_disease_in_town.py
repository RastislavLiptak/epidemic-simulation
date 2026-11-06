"""How the mechanisms of the disease models show in a town: who spreads, who stays at home."""

import numpy as np
import pytest
from config_factory import disease_definition
from synthetic_models import (
    HOSPITAL,
    MILD,
    VACCINATED,
    ForkModel,
    PlainModel,
    VaccinationModel,
)
from test_epidemic_model import arranged_model, put_into_state, run

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import EXPOSED, INFECTIOUS
from epidemic_sandbox.person_info import describe_person


@pytest.fixture(autouse=True)
def fork_in_registry(monkeypatch):
    monkeypatch.setitem(DISEASE_MODELS, "fork", ForkModel)
    monkeypatch.setitem(DISEASE_MODELS, "vaccination", VaccinationModel)
    monkeypatch.setitem(DISEASE_MODELS, "plain", PlainModel)


def fork_model(**parameters):
    diseases = {"a": disease_definition("fork", 0, **parameters)}
    return arranged_model(diseases, {})


def test_people_in_different_states_spread_with_the_infectiousness_of_their_state():
    model = fork_model(mild_infectiousness=0.25)
    put_into_state(model, 0, "a", INFECTIOUS)
    put_into_state(model, 1, "a", MILD)
    put_into_state(model, 2, "a", EXPOSED)
    put_into_state(model, 3, "a", HOSPITAL)
    assert model._infectiousness("a")[:5].tolist() == [1.0, 0.25, 0.0, 0.0, 0.0]


def test_the_trait_of_a_person_and_the_weight_of_their_state_multiply():
    model = fork_model(mild_infectiousness=0.5)
    put_into_state(model, 1, "a", MILD)
    model.disease_states["a"].infectiousness[1] = 3.0
    assert model._infectiousness("a")[1] == pytest.approx(1.5)


def test_only_the_symptomatic_stay_at_home_and_the_confined_always_do():
    model = fork_model(stay_home=1.0)
    disease_state = model.disease_states["a"]
    for person, state in ((0, INFECTIOUS), (1, MILD), (2, EXPOSED), (3, HOSPITAL)):
        model.diseases["a"]._enter(
            disease_state, np.array([person]), state, np.random.default_rng(person)
        )
    assert model._goes_home()[:4].tolist() == [True, False, False, True]


def test_the_confined_are_at_home_even_when_their_day_takes_them_out():
    model = fork_model(stay_home=0.0)
    model.disease_states["a"].timer[:] = 1e6
    model.diseases["a"]._enter(
        model.disease_states["a"], np.arange(model.population), HOSPITAL, np.random.default_rng(0)
    )
    run(model, 24)
    away = np.hypot(model.x - model.town.home_x, model.y - model.town.home_y)
    assert away.max() < 1e-6


def test_the_hover_card_shows_the_infectiousness_of_the_state_and_the_time_left():
    model = fork_model(mild_infectiousness=0.5)
    put_into_state(model, 1, "a", MILD)
    health = describe_person(model, 1)["health"]["a"]
    assert health["state"] == MILD and health["infectious"]
    assert health["infectiousness"] == pytest.approx(0.5)
    assert health["days_left"] == pytest.approx(1e6)
    assert health["model"] == "FORK"


def test_the_hover_card_shows_no_days_left_for_a_stay_for_life():
    model = arranged_model({"a": disease_definition("plain", 0)}, {})
    put_into_state(model, 1, "a", "recovered")
    put_into_state(model, 2, "a", "infectious")
    assert describe_person(model, 1)["health"]["a"]["days_left"] is None
    assert describe_person(model, 2)["health"]["a"]["days_left"] is not None


def test_the_hover_card_shows_the_susceptibility_that_includes_the_protection_of_the_state():
    diseases = {
        "a": disease_definition(
            "vaccination", 0, vaccine_protection=0.75, vaccination_rate=0.0, susceptibility_cv=0.0
        )
    }
    model = arranged_model(diseases, {})
    put_into_state(model, 1, "a", VACCINATED)
    assert describe_person(model, 1)["health"]["a"]["susceptibility"] == pytest.approx(0.25)
    assert describe_person(model, 2)["health"]["a"]["susceptibility"] == pytest.approx(1.0)
