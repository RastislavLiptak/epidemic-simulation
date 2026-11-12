"""The air of a room: everybody in an indoor venue is exposed to the infectious people in it,
whatever the distance (`room_exposure`), and nobody outdoors or in another venue."""

import numpy as np
import pytest
from config_factory import disease_definition
from model_helpers import arranged_model, make_infectious, state_of

pytestmark = pytest.mark.usefixtures("synthetic_in_registry")

# the neighbour of the infectious person stands 5 m away: beyond the contact distance of 2 m, so
# only the air of the room can carry the disease
FAR = {1: (5, 10)}


def room_model(room_exposure, seed=0, **parameters):
    diseases = {
        "a": disease_definition(
            "plain",
            0,
            transmission=0.5,
            infectious_days=60,
            susceptibility_cv=0.0,
            near_distance=2.0,
            room_exposure=room_exposure,
            **parameters,
        )
    }
    model = arranged_model(diseases, FAR, seed=seed)
    make_infectious(model, 0, "a")
    model.venue = np.zeros(model.population, np.int64)  # one room, as after a step
    return model


def test_everybody_in_the_venue_is_exposed_to_the_infectious_whatever_the_distance():
    model = room_model(0.4)
    model.disease_states["a"].infectiousness[0] = 2.0
    weights = model._infectiousness("a")
    room = model._room_exposure(model.diseases["a"], weights)
    assert room[0] == 0.0, "nobody is exposed to themselves"
    assert room[1:] == pytest.approx(0.4 * 2.0), "the others, near or far"


def test_a_disease_without_the_route_has_none_of_it():
    model = room_model(0.0)
    assert model._room_exposure(model.diseases["a"], model._infectiousness("a")) == 0.0


def test_people_in_other_venues_and_on_the_way_are_not_exposed():
    model = room_model(1.0)
    venue = np.zeros(model.population, np.int64)
    venue[10:20] = 1
    venue[20:25] = -1
    model.venue = venue
    room = model._room_exposure(model.diseases["a"], model._infectiousness("a"))
    assert (room[10:25] == 0.0).all()
    assert (room[1:10] > 0).all()


def test_the_air_of_an_outdoor_venue_is_not_shared():
    model = room_model(1.0)
    model._indoor_venue[0] = False
    room = model._room_exposure(model.diseases["a"], model._infectiousness("a"))
    assert (room == 0.0).all()


def test_the_exposure_adds_the_infectiousness_of_everybody_in_the_venue():
    model = room_model(0.25)
    for person in (2, 3, 4):
        make_infectious(model, person, "a")
    model.disease_states["a"].infectiousness[3] = 0.5
    weights = model._infectiousness("a")
    room = model._room_exposure(model.diseases["a"], weights)
    assert room[10] == pytest.approx(0.25 * (1 + 1 + 0.5 + 1))
    assert room[2] == pytest.approx(0.25 * (1 + 0.5 + 1)), "not their own"


def catch_rate(room_exposure, repetitions=1500):
    caught = 0
    for seed in range(repetitions):
        model = room_model(room_exposure, seed=seed)
        model.step()
        caught += state_of(model, 1, "a") == "infectious"
    return caught / repetitions


def test_the_chance_of_catching_the_disease_in_the_air_of_a_room_follows_the_definition():
    assert catch_rate(0.0) == 0.0, "5 m is out of the close range"
    # one step of 1/24 of a day with the hazard -ln(1 - 0.5) per day times the exposure
    assert catch_rate(1.0) == pytest.approx(1 - 0.5 ** (1 / 24), abs=0.012)
    assert catch_rate(0.3) == pytest.approx(1 - 0.5 ** (0.3 / 24), abs=0.01)


def test_somebody_asleep_in_bed_exposes_others_less():
    model = room_model(0.4, asleep_emission=0.2)
    model._at_home[:] = True
    assert model.town_parameters.dwelling.asleep(model.hour), "step 0 is midnight"
    assert model._infectiousness("a")[0] == pytest.approx(0.2)
    model.steps = model.world["steps_per_day"] // 2  # noon
    assert not model.town_parameters.dwelling.asleep(model.hour)
    assert model._infectiousness("a")[0] == pytest.approx(1.0)


def test_only_those_at_home_are_asleep_at_night():
    model = room_model(0.4, asleep_emission=0.2)
    model._at_home[:] = False
    assert model._infectiousness("a")[0] == pytest.approx(1.0), "out at night, awake"
