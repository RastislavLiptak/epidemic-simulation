"""Trips take the time of the travel survey, and passengers of public transport meet in vehicles."""

import numpy as np
from config_factory import disease_definition, make_model
from model_helpers import run

from epidemic_sandbox.world.town import TRANSIT

DISEASE = {"a": disease_definition("sirs", 0)}


def town_model(population=3000, seed=2, steps_per_day=48):
    return make_model(
        DISEASE, population=population, area_km2=4.0, seed=seed, steps_per_day=steps_per_day
    )


def test_a_timed_trip_arrives_after_its_time_whatever_the_distance():
    model = town_model()
    run(model, 1)
    person = int(np.flatnonzero(model.town.guardian < 0)[5])
    model._target_x[person] = model.x[person] + 1000.0
    model._target_y[person] = model.y[person]
    model._departure_time[person] = model.steps * model.step_hours
    model._set_off(np.array([person]), np.array([1.0]))  # one hour for a kilometre
    assert model._speed[person] == 1000.0 and model._trip_hours[person] == 1.0
    model._set_off(np.array([person]), np.array([np.nan]))
    assert model._speed[person] == model.town.speed_kmh[person] * 1000
    assert np.isnan(model._trip_hours[person])


def test_passengers_ride_in_vehicles_in_the_middle_of_their_trip():
    model = town_model(population=4000)
    vehicles = model._vehicles
    seen = 0
    for _ in range(24 * 2):  # until the morning trips
        model.step()
        riding = np.flatnonzero(model.venue >= vehicles.first_venue)
        if not len(riding):
            continue
        assert (model.town.travel_mode[riding] == TRANSIT).all()
        number = model.venue[riding] - vehicles.first_venue
        assert np.bincount(number).max() <= vehicles.load
        now = model.steps * model.step_hours
        into = (now - model._departure_time[riding]) / model._trip_hours[riding]
        start = (1 - vehicles.share) / 2
        assert ((into >= start - 1e-9) & (into <= start + vehicles.share + 1e-9)).all()
        seen += len(riding)
    assert seen > 50


def test_the_seats_of_a_vehicle_are_on_its_floor():
    model = town_model(population=4000)
    vehicles = model._vehicles
    run(model, 18)
    riders = np.flatnonzero(vehicles.number >= 0)
    assert len(riders)
    for number in np.unique(vehicles.number[riders])[:20]:
        seats = riders[vehicles.number[riders] == number]
        middle_x, middle_y = vehicles.x[seats].mean(), vehicles.y[seats].mean()
        spread = np.hypot(vehicles.x[seats] - middle_x, vehicles.y[seats] - middle_y)
        assert spread.max() <= 2 * vehicles.radius + 1e-6


def test_nobody_under_the_driving_age_drives():
    model = town_model(population=6000)
    town = model.town
    from epidemic_sandbox.world.town import CAR_DRIVER, DRIVING_AGE

    drivers = town.travel_mode == CAR_DRIVER
    on_own = town.guardian < 0
    assert not (drivers & on_own & (town.age < DRIVING_AGE)).any()
    assert drivers[on_own & (town.age >= 30) & (town.age < 50)].mean() > 0.3
