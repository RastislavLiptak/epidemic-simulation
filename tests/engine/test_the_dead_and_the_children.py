"""The dead go nowhere, and children are not left without a carer by the dead or by illness."""

import numpy as np
from config_factory import TEST_STEPS_PER_DAY, disease_definition, make_model
from model_helpers import run

from epidemic_sandbox.world.population import ADULT_AGE, FEMALE, SCHOOL, WORKER
from epidemic_sandbox.world.schedule import WORK_EVENTS, draw_day_schedule, free_adults

DISEASE = {"a": disease_definition("sirs", 0)}


def town_model(population=2000):
    return make_model(DISEASE, population=population, area_km2=1.0)


def kill(model, people) -> None:
    model.disease_states["a"].state[people] = model.diseases["a"].dead_index


def next_day(model) -> None:
    run(model, TEST_STEPS_PER_DAY - model.steps % TEST_STEPS_PER_DAY + 1)


def household_adults(model, person: int) -> np.ndarray:
    town = model.town
    return np.flatnonzero((town.household == town.household[person]) & (town.age >= ADULT_AGE))


def test_the_dead_have_no_place_and_no_outings():
    model = town_model()
    dead = np.flatnonzero(model.town.role == WORKER)[:20]
    kill(model, dead)
    next_day(model)
    assert not model.schedule.attends[dead].any()
    assert np.isinf(model.schedule.times[dead]).all()


def test_a_child_whose_guardian_died_goes_out_with_the_next_adult_of_the_household():
    model = town_model()
    town = model.town
    child = next(
        child
        for child in np.flatnonzero(town.guardian >= 0)
        if len(household_adults(model, child)) >= 2
    )
    guardian = int(town.guardian[child])
    kill(model, [guardian])
    next_day(model)
    others = [adult for adult in household_adults(model, child) if adult != guardian]
    # the youngest woman, else the youngest man, as guardians are chosen
    expected = min(others, key=lambda adult: (town.sex[adult] != FEMALE, town.age[adult]))
    assert town.guardian[child] == expected
    for index, venues in enumerate(town.activity_venue):
        assert (venues[child] == venues[expected]).all()
        stacked = model._activity_venue[index, child, : venues.shape[1]]  # padded to the most
        assert (stacked == venues[expected]).all()
    assert town.speed_kmh[child] == town.speed_kmh[expected]
    assert (model.schedule.choice[child] == model.schedule.choice[expected]).all()


def test_a_child_left_with_no_living_adult_goes_on_no_outing():
    model = town_model()
    town = model.town
    children = np.flatnonzero(town.guardian >= 0)
    adults = np.unique(np.concatenate([household_adults(model, child) for child in children]))
    kill(model, adults)
    next_day(model)
    assert np.isinf(model.schedule.times[children, WORK_EVENTS:]).all()


def test_a_dead_guardian_sends_the_child_home_from_an_outing():
    model = town_model()
    town = model.town
    child = int(np.flatnonzero(town.guardian >= 0)[0])
    guardian = int(town.guardian[child])
    model._code[[child, guardian]] = model._visits[0] + 3  # both on a visit
    assert not model._detours()[0][child]
    kill(model, [guardian])
    assert model._detours()[0][child]


def test_a_young_child_ill_at_home_keeps_its_guardian_at_home_when_nobody_else_is_free():
    model = town_model()
    town = model.town
    care = model.town_parameters.schedule.childcare
    child = int(np.flatnonzero((town.guardian >= 0) & (town.age < care.alone_all_day_from))[0])
    guardian = int(town.guardian[child])
    model.disease_states["a"].stays_home[child] = True
    model._free_adults[:] = 0
    assert model._detours()[0][guardian]
    model._free_adults[town.household[child]] = 1
    assert not model._detours()[0][guardian]


def test_no_child_under_ten_spends_a_closed_school_day_without_an_adult():
    model = town_model(4000)
    town, params = model.town, model.town_parameters.schedule
    care = params.childcare
    schedule = draw_day_schedule(
        town.role, town.traits, params, False, np.random.default_rng(3), (SCHOOL,)
    )
    alive = np.ones(model.population, np.bool_)
    free = free_adults(town.traits, schedule.attends, alive)
    young = np.flatnonzero(
        (town.guardian >= 0) & (town.role == SCHOOL) & (town.age < care.alone_all_day_from)
    )
    assert len(young)
    looked_after = (~schedule.attends[town.guardian[young]]) | (free[town.household[young]] > 0)
    assert looked_after.all()
