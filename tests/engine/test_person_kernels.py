"""The kernels that look at every person once give what the array expressions they replaced gave.

The expressions are kept here as the reference; they are the plain statement of the rules.
"""

import numpy as np
import pytest
from config_factory import LIBRARY

from epidemic_sandbox.engine.epidemic_model import EpidemicModel
from epidemic_sandbox.engine.person_kernels import (
    due_events,
    settled_movers,
    sort_events,
)
from epidemic_sandbox.world.schedule import FIRST_ACTIVITY, HOME, LUNCH, WORK
from epidemic_sandbox.world.town import destination_venues

POPULATION = 1500


@pytest.fixture(scope="module")
def messy_model() -> EpidemicModel:
    """Every disease of the catalog, three days in, and then people put into arbitrary states, so
    that every case of the rules (hospital, isolation, a guardian, a visit, night) occurs."""
    names = sorted(LIBRARY.catalog)
    config = LIBRARY.parse(
        {
            "world": {"population": POPULATION, "area_km2": 2.0, "seed": 11},
            "diseases": {name: {"initial_infected": 60} for name in names},
        }
    )
    model = EpidemicModel(config, LIBRARY.town)
    for _ in range(3 * 48 + 20):
        model.step()
    rng = np.random.default_rng(1)
    for state in model.disease_states.values():
        states = int(state.state.max()) + 1
        picked = rng.random(POPULATION) < 0.2
        state.state[picked] = rng.integers(0, states + 2, picked.sum()).clip(0, states - 1)
        state.stays_home[:] = rng.random(POPULATION) < 0.1
        state.limits_outings[:] = rng.random(POPULATION) < 0.15
    model._inside[:] = rng.random(POPULATION) < 0.8
    model._in_common[:] = rng.random(POPULATION) < 0.3
    model._code[:] = rng.choice(
        [HOME, WORK, LUNCH, *range(FIRST_ACTIVITY, FIRST_ACTIVITY + 4)], POPULATION
    )
    return model


def reference_detours(model):
    """The rules as array expressions (what `EpidemicModel._detours` was before)."""
    home = np.zeros(model.population, np.bool_)
    skips = np.zeros(model.population, np.bool_)
    isolated = np.zeros(model.population, np.bool_)
    ill = np.zeros(model.population, np.bool_)
    removed = np.zeros(model.population, np.bool_)  # in hospital or dead
    for name, disease in model.diseases.items():
        state = model.disease_states[name]
        removed |= disease.away_mask(state) | (state.state == disease.dead_index)
        ill |= state.stays_home
        skips |= state.limits_outings
        isolated |= disease.confined_mask(state)
    home |= removed | ill
    # a child too young to spend a day alone, ill at home, keeps its guardian at home when no
    # other adult of the household is free today
    town = model.town
    care = model.town_parameters.schedule.childcare
    wards = np.flatnonzero(town.guardian >= 0)
    needs = wards[
        ill[wards]
        & ~removed[wards]
        & (town.age[wards] < care.alone_all_day_from)
        & (model._free_adults[town.household[wards]] == 0)
    ]
    home[town.guardian[needs]] = True
    outing = model._code >= FIRST_ACTIVITY
    households = int(model.town.household.max()) + 1
    unwell = np.bincount(model.town.household, weights=isolated, minlength=households) > 0
    for index in model._visit_activities:
        visiting = np.flatnonzero(model._code == FIRST_ACTIVITY + index)
        choice = model._schedule.choice[visiting, index]
        host = model.town.activity_venue[index][visiting, choice]
        home[visiting[unwell[host]]] = True
    home |= skips & outing
    children = np.flatnonzero((model.town.guardian >= 0) & outing)
    home[children] |= home[model.town.guardian[children]]
    work = skips & (model._code == LUNCH) & ~home
    return home, work


def reference_venues(model, home, work):
    town = model.town
    venue = destination_venues(town, model._code, model._people, model._schedule)
    common = model._in_common & (model._code == WORK) & ~home & ~work
    venue[common] = town.school_venue[common]
    venue[home] = town.household[home]
    venue[work] = town.work_venue[work]
    start, end = model.town_parameters.care_home.day_hours
    in_home = model._residents & (venue == town.household)
    night = not start <= model.hour < end
    venue[in_home] = np.where(
        night | town.bedridden[in_home], town.room_venue[in_home], town.ward_venue[in_home]
    )
    venue[~model._inside | model._away()] = -1
    # a teacher who goes from class to class is in the class of the lesson, a carer in the room
    # of the resident they look after
    inside = (model._code == WORK) & (venue >= 0) & ~model._in_common & ~work & ~home
    for room in (model._lesson_venue, model._room_visit):
        venue[inside & (room >= 0)] = room[inside & (room >= 0)]
    return venue


def test_who_goes_home_or_back_to_work_is_what_the_rules_say(messy_model):
    home, work = messy_model._detours()
    expected_home, expected_work = reference_detours(messy_model)
    assert np.array_equal(home, expected_home) and np.array_equal(work, expected_work)
    assert home.sum() > 100 and work.sum() > 5, "the case has to contain what it tests"


@pytest.mark.parametrize("hour", [3.0, 13.0], ids=["night", "day"])
def test_the_venue_everybody_is_in_is_what_the_rules_say(messy_model, hour, monkeypatch):
    monkeypatch.setattr(type(messy_model), "hour", property(lambda self: hour))
    home, work = reference_detours(messy_model)
    venue = messy_model._venues(home, work)
    assert np.array_equal(venue, reference_venues(messy_model, home, work))
    assert (venue < 0).sum() > 100


def test_who_may_move_to_another_spot_is_what_the_rules_say(messy_model):
    model, town = messy_model, messy_model.town
    rng = np.random.default_rng(4)
    model._target_x[:] = model.x
    model._target_y[:] = model.y
    unsettled = rng.random(POPULATION) < 0.4
    model._target_x[unsettled] += 1.0
    home, _ = reference_detours(model)
    for up, awake in ((True, True), (False, False), (True, False)):
        settled = (model.x == model._target_x) & (model.y == model._target_y)
        away = (model._code != HOME) & ~home
        on_their_own = (town.guardian < 0) | (model._code == WORK)
        at_home = (model._code == HOME) & ~home
        residents = model._residents & at_home & ~town.bedridden
        dwellers = ~model._residents & at_home
        expected = np.flatnonzero(
            settled & ((away & on_their_own) | (residents & up) | (dwellers & awake))
        )
        found = settled_movers(
            model.x,
            model.y,
            model._target_x,
            model._target_y,
            model._code,
            home,
            town.guardian,
            model._residents,
            town.bedridden,
            up,
            awake,
            HOME,
            WORK,
        )
        assert np.array_equal(found, expected)
        assert len(found) > 50


def test_whose_event_is_due_is_what_the_schedule_says(messy_model):
    schedule = messy_model._schedule
    rng = np.random.default_rng(2)
    pointer = rng.integers(0, schedule.times.shape[1] - 1, POPULATION).astype(np.int16)
    for until in (0.0, 7.5, 12.0, 23.9):
        expected = np.flatnonzero(schedule.times[np.arange(POPULATION), pointer] <= until)
        assert np.array_equal(due_events(schedule.times, pointer, until), expected)


def test_the_events_of_a_day_are_sorted_as_a_stable_sort_does(messy_model):
    rng = np.random.default_rng(3)
    times = rng.choice([0.5, 7.0, 7.0, 12.25, np.inf, 20.0], (800, 13))
    codes = rng.integers(0, 9, (800, 13)).astype(np.int8)
    travel = rng.random((800, 13))
    order = np.argsort(times, axis=1, kind="stable")
    for kernel in (sort_events.serial, sort_events.parallel):
        ordered_times, ordered_codes, ordered_travel = kernel(times, codes, travel)
        assert np.array_equal(ordered_times, np.take_along_axis(times, order, axis=1))
        assert np.array_equal(ordered_codes, np.take_along_axis(codes, order, axis=1))
        assert np.array_equal(ordered_travel, np.take_along_axis(travel, order, axis=1))
    day = messy_model._schedule
    assert (day.times[:, :-1] <= day.times[:, 1:]).all()


def test_a_town_without_diseases_can_be_stepped():
    config = LIBRARY.parse(
        {"world": {"population": 300, "area_km2": 0.3, "seed": 2}, "diseases": {}},
        allow_no_diseases=True,
    )
    model = EpidemicModel(config, LIBRARY.town)
    for _ in range(48):
        model.step()
    home, work = model._detours()
    assert not home.any() and not work.any()


def test_somebody_ill_at_home_is_in_their_household_not_in_the_room_or_class_of_their_work(
    messy_model, monkeypatch
):
    monkeypatch.setattr(type(messy_model), "hour", property(lambda self: 13.0))
    home, work = reference_detours(messy_model)
    # everybody on a detour with a room or lesson of the day: put them there, as the illness is
    messy_model._room_visit[:] = np.maximum(messy_model._room_visit, 0) + 1
    messy_model._lesson_venue[:] = np.maximum(messy_model._lesson_venue, 0) + 1
    venue = messy_model._venues(home, work)
    ill_at_work_code = home & (messy_model._code == WORK) & messy_model._inside
    ill_at_work_code &= ~messy_model._away() & ~messy_model._in_common & ~messy_model._residents
    assert ill_at_work_code.sum() > 0, "the case has to contain what it tests"
    assert np.array_equal(venue[ill_at_work_code], messy_model.town.household[ill_at_work_code])


def test_the_dwell_at_places_is_not_overwritten_by_the_dwell_at_home(messy_model):
    messy_model._ensure_interiors()
    assert messy_model._dwell is not messy_model._place_dwell
