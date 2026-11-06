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
    for name, disease in model.diseases.items():
        state = model.disease_states[name]
        home |= disease.away_mask(state) | state.stays_home
        skips |= state.limits_outings
        isolated |= disease.confined_mask(state)
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
    venue = destination_venues(model.town, model._code, model._people, model._schedule)
    common = model._in_common & (model._code == WORK) & ~home & ~work
    venue[common] = model.town.school_venue[common]
    venue[home] = model.town.household[home]
    venue[work] = model.town.work_venue[work]
    start, end = model.town_parameters.care_home.day_hours
    if not start <= model.hour < end:
        asleep = model._residents & (venue == model.town.household)
        venue[asleep] = model.town.room_venue[asleep]
    venue[~model._inside | model._away()] = -1
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
    assert (venue < 0).sum() > 100 and (venue == messy_model.town.room_venue).sum() >= 0


def test_who_may_move_to_another_spot_is_what_the_rules_say(messy_model):
    model, town = messy_model, messy_model.town
    rng = np.random.default_rng(4)
    model._target_x[:] = model.x
    model._target_y[:] = model.y
    unsettled = rng.random(POPULATION) < 0.4
    model._target_x[unsettled] += 1.0
    home, _ = reference_detours(model)
    for up in (True, False):
        settled = (model.x == model._target_x) & (model.y == model._target_y)
        away = (model._code != HOME) & ~home
        on_their_own = (town.guardian < 0) | (model._code == WORK)
        residents = model._residents & (model._code == HOME) & ~home
        expected = np.flatnonzero(settled & ((away & on_their_own) | (residents & up)))
        found = settled_movers(
            model.x,
            model.y,
            model._target_x,
            model._target_y,
            model._code,
            home,
            town.guardian,
            model._residents,
            up,
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
    order = np.argsort(times, axis=1, kind="stable")
    for kernel in (sort_events.serial, sort_events.parallel):
        ordered_times, ordered_codes = kernel(times, codes)
        assert np.array_equal(ordered_times, np.take_along_axis(times, order, axis=1))
        assert np.array_equal(ordered_codes, np.take_along_axis(codes, order, axis=1))
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
