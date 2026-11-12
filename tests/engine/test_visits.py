"""Visits go to a household where somebody is at home, into the host's dwelling."""

import numpy as np
from config_factory import disease_definition, make_model

from epidemic_sandbox.world.schedule import FIRST_ACTIVITY, HOME

DISEASE = {"a": disease_definition("sirs", 0)}


def visit_model():
    model = make_model(DISEASE, population=2000, area_km2=1.0)
    return model, int(model._visits[0])


def visitor(model, activity):
    """Somebody on their own (not a child that goes with a guardian)."""
    return int(np.flatnonzero(model.town.guardian < 0)[0])


def set_off(model, person, activity, choice):
    """`person` sets off on a visit to their favourite `choice` in this step."""
    model._code[person] = FIRST_ACTIVITY + activity
    model._schedule.choice[person, activity] = choice
    model._visit_hosts_at_home(np.array([person]))


def hosts(model, person, activity):
    return model._activity_venue[activity, person]


def empty_households(model, households):
    """Nobody of `households` is at home: all of them are on the way somewhere."""
    model._inside[np.isin(model.town.household, households)] = False


def test_who_is_at_home_is_who_is_inside_and_heading_home():
    model, _ = visit_model()
    assert model._members_at_home().all(), "everybody starts at home"
    model._inside[3] = False
    model._code[4] = FIRST_ACTIVITY
    home, work = model._detour
    home[5], model._code[5] = True, FIRST_ACTIVITY  # an ill person heading home
    work[6] = True  # kept at the workplace over lunch
    at_home = model._members_at_home()
    assert not at_home[[3, 4, 6]].any() and at_home[5]


def test_a_visitor_goes_to_the_planned_household_when_somebody_is_at_home():
    model, activity = visit_model()
    person = visitor(model, activity)
    set_off(model, person, activity, 1)
    assert model._code[person] == FIRST_ACTIVITY + activity
    assert model._schedule.choice[person, activity] == 1


def test_a_visitor_goes_to_another_favourite_when_nobody_is_at_home():
    model, activity = visit_model()
    person = visitor(model, activity)
    favourites = hosts(model, person, activity)
    empty_households(model, favourites[1])
    set_off(model, person, activity, 1)
    assert model._code[person] == FIRST_ACTIVITY + activity
    chosen = model._schedule.choice[person, activity]
    assert favourites[chosen] != favourites[1]


def test_a_visit_is_dropped_when_nobody_is_at_home_in_any_favourite():
    model, activity = visit_model()
    person = visitor(model, activity)
    empty_households(model, hosts(model, person, activity))
    set_off(model, person, activity, 0)
    assert model._code[person] == HOME


def test_a_child_on_a_visit_goes_where_its_guardian_goes():
    model, activity = visit_model()
    town = model.town
    child = int(np.flatnonzero(town.guardian >= 0)[0])
    guardian = int(town.guardian[child])
    favourites = hosts(model, guardian, activity)
    empty_households(model, favourites[0])
    code = FIRST_ACTIVITY + activity
    model._code[[child, guardian]] = code
    model._schedule.choice[[child, guardian], activity] = 0
    model._visit_hosts_at_home(np.array([child, guardian]))
    assert model._code[child] == model._code[guardian] == code
    chosen = model._schedule.choice[[child, guardian], activity]
    assert chosen[0] == chosen[1] != 0
    empty_households(model, favourites)
    model._code[[child, guardian]] = code
    model._visit_hosts_at_home(np.array([child, guardian]))
    assert model._code[child] == model._code[guardian] == HOME


def test_visitors_stand_in_the_common_room_of_the_host():
    model, activity = visit_model()
    town = model.town
    host_member = {}
    for person in range(model.population):
        host_member.setdefault(int(town.household[person]), person)
    for person in range(0, model.population, 37):
        for favourite, host in enumerate(model._activity_venue[activity, person]):
            if host < 0:
                continue
            if host >= model._households:  # a ward or a room of a care home, not a dwelling
                assert host in town.ward_venue or host in town.room_venue
                continue
            member = host_member[int(host)]
            assert town.activity_x[activity][person, favourite] == town.home_center_x[member]
            assert town.activity_radius[activity][person, favourite] == town.home_radius[member]
