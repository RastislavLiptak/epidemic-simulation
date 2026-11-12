import json

import numpy as np
import pytest
from config_factory import disease_definition, interaction, make_model

from epidemic_sandbox.app.person_info import describe_person
from epidemic_sandbox.world.population import ROLES

pytestmark = pytest.mark.usefixtures("synthetic_in_registry")


def diseases(**parameters):
    return {
        "a": disease_definition("plain", 3, transmission=0.0, infectious_days=60.0, **parameters),
        "b": disease_definition("latent", 0),
    }


def workers(model):
    town = model.town
    away = np.hypot(town.work_x - town.home_x, town.work_y - town.home_y) > 100
    return np.flatnonzero((town.role == ROLES.index("worker")) & away & ~town.at_school)


def test_description_is_json_and_names_age_role_and_household():
    model = make_model(diseases())
    person = int(workers(model)[0])
    info = describe_person(model, person)
    json.dumps(info)
    assert info["id"] == person and info["age"] == model.town.age[person]
    assert info["role"] == "worker" and info["place"]["kind"] == "workplace"
    household = info["household"]
    assert person in [member["id"] for member in household["members"]]
    assert {model.town.household[member["id"]] for member in household["members"]} == {
        model.town.household[person]
    }


def test_traits_are_the_personal_ones_of_the_person():
    model = make_model(diseases())
    info = describe_person(model, 5)
    assert info["traits"]["speed_kmh"] == pytest.approx(model.town.speed_kmh[5])
    assert info["traits"]["sociability"] == pytest.approx(model.town.sociability[5])


def test_timetable_lists_the_events_of_today_in_order():
    model = make_model(diseases())
    person = int(workers(model)[0])
    entries = describe_person(model, person)["timetable"]
    timed = [entry["at"] for entry in entries if entry["at"] is not None]
    assert timed == sorted(timed)
    if not model.schedule.absent[person]:
        assert entries[0]["what"] == "to workplace" and entries[-1]["what"] == "to home"


def test_a_person_in_the_morning_is_at_home_and_later_on_the_way_to_work():
    model = make_model(diseases(), population=1000, arena_size=6000, steps_per_day=144)
    # Up to two steps pass before the check, so the trip must take longer than that.
    long_trip = model.schedule.travel[:, 0] > 2.5 * model.step_hours
    leaves = model.schedule.times[:, 0]
    person = int(
        next(
            p
            for p in workers(model)
            if long_trip[p] and not model.schedule.absent[p] and leaves[p] > 1
        )
    )
    assert describe_person(model, person)["now"]["activity"] == "at home"
    while model.hour < leaves[person] + model.step_hours:
        model.step()
    now = describe_person(model, person)["now"]
    assert now["activity"].startswith("on the way to") and now["arrives"] is not None


def test_a_person_who_skips_work_today_is_explained():
    model = make_model(diseases(), population=1000, arena_size=1500)
    skipping = [p for p in workers(model) if model.schedule.absent[p]]
    assert skipping
    now = describe_person(model, int(skipping[0]))["now"]
    assert now["activity"] == "at home" and "does not go to" in now["reason"]


def test_an_infectious_person_who_stays_home_is_said_to_do_so():
    model = make_model(diseases(stay_home=1.0))
    state = model.disease_states["a"]
    sick = int(np.flatnonzero(state.state == 1)[0])
    assert "stays at home" in describe_person(model, sick)["now"]["reason"]


def test_health_reports_the_state_of_every_disease_and_interaction_effects():
    rule = interaction("a", ["infectious"], "b", "susceptibility", 0.25)
    model = make_model(diseases(), [rule])
    state = model.disease_states["a"]
    sick = int(np.flatnonzero(state.state == 1)[0])
    health = describe_person(model, sick)["health"]
    assert health["a"]["infectious"] and health["a"]["state"] == "infectious"
    assert health["b"]["state"] == "susceptible" and not health["b"]["ever_infected"]
    base = model.disease_states["b"].susceptibility[sick]
    assert health["b"]["susceptibility"] == pytest.approx(base * 0.25)
    assert health["a"]["days_left"] is not None and health["b"]["days_left"] is None


def test_nearby_people_are_sorted_by_distance_and_within_the_contact_distance():
    model = make_model(diseases())
    model.venue[:3] = 7  # one room
    model.x[1], model.y[1] = model.x[0] + 1.0, model.y[0]
    model.x[2], model.y[2] = model.x[0] + 1.5, model.y[0]
    nearby = describe_person(model, 0)["nearby"]
    assert nearby["count"] >= 2
    distances = [other["distance_m"] for other in nearby["people"]]
    assert distances == sorted(distances) and max(distances) <= 2.0
    assert nearby["people"][0]["id"] == 1


def test_people_behind_a_wall_or_on_the_way_are_not_counted_as_nearby():
    model = make_model(diseases())
    model.x[:4], model.y[:4] = model.x[0], model.y[0]
    model.venue[:4] = [7, 7, 8, -1]  # the same room, the next room, on the way
    nearby = describe_person(model, 0)["nearby"]
    assert [other["id"] for other in nearby["people"]] == [1] and nearby["count"] == 1
    model.venue[0] = -1
    assert describe_person(model, 0)["nearby"]["count"] == 0


def test_the_dead_are_described_as_dead_and_the_gravely_ill_as_in_hospital():
    model = make_model(
        {
            "a": disease_definition("plain", 0, fatality=1.0, death_delay_days=5.0),
            "b": disease_definition("latent", 0),
        }
    )
    dead, dying = 3, 4
    a = model.diseases["a"]
    model.disease_states["a"].state[dead] = a.dead_index
    model.cause_of_death[dead] = 0
    model.death_step[dead] = 2 * model.world["steps_per_day"] + 5
    model.disease_states["a"].state[dying] = a.state_index("dying")
    assert describe_person(model, dead)["now"] == {
        "activity": "died of a",
        "arrives": None,
        "reason": "on day 2",
    }
    now = describe_person(model, dying)["now"]
    assert now["activity"] == "in hospital" and "with a" in now["reason"]


def test_an_activity_without_a_label_is_named_by_its_name():
    import dataclasses

    from epidemic_sandbox.app.person_info import _label
    from epidemic_sandbox.world.schedule import FIRST_ACTIVITY

    model = make_model(diseases())
    params = model.town_parameters
    renamed = tuple(
        dataclasses.replace(activity, name="board_games") if index == 0 else activity
        for index, activity in enumerate(params.schedule.activities)
    )
    model.town_parameters = dataclasses.replace(
        params, schedule=dataclasses.replace(params.schedule, activities=renamed)
    )
    assert _label(model, "worker", FIRST_ACTIVITY) == "board games"


@pytest.mark.parametrize("person", [-1, 200, 10_000, "3", None, 1.5, True])
def test_unknown_people_are_rejected(person):
    with pytest.raises(ValueError, match="Unknown person"):
        describe_person(make_model(diseases()), person)
