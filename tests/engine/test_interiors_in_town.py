"""Where people are inside the places of a town: classes and their breaks, teachers who go from
class to class, carers on their rounds, the rooms of a dwelling."""

import dataclasses

import numpy as np
import pytest
from config_factory import disease_definition, make_model
from model_helpers import run

from epidemic_sandbox.world.population import RESIDENT
from epidemic_sandbox.world.schedule import HOME, WORK

DISEASE = {"a": disease_definition("sirs", 0)}
SATURDAY_AFTERNOON = 5 * 24 + 14  # a run starts on Monday at midnight; 24 steps make a day


def town_model(**world):
    model = make_model(DISEASE, population=3000, arena_size=2000, **world)
    model._ensure_interiors()
    return model


def nobody(model) -> np.ndarray:
    return np.zeros(model.population, np.bool_)


def test_every_lesson_each_teacher_who_goes_from_class_to_class_has_a_class_of_their_school():
    model = town_model()
    rotation, town = model._rotation, model.town
    teachers = rotation.teacher
    assert len(teachers) > 0
    model._code[teachers] = WORK
    taught = []
    for _ in range(3):
        model._last_lesson = -1
        model._teach(nobody(model))
        venue = model._lesson_venue[teachers]
        teaching = venue >= 0
        assert teaching.any()
        # a class has one teacher at a time, from the school of the teacher
        assert len(np.unique(venue[teaching])) == teaching.sum()
        for school in range(len(rotation.start) - 1):
            classes = rotation.class_venue[rotation.start[school] : rotation.start[school + 1]]
            mine = rotation.school == school
            assert np.isin(venue[mine & teaching], classes).all()
            assert (mine & teaching).sum() == min(mine.sum(), len(classes))
        # they stand in the class, within its radius
        index = np.searchsorted(rotation.class_venue, venue[teaching])
        who = teachers[teaching]
        far = np.hypot(
            model._target_x[who] - rotation.class_x[index],
            model._target_y[who] - rotation.class_y[index],
        )
        assert (far <= rotation.class_radius[index] + 1e-6).all()
        assert not model._in_common[who].any()
        taught.append(venue.copy())
    assert any((first != second).any() for first, second in zip(taught, taught[1:], strict=False))
    assert town.work_venue[teachers].min() >= 0


def test_teachers_do_not_change_class_inside_a_lesson_and_those_away_have_none():
    model = town_model()
    teachers = model._rotation.teacher
    model._code[teachers] = WORK
    model._teach(nobody(model))
    first = model._lesson_venue[teachers].copy()
    model._teach(nobody(model))  # the same lesson: nothing changes
    assert (model._lesson_venue[teachers] == first).all()
    model._last_lesson = -1
    away = nobody(model)
    away[teachers[:3]] = True  # ill at home: they do not teach
    model._teach(away)
    assert (model._lesson_venue[teachers[:3]] == -1).all()


def test_the_spots_of_a_place_at_the_edge_of_the_arena_are_inside_the_arena():
    model = town_model()
    rotation, town = model._rotation, model.town
    arena = float(model.world["arena_size"])
    # a school at the corner of the town: its classes, its common rooms and its seats spill over
    rotation.class_x[:] = -0.5
    rotation.class_y[:] = arena + 0.5
    model._code[rotation.teacher] = WORK
    model._teach(nobody(model))
    pupils = model._pupils
    model._code[pupils] = WORK
    town.school_x[pupils] = -0.5
    town.school_y[pupils] = arena + 0.5
    model._class_share[:] = 1.0
    model._breaks(nobody(model))
    for who in (rotation.teacher, pupils):
        assert model._target_x[who].min() >= 0 and model._target_y[who].max() <= arena
    town.work_center_x[pupils] = -0.5
    model._class_share[:] = 0.0
    model._breaks(nobody(model))
    assert model._target_x[pupils].min() >= 0


def test_the_pupils_of_a_class_are_in_the_common_rooms_of_their_school_together():
    model = town_model()
    town = model.town
    pupils = model._pupils
    assert len(pupils) > 100
    model._code[pupils] = WORK
    model._class_share[:] = 1.0  # a break in every class
    model._breaks(nobody(model))
    assert model._in_common[pupils].all()
    near = np.hypot(
        model._target_x[pupils] - town.school_x[pupils],
        model._target_y[pupils] - town.school_y[pupils],
    )
    assert (near <= town.school_radius[pupils] + 1e-6).all()
    model._class_share[:] = 0.0  # the break is over: back to the seats
    model._breaks(nobody(model))
    assert not model._in_common[pupils].any()
    seated = model.town.seat_fixed[pupils]
    assert seated.any()
    assert (model._target_x[pupils][seated] == town.work_x[pupils][seated]).all()
    # with a chance a step, the classes take their breaks one by one, all pupils of a class at once
    model._class_share[:] = 0.3
    for _ in range(5):
        model._breaks(nobody(model))
        on_break = np.bincount(model._pupil_class, weights=model._in_common[pupils])
        size = np.bincount(model._pupil_class)
        assert set(np.unique(on_break / size)) <= {0.0, 1.0}
    assert 0 < model._in_common[pupils].mean() < 1


def test_carers_on_their_rounds_stand_by_the_bed_of_a_resident_of_their_ward():
    model = town_model()
    town = model.town
    care = dataclasses.replace(model.town_parameters.care_home, staff_room_share=1.0)
    model.town_parameters = dataclasses.replace(model.town_parameters, care_home=care)
    carers = np.flatnonzero(model._care_staff)
    assert len(carers) > 0
    model._code[carers] = WORK
    model._new_spots(carers)
    visiting = model._room_visit[carers] >= 0
    assert visiting.all()
    rooms = model._room_visit[carers]
    beds = np.flatnonzero(np.isin(town.room_venue, rooms) & (town.role == RESIDENT))
    assert np.isin(rooms, town.room_venue[beds]).all()
    # the bed they stand by is that of a resident of their own ward
    for carer, room in zip(carers, rooms, strict=True):
        resident = beds[town.room_venue[beds] == room]
        assert (town.ward[resident] == town.ward[carer]).all()
        distance = np.hypot(
            model._target_x[carer] - town.home_x[resident],
            model._target_y[carer] - town.home_y[resident],
        )
        assert distance.min() <= 0.5 + 1e-6
    # with no rounds they stay in the ward's day room
    care = dataclasses.replace(care, staff_room_share=0.0)
    model.town_parameters = dataclasses.replace(model.town_parameters, care_home=care)
    model._new_spots(carers)
    assert (model._room_visit[carers] == -1).all()


def test_at_home_everybody_sleeps_in_bed_and_awake_is_mostly_in_the_common_room():
    model = town_model()
    town, params = model.town, model.town_parameters
    run(model, 3 * 24 + 3)  # three o'clock at night
    at_home = (model._code == HOME) & ~model._residents
    in_bed = np.hypot(model.x - town.home_x, model.y - town.home_y) < 1e-6
    assert in_bed[at_home].all()
    run(model, SATURDAY_AFTERNOON - model.steps)
    at_home = (model._code == HOME) & model._inside & ~model._residents & ~model._away()
    assert at_home.sum() > 500
    middle = np.hypot(model.x - town.home_center_x, model.y - town.home_center_y)
    common = middle <= town.home_radius + 1e-6
    in_bed = np.hypot(model.x - town.home_x, model.y - town.home_y) < 1e-6
    # everybody is in the common room or in their own room, and most are together
    # (a few are on their short walk between the two rooms at any moment)
    assert (common | in_bed)[at_home].mean() > 0.99
    assert common[at_home].mean() == pytest.approx(params.dwelling.together_share, abs=0.07)
    assert in_bed[at_home].any()


def test_whoever_comes_home_by_day_enters_the_common_room_and_at_night_goes_to_bed():
    model = town_model()
    town, params = model.town, model.town_parameters
    people = np.flatnonzero(~model._residents)[:1500]
    model._code[people] = HOME
    model.steps = 14 * model.world["steps_per_day"] // 24  # two in the afternoon
    model._aim(people)
    middle = np.hypot(
        model._target_x[people] - town.home_center_x[people],
        model._target_y[people] - town.home_center_y[people],
    )
    in_bed = (model._target_x[people] == town.home_x[people]) & (
        model._target_y[people] == town.home_y[people]
    )
    common = middle <= town.home_radius[people] + 1e-6
    assert (common | in_bed).all()
    assert common.mean() == pytest.approx(params.dwelling.together_share, abs=0.03)
    model.steps = 2 * model.world["steps_per_day"] // 24  # two at night
    model._aim(people)
    assert (model._target_x[people] == town.home_x[people]).all()
