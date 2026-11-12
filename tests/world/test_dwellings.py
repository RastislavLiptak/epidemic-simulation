import math

import numpy as np
import pytest
from config_factory import LIBRARY

from epidemic_sandbox.world.dwellings import (
    bedrooms,
    beds,
    layout,
    load_dwelling_parameters,
)

DWELLING = LIBRARY.town.dwelling
RAW = {
    "rooms_per_person": [[2.9, 1.3, 1.3, 1.3], [1.8, 1.2, 1.0, 0.8], [1.4, 1.1, 1.1, 1.1]],
    "common_area": 4.0,
    "together_share": 0.8,
    "bed_gap": 0.6,
    "room_bed_gap": 2.0,
    "bedroom_distance": 5.0,
    "sleep_hours": [22.5, 6.5],
    "dwell_hours": 0.75,
}


def test_the_hours_in_bed_may_run_over_midnight():
    assert DWELLING.asleep(23.0) and DWELLING.asleep(2.0) and DWELLING.asleep(6.0)
    assert not DWELLING.asleep(6.5) and not DWELLING.asleep(12.0) and not DWELLING.asleep(22.0)
    day = load_dwelling_parameters({**RAW, "sleep_hours": [1.0, 5.0]})
    assert day.asleep(3.0) and not day.asleep(0.5) and not day.asleep(5.0)


@pytest.mark.parametrize(
    "change",
    [
        {"common_area": 0},
        {"together_share": 1.2},
        {"sleep_hours": [8, 8]},
        {"sleep_hours": [25, 3]},
        {"rooms_per_person": [[1, 1, 1, 1]] * 2},
        {"rooms_per_person": [[0, 1, 1, 1]] * 3},
    ],
)
def test_dwelling_parameters_that_cannot_be_are_refused(change):
    with pytest.raises(ValueError):
        load_dwelling_parameters({**RAW, **change})


def test_the_common_radius_is_that_of_a_disk_of_the_common_area():
    assert math.pi * DWELLING.common_radius**2 == pytest.approx(DWELLING.common_area)


def test_the_bedrooms_are_the_rooms_of_the_dwelling_without_the_common_room_and_at_least_one():
    # a person alone (2.9 rooms): 3 rooms, 2 bedrooms; two adults with two children (1.0 a
    # person): 4 rooms, 3 bedrooms; a lone adult with one child (1.3 a person): 3 rooms, 2 bedrooms
    age = np.array([70, 40, 38, 10, 5, 35, 8, 50, 52, 18, 16, 14, 12])
    household = np.array([0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    rooms = bedrooms(age, household, DWELLING)
    assert rooms.tolist() == [2, 3, 2, 6]  # the last: 2 adults + 1 grown-up + 3 children = 6
    # the smallest dwellings still have a bedroom
    tiny = load_dwelling_parameters({**RAW, "rooms_per_person": [[0.1] * 4] * 3})
    assert bedrooms(age, household, tiny).min() == 1


def test_a_couple_shares_a_bed_and_the_children_share_the_rooms_that_are_left_evenly():
    # a couple (persons 0 and 1) with children aged 10, 7, 3, 1 and three bedrooms
    age = np.array([40, 38, 10, 7, 3, 1])
    household = np.zeros(6, np.int64)
    partner = np.array([1, 0, -1, -1, -1, -1])
    bedroom, place = beds(age, household, partner, np.array([3]), np.random.default_rng(0))
    assert bedroom[0] == bedroom[1] == 0 and place[0] == place[1] == 0  # the parents' own room
    children = bedroom[2:]
    assert sorted(set(children)) == [1, 2]  # the two other rooms
    assert np.sum(children == 1) == np.sum(children == 2) == 2  # two children to a room
    for room in (1, 2):  # the places in a room are 0, 1, ...
        assert sorted(place[2:][children == room]) == [0, 1]
    # the eldest child has a room of its own place first: the oldest children come first
    assert place[2] == 0 and place[3] == 0


def test_with_one_bedroom_everybody_sleeps_in_it_and_nobody_shares_a_place():
    age = np.array([40, 38, 10, 70])
    household = np.array([0, 0, 0, 1])
    partner = np.array([1, 0, -1, -1])
    bedroom, place = beds(age, household, partner, np.array([1, 1]), np.random.default_rng(0))
    assert (bedroom == 0).all()
    assert sorted(place[:3].tolist()) == [0, 0, 1]  # the couple's double bed and one more bed


def test_the_beds_lie_around_the_common_room_as_far_apart_as_the_parameters_say():
    age = np.array([40, 38, 10, 7, 3, 1, 80])
    household = np.array([0, 0, 0, 0, 0, 0, 1])
    partner = np.array([1, 0, -1, -1, -1, -1, -1])
    centers = np.array([100.0, 500.0]), np.array([100.0, 300.0])
    x, y = layout(age, household, partner, *centers, DWELLING, np.random.default_rng(0))
    # the two halves of the double bed are the bed gap apart
    assert math.hypot(x[0] - x[1], y[0] - y[1]) == pytest.approx(DWELLING.bed_gap)
    # a bedroom is the bedroom distance from the middle of the common room (the bed of a couple
    # is its middle), the beds of a room the bed gap of a room apart
    middle_x, middle_y = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
    assert math.hypot(middle_x - 100, middle_y - 100) == pytest.approx(DWELLING.bedroom_distance)
    assert math.hypot(x[6] - 500, y[6] - 300) == pytest.approx(DWELLING.bedroom_distance)
    # 4.8 rooms: 5 rooms, 4 bedrooms; the couple has one, so four children share three: one pair
    # of them is a room gap apart, the other two are alone in their rooms
    children = np.arange(2, 6)
    gaps = np.hypot(x[children, None] - x[children], y[children, None] - y[children])
    np.fill_diagonal(gaps, np.inf)
    nearest = gaps.min(axis=1)
    assert np.isclose(nearest, DWELLING.room_bed_gap).sum() == 2
    assert (nearest[~np.isclose(nearest, DWELLING.room_bed_gap)] > DWELLING.room_bed_gap).all()
