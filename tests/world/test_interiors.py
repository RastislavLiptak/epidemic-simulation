import numpy as np
import pytest

from epidemic_sandbox.world.interiors import (
    Rotation,
    SeatLayout,
    around,
    classes,
    desks,
    ranks_within,
    ring_positions,
    seats,
    split_evenly,
)

RNG = np.random.default_rng


def test_points_around_a_center_lie_in_the_disk_and_fill_it_evenly():
    x, y = around(np.full(20_000, 3.0), np.full(20_000, -2.0), 5.0, RNG(0))
    distance = np.hypot(x - 3.0, y + 2.0)
    assert distance.max() <= 5.0
    assert (distance < 2.5).mean() == pytest.approx(0.25, abs=0.02)  # the area, a quarter


def test_ranks_within_each_group_run_from_zero_in_a_random_order():
    group = np.array([2, 0, 2, 2, 1, 0, 2, 5])
    rank, size = ranks_within(group, RNG(1))
    assert size.tolist() == [4, 2, 4, 4, 1, 2, 4, 1]
    for value in np.unique(group):
        assert sorted(rank[group == value]) == list(range((group == value).sum()))
    # random: the order of the members differs between draws
    other, _ = ranks_within(np.zeros(50, np.int64), RNG(2))
    again, _ = ranks_within(np.zeros(50, np.int64), RNG(3))
    assert (other != again).any()


def test_a_group_is_split_into_parts_of_about_the_given_size_and_never_below_one_part():
    group = np.repeat([0, 1, 2, 3], [100, 7, 1, 52])
    part = split_evenly(group, 25.0, RNG(0))
    assert len(np.unique(part)) == 4 + 1 + 1 + 2  # 100 people, 7 people, 1 person, 52 people
    assert (np.unique(part[group == 0], return_counts=True)[1] == 25).all()
    sizes = np.unique(part[group == 3], return_counts=True)[1]
    assert sizes.tolist() == [26, 26]
    assert not set(part[group == 0]) & set(part[group == 1])  # numbers are unique over groups


def test_pupils_of_a_school_are_taught_by_year_and_small_years_together():
    # school 0: 40 pupils of 6, 40 of 7; school 1: 3 pupils of 6, 4 of 7, 2 of 8 (a small school)
    site = np.repeat([0, 0, 1, 1, 1], [40, 40, 3, 4, 2])
    age = np.repeat([6, 7, 6, 7, 8], [40, 40, 3, 4, 2])
    group = classes(
        site, age, class_size=20, min_class_size=8, age_span=1, oldest_age=15, rng=RNG(0)
    )
    # school 0: two years of two classes of 20; the years never share a class
    for year in (6, 7):
        mine = (site == 0) & (age == year)
        assert len(np.unique(group[mine])) == 2
    assert not set(group[(site == 0) & (age == 6)]) & set(group[(site == 0) & (age == 7)])
    # school 1: nine pupils of three years are one class; no class has pupils of two schools
    assert len(np.unique(group[site == 1])) == 1
    assert not set(group[site == 0]) & set(group[site == 1])


def test_the_oldest_ages_are_one_year_and_pupils_of_a_school_are_never_alone_in_a_class():
    site = np.zeros(61, np.int64)
    age = np.concatenate([np.full(30, 14), np.full(30, 15), [19]])  # 15 and over are one year
    group = classes(
        site, age, class_size=30, min_class_size=5, age_span=1, oldest_age=15, rng=RNG(0)
    )
    assert group[-1] == group[30]  # the 19-year-old is taught with the 15-year-olds
    assert len(np.unique(group)) == 2


def test_the_members_of_a_room_sit_in_double_desks_each_with_a_desk_mate():
    layout = SeatLayout(pair_gap=0.8, column_gap=1.6, row_gap=1.2)
    room = np.repeat([0, 1], [24, 9])
    x, y = seats(room, np.array([10.0, 50.0]), np.array([0.0, 20.0]), layout, RNG(0))
    for index, size in ((0, 24), (1, 9)):
        mine = room == index
        spots = set(zip(np.round(x[mine], 6), np.round(y[mine], 6), strict=True))
        assert len(spots) == size  # one seat each
        gaps = np.hypot(x[mine, None] - x[mine], y[mine, None] - y[mine])
        np.fill_diagonal(gaps, np.inf)
        nearest = gaps.min(axis=1)
        assert nearest.max() == pytest.approx(0.8)  # everybody has a desk-mate...
        assert (nearest > 0.79).all()
        assert abs(x[mine].mean() - (10.0 if index == 0 else 50.0)) < 1.0  # ...around the middle
    # with an odd number of members one sits at a desk alone
    x, y = seats(np.zeros(5, np.int64), np.zeros(1), np.zeros(1), layout, RNG(1))
    gaps = np.hypot(x[:, None] - x, y[:, None] - y)
    np.fill_diagonal(gaps, np.inf)
    assert (gaps.min(axis=1) > 0.79).all() and (gaps.min(axis=1) < 0.81).sum() == 4


def test_desks_form_a_grid_around_the_middle_of_the_room():
    room = np.repeat([0, 1], [9, 5])
    x, y = desks(room, np.array([0.0, 100.0]), np.array([0.0, 0.0]), 2.0, RNG(0))
    nine = sorted(zip(np.round(x[:9], 6), np.round(y[:9], 6), strict=True))
    assert nine == sorted((a, b) for a in (-2.0, 0.0, 2.0) for b in (-2.0, 0.0, 2.0))
    assert len(set(zip(np.round(x[9:], 6), np.round(y[9:], 6), strict=True))) == 5
    assert abs(x[9:].mean() - 100.0) <= 2.0


def rotation_of(teachers_per_school, classes_per_school) -> Rotation:
    school = np.repeat(np.arange(len(teachers_per_school)), teachers_per_school)
    start = np.concatenate([[0], np.cumsum(classes_per_school)])
    count = int(start[-1])
    return Rotation(
        teacher=np.arange(len(school)) + 100,
        school=school,
        start=start,
        class_venue=np.arange(count) + 1000,
        class_x=np.arange(count, dtype=float),
        class_y=np.zeros(count),
        class_radius=np.full(count, 3.0),
    )


def test_every_class_has_at_most_one_teacher_and_every_teacher_at_most_one_class():
    rotation = rotation_of([5, 2, 3], [3, 4, 3])
    chosen = rotation.assign(np.ones(10, np.bool_), RNG(0))
    taught = chosen[chosen >= 0]
    assert len(set(taught.tolist())) == len(taught)  # no class twice
    # school 0: 5 teachers, 3 classes: three teach, two stay in the staff room
    assert (chosen[:5] >= 0).sum() == 3 and np.isin(chosen[:5][chosen[:5] >= 0], [0, 1, 2]).all()
    # school 1: 2 teachers, 4 classes: both teach
    assert (chosen[5:7] >= 0).all() and np.isin(chosen[5:7], [3, 4, 5, 6]).all()
    assert (chosen[7:] >= 0).all() and np.isin(chosen[7:], [7, 8, 9]).all()


def test_teachers_who_are_not_present_teach_nothing_and_the_others_take_their_place():
    rotation = rotation_of([4], [4])
    present = np.array([True, False, True, False])
    chosen = rotation.assign(present, RNG(0))
    assert (chosen[~present] == -1).all() and (chosen[present] >= 0).all()
    assert (rotation.assign(np.zeros(4, np.bool_), RNG(0)) == -1).all()
    # the classes the teachers take differ from lesson to lesson
    lessons = {
        tuple(rotation.assign(np.ones(4, np.bool_), RNG(seed)).tolist()) for seed in range(20)
    }
    assert len(lessons) > 5


def test_joined_rotations_keep_their_schools_and_classes_apart():
    first, second = rotation_of([2, 2], [2, 3]), rotation_of([3], [2])
    both = first.extend(second)
    assert len(both.teacher) == 7 and len(both.start) - 1 == 3
    assert both.start.tolist() == [0, 2, 5, 7]
    assert both.school.tolist() == [0, 0, 1, 1, 2, 2, 2]
    chosen = both.assign(np.ones(7, np.bool_), RNG(0))
    assert np.isin(chosen[4:], [5, 6]).sum() == 2  # the third school's classes, numbered after


def test_points_on_a_ring_are_the_given_distance_from_the_middle_and_evenly_spread():
    # two circles, of 6 and 3 points; each circle is turned as a whole
    index, count = np.array([*range(6), *range(3)]), np.array([6] * 6 + [3] * 3)
    circle = np.array([0] * 6 + [1] * 3)
    centers_x, centers_y = np.where(circle == 0, 10.0, -4.0), np.where(circle == 0, 1.0, 7.0)
    for seed in range(3):
        x, y = ring_positions(index, count, circle, centers_x, centers_y, 1.5, RNG(seed))
        assert np.hypot(x - centers_x, y - centers_y) == pytest.approx(np.full(9, 1.5))
        for which in (0, 1):  # evenly spread points are the vertices of a regular polygon: the
            mine = circle == which  # center is the mean of the points
            assert x[mine].mean() == pytest.approx(centers_x[mine][0])
            assert y[mine].mean() == pytest.approx(centers_y[mine][0])
    x, y = ring_positions(index, count, circle, centers_x, centers_y, 1.5, RNG(0))
    side = np.hypot(np.diff(x[:6]), np.diff(y[:6]))
    assert side == pytest.approx(
        np.full(5, 1.5)
    )  # a regular hexagon has sides as long as its radius
