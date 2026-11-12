import dataclasses
import math

import numpy as np
import pytest
from config_factory import LIBRARY

from epidemic_sandbox.engine.spatial_kernels import build_grid, count_infectious_contacts
from epidemic_sandbox.world.population import (
    FEMALE,
    HOMEBOUND,
    PRESCHOOL,
    REMOTE,
    RESIDENT,
    ROLES,
    SCHOOL,
    UNIVERSITY,
    WORKER,
)
from epidemic_sandbox.world.schedule import (
    FIRST_ACTIVITY,
    HOME,
    LUNCH,
    WORK,
    destination_codes_at,
    draw_day_schedule,
)
from epidemic_sandbox.world.town import (
    build_town,
    destination_coordinates,
    load_town_parameters,
    mean_contacts,
    population_statistics,
)

PARAMS = LIBRARY.town
ARENA = 4000.0
WORLD = {"population": 3000, "arena_size": 2000}


def town_of(population=6000, arena=ARENA, params=PARAMS, seed=1):
    return build_town(population, arena, params, np.random.default_rng(seed))


def mean_contacts_at(x, y, mask, arena, venue, radius=2.0):
    """Average number of other people in `mask` within `radius` of a person in `mask`, in the
    same venue."""
    cells = max(1, min(int(arena // radius), 2 * math.isqrt(len(x))))
    contacts = count_infectious_contacts(
        x, y, mask, mask.astype(np.float32), venue, *build_grid(x, y, arena, cells), cells, radius
    )
    return contacts[mask].mean() - 1.0  # a person is within contact distance of themselves


def distance_to_place(town):
    return np.hypot(town.work_x - town.home_x, town.work_y - town.home_y)


def test_demo_town_parameters_load():
    assert set(PARAMS.places) == {"preschool", "basic", "secondary", "university", "work"}
    assert {activity.name for activity in PARAMS.schedule.activities} == {
        "shopping",
        "food",
        "services",
        "outdoor",
        "culture",
        "visit",
    }


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("work = [7, 16]", "work = [16, 7]", "invalid hours"),
        ("work = [7, 16]", "work = [7, 30]", "invalid hours"),
        ("absence = 0.079", "absence = 1.5", "between 0 and 1"),
        ('kind = "visit"', 'kind = "nonsense"', "kind must be"),
        (
            "[places.work]\nsolo_share = 0.132\nmean_size = ",
            "[places.work]\nsolo_share = 0.132\nmean_size = -",
            "mean_size must be positive",
        ),
        ("area_mean = 14.0", "area_mean = 0", "area_mean must be positive"),
        ("candidate_sites = 12", "candidate_sites = 0", "candidate_sites"),
        ("core_share = 0.6", "core_share = 3", "core_share must be between"),
        ("min_area_per_person = 25", "min_area_per_person = 0", "min_area_per_person"),
        ("sociability_sigma = 0.5", "sociability_sigma = -1", "sociability_sigma"),
        ("walk = 4.5", "walk = -4.5", "speeds must be positive"),
        ("mean_regulars = 315", "mean_regulars = 0", "mean_regulars must be positive"),
        ("favourites = 3", "favourites = 0", "favourites must be a whole number"),
        (
            "duration_mean = { weekday = 0.84",
            "duration_mean = { weekday = 0",
            "duration_mean weekday must be positive",
        ),
        ("class_size = 22.6", "class_size = 0", "class_size must be at least 1"),
        ("team_size = 10", "team_size = 0", "team_size must be positive"),
        ("together_share = 0.8", "together_share = 1.5", "together_share"),
        ("seats = [0.65, 1.9, 1.1]", "seats = [0.65, 1.9]", "seats"),
        ("[lunch]\nduration_mean = 0.75", "[lunch]\nduration_mean = 0", "lunch duration_mean"),
        ("remote_share = 0.046", "remote_share = 2", "remote_share must be between"),
        ("common_area = 4.0", "common_area = 0", "common_area must be positive"),
    ],
)
def test_invalid_town_settings_are_rejected(tmp_path, old, new, message):
    from town_files import write_town

    with pytest.raises((ValueError, KeyError, TypeError), match=message):
        load_town_parameters(write_town(tmp_path, old, new))


def test_everybody_lives_inside_the_arena_and_has_all_traits():
    town = town_of(population=30_000)
    coordinates = (town.home_x, town.home_y, town.work_x, town.work_y, *town.activity_x)
    for array in coordinates:
        assert array.min() >= 0 and array.max() <= ARENA
    assert len(town.household) == len(town.age) == len(town.role) == 30_000
    assert len(town.sex) == len(town.speed_kmh) == len(town.guardian) == 30_000
    assert len(town.start_habit) == len(town.weekend_work) == 30_000


def test_couples_share_a_bed_and_the_others_sleep_in_bedrooms_around_the_common_room():
    town = town_of()
    dwelling = PARAMS.dwelling
    private = town.role != RESIDENT
    couples = np.flatnonzero((town.partner >= 0) & private)
    bed = np.hypot(
        town.home_x[couples] - town.home_x[town.partner[couples]],
        town.home_y[couples] - town.home_y[town.partner[couples]],
    )
    assert bed.max() <= dwelling.bed_gap + 1e-6
    to_common = np.hypot(town.home_x - town.home_center_x, town.home_y - town.home_center_y)
    assert np.abs(to_common[private] - dwelling.bedroom_distance).max() < 3 * dwelling.room_bed_gap
    assert town.home_radius[private] == pytest.approx(dwelling.common_radius)
    # the children of a family with as many bedrooms as sleepers sleep apart from their parents
    alone_in_room = np.flatnonzero(private & (town.partner < 0) & (town.age < 18))
    parents = town.partner >= 0
    first_parent = np.full(town.household.max() + 1, -1)
    first_parent[town.household[parents]] = np.flatnonzero(parents)
    with_parents = alone_in_room[first_parent[town.household[alone_in_room]] >= 0]
    gap = np.hypot(
        town.home_x[with_parents] - town.home_x[first_parent[town.household[with_parents]]],
        town.home_y[with_parents] - town.home_y[first_parent[town.household[with_parents]]],
    )
    assert (gap > 2.0).mean() > 0.9


def test_homes_are_denser_in_the_core():
    town = town_of(population=30_000)
    center = np.hypot(town.home_x - ARENA / 2, town.home_y - ARENA / 2) < ARENA / 4
    uniform_share = math.pi / 16
    assert center.mean() > 2 * uniform_share


def test_pupils_sit_at_double_desks_and_workers_at_desks_in_a_grid():
    town = town_of(population=12_000)
    pupils = np.flatnonzero((town.role == SCHOOL) & town.seat_fixed)
    workers = np.flatnonzero(
        (town.role == WORKER) & (town.place_kind == WORKER) & (town.work_people > 1)
    )
    workers = workers[~town.activity_staff[workers]]
    for members, nearest_expected in (
        (pupils, PARAMS.places["basic"].seats[0]),
        (workers, PARAMS.places["work"].desk_spacing),
    ):
        sample = members[:400]
        nearest = []
        for person in sample:
            same = members[town.work_venue[members] == town.work_venue[person]]
            same = same[same != person]
            if len(same):
                nearest.append(
                    np.hypot(
                        town.work_x[same] - town.work_x[person],
                        town.work_y[same] - town.work_y[person],
                    ).min()
                )
        assert np.median(nearest) == pytest.approx(nearest_expected, abs=0.05)


@pytest.mark.parametrize("role", [PRESCHOOL, SCHOOL, UNIVERSITY, WORKER])
def test_people_with_a_place_of_the_day_go_somewhere_else_than_home(role):
    town = town_of(population=20_000)
    members = town.role == role
    assert members.any()
    assert (distance_to_place(town)[members] > 20).mean() > 0.95


def test_commutes_are_shorter_than_random_pairs_and_shrink_with_distance_decay():
    def commute(params):
        town = town_of(params=params)
        return distance_to_place(town)[town.role == WORKER].mean()

    def with_scale(scale):
        work = dataclasses.replace(PARAMS.places["work"], distance_scale=scale)
        return dataclasses.replace(PARAMS, places={**PARAMS.places, "work": work})

    random_pair = 0.52 * ARENA
    assert commute(PARAMS) < 0.6 * random_pair
    assert commute(with_scale(0.02)) < commute(PARAMS) < commute(with_scale(5.0))


def test_young_children_go_to_closer_places_than_students():
    town = town_of(population=20_000)
    distance = distance_to_place(town)
    assert distance[town.role == PRESCHOOL].mean() < distance[town.role == SCHOOL].mean()
    assert distance[town.role == SCHOOL].mean() < distance[town.role == UNIVERSITY].mean()


def test_workplaces_are_shared_and_unequal_in_size():
    town = town_of(population=20_000)
    workers = np.flatnonzero((town.role == WORKER) & ~town.at_school)
    keys = np.round(town.work_x[workers] / 100) * 1e6 + np.round(town.work_y[workers] / 100)
    places, sizes = np.unique(keys, return_counts=True)
    assert len(places) < len(workers) / 1.5
    assert sizes.max() > 4 * np.median(sizes)


def test_a_share_of_the_workers_work_alone():
    town = town_of(population=20_000)
    workers = (town.role == WORKER) & (town.place_kind == WORKER)
    alone = (town.work_people[workers] == 1).mean()
    assert alone == pytest.approx(PARAMS.places["work"].solo_share, abs=0.03)


def test_teachers_are_workers_placed_with_the_pupils():
    town = town_of(population=20_000)
    teachers = town.at_school & (town.role == WORKER)
    pupils = np.isin(town.role, (PRESCHOOL, SCHOOL, UNIVERSITY))
    assert 0.02 * pupils.sum() < teachers.sum() < 0.15 * pupils.sum()
    assert town.age[teachers & (town.place_kind == PRESCHOOL)].min() >= 19
    assert town.age[teachers & (town.place_kind == SCHOOL)].min() >= 24
    assert town.at_school[pupils].all()


def test_subject_teachers_go_from_class_to_class_and_young_classes_have_their_own():
    town = town_of(population=20_000)
    rotation = town.rotation
    assert len(rotation.teacher) > 0
    assert (town.role[rotation.teacher] == WORKER).all()
    chosen = rotation.assign(np.ones(len(rotation.teacher), np.bool_), np.random.default_rng(0))
    taught = chosen[chosen >= 0]
    assert len(np.unique(taught)) == len(taught)  # one teacher in a class at a time
    classes = rotation.start[1:] - rotation.start[:-1]
    teachers = np.bincount(rotation.school, minlength=len(classes))
    assert ((teachers >= classes) | (classes == 0)).all()
    young = (town.role == SCHOOL) & (town.age < 10)
    venues = np.unique(town.work_venue[young])
    assert not np.isin(venues, rotation.class_venue).any()


def test_classes_consist_of_children_of_the_same_age_and_are_not_tiny():
    town = town_of(population=30_000)
    pupils = np.flatnonzero(town.role == SCHOOL)
    ages = {}
    for person in pupils:
        ages.setdefault(town.work_venue[person], []).append(town.age[person])
    assert np.mean([len(set(values)) for values in ages.values()]) < 2
    sizes = np.array([len(values) for values in ages.values()])
    assert (sizes >= PARAMS.places["basic"].min_class_size).mean() > 0.95


def test_every_activity_has_favourite_places_for_everybody():
    town = town_of()
    for activity, x, y, radius, regulars in zip(
        PARAMS.schedule.activities,
        town.activity_x,
        town.activity_y,
        town.activity_radius,
        town.activity_people,
        strict=True,
    ):
        assert x.shape == y.shape == radius.shape == regulars.shape == (6000, activity.favourites)


def test_visits_go_to_homes_of_people_of_similar_age_nearby():
    town = town_of(population=12_000)
    index = PARAMS.schedule.activity_index("visit")
    visited_x = town.activity_x[index][:, 0]
    visited_y = town.activity_y[index][:, 0]
    distance = np.hypot(visited_x - town.home_x, visited_y - town.home_y)
    assert np.median(distance) < 0.4 * ARENA
    assert (distance > 0).mean() > 0.99


def test_leisure_places_favour_the_ages_that_typically_visit_them():
    town = town_of(population=30_000)
    index = PARAMS.schedule.activity_index("food")
    place = np.round(town.activity_x[index][:, 0], 0) * 1e5 + np.round(town.activity_y[index][:, 0])
    ages = {}
    for person, key in enumerate(place):
        ages.setdefault(key, []).append(town.age[person])
    spread = np.mean([np.std(values) for values in ages.values() if len(values) > 5])
    assert spread < 0.9 * town.age.std()


def test_travel_speeds_are_personal_and_depend_on_age():
    town = town_of(population=20_000)
    assert (town.speed_kmh > 0).all()
    alone = town.guardian < 0
    assert len(np.unique(town.speed_kmh[alone])) > 0.99 * alone.sum()
    assert (
        town.speed_kmh[town.age >= 65].mean()
        < town.speed_kmh[(town.age >= 18) & (town.age < 65)].mean()
    )
    assert (
        town.speed_kmh[town.age < 12].mean()
        < town.speed_kmh[(town.age >= 18) & (town.age < 65)].mean()
    )


def test_sociability_is_a_personal_trait_with_mean_one():
    town = town_of(population=30_000)
    below_the_frail_ages = town.sociability[town.age < 65]
    assert below_the_frail_ages.mean() == pytest.approx(1.0, rel=0.03)
    assert below_the_frail_ages.std() > 0.3


def day_in(town, seed=0, weekend=False):
    return draw_day_schedule(
        town.role, town.traits, PARAMS.schedule, weekend, np.random.default_rng(seed)
    )


def test_destinations_are_translated_to_coordinates():
    town = town_of()
    people = np.arange(len(town.age))
    day = day_in(town)
    x, y = destination_coordinates(town, np.full(len(people), HOME), people, day)
    assert np.array_equal(x, town.home_x) and np.array_equal(y, town.home_y)
    x, _ = destination_coordinates(town, np.full(len(people), WORK), people, day)
    assert np.array_equal(x, town.work_x)


def test_a_person_stands_somewhere_else_in_a_shop_on_every_visit():
    town = town_of()
    people = np.arange(len(town.age))
    shop = PARAMS.schedule.activity_index("shopping")
    codes = np.full(len(people), FIRST_ACTIVITY + shop)
    first, second = day_in(town, seed=1), day_in(town, seed=2)
    second.choice[:] = first.choice
    x_first, y_first = destination_coordinates(town, codes, people, first)
    x_second, y_second = destination_coordinates(town, codes, people, second)
    moved = np.hypot(x_first - x_second, y_first - y_second)
    adults = town.guardian < 0
    assert (moved[adults] > 0.01).mean() > 0.95
    center = town.activity_x[shop][people, first.choice[:, shop]]
    radius = town.activity_radius[shop][people, first.choice[:, shop]]
    assert (np.abs(x_first - center) <= radius + 1e-9).all()


def test_lunch_is_near_the_workplace_and_not_near_home():
    town = town_of(population=30_000)
    workers = (town.role == WORKER) & (distance_to_place(town) > 0.2 * ARENA)
    to_work = np.hypot(town.lunch_x - town.work_x, town.lunch_y - town.work_y)[workers]
    to_home = np.hypot(town.lunch_x - town.home_x, town.lunch_y - town.home_y)[workers]
    assert np.median(to_work) < 0.7 * np.median(to_home)
    people = np.arange(len(town.age))
    x, _ = destination_coordinates(town, np.full(len(people), LUNCH), people, day_in(town))
    assert np.abs(x - town.lunch_x).max() < town.lunch_radius.max() + 1e-9


def test_presence_shares_are_the_measured_ones():
    """The share of regulars present at the busiest hour sets the floor space of a place."""
    town = town_of(population=60_000, arena=2000 * math.sqrt(20))
    days = [day_in(town, seed=day, weekend=weekend) for day, weekend in enumerate((False, True))]
    for index, activity in enumerate(PARAMS.schedule.activities):
        if activity.kind != "site":
            continue
        code = FIRST_ACTIVITY + index
        peak = max(
            (destination_codes_at(day, hour / 2, PARAMS.schedule.max_events) == code).mean()
            for day in days
            for hour in range(48)
        )
        if activity.name == "food":
            at_lunch = max(
                (
                    destination_codes_at(days[0], 12 + quarter / 4, PARAMS.schedule.max_events)
                    == LUNCH
                ).mean()
                for quarter in range(8)
            )
            peak = max(peak, at_lunch)
        assert peak / activity.favourites == pytest.approx(activity.presence_share, rel=0.3)


def test_small_children_go_out_with_their_guardian_to_the_same_place():
    town = town_of(population=30_000)
    children = np.flatnonzero(town.guardian >= 0)
    assert (town.age[children] < PARAMS.accompanied_under).all()
    assert ((town.age < PARAMS.accompanied_under) == (town.guardian >= 0)).mean() > 0.999
    guardians = town.guardian[children]
    assert (town.household[children] == town.household[guardians]).all()
    assert (town.age[guardians] >= 18).all()
    assert (town.sex[guardians] == FEMALE).mean() > 0.9
    day = day_in(town, weekend=True)
    for index in range(len(town.activity_x)):
        code = FIRST_ACTIVITY + index
        for person, carer in zip(children[:500], guardians[:500], strict=True):
            mine = day.times[person][(day.codes[person] == code) & np.isfinite(day.times[person])]
            theirs = day.times[carer][(day.codes[carer] == code) & np.isfinite(day.times[carer])]
            assert set(mine) <= set(theirs)
    assert np.isfinite(day.times[children]).any(axis=1).mean() > 0.2


def test_a_child_at_school_does_not_go_out_with_the_guardian():
    town = town_of(population=30_000)
    day = day_in(town)
    pupils = np.flatnonzero((town.guardian >= 0) & (town.role == SCHOOL) & ~day.absent)
    for person in pupils[:1000]:
        finite = np.isfinite(day.times[person])
        codes = day.codes[person][finite].tolist()
        assert codes[0] == WORK or not codes
        assert all(code in (HOME, WORK) for code in codes[:2])


def test_visits_never_go_to_the_own_household():
    town = town_of(population=12_000)
    index = PARAMS.schedule.activity_index("visit")
    distance = np.hypot(
        town.activity_x[index] - town.home_x[:, None], town.activity_y[index] - town.home_y[:, None]
    )
    assert (distance[town.role != RESIDENT] > 1.0).mean() > 0.999


def test_older_people_go_out_less():
    town = town_of(population=60_000, arena=8000.0)

    def outings(low, high):
        return town.sociability[(town.age >= low) & (town.age < high)].mean()

    assert outings(30, 60) == pytest.approx(1.0, abs=0.03)
    assert outings(85, 100) == pytest.approx(0.660, abs=0.08)


def test_care_homes_are_places_of_old_people_with_their_staff():
    town = town_of(population=60_000, arena=8000.0)
    residents = town.role == RESIDENT
    assert residents.mean() == pytest.approx(0.0055, abs=0.0005)
    assert (town.age[residents] >= 65).mean() > 0.9
    assert residents[town.age >= 85].mean() > 0.08
    homes, sizes = np.unique(town.household[residents], return_counts=True)
    assert sizes.mean() == pytest.approx(66, rel=0.25)
    staff = (town.place_kind == RESIDENT) & (town.role == WORKER)
    # `staff_per_resident` of them (head count of the day shifts) per resident: 0.40 against the
    # 0.574 of all shifts (31,646 care staff for 55,108 residents), not one per resident
    care = PARAMS.care_home
    assert staff.sum() == pytest.approx(care.staff_per_resident * residents.sum(), rel=0.1)
    assert staff.sum() < 0.574 * residents.sum()
    assert (town.shift_window[staff] == care.shift_hours).all()
    assert (town.sex[staff] == FEMALE).mean() > 0.75
    assert town.bedridden[residents].mean() == pytest.approx(care.bedridden_share, abs=0.05)
    wards, in_ward = np.unique(town.ward_venue[residents], return_counts=True)
    assert in_ward.mean() == pytest.approx(care.ward_size, rel=0.35)
    assert (town.room_venue[residents] >= 0).all() and not town.bedridden[~residents].any()


def test_the_places_of_the_lunch_break_have_their_staff_over_the_lunch_breaks():
    town = town_of(population=6000)
    food = next(a for a in PARAMS.schedule.activities if a.name == "food")
    staff = town.activity_staff & ~np.isnan(town.shift_window[:, 0, 0])
    lunch_places = np.unique(town.lunch_venue[town.lunch_venue >= 0])
    serves_lunch = np.isin(town.work_venue[staff], lunch_places)
    weekday = town.shift_window[staff, 0][serves_lunch]
    assert len(weekday) > 0
    lunch = np.nanmin(PARAMS.schedule.lunch_window[:, 0, 0])
    assert (weekday[:, 0] <= lunch).mean() > 0.2, "some begin before the lunch break"
    assert (weekday[:, 1] >= food.window[0][1]).mean() > 0.2, "and some stay into the evening"
    assert (weekday[:, 1] - weekday[:, 0] <= 8.0 + 1e-9).all(), "shifts of a working day"


def test_residents_stay_in_the_care_home():
    town = town_of(population=30_000)
    day = day_in(town)
    assert not np.isfinite(day.times[town.role == RESIDENT]).any()


def test_teachers_live_as_near_to_work_as_other_workers():
    town = town_of(population=40_000, arena=2000 * math.sqrt(40_000 / 3000))
    distance = distance_to_place(town)
    teachers = town.at_school & (town.role == WORKER)
    others = (town.place_kind == WORKER) & (town.role == WORKER)
    assert distance[teachers].mean() < 1.25 * distance[others].mean()


def test_staff_are_of_the_usual_sex_for_the_level():
    town = town_of(population=60_000, arena=8000.0)

    def women(kind):
        staff = (town.place_kind == kind) & (town.role == WORKER)
        return (town.sex[staff] == FEMALE).mean()

    assert women(PRESCHOOL) > 0.9 > women(SCHOOL) > women(UNIVERSITY)


def test_there_are_about_as_many_teachers_as_in_reality():
    # CZSO 2018/19: 11.9 children per kindergarten teacher; 940,928 + 420,814 pupils of basic and
    # secondary schools per 64,345 + 38,223 teachers (13.3); Eurostat 2019: 16.8 students per
    # university teacher (full-time equivalents and head counts)
    town = town_of(population=60_000, arena=8000.0)
    staff = lambda kind: ((town.place_kind == kind) & (town.role == WORKER)).sum()  # noqa: E731
    assert (town.role == PRESCHOOL).sum() / staff(PRESCHOOL) == pytest.approx(11.9, rel=0.2)
    assert (town.role == SCHOOL).sum() / staff(SCHOOL) == pytest.approx(13.3, rel=0.2)
    assert (town.role == UNIVERSITY).sum() / staff(UNIVERSITY) == pytest.approx(16.8, rel=0.3)


def test_teachers_keep_school_hours_without_a_late_shift():
    town = town_of(population=20_000)
    teachers = (town.role == WORKER) & np.isin(town.place_kind, (SCHOOL, UNIVERSITY))
    window = town.shift_window[teachers, 0]
    assert not np.isnan(window).any() and window[:, 1].max() <= 16.0
    assert (town.start_habit[teachers] == 0).all()
    assert (town.shift_attendance[teachers, 1] == 0).all()
    preschool = (town.role == WORKER) & (town.place_kind == PRESCHOOL)
    shifts = np.unique(town.shift_window[preschool, 0], axis=0)
    assert len(shifts) == len(PARAMS.places["preschool"].staff_shifts)


def test_remote_workers_and_the_homebound_have_no_place():
    town = town_of(population=20_000)
    for role in (REMOTE, HOMEBOUND):
        assert (distance_to_place(town)[town.role == role] == 0).all()


def test_population_statistics_describe_the_sampled_town():
    statistics = population_statistics(WORLD, PARAMS)
    assert statistics["mean_age"] == pytest.approx(42.7, abs=2.0)
    assert statistics["mean_household_size"] == pytest.approx(2.15, abs=0.25)
    assert (
        sum(statistics["age_shares"]) == pytest.approx(1.0) and len(statistics["age_shares"]) == 18
    )
    assert sum(statistics["role_shares"].values()) == pytest.approx(1.0)
    assert set(statistics["role_shares"]) == set(ROLES)
    assert statistics["role_shares"]["school"] > 0.08
    assert statistics["role_shares"]["resident"] > 0


def test_estimated_mean_contacts_grow_with_the_contact_distance_and_are_cached():
    small = mean_contacts(WORLD, PARAMS, 2.0)
    assert 0.1 < small < 20
    assert mean_contacts(WORLD, PARAMS, 6.0) > small
    assert mean_contacts(WORLD, PARAMS, 2.0) == small


def test_mean_contacts_do_not_depend_on_the_area_of_the_town():
    # People meet inside homes and places, whose floor space per person is a property of the
    # place; a larger town only puts them further apart (contacts through walls do not count).
    # The two towns are drawn independently: over the seeds 0-3 their difference spread from -3.7
    # to +5.8 % (a few large workplaces carry much of it), so only a systematic one is caught.
    crowded = mean_contacts({"population": 3000, "arena_size": 1200}, PARAMS, 4.0)
    sparse = mean_contacts({"population": 3000, "arena_size": 4000}, PARAMS, 4.0)
    assert sparse == pytest.approx(crowded, rel=0.08)


def test_town_is_reproducible_from_the_seed():
    first, second, other = town_of(seed=5), town_of(seed=5), town_of(seed=6)
    assert np.array_equal(first.work_x, second.work_x) and np.array_equal(first.age, second.age)
    assert np.array_equal(first.speed_kmh, second.speed_kmh)
    assert not np.array_equal(first.work_x, other.work_x)


def test_the_caches_of_the_setup_preview_stay_small():
    from epidemic_sandbox.world import town as town_module

    for population in range(1000, 1000 + 3 * town_module.CACHED_SAMPLES):
        population_statistics({"population": population, "arena_size": 2000}, PARAMS)
    assert len(town_module._SAMPLES) == town_module.CACHED_SAMPLES
    for radius in range(2, 2 + town_module.CACHED_CONTACTS + 5):
        mean_contacts({"population": 300, "arena_size": 600}, PARAMS, float(radius))
    assert len(town_module._CONTACTS) == town_module.CACHED_CONTACTS


def test_a_recently_used_preview_is_kept_and_gives_the_same_answer():
    world = {"population": 777, "arena_size": 1500}
    first = mean_contacts(world, PARAMS, 3.0)
    mean_contacts({"population": 778, "arena_size": 1500}, PARAMS, 3.0)
    assert mean_contacts(world, PARAMS, 3.0) == first


def test_relatives_among_the_friends_are_a_generation_older_or_younger():
    town = build_town(6000, ARENA, PARAMS, np.random.default_rng(4))
    activity = PARAMS.schedule.activities[PARAMS.schedule.activity_index("visit")]
    on_own = town.guardian < 0  # children share the favourites of their guardian
    kin = town.visit_kin[on_own]
    host_age = np.full(int(town.household.max()) + 1, -1.0)
    np.maximum.at(host_age, town.household, town.age)  # the oldest member of each household
    hosts = town.visit_host[on_own]  # the household of the host (a venue may be a ward)
    gap = host_age[hosts] - town.age[on_own][:, None]
    older, younger, friends = kin == 1, kin == -1, kin == 0
    assert older.any() and younger.any() and friends.any()
    assert np.median(gap[older]) > activity.generation_gap - 8
    assert np.median(np.abs(gap[friends])) < activity.generation_gap / 2
    share = (kin != 0).mean()
    family = activity.family_share[town.age[on_own]].mean()
    assert share == pytest.approx(family, abs=0.05)


def test_a_small_child_goes_out_with_somebody_old_enough_to_be_a_parent_not_with_a_sibling():
    town = town_of(population=30_000, arena=6500.0)
    children = np.flatnonzero(town.guardian >= 0)
    gap = town.age[town.guardian[children]] - town.age[children]
    # an adult sibling was the guardian of 1.1 % of the children before; what remains are the
    # households in which nobody is old enough to be a parent
    assert len(children) > 1000
    assert (gap >= PARAMS.population.parent_gap_min).mean() > 0.995
