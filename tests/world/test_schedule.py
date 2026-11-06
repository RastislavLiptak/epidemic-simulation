import dataclasses
from datetime import date

import numpy as np
import pytest
from config_factory import LIBRARY

from epidemic_sandbox.world.population import PRESCHOOL, RETIRED, ROLES, SCHOOL, WORKER
from epidemic_sandbox.world.schedule import (
    HOME,
    LATEST_END,
    LUNCH,
    WORK,
    PersonalTraits,
    destination_codes_at,
    draw_day_schedule,
    draw_habits,
    draw_weekend_work,
)

PARAMS = LIBRARY.town.schedule
COUNT = 40_000
WEEKDAY, WEEKEND = False, True
NO_HABIT = np.zeros(COUNT)


def traits_of(role, start_habit=NO_HABIT, sociability=None, weekend_work=0.0, guardian=None):
    return PersonalTraits(
        start_habit,
        NO_HABIT,
        np.ones(COUNT) if sociability is None else sociability,
        np.full(COUNT, weekend_work),
        np.full(COUNT, -1) if guardian is None else guardian,
        np.full(COUNT, role),
    )


def day_of(
    role,
    weekend=WEEKDAY,
    start_habit=NO_HABIT,
    sociability=None,
    seed=0,
    weekend_work=0.0,
    closed=(),
):
    traits = traits_of(role, start_habit, sociability, weekend_work)
    return draw_day_schedule(
        np.full(COUNT, role), traits, PARAMS, weekend, np.random.default_rng(seed), closed
    )


def work_events(day):
    return day.times[day.codes == WORK]


def first_arrival(day):
    arrival = np.where(day.codes == WORK, day.times, np.inf).min(axis=1)
    return arrival[np.isfinite(arrival)]


def test_events_of_each_person_are_in_time_order_and_end_with_infinity():
    day = day_of(WORKER)
    assert (day.times[:, 1:] >= day.times[:, :-1]).all()
    assert np.isinf(day.times[:, -1]).all()
    assert day.codes.shape == day.times.shape


def test_nothing_happens_late_at_night():
    day = day_of(WORKER)
    assert day.times[np.isfinite(day.times)].max() <= LATEST_END


def test_workers_arrive_around_the_usual_start_but_each_at_their_own_time():
    arrival = first_arrival(day_of(WORKER))
    usual = PARAMS.work_window[WORKER, 0, 0]
    assert arrival.mean() == pytest.approx(usual, abs=0.2)
    assert len(np.unique(arrival)) > 0.99 * len(arrival), "times are continuous"
    assert arrival.std() > 0.2, "not everybody arrives at once"


def test_a_personal_habit_shifts_the_whole_day_of_that_person():
    early = first_arrival(day_of(WORKER, start_habit=np.full(COUNT, -1.0)))
    late = first_arrival(day_of(WORKER, start_habit=np.full(COUNT, 1.0)))
    assert late.mean() - early.mean() == pytest.approx(2.0, abs=0.15)


def test_some_people_skip_their_place_on_a_given_day():
    day = day_of(WORKER)
    expected = PARAMS.variability["absence"][WORKER]
    assert day.absent.mean() == pytest.approx(expected, abs=0.01)
    absent_stays = np.isfinite(day.times[day.absent]) & (day.codes[day.absent] == WORK)
    assert not absent_stays.any()


def test_absence_differs_between_days():
    first, second = day_of(WORKER, seed=1), day_of(WORKER, seed=2)
    assert (first.absent != second.absent).any()


def work_end(day):
    """Time of the first event after the last stay at work of those who worked."""
    stays = np.isfinite(day.times) & (day.codes == WORK)
    last = np.where(stays.any(axis=1), day.times.shape[1] - 1 - stays[:, ::-1].argmax(axis=1), -1)
    rows = np.flatnonzero(last >= 0)
    return day.times[rows, last[rows] + 1]


def test_overtime_makes_some_people_leave_later_than_the_usual_end():
    ends = work_end(day_of(WORKER))
    usual_end = PARAMS.work_window[WORKER, 0, 1]
    overtime = PARAMS.variability["overtime_probability"][WORKER]
    assert ends.mean() == pytest.approx(
        usual_end + overtime * PARAMS.variability["overtime_mean"][WORKER], abs=0.15
    )
    assert 0.02 < (ends > usual_end + 1.5).mean() < overtime


def test_most_workers_do_not_work_on_weekends():
    day = day_of(WORKER, WEEKEND)
    assert day.absent.all()
    assert not np.isfinite(day.times[day.codes == WORK]).any()


def test_weekend_workers_work_on_weekends_with_their_chance_and_take_weekdays_off():
    day = day_of(WORKER, WEEKEND, weekend_work=0.75)
    assert (~day.absent).mean() == pytest.approx(0.75, abs=0.01)
    assert np.isfinite(day.times[day.codes == WORK]).any()
    base = PARAMS.variability["absence"][WORKER]
    weekday = day_of(WORKER, weekend_work=0.75)
    assert weekday.absent.mean() == pytest.approx(base + 0.75 * 2 / 5, abs=0.01)


def test_the_share_of_people_working_on_a_weekend_day_is_the_published_one():
    roles = np.full(COUNT, WORKER)
    chance = draw_weekend_work(roles, PARAMS, np.random.default_rng(0))
    assert (chance == PARAMS.weekend_usual_chance).mean() == pytest.approx(0.137, abs=0.01)
    assert chance.mean() == pytest.approx(0.137 * 0.75 + 0.177 * 0.25, abs=0.01)
    assert (draw_weekend_work(np.full(COUNT, SCHOOL), PARAMS, np.random.default_rng(0)) == 0).all()


def test_a_share_of_workers_is_on_the_late_shift_and_runs_errands_before_work():
    start, end = draw_habits(np.full(COUNT, WORKER), PARAMS, np.random.default_rng(0))
    late = start > 4
    assert late.mean() == pytest.approx(PARAMS.late_shift_share, abs=0.01)
    day = draw_day_schedule(
        np.full(COUNT, WORKER),
        PersonalTraits(
            start,
            end,
            np.ones(COUNT),
            np.zeros(COUNT),
            np.full(COUNT, -1),
            np.full(COUNT, WORKER),
        ),
        PARAMS,
        WEEKDAY,
        np.random.default_rng(1),
    )
    arrival = np.where(day.codes == WORK, day.times, np.inf).min(axis=1)
    at_work = np.isfinite(arrival)
    assert np.median(arrival[late & at_work]) == pytest.approx(14, abs=0.5)
    first = day.times[:, 0]
    assert (first[late & at_work] < arrival[late & at_work]).mean() > 0.1


def test_a_closed_place_keeps_everybody_at_home():
    day = day_of(SCHOOL, closed=(SCHOOL,))
    assert not day.absent.any()
    assert not np.isfinite(day.times[day.codes == WORK]).any()
    assert np.isfinite(day_of(SCHOOL, closed=(PRESCHOOL,)).times[:, 0]).any()


def test_staff_stay_away_from_a_closed_school():
    traits = traits_of(SCHOOL)
    day = draw_day_schedule(
        np.full(COUNT, WORKER), traits, PARAMS, WEEKDAY, np.random.default_rng(0), (SCHOOL,)
    )
    assert not np.isfinite(day.times[day.codes == WORK]).any()


def test_the_calendar_knows_weekends_public_holidays_and_school_holidays():
    calendar = PARAMS.calendar
    assert calendar.first_day.weekday() == 0
    assert [calendar.is_day_off(day) for day in range(7)] == [False] * 5 + [True, True]
    new_year = (date(2018, 12, 31) - calendar.first_day).days  # a Monday
    assert calendar.date_of(new_year) == date(2018, 12, 31)
    assert SCHOOL in calendar.closed_places(new_year)
    assert SCHOOL not in calendar.closed_places(0)
    summer = (date(2019, 7, 9) - calendar.first_day).days  # a Tuesday
    assert SCHOOL in calendar.closed_places(summer)
    assert calendar.is_day_off(summer - 4)  # Friday, 5 July
    assert not calendar.is_day_off(summer - 1)
    labour_day = (date(2019, 5, 1) - calendar.first_day).days  # a Wednesday
    assert calendar.is_day_off(labour_day)


def test_school_days_in_a_year_are_about_the_real_number():
    calendar = PARAMS.calendar
    days = [
        day
        for day in range(365)
        if not calendar.is_day_off(day) and SCHOOL not in calendar.closed_places(day)
    ]
    assert len(days) == pytest.approx(190, abs=8)


def test_lunch_is_a_trip_between_two_stays_at_work():
    day = day_of(WORKER)
    lunches = 0
    for person in range(5000):
        sequence = day.codes[person][np.isfinite(day.times[person])].tolist()
        for index in range(1, len(sequence) - 1):
            if sequence[index] == LUNCH and sequence[index - 1] == WORK:
                assert sequence[index + 1] == WORK
                lunches += 1
    assert lunches > 0.1 * 5000


def test_pupils_have_school_days_that_differ_from_workers_days():
    school = first_arrival(day_of(SCHOOL))
    work = first_arrival(day_of(WORKER))
    assert school.mean() == pytest.approx(PARAMS.work_window[SCHOOL, 0, 0], abs=0.2)
    assert school.mean() != pytest.approx(work.mean(), abs=0.2)


def test_retired_people_have_no_place_of_the_day_but_still_go_out():
    day = day_of(RETIRED)
    finite = np.isfinite(day.times)
    assert not (day.codes[finite] == WORK).any()
    assert finite.any(axis=1).mean() > 0.3


def test_sociable_people_go_out_more_often():
    sociability = np.where(np.arange(COUNT) < COUNT // 2, 0.4, 2.0)
    day = day_of(RETIRED, sociability=sociability)
    outings = np.isfinite(day.times).sum(axis=1) // 2
    assert outings[COUNT // 2 :].mean() > 1.8 * outings[: COUNT // 2].mean()


def test_no_errand_starts_while_somebody_is_at_work_except_the_lunch_trip():
    day = day_of(WORKER)
    for person in range(3000):
        sequence = day.codes[person][np.isfinite(day.times[person])].tolist()
        at_work = False
        for code in sequence:
            if code == WORK:
                at_work = True
            elif code == HOME:
                at_work = False
            elif at_work:
                assert code == LUNCH
                at_work = False  # the trip is followed by the return to work


def test_destination_follows_the_schedule_through_the_day():
    day = day_of(WORKER)
    attends = ~day.absent
    before = destination_codes_at(day, 3.0, PARAMS.max_events)
    midday = destination_codes_at(day, 10.5, PARAMS.max_events)
    night = destination_codes_at(day, 23.9, PARAMS.max_events)
    assert (before == HOME).all() and (night == HOME).all()
    assert (midday[attends] == WORK).mean() > 0.8


def test_habits_are_zero_for_roles_without_variation_and_spread_for_workers():
    roles = np.array([WORKER] * 20_000)
    start, end = draw_habits(roles, PARAMS, np.random.default_rng(0))
    day_shift = start < 3.5
    assert start[day_shift].std() == pytest.approx(
        PARAMS.variability["start_personal_sd"][WORKER], rel=0.05
    )
    assert abs(start[day_shift].mean()) < 0.05 and end.std() > 0


def test_invalid_variability_is_rejected():
    from epidemic_sandbox.world.schedule import load_schedule_parameters

    raw = {
        "schedule": {},
        "lunch": {"duration_mean": 0.5},
        "activities": {},
        "variability": {"worker": {"absence": 1.5}},
    }
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_schedule_parameters(raw)
    raw["variability"]["worker"] = {"start_daily_sd": -1.0}
    with pytest.raises(ValueError, match="must not be negative"):
        load_schedule_parameters(raw)


def test_every_role_has_a_variability_entry():
    assert all(len(PARAMS.variability[key]) == len(ROLES) for key in PARAMS.variability)
    assert dataclasses.is_dataclass(PARAMS)
