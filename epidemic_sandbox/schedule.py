"""Daily plans of individual people: when they leave, where they go, how long they stay.

The windows in `configs/town.toml` only describe the usual day of a role. Every person gets
their own times each day: a personal habit (some always come early, some late), daily
randomness, days off, overtime, a lunch break of their own length, and errands and visits at
their own times. A calendar closes schools and turns public holidays into days off.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from epidemic_sandbox.config_checks import (
    require,
    require_not_negative,
    require_positive,
    require_share,
)
from epidemic_sandbox.population import ADULT_AGE, ROLES, WORKER

DAY_TYPES = ("weekday", "weekend")
HOME, WORK, LUNCH = 0, 1, 2
FIRST_ACTIVITY = 3
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
WEEKEND_FROM_DAY = 5  # the week starts on Monday, so days 5 and 6 are Saturday and Sunday
MONDAY = 0
# Everything planned ends by a quarter to midnight, so nobody is out when the next day is drawn.
LATEST_END = 23.75
# Shortest stay at work or school, so that a shift drawn at random never ends before it starts.
MIN_PRESENCE_HOURS = 1.0
# Source: the Labour Code (262/2006 Coll., section 88) gives at least a 30 minute break, hence 0.3
# hours (18 minutes) as the shortest lunch; Assumption: two hours as the longest.
MIN_LUNCH_HOURS, MAX_LUNCH_HOURS = 0.3, 2.0
# Assumption: the shortest outing is a quick errand (15 minutes), the longest a long visit.
MIN_OUTING_HOURS, MAX_OUTING_HOURS = 0.25, 6.0
# Assumption: lunch lasts a quarter of an hour longer or shorter than the average, no source.
LUNCH_SPREAD_HOURS = 0.25
# Assumption: about nine minutes between the end of work and the next outing (getting out, a short
# stop on the way), no source.
BUFFER_HOURS = 0.15
WORK_EVENTS = 4  # to work, to lunch, back to work, home: the events of a day at work
MAX_AGE_OF_TABLES = 128  # tables by year of age reach this far
# A shift that starts this late leaves the morning free for errands, as a day shift the evening.
LATE_SHIFT_FROM = 12.0
VARIABILITY_KEYS = (
    "absence",
    "start_personal_sd",
    "start_daily_sd",
    "end_personal_sd",
    "end_daily_sd",
    "overtime_probability",
    "overtime_mean",
)


@dataclass(frozen=True, eq=False)
class ActivityParameters:
    """An errand or leisure activity: when it happens, how long, how likely, where."""

    name: str
    kind: str
    window: np.ndarray
    probability: np.ndarray
    duration_mean: float
    duration_shape: float
    favourites: int
    distance_scale: float
    mean_regulars: float = 0.0
    size_sigma: float = 0.0
    area_mean: float = 0.0
    area_sigma: float = 0.0
    presence_share: float = 0.0
    age_scale: float = 0.0
    staff_share: float = 0.0
    staff_days: float = 5.0
    family_share: np.ndarray | None = None  # by year of age
    generation_gap: float = 0.0


@dataclass(frozen=True, eq=False)
class Calendar:
    """Which days are off: weekends, public holidays and when places are closed.

    Dates are "MM-DD", the same every year; a closure may run over the new year.
    """

    first_day: date
    public_holidays: frozenset[str]
    closures: dict[int, tuple[tuple[str, str], ...]]

    def date_of(self, day_index: int) -> date:
        return self.first_day + timedelta(days=day_index)

    def is_day_off(self, day_index: int) -> bool:
        """True on Saturdays, Sundays and public holidays."""
        return (
            day_index % DAYS_PER_WEEK >= WEEKEND_FROM_DAY
            or self.date_of(day_index).strftime("%m-%d") in self.public_holidays
        )

    def closed_places(self, day_index: int) -> tuple[int, ...]:
        """The kinds of places, as role codes, that are closed on a day."""
        today = self.date_of(day_index).strftime("%m-%d")
        return tuple(
            place
            for place, ranges in self.closures.items()
            if any(_in_range(today, start, end) for start, end in ranges)
        )


def _in_range(today: str, start: str, end: str) -> bool:
    return start <= today <= end if start <= end else today >= start or today <= end


@dataclass(frozen=True, eq=False)
class Childcare:
    """How small children are looked after while their guardian works.

    `alone_from`: the age from which a child may be at home without an adult. `hours`: the opening
    hours [from, to] of the kindergarten and of the after-school club, by the role code of the
    child (a child with `aftercare` stays there until the guardian is back, within these hours).
    `club_share`: the share of the school children young enough to be accompanied who attend
    the after-school club.
    """

    alone_from: int
    hours: dict[int, tuple[float, float]]
    club_share: float


@dataclass(frozen=True, eq=False)
class ScheduleParameters:
    """Usual windows by role and day type, probabilities, and how much people deviate."""

    work_window: np.ndarray
    lunch_window: np.ndarray
    lunch_probability: np.ndarray
    lunch_mean_hours: float
    variability: dict[str, np.ndarray]
    activities: tuple[ActivityParameters, ...]
    calendar: Calendar
    weekend_usual_share: float
    weekend_sometimes_share: float
    weekend_usual_chance: float
    weekend_sometimes_chance: float
    late_shift_share: float
    late_shift_start: float
    late_shift_end: float
    childcare: "Childcare | None" = None

    @property
    def max_events(self) -> int:
        return WORK_EVENTS + 2 * len(self.activities)

    def activity_index(self, name: str) -> int:
        return [activity.name for activity in self.activities].index(name)


@dataclass(frozen=True)
class PersonalTraits:
    """What makes the days of a person their own.

    `start_habit` and `end_habit` are fixed shifts of the usual day in hours, `weekend_work` the
    chance of working on a weekend day, `guardian` the adult a child goes out with (-1 for nobody)
    and `place_kind` the role code of the kind of place somebody attends.

    Optional: `shift_window` [person, weekday/weekend, start/end] and `shift_attendance`
    [person, weekday/weekend] replace the usual hours and the chance of being at work of the
    people whose job has its own hours (NaN keeps the role's); `age`, `household` and `aftercare`
    (the child stays at kindergarten or in the after-school club until the guardian is back from
    work) let small children be looked after (see `_look_after_children`).
    """

    start_habit: np.ndarray
    end_habit: np.ndarray
    sociability: np.ndarray
    weekend_work: np.ndarray
    guardian: np.ndarray
    place_kind: np.ndarray
    shift_window: np.ndarray | None = None
    shift_attendance: np.ndarray | None = None
    age: np.ndarray | None = None
    household: np.ndarray | None = None
    aftercare: np.ndarray | None = None


@dataclass
class DaySchedule:
    """The events of one day for everybody: when they happen and where each leads.

    `times` (hours of the day, infinity for none) are sorted per person and have a final
    infinite column; `codes` are the destinations after each event: HOME, WORK, LUNCH, or
    FIRST_ACTIVITY + the index of an activity. `choice` is today's favourite place per activity
    and `spot_x`, `spot_y` are where in a place a person is today, within the unit disk.
    """

    times: np.ndarray
    codes: np.ndarray
    absent: np.ndarray
    choice: np.ndarray
    spot_x: np.ndarray
    spot_y: np.ndarray


def load_schedule_parameters(raw: dict) -> ScheduleParameters:
    """Build and validate the schedule parameters from the parsed town file."""
    shape = (len(ROLES), len(DAY_TYPES), 2)
    windows = {name: np.full(shape, np.nan) for name in ("work", "lunch")}
    for role_index, role in enumerate(ROLES):
        for day_index, day_type in enumerate(DAY_TYPES):
            plan = raw["schedule"].get(role, {}).get(day_type, {})
            for activity, array in windows.items():
                if activity in plan:
                    start, end = plan[activity]
                    if not 0 <= start < end <= HOURS_PER_DAY:
                        raise ValueError(f"invalid hours {start}-{end} for {role}")
                    array[role_index, day_index] = (start, end)
    lunch = np.array([raw.get("lunch_probability", {}).get(role, 0.0) for role in ROLES])
    variability = {
        key: np.array([raw.get("variability", {}).get(role, {}).get(key, 0.0) for role in ROLES])
        for key in VARIABILITY_KEYS
    }
    probabilities = (lunch, variability["absence"], variability["overtime_probability"])
    if any(((array < 0) | (array > 1)).any() for array in probabilities):
        raise ValueError("probabilities must be between 0 and 1")
    if any((variability[key] < 0).any() for key in VARIABILITY_KEYS):
        raise ValueError("variability values must not be negative")
    weekend, late = raw["weekend_work"], raw["late_shift"]
    childcare = _load_childcare(raw["childcare"]) if "childcare" in raw else None
    for key in ("usual_share", "sometimes_share", "usual_chance", "sometimes_chance"):
        require_share(f"weekend_work.{key}", weekend[key])
    require_share("late_shift.share", late["share"])
    return ScheduleParameters(
        work_window=windows["work"],
        lunch_window=windows["lunch"],
        lunch_probability=lunch,
        lunch_mean_hours=_lunch_hours(raw["lunch"]["duration_mean"]),
        variability=variability,
        activities=tuple(
            _load_activity(name, values) for name, values in raw["activities"].items()
        ),
        calendar=_load_calendar(raw["calendar"]),
        weekend_usual_share=weekend["usual_share"],
        weekend_sometimes_share=weekend["sometimes_share"],
        weekend_usual_chance=weekend["usual_chance"],
        weekend_sometimes_chance=weekend["sometimes_chance"],
        late_shift_share=late["share"],
        late_shift_start=late["start_hours_later"],
        late_shift_end=late["end_hours_later"],
        childcare=childcare,
    )


def _load_childcare(raw: dict) -> Childcare:
    require(raw["alone_from"] >= 0, "childcare.alone_from must not be negative")
    hours = {}
    for role, key in (("preschool", "kindergarten_hours"), ("school", "club_hours")):
        start, end = raw[key]
        require(0 <= start < end <= HOURS_PER_DAY, f"childcare.{key}: invalid hours")
        hours[ROLES.index(role)] = (float(start), float(end))
    require_share("childcare.club_share", raw["club_share"])
    return Childcare(int(raw["alone_from"]), hours, float(raw["club_share"]))


def _load_calendar(raw: dict) -> Calendar:
    first_day = date.fromisoformat(raw["first_day"])
    require(first_day.weekday() == MONDAY, "calendar.first_day must be a Monday")
    dates = [*raw["public_holidays"], *np.ravel(sum(raw["closures"].values(), []))]
    for text in dates:
        date.fromisoformat(f"2000-{text}")  # a leap year, so that any date of a year is accepted
    return Calendar(
        first_day=first_day,
        public_holidays=frozenset(raw["public_holidays"]),
        closures={
            ROLES.index(role): tuple(tuple(span) for span in spans)
            for role, spans in raw["closures"].items()
        },
    )


def _lunch_hours(hours: float) -> float:
    require(
        MIN_LUNCH_HOURS <= hours <= MAX_LUNCH_HOURS,
        f"lunch duration_mean must be between {MIN_LUNCH_HOURS} and {MAX_LUNCH_HOURS} hours",
    )
    return hours


def _check_activity(name: str, values: dict) -> None:
    """Reject values that would divide by zero or make no sense for an activity."""
    require(
        isinstance(values["favourites"], int) and values["favourites"] >= 1,
        f"activity {name}: favourites must be a whole number, at least 1",
    )
    for key in ("duration_mean", "duration_shape", "distance_scale"):
        require_positive(f"activity {name}: {key}", values[key])
    if "age_scale" in values:
        require_positive(f"activity {name}: age_scale", values["age_scale"])
    if values["kind"] == "site":
        for key in ("mean_regulars", "area_mean", "presence_share"):
            require(key in values, f"activity {name}: a site needs {key}")
            require_positive(f"activity {name}: {key}", values[key])
        require_share(f"activity {name}: presence_share", values["presence_share"])
        for key in ("size_sigma", "area_sigma"):
            require_not_negative(f"activity {name}: {key}", values.get(key, 0.0))
    if "staff_share" in values:
        require(values["kind"] == "site", f"activity {name}: only a site has staff")
        require_share(f"activity {name}: staff_share", values["staff_share"])
    if "family_share" in values:
        require(values["kind"] == "visit", f"activity {name}: only a visit can be to family")
        for start, end, share in values["family_share"]:
            require(0 <= start < end, f"activity {name}: family_share has an invalid age range")
            require_share(f"activity {name}: family_share", share)
        require_positive(f"activity {name}: generation_gap", values["generation_gap"])
    if "staff_days" in values:
        require(
            0 < values["staff_days"] <= DAYS_PER_WEEK,
            f"activity {name}: staff_days must be between 0 and 7",
        )


def _load_activity(name: str, values: dict) -> ActivityParameters:
    window = np.array([values["window"][day] for day in DAY_TYPES], dtype=float)
    if ((window[:, 0] < 0) | (window[:, 0] >= window[:, 1]) | (window[:, 1] > HOURS_PER_DAY)).any():
        raise ValueError(f"invalid window of activity {name}")
    probability = np.array([values["probability"][role] for role in ROLES], dtype=float)
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"probabilities of activity {name} must be between 0 and 1")
    if values["kind"] not in ("site", "visit"):
        raise ValueError(f"activity {name}: kind must be site or visit")
    _check_activity(name, values)
    optional = {
        key: values[key]
        for key in (
            "mean_regulars",
            "size_sigma",
            "area_mean",
            "area_sigma",
            "presence_share",
            "age_scale",
            "staff_share",
            "staff_days",
            "generation_gap",
        )
        if key in values
    }
    if "family_share" in values:
        shares = np.zeros(MAX_AGE_OF_TABLES)
        for start, end, share in values["family_share"]:
            shares[int(start) : int(end)] = share
        optional["family_share"] = shares
    return ActivityParameters(
        name=name,
        kind=values["kind"],
        window=window,
        probability=probability,
        duration_mean=values["duration_mean"],
        duration_shape=values["duration_shape"],
        favourites=values["favourites"],
        distance_scale=values["distance_scale"],
        **optional,
    )


def draw_habits(role: np.ndarray, params: ScheduleParameters, rng) -> tuple[np.ndarray, np.ndarray]:
    """Personal, fixed deviations (hours) of when somebody usually starts and ends their day.

    A share of the workers are on the late shift, which moves their whole day.
    """
    count = len(role)
    start = rng.normal(0.0, 1.0, count) * params.variability["start_personal_sd"][role]
    end = rng.normal(0.0, 1.0, count) * params.variability["end_personal_sd"][role]
    late = (role == WORKER) & (rng.random(count) < params.late_shift_share)
    start[late] += params.late_shift_start
    end[late] += params.late_shift_end
    return start, end


def draw_weekend_work(role: np.ndarray, params: ScheduleParameters, rng) -> np.ndarray:
    """Chance of working on a weekend day: some usually do, some sometimes, most never."""
    usual = params.weekend_usual_share
    draw = rng.random(len(role))
    chance = np.select(
        [draw < usual, draw < usual + params.weekend_sometimes_share],
        [params.weekend_usual_chance, params.weekend_sometimes_chance],
        0.0,
    )
    return np.where(role == WORKER, chance, 0.0)


def draw_day_schedule(
    role: np.ndarray,
    traits: PersonalTraits,
    params: ScheduleParameters,
    weekend: bool,
    rng,
    closed: tuple[int, ...] = (),
) -> DaySchedule:
    """Draw everybody's day: the place of the day, a lunch break, errands and visits.

    `closed` are the kinds of places that are closed today. Whoever works at weekends takes
    weekdays off instead, and on weekend days only they have a place to go.
    """
    count, day = len(role), int(weekend)
    work = params.work_window[role, day]
    if traits.shift_window is not None:
        own = ~np.isnan(traits.shift_window[:, day, 0])
        work = np.where(own[:, None], traits.shift_window[:, day], work)
    lunch = params.lunch_window[role, day]
    spread = {key: params.variability[key][role] for key in VARIABILITY_KEYS}

    has_place = ~np.isnan(work[:, 0]) & ~np.isin(traits.place_kind, closed)
    days_off = traits.weekend_work * (DAYS_PER_WEEK - WEEKEND_FROM_DAY) / WEEKEND_FROM_DAY
    chance_absent = 1.0 - traits.weekend_work if weekend else spread["absence"] + days_off
    if traits.shift_attendance is not None:
        own = ~np.isnan(traits.shift_attendance[:, day])
        chance_absent = np.where(own, 1.0 - traits.shift_attendance[:, day], chance_absent)
    absent = has_place & (rng.random(count) < chance_absent)
    attends = has_place & ~absent
    work_from = (
        work[:, 0] + traits.start_habit + rng.normal(0.0, 1.0, count) * spread["start_daily_sd"]
    )
    overtime = (rng.random(count) < spread["overtime_probability"]) * rng.gamma(
        2.0, np.maximum(spread["overtime_mean"], 1e-9) / 2.0, count
    )
    work_to = (
        work[:, 1]
        + traits.end_habit
        + rng.normal(0.0, 1.0, count) * spread["end_daily_sd"]
        + overtime
    )
    work_from = np.clip(work_from, 0.0, LATEST_END - MIN_PRESENCE_HOURS)
    work_to = np.clip(np.maximum(work_to, work_from + MIN_PRESENCE_HOURS), 0.0, LATEST_END)
    absent, attends = _look_after_children(traits, has_place, absent, work_from, work_to, params)

    wants_lunch = (
        attends & ~np.isnan(lunch[:, 0]) & (rng.random(count) < params.lunch_probability[role])
    )
    if traits.shift_window is not None:  # a shift takes its break at the workplace
        wants_lunch &= np.isnan(traits.shift_window[:, day, 0])
    lunch_from = rng.uniform(np.nan_to_num(lunch[:, 0]), np.nan_to_num(lunch[:, 1]))
    lunch_to = lunch_from + np.clip(
        rng.normal(params.lunch_mean_hours, LUNCH_SPREAD_HOURS, count),
        MIN_LUNCH_HOURS,
        MAX_LUNCH_HOURS,
    )
    has_lunch = wants_lunch & (lunch_from >= work_from + 0.5) & (lunch_to <= work_to - 0.5)

    times = np.full((count, params.max_events + 1), np.inf)
    codes = np.zeros((count, params.max_events + 1), np.int8)
    for column, (mask, when, code) in enumerate(
        (
            (attends, work_from, WORK),
            (has_lunch, lunch_from, LUNCH),
            (has_lunch, lunch_to, WORK),
            (attends, work_to, HOME),
        )
    ):
        times[mask, column] = when[mask]
        codes[:, column] = code

    late = attends & (work_from >= LATE_SHIFT_FROM)
    busy_until = np.where(attends & ~late, work_to, 0.0)
    limit = np.where(late, work_from - BUFFER_HOURS, LATEST_END)
    _add_outings(times, codes, role, traits.sociability, busy_until, limit, params, day, rng)
    choice = np.stack(
        [rng.integers(0, activity.favourites, count) for activity in params.activities], axis=1
    )
    distance = np.sqrt(rng.random(count))
    angle = rng.random(count) * 2.0 * np.pi
    spot_x, spot_y = distance * np.cos(angle), distance * np.sin(angle)
    children = np.flatnonzero(traits.guardian >= 0)
    _accompany(times, codes, children, traits.guardian[children], attends, work_to)
    guardians = traits.guardian[children]
    choice[children] = choice[guardians]
    spot_x[children], spot_y[children] = spot_x[guardians], spot_y[guardians]
    order = np.argsort(times, axis=1, kind="stable")
    return DaySchedule(
        times=np.take_along_axis(times, order, axis=1),
        codes=np.take_along_axis(codes, order, axis=1),
        absent=absent,
        choice=choice,
        spot_x=spot_x,
        spot_y=spot_y,
    )


def _look_after_children(traits, has_place, absent, work_from, work_to, params):
    """Nobody leaves a small child alone; returns who is absent from their place and who attends.

    A child with `aftercare` (kindergarten, after-school club) is brought before the guardian
    leaves for work and fetched when the guardian is back, within the opening hours of the place:
    `work_from` and `work_to` of the child change. A child younger than `alone_from` who would be
    at home while the guardian works and no other adult of the household is at home keeps the
    guardian at home that day (who then counts as absent from work).
    """
    attends = has_place & ~absent
    care = params.childcare
    if care is None or traits.age is None or traits.household is None:
        return absent, attends
    children = np.flatnonzero(traits.guardian >= 0)
    guardian = traits.guardian[children]
    away_from = work_from[guardian] - BUFFER_HOURS
    away_to = work_to[guardian] + BUFFER_HOURS
    opens = np.full(len(children), np.inf)
    closes = np.full(len(children), -np.inf)
    for kind, (start, end) in care.hours.items():
        at_place = traits.place_kind[children] == kind
        opens[at_place], closes[at_place] = start, end
    stays = attends[children] & traits.aftercare[children] & attends[guardian]
    kept = children[stays]
    work_from[kept] = np.maximum(np.minimum(work_from[kept], away_from[stays]), opens[stays])
    work_to[kept] = np.minimum(np.maximum(work_to[kept], away_to[stays]), closes[stays])
    covered = attends[children] & (work_from[children] <= away_from)
    covered &= work_to[children] >= away_to
    free_adult = (traits.age >= ADULT_AGE) & ~attends  # (care homes have no children)
    free = np.bincount(traits.household, weights=free_adult)
    small = traits.age[children] < care.alone_from
    alone = small & attends[guardian] & ~covered & (free[traits.household[children]] == 0)
    absent = absent.copy()
    absent[guardian[alone]] = True
    return absent, has_place & ~absent


def _accompany(times, codes, children, guardians, attends, work_to) -> None:
    """Children go out with their guardian, when they are not at school or kindergarten."""
    times[children, WORK_EVENTS:] = times[guardians, WORK_EVENTS:]
    codes[children, WORK_EVENTS:] = codes[guardians, WORK_EVENTS:]
    free_from = np.where(attends[children], work_to[children] + BUFFER_HOURS, 0.0)
    for column in range(WORK_EVENTS, times.shape[1] - 1, 2):
        clash = times[children, column] < free_from
        times[children[clash], column : column + 2] = np.inf


def _add_outings(times, codes, role, sociability, busy_until, limit, params, day, rng) -> None:
    """Add errands and visits one after another; one that ends after `limit` is dropped."""
    count, activities = len(role), len(params.activities)
    starts = np.full((count, activities), np.inf)
    durations = np.zeros((count, activities))
    for index, activity in enumerate(params.activities):
        chance = np.minimum(activity.probability[role, day] * sociability, 1.0)
        window = activity.window[day]
        start = rng.uniform(window[0], window[1], count)
        durations[:, index] = np.clip(
            rng.gamma(
                activity.duration_shape, activity.duration_mean / activity.duration_shape, count
            ),
            MIN_OUTING_HOURS,
            MAX_OUTING_HOURS,
        )
        starts[:, index] = np.where(rng.random(count) < chance, start, np.inf)
    rows = np.arange(count)
    busy_until = busy_until + BUFFER_HOURS
    for rank, activity_index in enumerate(np.argsort(starts, axis=1, kind="stable").T):
        start = np.maximum(starts[rows, activity_index], busy_until)
        end = start + durations[rows, activity_index]
        valid = np.isfinite(starts[rows, activity_index]) & (end <= limit)
        column = WORK_EVENTS + 2 * rank
        times[valid, column] = start[valid]
        times[valid, column + 1] = end[valid]
        codes[:, column] = FIRST_ACTIVITY + activity_index
        busy_until = np.where(valid, end + BUFFER_HOURS, busy_until)


def destination_codes_at(schedule: DaySchedule, hour: float, max_events: int) -> np.ndarray:
    """Where everybody is heading at `hour` of the day, assuming they follow the schedule."""
    passed = (schedule.times[:, :max_events] <= hour).sum(axis=1)
    rows = np.arange(len(passed))
    return np.where(passed > 0, schedule.codes[rows, np.maximum(passed - 1, 0)], HOME)
