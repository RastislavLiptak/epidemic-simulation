"""Daily plans of individual people: when they leave, where they go, how long they stay.

The windows in `configs/town.toml` only describe the usual day of a role. Every person gets
their own times each day: a personal habit (some always come early, some late), daily
randomness, days off, overtime, a lunch break of their own length, and errands and visits at
their own times. A calendar closes schools and turns public holidays into days off.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from epidemic_sandbox.config.checks import (
    require,
    require_not_negative,
    require_positive,
    require_share,
)
from epidemic_sandbox.engine.person_kernels import sort_events
from epidemic_sandbox.world.population import ADULT_AGE, ROLES, WORKER

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
    """An errand or leisure activity: when it happens, how long, how likely, where.

    `window` [weekday/weekend, from/to] are the hours in which an outing can begin (opening hours).
    Without `start_mean` its beginning is uniform in them. With it, it is triangular: from the
    opening, most often at the hour that puts the most people there at `start_peak` (the hour of
    the day with the most people out on it, as travel surveys report it), and on average at
    `start_mean`, by role on weekdays (`start_mean_weekend`, `start_peak_weekend` for everybody at
    weekends). `duration_mean` is the mean length of an outing [weekday, weekend], gamma
    distributed with the shape `duration_shape`.
    """

    name: str
    kind: str
    window: np.ndarray
    probability: np.ndarray
    duration_mean: np.ndarray
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
    indoor: bool = True  # False for an activity outdoors: the air is not shared
    start_mean: np.ndarray | None = None  # by role, weekdays
    start_peak: np.ndarray | None = None  # by role, weekdays
    start_mean_weekend: float | None = None
    start_peak_weekend: float | None = None


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

    `alone_from`: the age from which a child may be at home without an adult for a while (after
    school, until the guardian is back). `alone_all_day_from`: the age from which a child may
    spend a whole day at home without an adult (when its school or kindergarten is closed, or it
    is ill). `hours`: the opening hours [from, to] of the kindergarten and of the after-school
    club, by the role code of the child (a child with `aftercare` stays there until the guardian
    is back, within these hours). `club_share`: the share of the school children young enough to
    be accompanied who attend the after-school club. `grandparent_care` [from age, to age, share]:
    the children looked after by their grandparents, `grandparent_hours` [from age, to age, mean]
    their hours a week, gamma distributed with the shape `grandparent_hours_shape`.
    """

    alone_from: int
    hours: dict[int, tuple[float, float]]
    club_share: float
    alone_all_day_from: int = 0
    grandparent_care: tuple = ()
    grandparent_hours: tuple = ()
    grandparent_hours_shape: float = 1.0


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
    trip_hours: np.ndarray | None = None  # the mean time of a trip by year of age, hours
    trip_shape: float = 1.0
    stay_in: np.ndarray | None = None  # by role: a weekday without outings, see `_add_outings`

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
    work) let small children be looked after (see `_look_after_children`). `care_host`,
    `care_choice` and `care_hours` are the grandparent care of children (see `grandparent_care`).
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
    care_host: np.ndarray | None = None  # see `GrandparentCare`
    care_choice: np.ndarray | None = None
    care_hours: np.ndarray | None = None


@dataclass
class DaySchedule:
    """The events of one day for everybody: when they happen and where each leads.

    `times` (hours of the day, infinity for none) are sorted per person and have a final
    infinite column; `codes` are the destinations after each event: HOME, WORK, LUNCH, or
    FIRST_ACTIVITY + the index of an activity. `choice` is today's favourite place per activity
    and `spot_x`, `spot_y` are where in a place a person is today, within the unit disk.
    `attends` marks who goes to their kindergarten, school or workplace today, `cared` the
    children who spend a part of the day at their grandparents' (an event of the visit activity).
    `travel` is the time of the trip that each event begins (hours; NaN for a move at the speed
    of the person's mode, such as the way to lunch and back).
    """

    times: np.ndarray
    codes: np.ndarray
    absent: np.ndarray
    choice: np.ndarray
    spot_x: np.ndarray
    spot_y: np.ndarray
    attends: np.ndarray | None = None
    cared: np.ndarray | None = None
    travel: np.ndarray | None = None


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
            _load_activity(name, values, raw["households"]["partner_age_gap_mean"])
            for name, values in raw["activities"].items()
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
        **_load_outings(raw["outings"]),
        **_load_trips(raw["travel"]),
    )


def _load_trips(raw: dict) -> dict:
    hours = np.full(MAX_AGE_OF_TABLES, np.nan)
    for start, end, minutes in raw["trip_minutes"]:
        require(0 <= start < end, "travel.trip_minutes: invalid age range")
        require_positive("travel.trip_minutes", minutes)
        hours[int(start) : int(end)] = minutes / 60
    require(not np.isnan(hours).any(), "travel.trip_minutes must cover every age")
    require_positive("travel.trip_shape", raw["trip_shape"])
    return {"trip_hours": hours, "trip_shape": float(raw["trip_shape"])}


def _load_outings(raw: dict) -> dict:
    stay_in = np.array([raw["stay_in"].get(role, 0.0) for role in ROLES])
    require(((stay_in >= 0) & (stay_in < 1)).all(), "outings.stay_in must be in [0, 1)")
    return {"stay_in": stay_in}


def _load_childcare(raw: dict) -> Childcare:
    require(raw["alone_from"] >= 0, "childcare.alone_from must not be negative")
    hours = {}
    for role, key in (("preschool", "kindergarten_hours"), ("school", "club_hours")):
        start, end = raw[key]
        require(0 <= start < end <= HOURS_PER_DAY, f"childcare.{key}: invalid hours")
        hours[ROLES.index(role)] = (float(start), float(end))
    require_share("childcare.club_share", raw["club_share"])
    require(
        raw["alone_all_day_from"] >= raw["alone_from"],
        "childcare.alone_all_day_from must not be below alone_from",
    )
    bands = {}
    for key in ("grandparent_care", "grandparent_hours"):
        bands[key] = tuple(tuple(float(value) for value in row) for row in raw.get(key, ()))
        for start, end, value in bands[key]:
            require(0 <= start < end, f"childcare.{key}: invalid age range")
            require(value >= 0, f"childcare.{key} must not be negative")
    for _, _, share in bands["grandparent_care"]:
        require_share("childcare.grandparent_care", share)
    shape = raw.get("grandparent_hours_shape", 1.0)
    require_positive("childcare.grandparent_hours_shape", shape)
    return Childcare(
        int(raw["alone_from"]),
        hours,
        float(raw["club_share"]),
        int(raw["alone_all_day_from"]),
        bands["grandparent_care"],
        bands["grandparent_hours"],
        float(shape),
    )


def grandparent_care(guardian, age, kin, hosts, params, rng):
    """Which children their grandparents look after, and how many hours a week (see `Childcare`).

    `kin` and `hosts` are the kinship and the household of everybody's visit favourites (children
    have their guardian's). Returns, for everybody, the household of the grandparents (-1 for
    none), its position among the visit favourites and the hours a week (0 for none).
    """
    count = len(age)
    host, choice = np.full(count, -1, np.int64), np.full(count, -1, np.int64)
    hours = np.zeros(count)
    care = params.childcare
    if care is None or kin is None or not care.grandparent_care:
        return host, choice, hours
    children = np.flatnonzero(guardian >= 0)
    elder = kin[children] == 1
    has = elder.any(axis=1)
    first = np.argmax(elder, axis=1)
    chance = np.zeros(len(children))
    for start, end, share in care.grandparent_care:
        band = (age[children] >= start) & (age[children] < end)
        available = has[band].mean() if band.any() else 0.0
        chance[band] = min(1.0, share / available) if available > 0 else 0.0
    mean = np.zeros(len(children))
    for start, end, value in care.grandparent_hours:
        mean[(age[children] >= start) & (age[children] < end)] = value
    shape = care.grandparent_hours_shape
    drawn = rng.random(len(children)) < chance
    weekly = rng.gamma(shape, np.maximum(mean, 1e-9) / shape)
    picked = has & drawn & (mean > 0)
    cared = children[picked]
    host[cared] = hosts[cared, first[picked]]
    choice[cared] = first[picked]
    hours[cared] = np.minimum(weekly[picked], 40.0)
    return host, choice, hours


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
    for key in ("duration_shape", "distance_scale"):
        require_positive(f"activity {name}: {key}", values[key])
    durations = values["duration_mean"]
    for day in DAY_TYPES if isinstance(durations, dict) else ():
        require_positive(f"activity {name}: duration_mean {day}", durations[day])
    if not isinstance(durations, dict):
        require_positive(f"activity {name}: duration_mean", durations)
    if "start_mean" in values:
        for key in ("start_mean", "start_peak"):
            require(set(values[key]) <= set(ROLES), f"activity {name}: {key} of an unknown role")
            for role, hour in values[key].items():
                require(0 <= hour < HOURS_PER_DAY, f"activity {name}: {key} {hour} of {role}")
            weekend = values[f"{key}_weekend"]
            require(0 <= weekend < HOURS_PER_DAY, f"activity {name}: {key}_weekend")
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
        require_positive(f"activity {name}: mothers_age_at_birth", values["mothers_age_at_birth"])
    if "staff_days" in values:
        require(
            0 < values["staff_days"] <= DAYS_PER_WEEK,
            f"activity {name}: staff_days must be between 0 and 7",
        )


def _load_activity(name: str, values: dict, partner_age_gap: float) -> ActivityParameters:
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
            "indoor",
        )
        if key in values
    }
    if "family_share" in values:
        shares = np.zeros(MAX_AGE_OF_TABLES)
        for start, end, share in values["family_share"]:
            shares[int(start) : int(end)] = share
        optional["family_share"] = shares
        # fathers are older than mothers by the partner gap; parents of both sexes are visited
        optional["generation_gap"] = values["mothers_age_at_birth"] + abs(partner_age_gap) / 2
    if "start_mean" in values:
        for key in ("start_mean", "start_peak"):
            optional[key] = np.array([values[key].get(role, np.nan) for role in ROLES], float)
            optional[f"{key}_weekend"] = float(values[f"{key}_weekend"])
    durations = values["duration_mean"]
    if not isinstance(durations, dict):
        durations = dict.fromkeys(DAY_TYPES, durations)  # the same on all days
    durations = [durations[day] for day in DAY_TYPES]
    return ActivityParameters(
        name=name,
        kind=values["kind"],
        window=window,
        probability=probability,
        duration_mean=np.array(durations, dtype=float),
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
    alive: np.ndarray | None = None,
) -> DaySchedule:
    """Draw everybody's day: the place of the day, a lunch break, errands and visits.

    `closed` are the kinds of places that are closed today. Whoever works at weekends takes
    weekdays off instead, and on weekend days only they have a place to go. `alive` marks the
    living (everybody if None): the dead go nowhere, and nobody counts on them to look after a
    child. The random numbers drawn do not depend on who is alive.
    """
    count, day = len(role), int(weekend)
    living = np.ones(count, np.bool_) if alive is None else np.asarray(alive, np.bool_)
    work = params.work_window[role, day]
    if traits.shift_window is not None:
        own = ~np.isnan(traits.shift_window[:, day, 0])
        work = np.where(own[:, None], traits.shift_window[:, day], work)
    lunch = params.lunch_window[role, day]
    spread = {key: params.variability[key][role] for key in VARIABILITY_KEYS}

    has_place = ~np.isnan(work[:, 0]) & ~np.isin(traits.place_kind, closed) & living
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
    care = _look_after_children(
        traits, role, has_place, absent, work_from, work_to, params, living, rng
    )
    absent, attends = care.absent, care.attends

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
    travel = _trip_times(traits, params, count, rng)
    leaves = np.maximum(work_from - travel[:, 0], 0.0)  # sets off so as to be there on time
    for column, (mask, when, code) in enumerate(
        (
            (attends, leaves, WORK),
            (has_lunch, lunch_from, LUNCH),
            (has_lunch, lunch_to, WORK),
            (attends, work_to, HOME),
        )
    ):
        times[mask, column] = when[mask]
        codes[:, column] = code
    travel[:, 1:3] = np.nan  # the lunch break is near the workplace

    late = attends & (work_from >= LATE_SHIFT_FROM)
    busy_until = np.where(attends & ~late, work_to + travel[:, WORK_EVENTS - 1], 0.0)
    busy_until = np.maximum(busy_until, care.minding)  # who looks after a child goes out later
    limit = np.where(late, leaves - BUFFER_HOURS, LATEST_END)
    sociability = np.where(living, traits.sociability, 0.0)  # the dead have no outings
    _add_outings(
        times, codes, role, sociability, busy_until, limit, params, day, rng, attends, travel
    )
    choice = np.stack(
        [rng.integers(0, activity.favourites, count) for activity in params.activities], axis=1
    )
    distance = np.sqrt(rng.random(count))
    angle = rng.random(count) * 2.0 * np.pi
    spot_x, spot_y = distance * np.cos(angle), distance * np.sin(angle)
    children = np.flatnonzero(traits.guardian >= 0)
    free_from = np.where(attends, work_to + BUFFER_HOURS, 0.0)
    free_from[care.children] = care.end + BUFFER_HOURS
    _accompany(times, codes, children, traits.guardian[children], free_from[children], travel)
    guardians = traits.guardian[children]
    choice[children] = choice[guardians]
    spot_x[children], spot_y[children] = spot_x[guardians], spot_y[guardians]
    cared = _go_to_grandparents(times, codes, choice, care, attends, params, travel)
    times, codes, travel = sort_events(times, codes, travel)
    return DaySchedule(
        times=times,
        codes=codes,
        absent=absent,
        choice=choice,
        spot_x=spot_x,
        spot_y=spot_y,
        attends=attends,
        cared=cared,
        travel=travel,
    )


def _trip_times(traits, params, count, rng) -> np.ndarray:
    """The time of every trip of everybody's day [person, event], hours: gamma with the mean of
    the person's age (`params.trip_hours`); 0 everywhere without travel times."""
    if params.trip_hours is None or traits.age is None:
        return np.zeros((count, params.max_events + 1))
    shape = params.trip_shape
    mean = params.trip_hours[np.minimum(traits.age, MAX_AGE_OF_TABLES - 1)]
    return rng.gamma(shape, (mean / shape)[:, None], (count, params.max_events + 1))


@dataclass
class ChildcareToday:
    """Who looks after the children today (see `_look_after_children`): who is absent from their
    place and who attends; `minding` the hour until which an adult stays at home to look after a
    child (no outing before it); `children` who spend the hours from `start` to `end` at their
    grandparents', whose household is their visit favourite `choice`."""

    absent: np.ndarray
    attends: np.ndarray
    minding: np.ndarray
    children: np.ndarray
    start: np.ndarray
    end: np.ndarray
    choice: np.ndarray


def _look_after_children(traits, role, has_place, absent, work_from, work_to, params, alive, rng):
    """Nobody leaves a small child alone (see `ChildcareToday`).

    A child with `aftercare` (kindergarten, after-school club) is brought before the guardian
    leaves for work and fetched when the guardian is back, within the opening hours of the place:
    `work_from` and `work_to` of the child change. A child needs an adult while the guardian works
    when it is younger than `alone_from` and not at its place, or younger than `alone_all_day_from`
    and does not go to its place today. Then a free living adult of the household looks after it
    and goes out only when the guardian is back; else the grandparents who look after the child
    (see `grandparent_care`) take it while the guardian is away, if one of them is free today;
    else the guardian stays at home (absent from work). Besides, a child with grandparent care goes
    to them on a day the guardian works, from the end of its kindergarten or school (or when the
    guardian leaves) until the guardian is back, as often as its hours of care a week say.
    """
    attends = has_place & ~absent
    care = params.childcare
    nobody = np.zeros(0, np.int64)
    minding = np.zeros(len(role))
    if care is None or traits.age is None or traits.household is None:
        return ChildcareToday(absent, attends, minding, nobody, np.zeros(0), np.zeros(0), nobody)
    children = np.flatnonzero(traits.guardian >= 0)
    guardian = traits.guardian[children]
    away_from = work_from[guardian] - BUFFER_HOURS
    away_to = work_to[guardian] + BUFFER_HOURS
    opens = np.full(len(children), np.inf)
    closes = np.full(len(children), -np.inf)
    for kind, (start, end) in care.hours.items():
        at_place = traits.place_kind[children] == kind
        opens[at_place], closes[at_place] = start, end
    hours_out = away_to - np.where(attends[children], work_to[children], away_from)
    with_grandparents = _grandparent_days(traits, children, attends, guardian, hours_out, rng)
    stays = attends[children] & traits.aftercare[children] & attends[guardian]
    stays &= ~with_grandparents  # the grandparents fetch the child instead
    kept = children[stays]
    work_from[kept] = np.maximum(np.minimum(work_from[kept], away_from[stays]), opens[stays])
    work_to[kept] = np.minimum(np.maximum(work_to[kept], away_to[stays]), closes[stays])
    covered = attends[children] & (work_from[children] <= away_from)
    covered &= work_to[children] >= away_to
    free = free_adults(traits, attends, alive)
    small = traits.age[children] < care.alone_from
    young = traits.age[children] < care.alone_all_day_from
    needs_adult = (small & ~covered) | (young & ~attends[children])
    gone = attends[guardian]
    household = traits.household[children]
    minded = needs_adult & gone & (free[household] > 0) & ~with_grandparents
    carer = _first_free_adult(traits, attends, alive, role)
    minders = carer[household[minded]]
    np.maximum.at(minding, minders, away_to[minded])
    grandparents = _free_grandparents(traits, attends, alive, role, children)
    taken = needs_adult & gone & ~minded & (grandparents >= 0)
    with_grandparents |= taken & (grandparents >= 0)
    with_grandparents &= gone & (grandparents >= 0)
    alone = needs_adult & gone & ~minded & ~with_grandparents
    absent = absent.copy()
    absent[guardian[alone]] = True
    attends = has_place & ~absent
    cared = with_grandparents & attends[guardian]
    start = np.where(attends[children], work_to[children], away_from)[cared]
    end = away_to[cared]
    keep = end > start + MIN_OUTING_HOURS
    np.maximum.at(minding, grandparents[cared][keep], end[keep])
    going = children[cared][keep]
    choice = traits.care_choice[going] if traits.care_choice is not None else nobody
    return ChildcareToday(absent, attends, minding, going, start[keep], end[keep], choice)


def _grandparent_days(traits, children, attends, guardian, hours_out, rng) -> np.ndarray:
    """Which of `children` go to their grandparents today as their weekly hours of care say: on a
    day the guardian works, with the chance hours a week / (5 * `hours_out`, the hours from the end
    of the child's day, or the guardian's leaving, to the guardian's return), at most 1."""
    draw = rng.random(len(children))  # drawn also without care, the same numbers either way
    if traits.care_hours is None:
        return np.zeros(len(children), np.bool_)
    hours = traits.care_hours[children]
    chance = hours / (5.0 * np.maximum(hours_out, MIN_OUTING_HOURS))
    return (hours > 0) & attends[guardian] & (hours_out > MIN_OUTING_HOURS) & (draw < chance)


def _first_free_adult(traits, attends, alive, role) -> np.ndarray:
    """For every household, a living adult of it who has no place to go today (-1 for none)."""
    free = np.flatnonzero(
        (traits.age >= ADULT_AGE) & ~attends & alive & (role != ROLES.index("resident"))
    )
    first = np.full(int(traits.household.max()) + 1, -1, np.int64)
    first[traits.household[free[::-1]]] = free[::-1]
    return first


def _free_grandparents(traits, attends, alive, role, children) -> np.ndarray:
    """For every child, a free living adult of the household of its grandparents who look after
    it (-1 if it has no such care or none of them is free today)."""
    if traits.care_host is None:
        return np.full(len(children), -1, np.int64)
    host = traits.care_host[children]
    first = _first_free_adult(traits, attends, alive, role)
    return np.where(host >= 0, first[np.maximum(host, 0)], -1)


def _go_to_grandparents(times, codes, choice, care, attends, params, travel) -> np.ndarray:
    """Write the hours at the grandparents' into the days of the children of `care`: from the end
    of their kindergarten or school (or the guardian's leaving) to the guardian's return, as an
    event of the visit activity to the favourite that is the grandparents' household. Such a child
    does not join another visit of its guardian that day. Returns who goes."""
    cared = np.zeros(times.shape[0], np.bool_)
    visits = [index for index, a in enumerate(params.activities) if a.kind == "visit"]
    if not len(care.children) or not visits:
        return cared
    children, visit = care.children, visits[0]
    code = FIRST_ACTIVITY + visit
    for column in range(WORK_EVENTS, times.shape[1] - 1, 2):
        joined = codes[children, column] == code
        times[children[joined], column : column + 2] = np.inf
    at_place = attends[children]
    # after kindergarten or school: the event of going home becomes going to the grandparents
    goes = children[at_place]
    codes[goes, WORK_EVENTS - 1] = code
    times[goes, 1], codes[goes, 1] = care.end[at_place], HOME
    travel[goes, 1] = travel[goes, WORK_EVENTS - 1]  # as long back as there
    # a day without its place: the first and the last event of a work day carry the hours
    stays = children[~at_place]
    times[stays, 0], codes[stays, 0] = care.start[~at_place], code
    times[stays, WORK_EVENTS - 1] = care.end[~at_place]
    codes[stays, WORK_EVENTS - 1] = HOME
    choice[children, visit] = care.choice
    cared[children] = True
    return cared


def free_adults(traits, attends: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """The living adults of each household who do not go to a place today (care homes have no
    children, so their residents do not matter)."""
    free_adult = (traits.age >= ADULT_AGE) & ~attends & alive
    return np.bincount(traits.household, weights=free_adult)


def _accompany(times, codes, children, guardians, free_from, travel=None) -> None:
    """Children go out with their guardian when they are free (from `free_from`, the end of their
    kindergarten or school, or of their time at the grandparents'): an outing that has begun by
    then is joined when the child is fetched, if enough of it is left."""
    times[children, WORK_EVENTS:] = times[guardians, WORK_EVENTS:]
    codes[children, WORK_EVENTS:] = codes[guardians, WORK_EVENTS:]
    if travel is not None:  # they travel together
        travel[children, WORK_EVENTS:] = travel[guardians, WORK_EVENTS:]
    for column in range(WORK_EVENTS, times.shape[1] - 1, 2):
        begins, ends = times[children, column], times[children, column + 1]
        late = begins < free_from
        joins = late & (ends - free_from >= MIN_OUTING_HOURS)
        times[children[joins], column] = free_from[joins]
        times[children[late & ~joins], column : column + 2] = np.inf


def _add_outings(
    times, codes, role, sociability, busy_until, limit, params, day, rng, attends=None, travel=None
) -> None:
    """Add errands and visits one after another; one that ends after `limit` is dropped. With
    `travel` (the times of the trips, see `DaySchedule`) an outing lasts its length at the place:
    the person leaves it the trip there plus the length after setting off, and is back home the
    trip back later.

    On weekdays a share `params.stay_in` of the people with no place to go today (`attends` False)
    does not leave home at all; the outings of the others of their role are likelier by as much, so
    that the time a role spends out stays what the probabilities say.
    """
    count, activities = len(role), len(params.activities)
    scale = np.ones(count)
    if day == 0 and params.stay_in is not None and attends is not None:
        free = ~attends
        stay_in = params.stay_in[role]
        stays = free & (rng.random(count) < stay_in)
        scale = np.where(free, np.where(stays, 0.0, 1.0 / (1.0 - stay_in)), 1.0)
    starts = np.full((count, activities), np.inf)
    durations = np.zeros((count, activities))
    for index, activity in enumerate(params.activities):
        chance = np.minimum(activity.probability[role, day] * sociability * scale, 1.0)
        start = _outing_starts(activity, role, day, rng)
        mean = activity.duration_mean[day]
        durations[:, index] = np.clip(
            rng.gamma(activity.duration_shape, mean / activity.duration_shape, count),
            MIN_OUTING_HOURS,
            MAX_OUTING_HOURS,
        )
        starts[:, index] = np.where(rng.random(count) < chance, start, np.inf)
    rows = np.arange(count)
    busy_until = busy_until + BUFFER_HOURS
    for rank, activity_index in enumerate(np.argsort(starts, axis=1, kind="stable").T):
        column = WORK_EVENTS + 2 * rank
        there = travel[:, column] if travel is not None else 0.0
        back = travel[:, column + 1] if travel is not None else 0.0
        start = np.maximum(starts[rows, activity_index], busy_until)
        end = start + there + durations[rows, activity_index]
        valid = np.isfinite(starts[rows, activity_index]) & (end + back <= limit)
        times[valid, column] = start[valid]
        times[valid, column + 1] = end[valid]
        codes[:, column] = FIRST_ACTIVITY + activity_index
        busy_until = np.where(valid, end + back + BUFFER_HOURS, busy_until)


def _outing_starts(activity, role, day, rng) -> np.ndarray:
    """When everybody would begin the activity today: uniform in its window, or triangular within
    it (see `ActivityParameters`): from the opening, with its mode where the outings that begin
    there are under way at the middle of the peak hour, and with the mean of the role."""
    low, high = activity.window[day]
    count = len(role)
    if activity.start_mean is None:
        return rng.uniform(low, high, count)
    if day == 0:
        mean, peak = activity.start_mean[role], activity.start_peak[role]
    else:
        mean = np.full(count, activity.start_mean_weekend)
        peak = np.full(count, activity.start_peak_weekend)
    known = ~np.isnan(mean)
    start = rng.uniform(low, high, count)  # a role without hours of its own
    mode = np.clip(peak[known] + 0.5 - activity.duration_mean[day] / 2, low, high)
    right = np.clip(3 * mean[known] - low - mode, mode + 1e-6, high)
    start[known] = rng.triangular(low, mode, right)
    return start


def destination_codes_at(schedule: DaySchedule, hour: float, max_events: int) -> np.ndarray:
    """Where everybody is heading at `hour` of the day, assuming they follow the schedule."""
    passed = (schedule.times[:, :max_events] <= hour).sum(axis=1)
    rows = np.arange(len(passed))
    return np.where(passed > 0, schedule.codes[rows, np.maximum(passed - 1, 0)], HOME)
