"""The simulated town: where people live, study, work and spend their free time, and when."""

import hashlib
import math
import tomllib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from epidemic_sandbox.config_checks import (
    require,
    require_not_negative,
    require_positive,
    require_share,
    require_weights,
)
from epidemic_sandbox.population import (
    ADULT_AGE,
    AGES,
    FEMALE,
    PRESCHOOL,
    RESIDENT,
    ROLES,
    SCHOOL,
    UNIVERSITY,
    WORKER,
    Population,
    PopulationParameters,
    assign_roles,
    generate_population,
    load_population_parameters,
)
from epidemic_sandbox.schedule import (
    DAYS_PER_WEEK,
    FIRST_ACTIVITY,
    HOURS_PER_DAY,
    LUNCH,
    MAX_LUNCH_HOURS,
    WEEKEND_FROM_DAY,
    WORK,
    DaySchedule,
    PersonalTraits,
    ScheduleParameters,
    destination_codes_at,
    draw_day_schedule,
    draw_habits,
    draw_weekend_work,
    load_schedule_parameters,
)
from epidemic_sandbox.spatial_kernels import build_grid, count_infectious_contacts

CANDIDATE_CHUNK = 100_000  # people processed at once when choosing places, a bound on memory
SAMPLE_POPULATION = 4000  # size of the sample town for the preview; large enough for stable means
PLACE_KINDS = {PRESCHOOL: "preschool", SCHOOL: "school", UNIVERSITY: "university", WORKER: "work"}
EDUCATION = (PRESCHOOL, SCHOOL, UNIVERSITY)
MODES = ("walk", "bike", "transit", "car")
# The age groups of the travel mix: children under 15 (the end of compulsory school age, as in the
# census age groups) and seniors from 65 (the age group CZSO uses for old people).
CHILD_UNDER, SENIOR_FROM = 15, 65
# Assumption: a teacher has finished university studies, which takes until about 24-25.
MIN_TEACHER_AGE = 25
LUNCH_ACTIVITY = "food"  # the lunch break is spent at a place of this activity near the workplace
METERS_PER_KILOMETER = 1000


@dataclass(frozen=True)
class PlaceParameters:
    """Places of one kind: mean size, spread of sizes, distance decay and floor space per person."""

    mean_size: float
    size_sigma: float
    distance_scale: float
    area_mean: float
    area_sigma: float
    class_size: int = 0
    class_age_span: int = 1
    class_oldest_age: int = 127
    pupils_per_teacher: float = 0.0
    staff_female_share: float = 0.5
    solo_share: float = 0.0
    common_share: float = 0.0

    def __post_init__(self):
        for name in ("mean_size", "distance_scale", "area_mean"):
            require_positive(name, getattr(self, name))
        for name in ("size_sigma", "area_sigma", "class_size", "pupils_per_teacher"):
            require_not_negative(name, getattr(self, name))
        require_share("staff_female_share", self.staff_female_share)
        require_share("common_share", self.common_share)


@dataclass(frozen=True)
class CareHomeParameters:
    """The buildings of the care homes and their staff (who lives there is in the population).

    `day_hours` are the hours in which the residents are up and about in the home; the rest of
    the day they are in their rooms.
    """

    area_mean: float
    staff_per_resident: float
    staff_female_share: float
    day_hours: tuple[float, float] = (0.0, 0.0)
    shift_hours: tuple[float, float] = (0.0, 0.0)
    weekly_hours: float = 0.0
    room_shares: np.ndarray = field(default_factory=lambda: np.ones(1))

    @property
    def attendance(self) -> float:
        """Chance that a member of the staff is on a shift on a given day."""
        start, end = self.shift_hours
        return min(1.0, self.weekly_hours / ((end - start) * DAYS_PER_WEEK))


@dataclass(frozen=True, eq=False)
class TravelParameters:
    """How people get around: speeds, how distance favours modes, and the mix by age group."""

    speed_sigma: float
    speeds: np.ndarray
    distances: np.ndarray
    mix: np.ndarray


@dataclass(frozen=True, eq=False)
class TownParameters:
    """Structure of the town, loaded from `configs/town.toml`."""

    fingerprint: str
    population: PopulationParameters
    schedule: ScheduleParameters
    places: dict[str, PlaceParameters]
    travel: TravelParameters
    core_share: float
    core_spread: float
    candidate_sites: int
    min_area_per_person: float
    max_site_contacts: float
    sociability_sigma: float
    dwell_hours: float
    care_home: CareHomeParameters
    going_out: np.ndarray
    accompanied_under: int


class Venues:
    """Hands out the numbers of the venues: a home, a class, a school, a workplace, a shop.

    People can infect each other only inside the same venue, so that walls separate homes from
    each other and from the places around them. The homes are numbered first, by household.
    """

    def __init__(self, households: int):
        self.count = households

    def allocate(self, count: int) -> int:
        """The first of `count` new venue numbers."""
        first = self.count
        self.count += int(count)
        return first


@dataclass
class Town:
    """Everybody's traits and fixed places.

    Activity places have a column per favourite: `activity_x`, `activity_y` are the middles of
    the places and `activity_radius` their sizes, since a person stands somewhere else each visit.
    `place_kind` is the role code of the kind of place somebody attends, also for its staff.
    `activity_staff` marks the workers whose workplace is a shop, restaurant or another place that
    people also visit (their work position is inside it).

    Venues (see `Venues`): a person's home is the venue of their household; `work_venue` is the
    class of a pupil or a teacher, the workplace of a worker (-1 without a place), `school_venue`
    the whole school of a pupil or a teacher (its corridors and canteen, -1 for everybody else),
    `activity_venue` the venue of each favourite place (a friend's home for a visit) and
    `lunch_venue` the place of the lunch break.
    """

    arena_size: float
    age: np.ndarray
    sex: np.ndarray
    household: np.ndarray
    role: np.ndarray
    home_x: np.ndarray
    home_y: np.ndarray
    work_x: np.ndarray
    work_y: np.ndarray
    work_people: np.ndarray
    place_kind: np.ndarray
    activity_x: list[np.ndarray]
    activity_y: list[np.ndarray]
    activity_radius: list[np.ndarray]
    activity_people: list[np.ndarray]
    lunch_x: np.ndarray
    lunch_y: np.ndarray
    lunch_radius: np.ndarray
    start_habit: np.ndarray
    end_habit: np.ndarray
    sociability: np.ndarray
    weekend_work: np.ndarray
    guardian: np.ndarray
    speed_kmh: np.ndarray
    activity_staff: np.ndarray | None = None
    work_center_x: np.ndarray | None = None
    work_center_y: np.ndarray | None = None
    work_radius: np.ndarray | None = None
    work_venue: np.ndarray | None = None
    school_venue: np.ndarray | None = None
    activity_venue: list[np.ndarray] | None = None
    lunch_venue: np.ndarray | None = None
    school_x: np.ndarray | None = None
    school_y: np.ndarray | None = None
    school_radius: np.ndarray | None = None
    home_center_x: np.ndarray | None = None
    home_center_y: np.ndarray | None = None
    home_radius: np.ndarray | None = None
    shift_window: np.ndarray | None = None
    shift_attendance: np.ndarray | None = None
    aftercare: np.ndarray | None = None
    room_venue: np.ndarray | None = None

    @property
    def traits(self) -> PersonalTraits:
        return PersonalTraits(
            self.start_habit,
            self.end_habit,
            self.sociability,
            self.weekend_work,
            self.guardian,
            self.place_kind,
            self.shift_window,
            self.shift_attendance,
            self.age,
            self.household,
            self.aftercare,
        )

    @property
    def at_school(self) -> np.ndarray:
        """Who spends the day at a kindergarten, school or university, pupils and staff."""
        return np.isin(self.place_kind, EDUCATION)


def load_town_parameters(path: Path) -> TownParameters:
    """Load and validate the town structure; a bad value raises ValueError naming the file."""
    text = path.read_text()
    raw = tomllib.loads(text)
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    try:
        population = load_population_parameters(raw, fingerprint, path.parent)
        schedule = load_schedule_parameters(raw)
        places = {kind: PlaceParameters(**values) for kind, values in raw["places"].items()}
        _check_places(places)
        layout = _check_layout(raw["layout"])
        travel = _load_travel(raw["travel"])
        care_home = _load_care_home(raw["care_homes"])
        going_out, accompanied_under = _load_outdoors(raw["outdoors"])
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return TownParameters(
        fingerprint=fingerprint,
        population=population,
        schedule=schedule,
        places=places,
        travel=travel,
        core_share=layout["core_share"],
        core_spread=layout["core_spread"],
        candidate_sites=layout["candidate_sites"],
        min_area_per_person=layout["min_area_per_person"],
        max_site_contacts=layout["max_site_contacts"],
        sociability_sigma=layout["sociability_sigma"],
        dwell_hours=layout["dwell_hours"],
        care_home=care_home,
        going_out=going_out,
        accompanied_under=accompanied_under,
    )


def _load_care_home(raw: dict) -> CareHomeParameters:
    require_positive("care_homes.area_mean", raw["area_mean"])
    require_not_negative("care_homes.staff_per_resident", raw["staff_per_resident"])
    require_share("care_homes.staff_female_share", raw["staff_female_share"])
    hours = {}
    for key in ("day_hours", "shift_hours"):
        start, end = raw[key]
        require(0 <= start < end <= HOURS_PER_DAY, f"care_homes.{key}: invalid hours")
        hours[key] = (float(start), float(end))
    require_positive("care_homes.weekly_hours", raw["weekly_hours"])
    require_weights("care_homes.room_bed_shares", raw["room_bed_shares"])
    shares = np.array(raw["room_bed_shares"], float)
    return CareHomeParameters(
        raw["area_mean"],
        raw["staff_per_resident"],
        raw["staff_female_share"],
        hours["day_hours"],
        hours["shift_hours"],
        raw["weekly_hours"],
        shares / shares.sum(),
    )


def _load_outdoors(raw: dict) -> tuple[np.ndarray, int]:
    """The chance by age of going out at all, and the age up to which children are accompanied."""
    going_out = np.ones(AGES)
    for start, end, share in raw["going_out"]:
        require(0 <= start < end <= AGES, "outdoors.going_out: invalid age range")
        require_share("outdoors.going_out", share)
        going_out[start:end] = share
    require(raw["accompanied_under"] >= 0, "outdoors.accompanied_under must not be negative")
    return going_out, raw["accompanied_under"]


def _check_places(places: dict[str, PlaceParameters]) -> None:
    require(
        set(places) == set(PLACE_KINDS.values()), f"places must be {sorted(PLACE_KINDS.values())}"
    )
    for kind in ("preschool", "school", "university"):
        require(places[kind].class_size >= 1, f"{kind}: class_size must be at least 1")
        require_positive(f"{kind}: pupils_per_teacher", places[kind].pupils_per_teacher)


def _check_layout(layout: dict) -> dict:
    require_share("core_share", layout["core_share"])
    require_positive("core_spread", layout["core_spread"])
    require(
        isinstance(layout["candidate_sites"], int) and layout["candidate_sites"] >= 1,
        "candidate_sites must be a whole number, at least 1",
    )
    require_positive("min_area_per_person", layout["min_area_per_person"])
    require_positive("max_site_contacts", layout["max_site_contacts"])
    require_not_negative("sociability_sigma", layout["sociability_sigma"])
    require_positive("dwell_hours", layout["dwell_hours"])
    return layout


def _load_travel(travel: dict) -> TravelParameters:
    mixes = np.array(
        [[travel["mix"][group][mode] for mode in MODES] for group in ("child", "adult", "senior")]
    )
    require((mixes >= 0).all() and (mixes.sum(axis=1) > 0).all(), "invalid travel mix")
    speeds = np.array([travel["speed"][mode] for mode in MODES], dtype=float)
    distances = np.array([travel[f"{mode}_distance"] for mode in MODES], dtype=float)
    require((speeds > 0).all(), "travel speeds must be positive")
    require((distances > 0).all(), "travel distances must be positive")
    require_not_negative("speed_sigma", travel["speed_sigma"])
    return TravelParameters(
        speed_sigma=travel["speed_sigma"],
        speeds=speeds,
        distances=distances,
        mix=mixes / mixes.sum(axis=1, keepdims=True),
    )


def build_town(population: int, arena_size: float, params: TownParameters, rng) -> Town:
    """Create people, households and the places they go to, all inside the square arena."""
    people = generate_population(population, params.population, rng)
    households = int(people.household.max()) + 1
    household_x, household_y = _scatter(households, arena_size, params, rng)
    home_radius = _home_radius(people, params)
    home_x, home_y = (
        np.clip(values, 0.0, arena_size)
        for values in _around(
            household_x[people.household], household_y[people.household], home_radius, rng
        )
    )
    venues = Venues(households)
    room_venue = np.full(population, -1, np.int64)
    residents = np.flatnonzero(people.resident)
    if len(residents):
        room, room_home = _rooms(people.household[residents], params.care_home.room_shares, rng)
        home_size = np.zeros(households)
        home_size[people.household[residents]] = home_radius[residents]
        room_x, room_y = _around(
            household_x[room_home], household_y[room_home], home_size[room_home], rng
        )
        home_x[residents], home_y[residents] = (
            np.clip(values, 0.0, arena_size)
            for values in _around(
                room_x[room], room_y[room], params.population.household_radius, rng
            )
        )
        room_venue[residents] = venues.allocate(len(room_home)) + room
    role = assign_roles(people, params.population, rng)

    work_x, work_y = home_x.copy(), home_y.copy()
    disc = (
        home_x.copy(),
        home_y.copy(),
        np.zeros(population),
    )  # centre and radius of the workplace
    work_people = np.zeros(population, np.int32)
    work_venue = np.full(population, -1, np.int64)
    school = (np.full(population, -1, np.int64), home_x.copy(), home_y.copy(), np.zeros(population))
    unassigned_workers = role == WORKER
    place_kind = role.copy()
    for role_code, kind in PLACE_KINDS.items():
        if role_code == WORKER:
            continue
        members = np.flatnonzero(role == role_code)
        if len(members):
            staff = _place_education(
                members,
                people,
                unassigned_workers,
                home_x,
                home_y,
                work_x,
                work_y,
                work_people,
                disc,
                arena_size,
                params.places[kind],
                params,
                rng,
                (venues, work_venue, school),
            )
            unassigned_workers[staff] = False
            place_kind[staff] = role_code
    carers, carers_venue = _place_care_staff(
        people,
        unassigned_workers,
        (household_x, household_y),
        home_radius,
        (home_x, home_y),
        (work_x, work_y),
        work_people,
        disc,
        arena_size,
        params,
        rng,
    )
    work_venue[carers] = carers_venue
    unassigned_workers[carers] = False
    place_kind[carers] = RESIDENT
    activity_places = []
    for activity in params.schedule.activities:
        if activity.kind == "visit":
            places, sites = (
                _friends_homes(people, home_x, home_y, arena_size, activity, params, rng),
                None,
            )
        else:
            places, sites = _activity_sites(
                home_x, home_y, people.age, arena_size, activity, params, rng, venues
            )
        activity_places.append((activity, places, sites))
    activity_staff = np.zeros(population, np.bool_)
    employed = int(np.count_nonzero(role == WORKER))
    for activity, places, sites in activity_places:
        if activity.staff_share > 0 and sites is not None:
            staff, staff_venue = _place_activity_staff(
                activity,
                places,
                int(np.floor(activity.staff_share * employed + rng.random())),
                people,
                unassigned_workers,
                (home_x, home_y),
                (work_x, work_y),
                work_people,
                disc,
                arena_size,
                params,
                rng,
            )
            unassigned_workers[staff] = False
            activity_staff[staff] = True
            work_venue[staff] = staff_venue
    workers = np.flatnonzero(unassigned_workers)
    if len(workers):
        (
            work_x[workers],
            work_y[workers],
            work_people[workers],
            disc[0][workers],
            disc[1][workers],
            disc[2][workers],
            work_venue[workers],
        ) = _place_in_sites(
            workers, home_x, home_y, arena_size, params.places["work"], params, rng, venues
        )

    guardian = _guardians(people, params.accompanied_under)
    activity_x, activity_y, activity_radius, activity_people = [], [], [], []
    activity_venue = []
    lunch_x, lunch_y, lunch_radius, lunch_venue = None, None, None, None
    for activity, places, sites in activity_places:
        if activity.name == LUNCH_ACTIVITY and sites is not None:
            chosen = _choose_sites(work_x, work_y, *sites[:-1], params, rng, sites[-1])
            center_x, center_y, site_radius = places[4][:3]
            lunch_x, lunch_y = center_x[chosen], center_y[chosen]
            lunch_radius = site_radius[chosen]
            lunch_venue = places[5][1] + chosen
        for values, array in zip(
            (activity_x, activity_y, activity_radius, activity_people, activity_venue),
            (*places[:4], places[5][0]),
            strict=True,
        ):
            values.append(_with_guardian(array, guardian))

    sociability = (
        np.exp(rng.normal(-(params.sociability_sigma**2) / 2, params.sociability_sigma, population))
        * params.going_out[people.age]
    )
    start_habit, end_habit = draw_habits(role, params.schedule, rng)
    weekend_work = draw_weekend_work(role, params.schedule, rng)
    weekend_work[np.isin(place_kind, EDUCATION)] = 0.0
    shift_window, shift_attendance = _shifts(
        population, carers, activity_places, activity_staff, work_venue, params
    )
    on_shift = ~np.isnan(shift_window[:, 0, 0])
    start_habit[on_shift] = 0.0  # a shift starts and ends on time
    end_habit[on_shift] = 0.0
    aftercare = _aftercare(people.age, role, params, rng)
    trip = np.hypot(work_x - home_x, work_y - home_y)
    errand = np.hypot(activity_x[0][:, 0] - home_x, activity_y[0][:, 0] - home_y)
    speed = _travel_speeds(people.age, np.where(trip > 0, trip, errand), params.travel, rng)
    speed = _with_guardian(speed, guardian)
    return Town(
        arena_size,
        people.age,
        people.sex,
        people.household,
        role,
        home_x,
        home_y,
        np.clip(work_x, 0.0, arena_size),
        np.clip(work_y, 0.0, arena_size),
        work_people,
        place_kind,
        activity_x,
        activity_y,
        activity_radius,
        activity_people,
        lunch_x,
        lunch_y,
        lunch_radius,
        start_habit,
        end_habit,
        sociability,
        weekend_work,
        guardian,
        speed,
        activity_staff,
        disc[0],
        disc[1],
        disc[2],
        work_venue,
        school[0],
        activity_venue,
        lunch_venue,
        school[1],
        school[2],
        school[3],
        household_x[people.household],
        household_y[people.household],
        home_radius,
        shift_window,
        shift_attendance,
        aftercare,
        room_venue,
    )


def _shifts(population, carers, activity_places, activity_staff, work_venue, params):
    """The working hours [person, weekday/weekend, start/end] and the chance of being at work
    [person, weekday/weekend] of the staff whose hours are those of their place (NaN for all
    others): care homes work in day shifts every day, shops and restaurants while they are open,
    and the places of the lunch break also over the lunch breaks of the workers around them.
    """
    window = np.full((population, 2, 2), np.nan)
    attendance = np.full((population, 2), np.nan)
    care = params.care_home
    window[carers] = care.shift_hours
    attendance[carers] = care.attendance
    for activity, places, sites in activity_places:
        if sites is None or activity.staff_share <= 0:
            continue
        first = places[5][1]
        staff = np.flatnonzero(
            activity_staff & (work_venue >= first) & (work_venue < first + len(places[4][0]))
        )
        hours = np.array(activity.window, float)  # [weekday/weekend, start/end]
        if activity.name == LUNCH_ACTIVITY:
            lunch = params.schedule.lunch_window  # [role, weekday/weekend, start/end]
            for day in range(2):
                if not np.isnan(lunch[:, day, 0]).all():
                    hours[day, 0] = min(hours[day, 0], np.nanmin(lunch[:, day, 0]))
                    latest = np.nanmax(lunch[:, day, 1]) + MAX_LUNCH_HOURS
                    hours[day, 1] = max(hours[day, 1], latest)
        window[staff] = hours
        attendance[staff] = activity.staff_days / DAYS_PER_WEEK
    return window, attendance


def _aftercare(age, role, params, rng) -> np.ndarray:
    """Who stays at kindergarten or in the after-school club until the guardian is back."""
    care = params.schedule.childcare
    if care is None:
        return np.zeros(len(age), np.bool_)
    young = (role == SCHOOL) & (age < params.accompanied_under)
    return (role == PRESCHOOL) | (young & (rng.random(len(age)) < care.club_share))


def destination_discs(
    town: Town, codes: np.ndarray, people: np.ndarray, schedule: DaySchedule
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Middle and radius of the place of each destination `codes` (a point at home): where a
    person moves about while they stay there."""
    at_place = codes == WORK
    x = np.where(at_place, town.work_center_x[people], town.home_x[people])
    y = np.where(at_place, town.work_center_y[people], town.home_y[people])
    radius = np.where(at_place, town.work_radius[people], 0.0)
    lunch = codes == LUNCH
    x[lunch], y[lunch] = town.lunch_x[people[lunch]], town.lunch_y[people[lunch]]
    radius[lunch] = town.lunch_radius[people[lunch]]
    for index in range(len(town.activity_x)):
        out = codes == FIRST_ACTIVITY + index
        if out.any():
            who = people[out]
            favourite = schedule.choice[who, index]
            x[out] = town.activity_x[index][who, favourite]
            y[out] = town.activity_y[index][who, favourite]
            radius[out] = town.activity_radius[index][who, favourite]
    return x, y, radius


def destination_coordinates(
    town: Town, codes: np.ndarray, people: np.ndarray, schedule: DaySchedule
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of the destinations `codes` of `people` under today's `schedule`."""
    at_place = codes == WORK
    x = np.where(at_place, town.work_x[people], town.home_x[people])
    y = np.where(at_place, town.work_y[people], town.home_y[people])
    lunch = codes == LUNCH
    if lunch.any():
        who = people[lunch]
        x[lunch] = town.lunch_x[who] + town.lunch_radius[who] * schedule.spot_x[who]
        y[lunch] = town.lunch_y[who] + town.lunch_radius[who] * schedule.spot_y[who]
    for index in range(len(town.activity_x)):
        out = codes == FIRST_ACTIVITY + index
        if out.any():
            who = people[out]
            favourite = schedule.choice[who, index]
            radius = town.activity_radius[index][who, favourite]
            x[out] = town.activity_x[index][who, favourite] + radius * schedule.spot_x[who]
            y[out] = town.activity_y[index][who, favourite] + radius * schedule.spot_y[who]
    return np.clip(x, 0.0, town.arena_size), np.clip(y, 0.0, town.arena_size)


def destination_venues(
    town: Town, codes: np.ndarray, people: np.ndarray, schedule: DaySchedule
) -> np.ndarray:
    """The venue (see `Venues`) of the destinations `codes` of `people` under today's `schedule`:
    the home of the household, the class or workplace, the place of lunch or of an activity."""
    at_work = codes == WORK
    venue = np.where(at_work, town.work_venue[people], town.household[people]).astype(np.int64)
    lunch = codes == LUNCH
    venue[lunch] = town.lunch_venue[people[lunch]]
    for index in range(len(town.activity_venue)):
        out = codes == FIRST_ACTIVITY + index
        if out.any():
            who = people[out]
            venue[out] = town.activity_venue[index][who, schedule.choice[who, index]]
    return venue


CACHED_SAMPLES = 8
CACHED_CONTACTS = 64
_SAMPLES: OrderedDict[tuple, tuple] = OrderedDict()
_CONTACTS: OrderedDict[tuple, float] = OrderedDict()


def _cached(cache: OrderedDict, limit: int, key: tuple, make: Callable):
    """The value for `key`, made once; only the `limit` most recently used values are kept."""
    if key in cache:
        cache.move_to_end(key)
    else:
        cache[key] = make()
        while len(cache) > limit:
            cache.popitem(last=False)
    return cache[key]


def _sample_town(world: dict, params: TownParameters):
    """A cached sample of the town with the same density and its days (a few thousand people)."""
    population = min(world["population"], SAMPLE_POPULATION)
    arena = world["arena_size"] * math.sqrt(population / world["population"])

    def make():
        rng = np.random.default_rng(0)
        town = build_town(population, arena, params, rng)
        days = {
            weekend: draw_day_schedule(town.role, town.traits, params.schedule, weekend, rng)
            for weekend in (False, True)
        }
        return town, days, arena

    return _cached(_SAMPLES, CACHED_SAMPLES, (population, arena, params.fingerprint), make)


def population_statistics(world: dict, params: TownParameters) -> dict:
    """Age, household and role statistics of a sample of the town."""
    town, _, _ = _sample_town(world, params)
    count = len(town.age)
    return {
        "mean_age": float(town.age.mean()),
        "age_shares": _age_shares(town.age, params).tolist(),
        "mean_household_size": _mean_household_size(town),
        "role_shares": dict(
            zip(ROLES, (np.bincount(town.role, minlength=len(ROLES)) / count).tolist(), strict=True)
        ),
    }


def mean_contacts(world: dict, params: TownParameters, radius: float) -> float:
    """Average number of people within `radius` of a person over a week of planned days."""
    key = (world["population"], world["arena_size"], radius, params.fingerprint)
    return _cached(
        _CONTACTS, CACHED_CONTACTS, key, lambda: _measure_contacts(world, params, radius)
    )


def _measure_contacts(world: dict, params: TownParameters, radius: float) -> float:
    town, days, arena = _sample_town(world, params)
    count = len(town.age)
    everybody = np.ones(count, np.bool_)
    weights = np.ones(count, np.float32)
    cells = max(1, min(int(arena // radius), 2 * math.isqrt(count)))
    people = np.arange(count)
    total = 0.0
    for weekend, weight in (
        (False, WEEKEND_FROM_DAY),
        (True, DAYS_PER_WEEK - WEEKEND_FROM_DAY),
    ):
        schedule = days[weekend]
        for hour in range(HOURS_PER_DAY):
            codes = destination_codes_at(schedule, hour + 0.5, params.schedule.max_events)
            x, y = destination_coordinates(town, codes, people, schedule)
            venue = destination_venues(town, codes, people, schedule)
            found = count_infectious_contacts(
                x,
                y,
                everybody,
                weights,
                venue,
                *build_grid(x, y, arena, cells),
                cells,
                float(radius),
            )
            others = found.mean() - 1.0  # everybody counts as in contact with themselves
            total += weight * others / (HOURS_PER_DAY * DAYS_PER_WEEK)
    return float(total)


def _mean_household_size(town: Town) -> float:
    """People per private household; a care home is not one."""
    private = town.household[town.role != RESIDENT]
    return float(len(private) / (private.max() + 1))


def _home_radius(people: Population, params: TownParameters) -> np.ndarray:
    """Radius around the home point: the rooms of a household or the building of a care home."""
    radius = np.full(len(people.age), params.population.household_radius)
    residents = people.household[people.resident]
    if len(residents):
        _, home, in_home = np.unique(residents, return_inverse=True, return_counts=True)
        radius[people.resident] = np.sqrt(in_home[home] * params.care_home.area_mean / math.pi)
    return radius


def _rooms(home: np.ndarray, bed_shares: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Rooms of the residents of care homes: the room of each resident (numbered from 0) and the
    home of each room. Rooms have 1, 2, 3, ... beds, with the share of the beds in each size
    given by `bed_shares`; the last room of a home may have fewer beds than drawn."""
    sizes = np.arange(1, len(bed_shares) + 1)
    room_chance = bed_shares / sizes / (bed_shares / sizes).sum()
    room = np.empty(len(home), np.int64)
    room_home = []
    for each in np.unique(home):
        members = rng.permutation(np.flatnonzero(home == each))
        drawn = rng.choice(sizes, len(members), p=room_chance)
        ends = np.minimum(np.cumsum(drawn), len(members))
        rooms = int(np.searchsorted(ends, len(members)) + 1)
        starts = np.concatenate([[0], ends[: rooms - 1]])
        for offset, (start, end) in enumerate(zip(starts, ends[:rooms], strict=True)):
            room[members[start:end]] = len(room_home) + offset
        room_home += [each] * rooms
    return room, np.array(room_home, np.int64)


def _guardians(people: Population, accompanied_under: int) -> np.ndarray:
    """The adult who goes out with each small child, -1 for everybody else.

    The youngest woman of the household, the mother as a rule, or else the youngest man.
    """
    guardian = np.full(len(people.age), -1, np.int64)
    adults = np.flatnonzero((people.age >= ADULT_AGE) & ~people.resident)
    order = adults[
        np.lexsort((people.age[adults], people.sex[adults] != FEMALE, people.household[adults]))
    ]
    households, first = np.unique(people.household[order], return_index=True)
    carer = np.full(int(people.household.max()) + 1, -1, np.int64)
    carer[households] = order[first]
    children = (people.age < accompanied_under) & ~people.resident
    guardian[children] = carer[people.household[children]]
    return guardian


def _with_guardian(values: np.ndarray, guardian: np.ndarray) -> np.ndarray:
    """A copy of a property of people in which children have the value of their guardian."""
    children = np.flatnonzero(guardian >= 0)
    copied = values.copy()
    copied[children] = values[guardian[children]]
    return copied


def _age_shares(age: np.ndarray, params: TownParameters) -> np.ndarray:
    """Share of people in each age group (the last group collects everybody older)."""
    groups = len(params.population.age_shares)
    index = np.minimum(age // params.population.age_group_width, groups - 1)
    return np.bincount(index, minlength=groups) / len(age)


def _scatter(
    count: int, arena_size: float, params: TownParameters, rng
) -> tuple[np.ndarray, np.ndarray]:
    """Random points, a share of them concentrated in a dense core in the middle."""
    in_core = rng.random(count) < params.core_share
    spread = params.core_spread * arena_size
    x = np.where(in_core, rng.normal(arena_size / 2, spread, count), rng.random(count) * arena_size)
    y = np.where(in_core, rng.normal(arena_size / 2, spread, count), rng.random(count) * arena_size)
    return np.clip(x, 0.0, arena_size), np.clip(y, 0.0, arena_size)


def _around(center_x, center_y, radius, rng) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random points in a disk of `radius` around each center."""
    distance = radius * np.sqrt(rng.random(len(center_x)))
    angle = rng.random(len(center_x)) * 2 * math.pi
    return center_x + distance * np.cos(angle), center_y + distance * np.sin(angle)


def _size_cdf(site_count: int, sigma: float, rng) -> np.ndarray:
    weights = rng.lognormal(0.0, sigma, site_count)
    return np.cumsum(weights) / weights.sum()


def _site_areas(site_count: int, mean: float, sigma: float, rng) -> np.ndarray:
    """Floor space per person of every place: log-normal around `mean`."""
    return mean * np.exp(rng.normal(-(sigma**2) / 2, sigma, site_count))


def _pick(weight: np.ndarray, rng) -> np.ndarray:
    """Index of one column per row, chosen with probability proportional to the weights."""
    threshold = rng.random(len(weight)) * weight.sum(axis=1)
    return np.minimum((weight.cumsum(axis=1) < threshold[:, None]).sum(axis=1), weight.shape[1] - 1)


def _choose_sites(
    home_x, home_y, site_x, site_y, cdf, scale, params, rng, affinity=None
) -> np.ndarray:
    """Gravity-like choice: big places and near places are more likely.

    Each person gets a few candidate places drawn by size, then picks one weighted by distance.
    `affinity` is (age of the people, typical age at each place, age scale): places favour the
    ages that typically visit them.
    """
    chosen = np.empty(len(home_x), np.int64)
    candidates = params.candidate_sites
    for start in range(0, len(home_x), CANDIDATE_CHUNK):
        block = slice(start, start + CANDIDATE_CHUNK)
        count = len(home_x[block])
        candidate = np.minimum(
            np.searchsorted(cdf, rng.random((count, candidates))), len(site_x) - 1
        )
        distance = np.hypot(
            site_x[candidate] - home_x[block, None], site_y[candidate] - home_y[block, None]
        )
        weight = np.exp(-distance / scale)
        if affinity is not None:
            age, site_age, age_scale = affinity
            weight = weight * np.exp(-np.abs(age[block, None] - site_age[candidate]) / age_scale)
        chosen[block] = candidate[np.arange(count), _pick(weight, rng)]
    return chosen


def _place_in_sites(members, home_x, home_y, arena_size, place, params, rng, venues: Venues):
    """Positions of `members` at their assigned places, how many people share each, and the
    venue of each.

    A share of them works alone, at a place of their own.
    """
    x, y, people, center_x, center_y, radius = (np.empty(len(members)) for _ in range(6))
    venue = np.empty(len(members), np.int64)
    solo = rng.random(len(members)) < place.solo_share
    alone = members[solo]
    x[solo], y[solo] = _around(home_x[alone], home_y[alone], arena_size * place.distance_scale, rng)
    center_x[solo], center_y[solo], radius[solo] = x[solo], y[solo], 0.0
    people[solo] = 1
    venue[solo] = venues.allocate(len(alone)) + np.arange(len(alone))
    (
        x[~solo],
        y[~solo],
        people[~solo],
        center_x[~solo],
        center_y[~solo],
        radius[~solo],
        site,
    ) = _place_in_shared_sites(members[~solo], home_x, home_y, arena_size, place, params, rng)
    venue[~solo] = venues.allocate(site.max() + 1 if len(site) else 0) + site
    return x, y, people.astype(np.int32), center_x, center_y, radius, venue


def _place_in_shared_sites(members, home_x, home_y, arena_size, place, params, rng):
    site_count = max(1, round(len(members) / place.mean_size))
    site_x, site_y = _scatter(site_count, arena_size, params, rng)
    cdf = _size_cdf(site_count, place.size_sigma, rng)
    site = _choose_sites(
        home_x[members],
        home_y[members],
        site_x,
        site_y,
        cdf,
        arena_size * place.distance_scale,
        params,
        rng,
    )
    people = np.bincount(site, minlength=site_count)
    area = _site_areas(site_count, place.area_mean, place.area_sigma, rng)
    site_radius = np.sqrt(people * area / math.pi)
    x, y = _around(site_x[site], site_y[site], site_radius[site], rng)
    return x, y, people[site].astype(np.int32), site_x[site], site_y[site], site_radius[site], site


def _recruit(slots, pool, people, homes, female_share, scale, params, rng) -> np.ndarray:
    """A person for each staff position of `slots` (x, y): near it and of the usual sex.

    Everybody picks among a few candidates like people pick their places, and whoever was picked
    twice goes to one position only, so a few rounds fill all of them or use up the pool.
    """
    slot_x, slot_y = slots
    home_x, home_y = homes
    chosen = np.full(len(slot_x), -1, np.int64)
    free = pool.copy()
    women = (people.sex[free] == FEMALE).mean() if len(free) else 0.5
    favour = np.array([(1 - female_share) / max(1 - women, 1e-9), female_share / max(women, 1e-9)])
    open_slots = rng.permutation(len(slot_x))
    while len(open_slots) and len(free):
        candidate = free[rng.integers(0, len(free), (len(open_slots), params.candidate_sites))]
        distance = np.hypot(
            home_x[candidate] - slot_x[open_slots, None],
            home_y[candidate] - slot_y[open_slots, None],
        )
        weight = np.exp(-distance / scale) * favour[(people.sex[candidate] == FEMALE).astype(int)]
        picked = candidate[np.arange(len(open_slots)), _pick(weight + 1e-300, rng)]
        taken, first = np.unique(picked, return_index=True)
        chosen[open_slots[first]] = taken
        free = free[~np.isin(free, taken)]
        open_slots = np.delete(open_slots, first)
    return chosen


def _place_care_staff(
    people, worker_pool, centers, radius, homes, work, work_people, disc, arena_size, params, rng
) -> np.ndarray:
    """Staff of the care homes, from the workers near each; returns who they are and the venue
    they work in, the home of the residents."""
    residents = np.flatnonzero(people.resident)
    if not len(residents):
        return residents, residents
    home_ids, in_home = np.unique(people.household[residents], return_counts=True)
    care = params.care_home
    # enough staff for `staff_per_resident` to be on the day shift on any day
    wanted = np.floor(
        care.staff_per_resident / care.attendance * in_home + rng.random(len(in_home))
    ).astype(np.int64)
    of_home = np.repeat(np.arange(len(in_home)), wanted)
    center_x, center_y = centers[0][home_ids], centers[1][home_ids]
    staff = _recruit(
        (center_x[of_home], center_y[of_home]),
        np.flatnonzero(worker_pool & (people.age >= ADULT_AGE)),
        people,
        homes,
        care.staff_female_share,
        arena_size * params.places["work"].distance_scale,
        params,
        rng,
    )
    hired = staff >= 0
    staff, of_home = staff[hired], of_home[hired]
    size = radius[residents[np.searchsorted(people.household[residents], home_ids)]]
    work[0][staff], work[1][staff] = _around(
        center_x[of_home], center_y[of_home], size[of_home], rng
    )
    work_people[staff] = in_home[of_home]
    disc[0][staff], disc[1][staff], disc[2][staff] = (
        center_x[of_home],
        center_y[of_home],
        size[of_home],
    )
    return staff, home_ids[of_home].astype(np.int64)


def _place_education(
    students,
    people,
    worker_pool,
    home_x,
    home_y,
    work_x,
    work_y,
    work_people,
    disc,
    arena_size,
    place,
    params,
    rng,
    venue_info,
):
    """Place students in classes of one age within schools, with teachers recruited from workers.

    Writes the positions into `work_x`, `work_y`, `work_people`, the venues (a class each, and
    the whole school for the breaks) into `venue_info` = (venues, work_venue, school) and returns
    the teachers.
    """
    venues, work_venue, school = venue_info
    age = people.age
    site_count = max(1, round(len(students) / place.mean_size))
    site_x, site_y = _scatter(site_count, arena_size, params, rng)
    cdf = _size_cdf(site_count, place.size_sigma, rng)
    site = _choose_sites(
        home_x[students],
        home_y[students],
        site_x,
        site_y,
        cdf,
        arena_size * place.distance_scale,
        params,
        rng,
    )
    in_site = np.bincount(site, minlength=site_count)
    area = _site_areas(site_count, place.area_mean, place.area_sigma, rng)
    site_radius = np.sqrt(in_site * area / math.pi)

    group = site * 128 + np.minimum(age[students], place.class_oldest_age) // place.class_age_span
    order = np.lexsort((rng.random(len(students)), group))
    sorted_group = group[order]
    first_of_group = np.searchsorted(sorted_group, sorted_group)
    in_group = np.searchsorted(sorted_group, sorted_group, side="right") - first_of_group
    classes_of_group = np.maximum(np.rint(in_group / place.class_size).astype(np.int64), 1)
    class_in_group = np.empty(len(students), np.int64)
    class_in_group[order] = (
        (np.arange(len(students)) - first_of_group) * classes_of_group // in_group
    )
    classes, class_of = np.unique(group * 4096 + class_in_group, return_inverse=True)
    class_site = (classes // 4096) // 128
    class_students = np.bincount(class_of)
    center_x, center_y = _around(
        site_x[class_site], site_y[class_site], site_radius[class_site], rng
    )
    class_radius = np.sqrt(class_students * area[class_site] / math.pi)
    work_x[students], work_y[students] = _around(
        center_x[class_of], center_y[class_of], class_radius[class_of], rng
    )
    work_people[students] = in_site[site]
    disc[0][students], disc[1][students], disc[2][students] = (
        center_x[class_of],
        center_y[class_of],
        class_radius[class_of],
    )

    # Every class is taught, so a small class has a teacher too; larger ones get teachers in the
    # published proportion.
    staff_wanted = np.maximum(
        np.floor(class_students / place.pupils_per_teacher + rng.random(len(classes))), 1
    ).astype(np.int64)
    staff_class = np.repeat(np.arange(len(classes)), staff_wanted)
    teachers = _recruit(
        (center_x[staff_class], center_y[staff_class]),
        np.flatnonzero(worker_pool & (age >= MIN_TEACHER_AGE)),
        people,
        (home_x, home_y),
        place.staff_female_share,
        arena_size * params.places["work"].distance_scale,
        params,
        rng,
    )
    hired = teachers >= 0
    teachers, staff_class = teachers[hired], staff_class[hired]
    work_x[teachers], work_y[teachers] = _around(
        center_x[staff_class], center_y[staff_class], class_radius[staff_class], rng
    )
    work_people[teachers] = in_site[class_site[staff_class]]
    disc[0][teachers], disc[1][teachers], disc[2][teachers] = (
        center_x[staff_class],
        center_y[staff_class],
        class_radius[staff_class],
    )
    first_class = venues.allocate(len(classes))
    first_school = venues.allocate(site_count)
    for members, of_class in ((students, class_of), (teachers, staff_class)):
        work_venue[members] = first_class + of_class
        site_of = class_site[of_class]
        school[0][members] = first_school + site_of
        school[1][members], school[2][members] = site_x[site_of], site_y[site_of]
        school[3][members] = site_radius[site_of]
    return teachers


def _place_activity_staff(
    activity,
    places,
    wanted,
    people,
    worker_pool,
    homes,
    work,
    work_people,
    disc,
    arena_size,
    params,
    rng,
) -> np.ndarray:
    """Workers of the shops, restaurants and other places of an activity; returns who they are.

    The positions go to the places in proportion to their regular visitors (a place with twice the
    customers employs twice the staff) and the staff are recruited from the workers near each, as
    teachers are. They stand inside the place, among the visitors.
    """
    site_x, site_y, radius, regulars = places[4]
    weight = regulars / regulars.sum()
    slot_site = np.minimum(np.searchsorted(np.cumsum(weight), rng.random(wanted)), len(site_x) - 1)
    pool = np.flatnonzero(worker_pool & (people.age >= ADULT_AGE))
    staff = _recruit(
        (site_x[slot_site], site_y[slot_site]),
        pool,
        people,
        homes,
        float((people.sex[pool] == FEMALE).mean()) if len(pool) else 0.5,
        arena_size * params.places["work"].distance_scale,
        params,
        rng,
    )
    hired = staff >= 0
    staff, slot_site = staff[hired], slot_site[hired]
    work[0][staff], work[1][staff] = _around(
        site_x[slot_site], site_y[slot_site], radius[slot_site], rng
    )
    work_people[staff] = np.maximum(np.rint(regulars[slot_site] * activity.presence_share), 1)
    disc[0][staff], disc[1][staff], disc[2][staff] = (
        site_x[slot_site],
        site_y[slot_site],
        radius[slot_site],
    )
    return staff, places[5][1] + slot_site


def _activity_sites(home_x, home_y, age, arena_size, activity, params, rng, venues: Venues):
    """Favourite places of an activity for everybody.

    Returns (middles x, y and radii by person and favourite, regulars, the places themselves,
    (venues by person and favourite, the venue of the first place)) and the arguments of a
    further choice among the places.
    """
    count, favourites = len(home_x), activity.favourites
    site_count = max(3, round(count * favourites / activity.mean_regulars))
    site_x, site_y = _scatter(site_count, arena_size, params, rng)
    cdf = _size_cdf(site_count, activity.size_sigma, rng)
    scale = arena_size * activity.distance_scale
    affinity = (age, rng.choice(age, site_count), activity.age_scale)
    chosen = np.stack(
        [
            _choose_sites(home_x, home_y, site_x, site_y, cdf, scale, params, rng, affinity)
            for _ in range(favourites)
        ],
        axis=1,
    )
    regulars = np.bincount(chosen.ravel(), minlength=site_count)
    area = _site_areas(site_count, activity.area_mean, activity.area_sigma, rng)
    radius = np.sqrt(np.maximum(regulars * activity.presence_share, 1.0) * area / math.pi)
    first_venue = venues.allocate(site_count)
    places = (
        site_x[chosen],
        site_y[chosen],
        radius[chosen],
        regulars[chosen].astype(np.int32),
        (site_x, site_y, radius, regulars),
        (first_venue + chosen, first_venue),
    )
    return places, (site_x, site_y, cdf, scale, affinity)


def _friends_homes(people, home_x, home_y, arena_size, activity, params, rng):
    """Homes of everybody's friends and relatives: nearby, of other households.

    A favourite is a relative of the next generation up or down (a parent, a grown-up child) with
    the chance `family_share`: about `generation_gap` years older or younger; otherwise a friend of
    a similar age. The venue of a visit is the home of the friend's household.
    """
    age, household = people.age, people.household
    count, favourites = len(home_x), activity.favourites
    x, y = np.empty((count, favourites)), np.empty((count, favourites))
    host = np.empty((count, favourites), np.int64)
    candidates = params.candidate_sites
    scale = arena_size * activity.distance_scale
    for favourite in range(favourites):
        friend = np.empty(count, np.int64)
        for start in range(0, count, CANDIDATE_CHUNK):
            block = slice(start, start + CANDIDATE_CHUNK)
            persons = np.arange(count)[block]
            candidate = rng.integers(0, count, (len(persons), candidates))
            distance = np.hypot(
                home_x[candidate] - home_x[block, None], home_y[candidate] - home_y[block, None]
            )
            gap = np.abs(age[candidate] - age[block, None])
            share = 0.0 if activity.family_share is None else activity.family_share[age[block]]
            family = rng.random(len(persons)) < share
            gap = np.where(family[:, None], np.abs(gap - activity.generation_gap), gap)
            weight = np.exp(-distance / scale - gap / activity.age_scale) * (
                household[candidate] != household[block, None]
            )
            friend[block] = candidate[np.arange(len(persons)), _pick(weight + 1e-300, rng)]
        x[:, favourite], y[:, favourite] = home_x[friend], home_y[friend]
        host[:, favourite] = household[friend]
    radius = np.full((count, favourites), params.population.household_radius)
    return x, y, radius, np.zeros((count, favourites), np.int32), None, (host, None)


def _travel_speeds(age, trip_distance, travel: TravelParameters, rng) -> np.ndarray:
    """Speed (km/h) of everybody: the mode suits their age and the length of their usual trip."""
    group = np.where(age < CHILD_UNDER, 0, np.where(age >= SENIOR_FROM, 2, 1))
    distance = trip_distance[:, None]
    walk, bike, transit, car = travel.distances
    suits = np.hstack(
        [
            np.exp(-distance / walk),
            np.exp(-distance / bike),
            1 - np.exp(-distance / transit),
            1 - np.exp(-distance / car),
        ]
    )
    weight = travel.mix[group] * suits + 1e-12
    mode = np.minimum(
        (weight.cumsum(axis=1) < (rng.random(len(age)) * weight.sum(axis=1))[:, None]).sum(axis=1),
        3,
    )
    return travel.speeds[mode] * np.exp(rng.normal(0.0, travel.speed_sigma, len(age)))
