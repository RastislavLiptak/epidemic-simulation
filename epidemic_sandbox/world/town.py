"""The simulated town: where people live, study, work and spend their free time, and when."""

import hashlib
import math
import tomllib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from epidemic_sandbox.config.checks import (
    require,
    require_not_negative,
    require_positive,
    require_share,
    require_weights,
)
from epidemic_sandbox.engine.spatial_kernels import (
    build_grid,
    count_infectious_contacts,
    locate_in_cdf,
)
from epidemic_sandbox.world.dwellings import DwellingParameters, layout, load_dwelling_parameters
from epidemic_sandbox.world.interiors import (
    Rotation,
    SeatLayout,
    classes,
    desks,
    ranks_within,
    ring_positions,
    seats,
    split_evenly,
)
from epidemic_sandbox.world.population import (
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
from epidemic_sandbox.world.schedule import (
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

CANDIDATE_CHUNK = 100_000  # people processed at once when choosing places, a bound on memory
SAMPLE_POPULATION = 4000  # size of the sample town for the preview; large enough for stable means
# The kinds of places by the role of the people who attend them; pupils go to a basic or a
# secondary school (see `PopulationParameters.secondary`).
PLACE_KINDS = {PRESCHOOL: "preschool", SCHOOL: "basic", UNIVERSITY: "university", WORKER: "work"}
SCHOOL_KINDS = ("basic", "secondary")
EDUCATION_KINDS = ("preschool", "basic", "secondary", "university")
EDUCATION = (PRESCHOOL, SCHOOL, UNIVERSITY)
MODES = ("walk", "bike", "transit", "car")
# the modes of a person: those of MODES, a car as the driver or as a passenger
WALK, BIKE, TRANSIT, CAR_DRIVER, CAR_PASSENGER = range(5)
TRAVEL_MODES = ("walk", "bike", "transit", "car_driver", "car_passenger")
DRIVING_AGE = 18  # Source: a driving licence for a car from 18 (Act 361/2000 Coll., section 83)
# Source: a teacher of a kindergarten needs a secondary school of pedagogy, which ends at 19 (Act
# 563/2004 Coll., section 6); the others a master's degree, until about 24 (section 7).
MIN_TEACHER_AGE = {"preschool": 19, "basic": 24, "secondary": 24, "university": 24}
LUNCH_ACTIVITY = "food"  # the lunch break is spent at a place of this activity near the workplace
METERS_PER_KILOMETER = 1000
# Source: 40 hours a week (Labour Code 262/2006 Coll., section 79(1)) over five days: the shift of
# the staff of shops and restaurants, which open longer than one shift.
SHIFT_HOURS = 8.0
# Assumption: the places of the lunch break open an hour before it to prepare the lunch.
OPEN_BEFORE_LUNCH_HOURS = 1.0


@dataclass(frozen=True)
class PlaceParameters:
    """Places of one kind: mean size, spread of sizes, distance decay and floor space per person.

    Inside, a place has rooms (the classes of a school, the rooms of a workplace) and a common
    area, where its people spend `common_share` of their stay. `seats` (pair gap, column gap, row
    gap; m) gives the pupils fixed seats at double desks, `desk_spacing` (m) the workers fixed
    desks in a grid; without them people move about their room. `dwell_hours` is the mean time at
    one spot (or at the seat) before moving. Schools: a class takes the pupils of `class_age_span`
    years (together with the next years while fewer than `min_class_size`); classes whose pupils
    are younger than `class_teacher_until` have their own teacher, the others are taught by
    teachers who go from class to class; `staff_hours` are the teachers' hours at school,
    `staff_shifts` the shifts in which several teachers of a class take turns.
    """

    mean_size: float
    size_sigma: float
    distance_scale: float
    area_mean: float
    area_sigma: float
    class_size: int = 0
    class_age_span: int = 1
    class_oldest_age: int = 127
    min_class_size: int = 1
    pupils_per_teacher: float = 0.0
    staff_female_share: float = 0.5
    solo_share: float = 0.0
    common_share: float = 0.0
    dwell_hours: float = 0.75
    seats: tuple[float, float, float] | None = None
    desk_spacing: float = 0.0
    team_size: float = 0.0
    class_teacher_until: int = 0
    staff_hours: tuple[float, float] | None = None
    staff_shifts: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        for name in ("mean_size", "distance_scale", "area_mean", "dwell_hours"):
            require_positive(name, getattr(self, name))
        for name in ("size_sigma", "area_sigma", "class_size", "pupils_per_teacher"):
            require_not_negative(name, getattr(self, name))
        for name in ("desk_spacing", "team_size", "min_class_size", "class_teacher_until"):
            require_not_negative(name, getattr(self, name))
        require_share("staff_female_share", self.staff_female_share)
        require_share("common_share", self.common_share)
        require_share("solo_share", self.solo_share)
        if self.seats is not None:
            require(len(self.seats) == 3 and min(self.seats) > 0, "seats: three positive gaps")
        for start, end in (self.staff_hours,) if self.staff_hours else ():
            require(0 <= start < end <= HOURS_PER_DAY, "staff_hours: invalid hours")
        for start, end in self.staff_shifts or ():
            require(0 <= start < end <= HOURS_PER_DAY, "staff_shifts: invalid hours")


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
    bedridden_share: float = 0.0
    ward_size: float = 1e9
    staff_room_share: float = 0.0
    dwell_hours: float = 0.75

    @property
    def shift_share(self) -> float:
        """Chance that a member of the staff has a shift on a given day (before leave and
        sickness)."""
        start, end = self.shift_hours
        return min(1.0, self.weekly_hours / ((end - start) * DAYS_PER_WEEK))


@dataclass(frozen=True, eq=False)
class TravelParameters:
    """How people get around (see [travel] in town.toml): the speeds of the modes [walk, bike,
    transit, car], the shares of the TRAVEL_MODES by age band (`mode_ages` [band, from/to],
    `mode_shares` [band, mode]) and the public transport vehicles."""

    speed_sigma: float
    speeds: np.ndarray
    mode_ages: np.ndarray
    mode_shares: np.ndarray
    vehicle_load: int
    vehicle_floor_area: float
    in_vehicle_share: float


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
    dwelling: DwellingParameters | None = None
    staff_leave: float = 0.0
    staff_sickness: float = 0.0

    @property
    def staff_presence(self) -> float:
        """The share of their working days on which staff with their own hours are at work."""
        return 1.0 - self.staff_leave - self.staff_sickness


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
    visit_kin: np.ndarray | None = None  # see `_friends_homes`, for the first visit activity
    visit_host: np.ndarray | None = None  # the household of every favourite of that visit
    travel_mode: np.ndarray | None = None  # see TRAVEL_MODES; children have their guardian's
    common_share: np.ndarray | None = None
    dwell_hours: np.ndarray | None = None
    seat_fixed: np.ndarray | None = None
    rotation: Rotation | None = None
    bedridden: np.ndarray | None = None
    ward_venue: np.ndarray | None = None
    ward: np.ndarray | None = None
    ward_start: np.ndarray | None = None
    ward_residents: np.ndarray | None = None
    partner: np.ndarray | None = None
    household_kind: np.ndarray | None = None
    venue_count: int = 0  # venues handed out by `Venues`; vehicles are numbered after them
    indoor_venue: np.ndarray | None = None  # by venue: the air is shared (not an outdoor place)

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
        places = {kind: _place_parameters(values) for kind, values in raw["places"].items()}
        _check_places(places)
        layout = _check_layout(raw["layout"])
        travel = _load_travel(raw["travel"])
        care_home = _load_care_home(raw["care_homes"])
        going_out, accompanied_under = _load_outdoors(raw["outdoors"])
        dwelling = load_dwelling_parameters(raw["households"]["dwelling"])
        employment = raw["employment"]
        for key in ("staff_leave", "staff_sickness"):
            require_share(f"employment.{key}", employment[key])
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
        dwelling=dwelling,
        staff_leave=float(employment["staff_leave"]),
        staff_sickness=float(employment["staff_sickness"]),
    )


def _place_parameters(values: dict) -> PlaceParameters:
    """A place's parameters from its table; lists become tuples (the dataclass is frozen)."""
    values = dict(values)
    for key in ("seats", "staff_hours"):
        if key in values:
            values[key] = tuple(float(value) for value in values[key])
    if "staff_shifts" in values:
        values["staff_shifts"] = tuple(tuple(map(float, shift)) for shift in values["staff_shifts"])
    return PlaceParameters(**values)


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
    require_share("care_homes.bedridden_share", raw["bedridden_share"])
    require_share("care_homes.staff_room_share", raw["staff_room_share"])
    require_positive("care_homes.ward_size", raw["ward_size"])
    shares = np.array(raw["room_bed_shares"], float)
    return CareHomeParameters(
        raw["area_mean"],
        raw["staff_per_resident"],
        raw["staff_female_share"],
        hours["day_hours"],
        hours["shift_hours"],
        raw["weekly_hours"],
        shares / shares.sum(),
        float(raw["bedridden_share"]),
        float(raw["ward_size"]),
        float(raw["staff_room_share"]),
        float(raw["dwell_hours"]),
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
    kinds = {*EDUCATION_KINDS, "work"}
    require(set(places) == kinds, f"places must be {sorted(kinds)}")
    for kind in EDUCATION_KINDS:
        require(places[kind].class_size >= 1, f"{kind}: class_size must be at least 1")
        require_positive(f"{kind}: pupils_per_teacher", places[kind].pupils_per_teacher)
    require_positive("work: team_size", places["work"].team_size)
    require_positive("work: desk_spacing", places["work"].desk_spacing)


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
    modes = travel["modes"]
    ages = np.array(modes["ages"], dtype=float)
    shares = np.array([modes[mode] for mode in TRAVEL_MODES], dtype=float).T
    require(ages.ndim == 2 and len(ages) == len(shares), "travel.modes: one row of ages a band")
    require((ages[:, 0] < ages[:, 1]).all(), "travel.modes: invalid age band")
    require((shares >= 0).all() and (shares.sum(axis=1) > 0).all(), "invalid travel modes")
    speeds = np.array([travel["speed"][mode] for mode in MODES], dtype=float)
    require((speeds > 0).all(), "travel speeds must be positive")
    require_not_negative("speed_sigma", travel["speed_sigma"])
    vehicles = travel["vehicles"]
    require(
        isinstance(vehicles["load"], int) and vehicles["load"] >= 1,
        "travel.vehicles.load must be a whole number, at least 1",
    )
    require_positive("travel.vehicles.floor_area", vehicles["floor_area"])
    require_share("travel.vehicles.in_vehicle_share", vehicles["in_vehicle_share"])
    return TravelParameters(
        speed_sigma=travel["speed_sigma"],
        speeds=speeds,
        mode_ages=ages,
        mode_shares=shares / shares.sum(axis=1, keepdims=True),
        vehicle_load=vehicles["load"],
        vehicle_floor_area=float(vehicles["floor_area"]),
        in_vehicle_share=float(vehicles["in_vehicle_share"]),
    )


@dataclass
class _Placement:
    """The places of the day of everybody while the town is being built (all by person): the
    fixed spot (`x`, `y`: a seat, a desk; the bed when there is no place), the room (`center_x`,
    `center_y`, `radius`, `venue`: a class, a team's room, a ward), the common area of the place
    (`common_*`), the share of the stay spent in it, the mean time at one spot (`dwell`), whether
    people come back to their fixed spot (`seat_fixed`) and how many people the place has."""

    x: np.ndarray
    y: np.ndarray
    people: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    radius: np.ndarray
    venue: np.ndarray
    common_venue: np.ndarray
    common_x: np.ndarray
    common_y: np.ndarray
    common_radius: np.ndarray
    common_share: np.ndarray
    dwell: np.ndarray
    seat_fixed: np.ndarray

    @classmethod
    def at_home(cls, home_x, home_y, dwell_hours: float) -> "_Placement":
        count = len(home_x)
        return cls(
            home_x.copy(),
            home_y.copy(),
            np.zeros(count, np.int32),
            home_x.copy(),
            home_y.copy(),
            np.zeros(count),
            np.full(count, -1, np.int64),
            np.full(count, -1, np.int64),
            home_x.copy(),
            home_y.copy(),
            np.zeros(count),
            np.zeros(count),
            np.full(count, dwell_hours),
            np.zeros(count, np.bool_),
        )


@dataclass
class _Homes:
    """Dwellings and care homes: beds (`x`, `y`), the common room or ward day room of everybody
    (`center_*`, `radius`), and in care homes the room and ward venues, the bedridden and, by
    ward, its residents (`ward_start`, `ward_residents`)."""

    x: np.ndarray
    y: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    radius: np.ndarray
    room_venue: np.ndarray
    ward_venue: np.ndarray
    ward: np.ndarray
    bedridden: np.ndarray
    ward_start: np.ndarray
    ward_residents: np.ndarray


def build_town(population: int, arena_size: float, params: TownParameters, rng) -> Town:
    """Create people, households and the places they go to, all inside the square arena."""
    people = generate_population(population, params.population, rng)
    households = int(people.household.max()) + 1
    household_x, household_y = _scatter(households, arena_size, params, rng)
    venues = Venues(households)
    homes = _homes(people, household_x, household_y, arena_size, params, venues, rng)
    home_x, home_y = homes.x, homes.y
    role = assign_roles(people, params.population, rng)

    placed = _Placement.at_home(home_x, home_y, params.dwell_hours)
    unassigned_workers = role == WORKER
    place_kind = role.copy()
    rotation = Rotation.empty()
    staff_kind = np.full(population, -1, np.int64)
    kind_of = _education_kinds(people, role, params, rng)
    for kind in EDUCATION_KINDS:
        members = np.flatnonzero(kind_of == EDUCATION_KINDS.index(kind))
        if not len(members):
            continue
        staff, taught = _place_education(
            members, kind, people, unassigned_workers, placed, arena_size, params, rng, venues
        )
        unassigned_workers[staff] = False
        place_kind[staff] = role[members[0]]
        staff_kind[staff] = EDUCATION_KINDS.index(kind)
        rotation = rotation.extend(taught)
    carers = _place_care_staff(people, unassigned_workers, homes, placed, arena_size, params, rng)
    unassigned_workers[carers] = False
    place_kind[carers] = RESIDENT
    activity_places = []
    for activity in params.schedule.activities:
        if activity.kind == "visit":
            # a visitor is in the common room of the host's dwelling, in the day room of the
            # ward of a mobile resident, at the bed of a bedridden one
            visited = _visited_rooms(people, homes, params)
            places, sites = (
                _friends_homes(people, visited, arena_size, activity, params, rng),
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
            staff = _place_activity_staff(
                activity,
                places,
                int(np.floor(activity.staff_share * employed + rng.random())),
                people,
                unassigned_workers,
                (home_x, home_y),
                placed,
                arena_size,
                params,
                rng,
            )
            unassigned_workers[staff] = False
            activity_staff[staff] = True
    workers = np.flatnonzero(unassigned_workers)
    if len(workers):
        _place_workers(workers, home_x, home_y, arena_size, placed, params, rng, venues)

    guardian = _guardians(
        people, params.accompanied_under, params.population.parent_gap_min
    )
    activity_x, activity_y, activity_radius, activity_people = [], [], [], []
    activity_venue = []
    lunch_x, lunch_y, lunch_radius, lunch_venue = None, None, None, None
    visit_kin = visit_host = None
    for activity, places, sites in activity_places:
        if activity.kind == "visit" and visit_kin is None:
            visit_kin = _with_guardian(places[6], guardian)
            visit_host = _with_guardian(places[7], guardian)
        if activity.name == LUNCH_ACTIVITY and sites is not None:
            chosen = _choose_sites(placed.x, placed.y, *sites[:-1], params, rng, sites[-1])
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
        people,
        role,
        place_kind,
        staff_kind,
        carers,
        activity_places,
        activity_staff,
        placed,
        params,
        rng,
    )
    on_shift = ~np.isnan(shift_window[:, 0, 0]) | ~np.isnan(shift_window[:, 1, 0])
    start_habit[on_shift] = 0.0  # a shift starts and ends on time
    end_habit[on_shift] = 0.0
    weekend_work[on_shift] = 0.0  # the shift says when they work
    aftercare = _aftercare(people.age, role, params, rng)
    _part_time_students(role, shift_attendance, params, rng)
    mode, speed = _travel_modes(people.age, params.travel, rng)
    mode, speed = _with_guardian(mode, guardian), _with_guardian(speed, guardian)
    # the venues whose air is shared: all but the places of an activity outdoors
    indoor_venue = np.ones(venues.count, np.bool_)
    for activity, places, sites in activity_places:
        if not activity.indoor and sites is not None:
            first = places[5][1]
            indoor_venue[first : first + len(places[4][0])] = False
    return Town(
        arena_size,
        people.age,
        people.sex,
        people.household,
        role,
        home_x,
        home_y,
        np.clip(placed.x, 0.0, arena_size),
        np.clip(placed.y, 0.0, arena_size),
        placed.people,
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
        placed.center_x,
        placed.center_y,
        placed.radius,
        placed.venue,
        placed.common_venue,
        activity_venue,
        lunch_venue,
        placed.common_x,
        placed.common_y,
        placed.common_radius,
        homes.center_x,
        homes.center_y,
        homes.radius,
        shift_window,
        shift_attendance,
        aftercare,
        homes.room_venue,
        visit_kin,
        visit_host,
        mode,
        placed.common_share,
        placed.dwell,
        placed.seat_fixed,
        rotation,
        homes.bedridden,
        homes.ward_venue,
        homes.ward,
        homes.ward_start,
        homes.ward_residents,
        people.partner,
        people.kind,
        venue_count=venues.count,
        indoor_venue=indoor_venue,
    )


def _homes(people, household_x, household_y, arena_size, params, venues, rng) -> _Homes:
    """Beds and common rooms of the dwellings; the wards, rooms and beds of the care homes."""
    count = len(people.age)
    private = ~people.resident
    dwelling = params.dwelling
    x, y = np.zeros(count), np.zeros(count)
    center_x = household_x[people.household].copy()
    center_y = household_y[people.household].copy()
    radius = np.full(count, dwelling.common_radius)
    partner = people.partner if people.partner is not None else np.full(count, -1)
    members = np.flatnonzero(private)
    if len(members):
        x[members], y[members] = layout(
            people.age[members],
            people.household[members],
            partner[members],
            household_x,
            household_y,
            dwelling,
            rng,
        )
    homes = _Homes(
        x,
        y,
        center_x,
        center_y,
        radius,
        np.full(count, -1, np.int64),
        np.full(count, -1, np.int64),
        np.full(count, -1, np.int64),
        np.zeros(count, np.bool_),
        np.zeros(1, np.int64),
        np.zeros(0, np.int64),
    )
    residents = np.flatnonzero(people.resident)
    if len(residents):
        _care_home_wards(residents, people, household_x, household_y, homes, params, venues, rng)
    homes.x, homes.y = np.clip(homes.x, 0.0, arena_size), np.clip(homes.y, 0.0, arena_size)
    return homes


def _care_home_wards(residents, people, home_x, home_y, homes: _Homes, params, venues, rng):
    """Wards of the care homes, each with a day room, and the rooms of the residents in them.

    The residents of a home are divided into wards of about `ward_size`; a ward's day room has
    the floor space per resident for its mobile residents and lies on a circle around the
    middle of the home; the rooms of a ward around its day room; the beds of a room a couple of
    metres apart. A share of the residents is bedridden.
    """
    care = params.care_home
    home = people.household[residents]
    ward_of = split_evenly(home, care.ward_size, rng)
    wards = int(ward_of.max()) + 1
    bedridden = rng.random(len(residents)) < care.bedridden_share
    ward_home = np.zeros(wards, np.int64)
    ward_home[ward_of] = home
    ward_index, ward_count = ranks_within(ward_home, rng)  # (by ward)
    home_size = np.bincount(home, minlength=int(home.max()) + 1)
    mobile = np.bincount(ward_of, weights=~bedridden, minlength=wards)
    ward_radius = np.sqrt(np.maximum(mobile, 1) * care.area_mean / math.pi)
    spread = np.sqrt(home_size[ward_home] * care.area_mean / math.pi)
    ward_x, ward_y = ring_positions(
        ward_index, ward_count, ward_home, home_x[ward_home], home_y[ward_home], spread, rng
    )
    room, room_ward = _rooms(ward_of, care.room_shares, rng)
    room_index, room_count = ranks_within(room_ward, rng)
    room_x, room_y = ring_positions(
        room_index,
        room_count,
        room_ward,
        ward_x[room_ward],
        ward_y[room_ward],
        ward_radius[room_ward] + 3.0 * params.dwelling.room_bed_gap,
        rng,
    )
    place, _ = ranks_within(room, rng)
    homes.x[residents] = room_x[room] + place * params.dwelling.room_bed_gap
    homes.y[residents] = room_y[room]
    homes.center_x[residents] = ward_x[ward_of]
    homes.center_y[residents] = ward_y[ward_of]
    homes.radius[residents] = ward_radius[ward_of]
    homes.room_venue[residents] = venues.allocate(len(room_ward)) + room
    homes.ward_venue[residents] = venues.allocate(wards) + ward_of
    homes.ward[residents] = ward_of
    homes.bedridden[residents] = bedridden
    order = np.argsort(ward_of, kind="stable")
    homes.ward_residents = residents[order]
    homes.ward_start = np.concatenate([[0], np.cumsum(np.bincount(ward_of, minlength=wards))])


def _education_kinds(people, role, params, rng) -> np.ndarray:
    """The kind of education place of everybody (an index of `EDUCATION_KINDS`, -1 for none):
    the pupils of an age go to a secondary school with the published share, else to a basic one.
    """
    kind = np.full(len(role), -1, np.int64)
    kind[role == PRESCHOOL] = EDUCATION_KINDS.index("preschool")
    kind[role == UNIVERSITY] = EDUCATION_KINDS.index("university")
    pupils = np.flatnonzero(role == SCHOOL)
    share = params.population.secondary[people.age[pupils], people.sex[pupils]]
    secondary = rng.random(len(pupils)) < share
    kind[pupils] = np.where(
        secondary, EDUCATION_KINDS.index("secondary"), EDUCATION_KINDS.index("basic")
    )
    return kind


def _place_education(
    students, kind, people, worker_pool, placed: _Placement, arena_size, params, rng, venues
):
    """Schools of one kind: classes of pupils of an age with seats (or room to move about), the
    common rooms, and teachers recruited from the workers near the school.

    Classes of pupils younger than `class_teacher_until` have a teacher of their own (in a
    kindergarten a class's teachers take turns in shifts); the others are taught by teachers who go
    from class to class: as many as the pupils per teacher give, at least one per class. Returns
    the teachers and their rotation.
    """
    place = params.places[kind]
    age = people.age
    site_count = max(1, round(len(students) / place.mean_size))
    site_x, site_y = _scatter(site_count, arena_size, params, rng)
    cdf = _size_cdf(site_count, place.size_sigma, rng)
    site = _choose_sites(
        placed.x[students],
        placed.y[students],
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
    class_of = classes(
        site,
        age[students],
        place.class_size,
        max(place.min_class_size, 1),
        place.class_age_span,
        place.class_oldest_age,
        rng,
    )
    count = int(class_of.max()) + 1
    class_site = np.zeros(count, np.int64)
    class_site[class_of] = site
    class_students = np.bincount(class_of, minlength=count)
    oldest = np.zeros(count, np.int64)
    np.maximum.at(oldest, class_of, age[students])
    center_x, center_y = _around(
        site_x[class_site], site_y[class_site], site_radius[class_site], rng
    )
    class_radius = np.sqrt(class_students * area[class_site] / math.pi)
    if place.seats is not None:
        seat_x, seat_y = seats(class_of, center_x, center_y, SeatLayout(*place.seats), rng)
    else:
        seat_x, seat_y = _around(
            center_x[class_of], center_y[class_of], class_radius[class_of], rng
        )
    first_class = venues.allocate(count)
    first_school = venues.allocate(site_count)
    common_radius = site_radius * math.sqrt(max(place.common_share, 1e-9))

    # teachers: one of its own for a class of young pupils (several taking turns where the pupils
    # per teacher make several), teachers going from class to class for the others
    own_teacher = oldest < place.class_teacher_until
    per_class = np.where(
        own_teacher, np.maximum(np.rint(class_students / place.pupils_per_teacher), 1), 0
    ).astype(np.int64)
    rotating_pupils = np.bincount(
        class_site, weights=np.where(own_teacher, 0, class_students), minlength=site_count
    )
    rotating_classes = np.bincount(class_site, weights=~own_teacher, minlength=site_count)
    rotating_staff = np.where(
        rotating_classes > 0,
        np.maximum(np.rint(rotating_pupils / place.pupils_per_teacher), rotating_classes),
        0,
    ).astype(np.int64)
    slot_class = np.repeat(np.arange(count), per_class)
    slot_site = np.concatenate(
        [class_site[slot_class], np.repeat(np.arange(site_count), rotating_staff)]
    )
    slot_x = np.concatenate(
        [center_x[slot_class], site_x[np.repeat(np.arange(site_count), rotating_staff)]]
    )
    slot_y = np.concatenate(
        [center_y[slot_class], site_y[np.repeat(np.arange(site_count), rotating_staff)]]
    )
    slot_class = np.concatenate([slot_class, np.full(int(rotating_staff.sum()), -1)])
    hired = _recruit(
        (slot_x, slot_y),
        np.flatnonzero(worker_pool & (age >= MIN_TEACHER_AGE[kind])),
        people,
        (placed.x, placed.y),
        place.staff_female_share,
        arena_size * params.places["work"].distance_scale,
        params,
        rng,
    )
    found = hired >= 0
    teachers, slot_class, slot_site = hired[found], slot_class[found], slot_site[found]
    own = slot_class >= 0

    for members, of_class, of_site, spot in (
        (students, class_of, site, (seat_x, seat_y)),
        (teachers[own], slot_class[own], slot_site[own], None),
    ):
        if spot is None:
            spot = _around(center_x[of_class], center_y[of_class], class_radius[of_class], rng)
        placed.x[members], placed.y[members] = spot
        placed.center_x[members], placed.center_y[members] = center_x[of_class], center_y[of_class]
        placed.radius[members] = class_radius[of_class]
        placed.venue[members] = first_class + of_class
        placed.people[members] = in_site[of_site]
    rotating = teachers[~own]
    rotating_site = slot_site[~own]
    placed.x[rotating], placed.y[rotating] = _around(
        site_x[rotating_site], site_y[rotating_site], common_radius[rotating_site], rng
    )
    placed.center_x[rotating], placed.center_y[rotating] = (
        site_x[rotating_site],
        site_y[rotating_site],
    )
    placed.radius[rotating] = common_radius[rotating_site]
    placed.venue[rotating] = first_school + rotating_site  # the staff room is the common area
    placed.people[rotating] = in_site[rotating_site]
    everybody = np.concatenate([students, teachers])
    of_site = np.concatenate([site, slot_site])
    placed.common_venue[everybody] = first_school + of_site
    placed.common_x[everybody], placed.common_y[everybody] = site_x[of_site], site_y[of_site]
    placed.common_radius[everybody] = common_radius[of_site]
    placed.common_share[everybody] = place.common_share
    placed.dwell[everybody] = place.dwell_hours
    placed.seat_fixed[students] = place.seats is not None
    placed.seat_fixed[teachers[own]] = False

    # the rotation: the classes without a teacher of their own, by school
    taught = np.flatnonzero(~own_teacher)
    taught = taught[np.argsort(class_site[taught], kind="stable")]
    schools, school_of = np.unique(rotating_site, return_inverse=True)
    position = np.searchsorted(schools, class_site[taught])
    keep = (position < len(schools)) & (
        schools[np.minimum(position, len(schools) - 1)] == class_site[taught]
    )
    taught, position = taught[keep], position[keep]
    start = np.concatenate([[0], np.cumsum(np.bincount(position, minlength=len(schools)))])
    rotation = Rotation(
        rotating,
        school_of.astype(np.int64),
        start.astype(np.int64),
        first_class + taught,
        center_x[taught],
        center_y[taught],
        class_radius[taught],
    )
    return teachers, rotation


def _place_care_staff(
    people, worker_pool, homes: _Homes, placed: _Placement, arena_size, params, rng
):
    """Direct care staff of the care homes, recruited from the workers near each: enough for
    `staff_per_resident` of them (head count of the day shifts) per resident, each working in one
    ward; how many are on duty on a day follows from their attendance (`shift_share`, absence).
    Returns who they are."""
    residents = np.flatnonzero(people.resident)
    if not len(residents):
        return residents
    care = params.care_home
    wards = len(homes.ward_start) - 1
    in_ward = np.diff(homes.ward_start)
    wanted = np.floor(care.staff_per_resident * in_ward + rng.random(wards)).astype(np.int64)
    of_ward = np.repeat(np.arange(wards), wanted)
    resident_of_ward = homes.ward_residents[homes.ward_start[:-1]]
    ward_x, ward_y = homes.center_x[resident_of_ward], homes.center_y[resident_of_ward]
    ward_radius = homes.radius[resident_of_ward]
    staff = _recruit(
        (ward_x[of_ward], ward_y[of_ward]),
        np.flatnonzero(worker_pool & (people.age >= ADULT_AGE)),
        people,
        (placed.x, placed.y),
        care.staff_female_share,
        arena_size * params.places["work"].distance_scale,
        params,
        rng,
    )
    hired = staff >= 0
    staff, of_ward = staff[hired], of_ward[hired]
    placed.x[staff], placed.y[staff] = _around(
        ward_x[of_ward], ward_y[of_ward], ward_radius[of_ward], rng
    )
    placed.center_x[staff], placed.center_y[staff] = ward_x[of_ward], ward_y[of_ward]
    placed.radius[staff] = ward_radius[of_ward]
    placed.venue[staff] = homes.ward_venue[resident_of_ward[of_ward]]
    placed.people[staff] = in_ward[of_ward]
    placed.dwell[staff] = care.dwell_hours
    homes.ward[staff] = of_ward
    return staff


def _place_workers(workers, home_x, home_y, arena_size, placed: _Placement, params, rng, venues):
    """Workplaces (local units) of the workers: a share works alone, at a place of their own;
    the others at shared workplaces, in rooms of a team each with a desk of their own, and in the
    common area of the workplace (meetings, breaks) for the share `common_share` of the day."""
    place = params.places["work"]
    solo = rng.random(len(workers)) < place.solo_share
    alone = workers[solo]
    x, y = _around(home_x[alone], home_y[alone], arena_size * place.distance_scale, rng)
    placed.x[alone], placed.y[alone] = x, y
    placed.center_x[alone], placed.center_y[alone], placed.radius[alone] = x, y, 0.0
    placed.people[alone] = 1
    placed.venue[alone] = venues.allocate(len(alone)) + np.arange(len(alone))
    placed.seat_fixed[alone] = True
    placed.dwell[alone] = place.dwell_hours
    shared = workers[~solo]
    if not len(shared):
        return
    site_count = max(1, round(len(shared) / place.mean_size))
    site_x, site_y = _scatter(site_count, arena_size, params, rng)
    cdf = _size_cdf(site_count, place.size_sigma, rng)
    site = _choose_sites(
        home_x[shared],
        home_y[shared],
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
    room_of = split_evenly(site, place.team_size, rng)
    rooms = int(room_of.max()) + 1
    room_site = np.zeros(rooms, np.int64)
    room_site[room_of] = site
    in_room = np.bincount(room_of, minlength=rooms)
    room_x, room_y = _around(site_x[room_site], site_y[room_site], site_radius[room_site], rng)
    room_radius = np.sqrt(in_room * area[room_site] / math.pi)
    desk_x, desk_y = desks(room_of, room_x, room_y, place.desk_spacing, rng)
    first_room = venues.allocate(rooms)
    first_site = venues.allocate(site_count)
    common_radius = site_radius * math.sqrt(max(place.common_share, 1e-9))
    placed.x[shared], placed.y[shared] = desk_x, desk_y
    placed.center_x[shared], placed.center_y[shared] = room_x[room_of], room_y[room_of]
    placed.radius[shared] = room_radius[room_of]
    placed.venue[shared] = first_room + room_of
    placed.people[shared] = in_site[site]
    placed.common_venue[shared] = first_site + site
    placed.common_x[shared], placed.common_y[shared] = site_x[site], site_y[site]
    placed.common_radius[shared] = common_radius[site]
    placed.common_share[shared] = place.common_share
    placed.dwell[shared] = place.dwell_hours
    placed.seat_fixed[shared] = True


def _place_activity_staff(
    activity,
    places,
    wanted,
    people,
    worker_pool,
    homes,
    placed: _Placement,
    arena_size,
    params,
    rng,
) -> np.ndarray:
    """Workers of the shops, restaurants and other places of an activity; returns who they are.

    The positions go to the places in proportion to their regular visitors (a place with twice the
    customers employs twice the staff) and the staff are recruited from the workers near each, as
    teachers are. They move about the place, among the visitors.
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
    placed.x[staff], placed.y[staff] = _around(
        site_x[slot_site], site_y[slot_site], radius[slot_site], rng
    )
    placed.people[staff] = np.maximum(np.rint(regulars[slot_site] * activity.presence_share), 1)
    placed.center_x[staff], placed.center_y[staff] = site_x[slot_site], site_y[slot_site]
    placed.radius[staff] = radius[slot_site]
    placed.venue[staff] = places[5][1] + slot_site
    placed.dwell[staff] = params.dwell_hours
    return staff


def _shifts(
    people,
    role,
    place_kind,
    staff_kind,
    carers,
    activity_places,
    activity_staff,
    placed,
    params,
    rng,
):
    """The working hours [person, weekday/weekend, start/end] and the chance of being at work
    [person, weekday/weekend] of the staff whose hours are those of their place (NaN for all
    others).

    Care homes work in day shifts every day; shops and restaurants in shifts of a working day
    within their opening hours (the places of the lunch break also over the lunch breaks of the
    workers around them), five days a week; teachers keep the hours of their school on weekdays,
    those of a kindergarten class taking turns in its shifts. Leave and sickness take their share
    of the days; teachers take their leave in the school holidays, when their place is closed.
    """
    count = len(role)
    window = np.full((count, 2, 2), np.nan)
    attendance = np.full((count, 2), np.nan)
    care = params.care_home
    window[carers] = care.shift_hours
    attendance[carers] = care.shift_share * params.staff_presence
    for activity, places, sites in activity_places:
        if sites is None or activity.staff_share <= 0:
            continue
        first = places[5][1]
        staff = np.flatnonzero(
            activity_staff & (placed.venue >= first) & (placed.venue < first + len(places[4][0]))
        )
        hours = np.array(activity.window, float)  # [weekday/weekend, start/end]
        if activity.name == LUNCH_ACTIVITY:
            lunch = params.schedule.lunch_window  # [role, weekday/weekend, start/end]
            for day in range(2):
                if not np.isnan(lunch[:, day, 0]).all():
                    opens = np.nanmin(lunch[:, day, 0]) - OPEN_BEFORE_LUNCH_HOURS
                    hours[day, 0] = min(hours[day, 0], opens)
                    latest = np.nanmax(lunch[:, day, 1]) + MAX_LUNCH_HOURS
                    hours[day, 1] = max(hours[day, 1], latest)
        # an early and a late shift of a working day cover the opening hours (the whole day when
        # it is shorter)
        length = np.minimum(SHIFT_HOURS, hours[:, 1] - hours[:, 0])
        late = rng.random((len(staff), 1)) < 0.5
        start = np.where(late, hours[None, :, 1] - length[None, :], hours[None, :, 0])
        window[staff, :, 0] = start
        window[staff, :, 1] = start + length[None, :]
        attendance[staff] = activity.staff_days / DAYS_PER_WEEK * params.staff_presence
    teachers = np.flatnonzero((role == WORKER) & np.isin(place_kind, EDUCATION) & (staff_kind >= 0))
    for index, kind in enumerate(EDUCATION_KINDS):
        place = params.places[kind]
        of_kind = teachers[staff_kind[teachers] == index]
        if not len(of_kind):
            continue
        if place.staff_shifts:
            turn, _ = ranks_within(placed.venue[of_kind], rng)
            shifts = np.array(place.staff_shifts, float)
            window[of_kind, 0] = shifts[turn % len(shifts)]
        elif place.staff_hours:
            window[of_kind, 0] = place.staff_hours
        else:
            continue
        attendance[of_kind, 0] = 1.0 - params.staff_sickness
        attendance[of_kind, 1] = 0.0
    return window, attendance


def _part_time_students(role, attendance, params, rng) -> None:
    """Students of the combined form go to their university only on a few weekdays (attendance
    [person, weekday/weekend] is changed)."""
    students = np.flatnonzero(role == UNIVERSITY)
    part_time = students[rng.random(len(students)) < params.population.part_time_students]
    attendance[part_time, 0] = params.population.part_time_weekdays / WEEKEND_FROM_DAY


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


def _guardians(people: Population, accompanied_under: int, parent_gap_min: float) -> np.ndarray:
    """The adult who goes out with each small child, -1 for everybody else.

    The youngest woman of the household, the mother as a rule, or else the youngest man. A child
    is not accompanied by somebody less than `parent_gap_min` years older (an adult sibling): the
    youngest woman, then man, who is at least that much older, when there is one.
    """
    guardian = np.full(len(people.age), -1, np.int64)
    adults = np.flatnonzero((people.age >= ADULT_AGE) & ~people.resident)
    order = adults[
        np.lexsort((people.age[adults], people.sex[adults] != FEMALE, people.household[adults]))
    ]
    households, first = np.unique(people.household[order], return_index=True)
    carer = np.full(int(people.household.max()) + 1, -1, np.int64)
    carer[households] = order[first]
    children = np.flatnonzero((people.age < accompanied_under) & ~people.resident)
    guardian[children] = carer[people.household[children]]
    too_close = children[
        (guardian[children] >= 0)
        & (people.age[guardian[children]] - people.age[children] < parent_gap_min)
    ]
    for child in too_close:  # about 1 % of the children
        home = order[people.household[order] == people.household[child]]
        older = home[people.age[home] - people.age[child] >= parent_gap_min]
        if len(older):
            guardian[child] = older[0]  # the order is: women first, the youngest first
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
        candidate = np.minimum(locate_in_cdf(rng.random((count, candidates)), cdf), len(site_x) - 1)
        distance = np.hypot(
            site_x[candidate] - home_x[block, None], site_y[candidate] - home_y[block, None]
        )
        weight = np.exp(-distance / scale)
        if affinity is not None:
            age, site_age, age_scale = affinity
            weight = weight * np.exp(-np.abs(age[block, None] - site_age[candidate]) / age_scale)
        chosen[block] = candidate[np.arange(count), _pick(weight, rng)]
    return chosen


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
    # a place lies inside the town: its middle is at least its radius from the edge, so that the
    # people in it are not pressed together at the border (where the arena is small for it)
    inner = np.minimum(radius, arena_size / 2)
    site_x, site_y = (np.clip(values, inner, arena_size - inner) for values in (site_x, site_y))
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


def _friends_homes(people, homes, arena_size, activity, params, rng):
    """Homes of everybody's friends and relatives: nearby, of other households.

    A favourite is a relative of the generation above or below (a parent, a grown-up child) with
    the chance `family_share`: about `generation_gap` years older or younger; otherwise a friend of
    a similar age. `homes` are, for every person, the centre and the radius of the room a visitor
    of theirs stays in (the common room of a dwelling, `Town.home_center_x`, `home_center_y`,
    `home_radius`; the day room of a care-home ward) and optionally the venue of that room (the
    household of a dwelling by default; the ward, or the room of a bedridden resident). Returns the
    places as `_activity_sites` does and, for every favourite, `kin` (1 for a relative of the
    generation above, -1 of the generation below, 0 for a friend) and the household of the host.
    """
    center_x, center_y, dwelling = homes[:3]
    visited = homes[3] if len(homes) > 3 else people.household
    age, household = people.age, people.household
    count, favourites = len(age), activity.favourites
    x, y, radius = (np.empty((count, favourites)) for _ in range(3))
    host, venue = (np.empty((count, favourites), np.int64) for _ in range(2))
    kin = np.zeros((count, favourites), np.int8)
    candidates = params.candidate_sites
    scale = arena_size * activity.distance_scale
    for favourite in range(favourites):
        friend = np.empty(count, np.int64)
        relative = np.zeros(count, np.bool_)
        for start in range(0, count, CANDIDATE_CHUNK):
            block = slice(start, start + CANDIDATE_CHUNK)
            persons = np.arange(count)[block]
            candidate = rng.integers(0, count, (len(persons), candidates))
            distance = np.hypot(
                center_x[candidate] - center_x[block, None],
                center_y[candidate] - center_y[block, None],
            )
            gap = np.abs(age[candidate] - age[block, None])
            share = 0.0 if activity.family_share is None else activity.family_share[age[block]]
            family = rng.random(len(persons)) < share
            gap = np.where(family[:, None], np.abs(gap - activity.generation_gap), gap)
            weight = np.exp(-distance / scale - gap / activity.age_scale) * (
                household[candidate] != household[block, None]
            )
            friend[block] = candidate[np.arange(len(persons)), _pick(weight + 1e-300, rng)]
            relative[block] = family
        x[:, favourite], y[:, favourite] = center_x[friend], center_y[friend]
        radius[:, favourite] = dwelling[friend]
        venue[:, favourite] = visited[friend]
        host[:, favourite] = household[friend]
        kin[:, favourite] = np.where(relative, np.sign(age[friend] - age), 0)
    regulars = np.zeros((count, favourites), np.int32)
    return x, y, radius, regulars, None, (venue, None), kin, host


def _travel_modes(age, travel: TravelParameters, rng) -> tuple[np.ndarray, np.ndarray]:
    """The mode of everybody (see TRAVEL_MODES), drawn with the shares of the trips of their age
    band (the youngest band for those below it), nobody under DRIVING_AGE as a driver; and their
    speed (km/h) on the moves that take no drawn time, around the speed of the mode."""
    band = np.searchsorted(travel.mode_ages[:, 1], age, side="right")
    band = np.minimum(band, len(travel.mode_ages) - 1)
    cdf = np.cumsum(travel.mode_shares[band], axis=1)
    mode = np.minimum((cdf < rng.random(len(age))[:, None] * cdf[:, -1:]).sum(axis=1), 4)
    mode[(mode == CAR_DRIVER) & (age < DRIVING_AGE)] = CAR_PASSENGER
    vehicle = np.array([0, 1, 2, 3, 3])  # the speed of the car for drivers and passengers
    speed = travel.speeds[vehicle[mode]] * np.exp(rng.normal(0.0, travel.speed_sigma, len(age)))
    return mode.astype(np.int8), speed


def _visited_rooms(people, homes, params):
    """For every person, where a visitor of theirs stays: the centre and radius of the room and
    its venue (see `_friends_homes`). The common room of the dwelling for everybody who lives in
    one, the day room of the ward for a mobile resident of a care home, the bed of a bedridden
    resident (a room of its own)."""
    center_x, center_y, radius = homes.center_x.copy(), homes.center_y.copy(), homes.radius.copy()
    venue = people.household.astype(np.int64)
    ward = homes.ward_venue >= 0
    venue[ward] = homes.ward_venue[ward]
    bed = homes.bedridden & (homes.room_venue >= 0)
    center_x[bed], center_y[bed] = homes.x[bed], homes.y[bed]
    radius[bed] = params.dwelling.room_bed_gap
    venue[bed] = homes.room_venue[bed]
    return center_x, center_y, radius, venue
