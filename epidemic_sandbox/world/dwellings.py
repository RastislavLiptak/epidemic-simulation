"""The inside of the dwellings: rooms, beds, and where the people of a household are at home.

A dwelling has a common room (the living room and the kitchen, where the household is together)
and bedrooms around it. Everybody has a bed: the two people of a couple share a double bed, the
others sleep alone in a bedroom while there are bedrooms left and share one otherwise. At night
everybody at home is in bed; awake at home a person is in the common room for a share of the time
and in their own room (by their bed) for the rest.

The geometry the rest of the model uses (see `town.Town`): `home_x`, `home_y` are a person's bed,
`home_center_x`, `home_center_y` and `home_radius` the middle and the radius of the common room
of their dwelling; a visitor stays in the common room of the host's dwelling.
"""

import math
from dataclasses import dataclass

import numpy as np

from epidemic_sandbox.config.checks import require, require_positive, require_share

ADULT_AGE = 18
# The rooms per person are published for one, two and three or more adults, each without children
# and with one, two and three or more dependent children.
ADULT_GROUPS, CHILD_GROUPS = 3, 4


@dataclass(frozen=True, eq=False)
class DwellingParameters:
    """Rooms and beds of the dwellings and how much of their time at home people spend together.

    `rooms_per_person` [adults 1, 2, 3+][children 0, 1, 2, 3+]: rooms of the dwelling per person.
    `common_area`: floor of the common room (m2). `together_share`: the share of the time awake at
    home spent in the common room. `bed_gap`: distance of the two people in a double bed (m);
    `room_bed_gap`: distance of two beds in one bedroom (m); `bedroom_distance`: distance of the
    bedrooms from the middle of the common room (m). `sleep_hours` [from, to]: the hours in bed.
    `dwell_hours`: the mean time awake at home before a person moves between the rooms.
    """

    rooms_per_person: np.ndarray
    common_area: float
    together_share: float
    bed_gap: float
    room_bed_gap: float
    bedroom_distance: float
    sleep_hours: tuple[float, float]
    dwell_hours: float

    @property
    def common_radius(self) -> float:
        return math.sqrt(self.common_area / math.pi)

    def asleep(self, hour: float) -> bool:
        """True in the hours in bed (they may run over midnight)."""
        start, end = self.sleep_hours
        return start <= hour or hour < end if start > end else start <= hour < end


def load_dwelling_parameters(raw: dict) -> DwellingParameters:
    rooms = np.array(raw["rooms_per_person"], float)
    require(rooms.shape == (ADULT_GROUPS, CHILD_GROUPS), "dwelling.rooms_per_person must be 3 x 4")
    require((rooms > 0).all(), "dwelling.rooms_per_person must be positive")
    for key in ("common_area", "bed_gap", "room_bed_gap", "bedroom_distance", "dwell_hours"):
        require_positive(f"dwelling.{key}", raw[key])
    require_share("dwelling.together_share", raw["together_share"])
    start, end = raw["sleep_hours"]
    require(0 <= start < 24 and 0 <= end < 24 and start != end, "dwelling.sleep_hours: invalid")
    return DwellingParameters(
        rooms_per_person=rooms,
        common_area=float(raw["common_area"]),
        together_share=float(raw["together_share"]),
        bed_gap=float(raw["bed_gap"]),
        room_bed_gap=float(raw["room_bed_gap"]),
        bedroom_distance=float(raw["bedroom_distance"]),
        sleep_hours=(float(start), float(end)),
        dwell_hours=float(raw["dwell_hours"]),
    )


def bedrooms(age: np.ndarray, household: np.ndarray, params: DwellingParameters) -> np.ndarray:
    """The number of bedrooms of every household (index), at least one: the rooms of the dwelling
    (rooms per person of its kind times its people, rounded) less the common room."""
    households = int(household.max()) + 1
    adults = np.bincount(household, weights=(age >= ADULT_AGE), minlength=households)
    children = np.bincount(household, weights=(age < ADULT_AGE), minlength=households)
    adult_group = np.clip(adults.astype(np.int64), 1, ADULT_GROUPS) - 1
    child_group = np.clip(children.astype(np.int64), 0, CHILD_GROUPS - 1)
    rooms = np.rint(params.rooms_per_person[adult_group, child_group] * (adults + children))
    return np.maximum(rooms.astype(np.int64) - 1, 1)


def beds(
    age: np.ndarray,
    household: np.ndarray,
    partner: np.ndarray,
    rooms: np.ndarray,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """The bedroom (numbered within the household) and the place in it of everybody.

    Sleepers are the people who sleep alone and the couples, who share a bed. The oldest sleeper
    (a couple counts as its older member) has the first bedroom; the others get the remaining
    bedrooms in turn from the oldest, so that when there are more sleepers than bedrooms the
    children share the rooms evenly. With a single bedroom everybody sleeps in it. Returns the
    bedroom of everybody and their place among the beds of the room (0, 1, ...; the two people of
    a couple have the same place).
    """
    count = len(age)
    people = np.arange(count)
    sleeper = np.where(partner >= 0, np.minimum(people, partner), people)
    eldest = age.astype(np.int64).copy()
    np.maximum.at(eldest, sleeper, age.astype(np.int64))
    order = np.flatnonzero(sleeper == people)
    order = order[np.lexsort((rng.random(len(order)), -eldest[order], household[order]))]
    starts = np.searchsorted(household[order], np.arange(len(rooms)))
    rank = np.arange(len(order)) - starts[household[order]]
    available = rooms[household[order]]
    single = available == 1
    others = np.maximum(available - 1, 1)
    room_of = np.where(rank == 0, 0, 1 + (rank - 1) % others)
    place = np.where(rank == 0, 0, (rank - 1) // others)
    room_of = np.where(single, 0, room_of)
    place = np.where(single, rank, place)
    bedroom = np.empty(count, np.int64)
    spot = np.empty(count, np.int64)
    bedroom[order], spot[order] = room_of, place
    bedroom, spot = bedroom[sleeper], spot[sleeper]
    return bedroom, spot


def layout(
    age: np.ndarray,
    household: np.ndarray,
    partner: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    params: DwellingParameters,
    rng,
) -> tuple[np.ndarray, np.ndarray]:
    """The bed of everybody (x, y) in the dwellings around the common rooms at `center_x`,
    `center_y` (by household).

    The bedrooms lie on a circle `bedroom_distance` around the middle of the common room, evenly
    spread; the beds of a room are `room_bed_gap` apart, a couple's two halves of a double bed
    `bed_gap` apart.
    """
    rooms = bedrooms(age, household, params)
    bedroom, place = beds(age, household, partner, rooms, rng)
    ring = params.bedroom_distance
    angle = (
        2.0 * math.pi * (bedroom / np.maximum(rooms[household], 1))
        + rng.random(int(household.max()) + 1)[household]
    )
    room_x = center_x[household] + ring * np.cos(angle)
    room_y = center_y[household] + ring * np.sin(angle)
    # beds of a room in a row along the tangent; a couple's two halves of a double bed
    tangent = angle + math.pi / 2
    offset = place * params.room_bed_gap
    couple = partner >= 0
    side = np.where(couple & (np.arange(len(age)) > partner), 0.5, -0.5) * params.bed_gap
    side = np.where(couple, side, 0.0)
    bed_x = room_x + (offset + side) * np.cos(tangent)
    bed_y = room_y + (offset + side) * np.sin(tangent)
    return bed_x, bed_y
