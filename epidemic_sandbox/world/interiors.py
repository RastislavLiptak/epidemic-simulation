"""The inside of the places: classes and their seats, the rooms of workplaces and their desks,
the teachers who go from class to class, the wards and rooms of care homes.

A place (a school, a workplace, a care home) has rooms, each a venue of its own (walls), and a
common area (corridors, canteen, kitchen, day room). The people of a room have a fixed spot in it
(a seat, a desk, a bed) or move about it; the functions here only lay the places out, the model
moves the people (see `EpidemicModel._move_about`).
"""

import math
from dataclasses import dataclass

import numpy as np


def around(center_x, center_y, radius, rng) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random points in a disk of `radius` around each center."""
    distance = radius * np.sqrt(rng.random(len(center_x)))
    angle = rng.random(len(center_x)) * 2 * math.pi
    return center_x + distance * np.cos(angle), center_y + distance * np.sin(angle)


def ranks_within(group: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """A random order within each group: the rank of every member (0, 1, ...) and the size of
    its group."""
    order = np.lexsort((rng.random(len(group)), group))
    sorted_group = group[order]
    first = np.searchsorted(sorted_group, sorted_group)
    size = np.searchsorted(sorted_group, sorted_group, side="right") - first
    rank = np.empty(len(group), np.int64)
    rank[order] = np.arange(len(group)) - first
    sizes = np.empty(len(group), np.int64)
    sizes[order] = size
    return rank, sizes


def split_evenly(group: np.ndarray, size: float, rng) -> np.ndarray:
    """Divide the members of every group into parts of about `size` people (at least one part a
    group, parts as equal as possible); returns a part number that is unique over all groups."""
    rank, members = ranks_within(group, rng)
    parts = np.maximum(np.rint(members / size).astype(np.int64), 1)
    part_in_group = rank * parts // members
    _, part = np.unique(
        group.astype(np.int64) * (int(parts.max()) + 1) + part_in_group, return_inverse=True
    )
    return part


def classes(
    site: np.ndarray,
    age: np.ndarray,
    class_size: float,
    min_class_size: int,
    age_span: int,
    oldest_age: int,
    rng,
) -> np.ndarray:
    """The class of every pupil. The pupils of a school are grouped by year of age (`age_span`
    years together, everybody from `oldest_age` with the oldest); going from the youngest year up,
    a year with fewer than `min_class_size` pupils is taught together with the next years until
    the group has that many, as small schools teach several years in one class, and a small
    oldest group joins the one before it. The groups are then divided into classes of about
    `class_size`."""
    year = np.minimum(age.astype(np.int64), oldest_age) // age_span
    keys, inverse, counts = np.unique(
        site.astype(np.int64) * 1000 + year, return_inverse=True, return_counts=True
    )
    label = np.empty(len(keys), np.int64)
    group, running = 0, 0
    for index in range(len(keys)):
        if index and keys[index] // 1000 != keys[index - 1] // 1000:  # a new school
            if running and running < min_class_size and group > 0:
                label[label == group] = group - 1  # its small oldest group joins the one before
            group, running = group + (running > 0), 0
        label[index] = group
        running += counts[index]
        if running >= min_class_size:
            group, running = group + 1, 0
    if running and running < min_class_size and group > 0:
        previous = label[label != group]
        if len(previous) and keys[np.flatnonzero(label != group)[-1]] // 1000 == keys[-1] // 1000:
            label[label == group] = group - 1
    return split_evenly(label[inverse], class_size, rng)


@dataclass(frozen=True)
class SeatLayout:
    """Double desks in rows: the two seats of a desk `pair_gap` apart, desks `column_gap` apart in
    a row and the rows `row_gap` apart (m)."""

    pair_gap: float
    column_gap: float
    row_gap: float


def seats(room: np.ndarray, center_x, center_y, layout: SeatLayout, rng):
    """A seat for every member of a room (`room` numbers from 0; `center_x`, `center_y` by
    room): double desks in rows around the middle of the room, the members in a random order, so
    that each has a desk-mate and neighbours in front and behind."""
    rank, members = ranks_within(room, rng)
    desks = (members + 1) // 2
    per_row = np.ceil(np.sqrt(desks)).astype(np.int64)
    rows = np.ceil(desks / per_row).astype(np.int64)
    desk = rank // 2
    column, row = desk % per_row, desk // per_row
    x = (column - (per_row - 1) / 2) * layout.column_gap + (rank % 2 - 0.5) * layout.pair_gap
    y = (row - (rows - 1) / 2) * layout.row_gap
    return center_x[room] + x, center_y[room] + y


def desks(room: np.ndarray, center_x, center_y, spacing: float, rng):
    """A desk for every member of a room: a square grid of desks `spacing` apart around the
    middle of the room, the members in a random order."""
    rank, members = ranks_within(room, rng)
    per_row = np.ceil(np.sqrt(members)).astype(np.int64)
    rows = np.ceil(members / per_row).astype(np.int64)
    column, row = rank % per_row, rank // per_row
    x = (column - (per_row - 1) / 2) * spacing
    y = (row - (rows - 1) / 2) * spacing
    return center_x[room] + x, center_y[room] + y


@dataclass
class Rotation:
    """Teachers who go from class to class (subject teachers): `teacher` are the people,
    `school` their school (0, 1, ...); the classes they teach of school s are
    `class_venue[start[s]:start[s + 1]]` with their middles and radii."""

    teacher: np.ndarray
    school: np.ndarray
    start: np.ndarray
    class_venue: np.ndarray
    class_x: np.ndarray
    class_y: np.ndarray
    class_radius: np.ndarray

    @classmethod
    def empty(cls) -> "Rotation":
        nothing = np.zeros(0, np.int64)
        return cls(nothing, nothing, np.zeros(1, np.int64), nothing, *(np.zeros(0),) * 3)

    def assign(self, present: np.ndarray, rng) -> np.ndarray:
        """The class (an index into `class_venue`) every teacher teaches this lesson, -1 for a
        teacher who has none (in the staff room) or is not `present` (by teacher)."""
        chosen = np.full(len(self.teacher), -1, np.int64)
        here = np.flatnonzero(present)
        if not len(here):
            return chosen
        rank, _ = ranks_within(self.school[here], rng)
        school = self.school[here]
        classes_of = self.start[school + 1] - self.start[school]
        teaches = rank < classes_of
        chosen[here[teaches]] = self.start[school[teaches]] + rank[teaches]
        return chosen

    def extend(self, other: "Rotation") -> "Rotation":
        """The teachers of both, the schools of `other` numbered after these."""
        schools = len(self.start) - 1
        offset = len(self.class_venue)
        return Rotation(
            np.concatenate([self.teacher, other.teacher]),
            np.concatenate([self.school, other.school + schools]),
            np.concatenate([self.start[:-1], other.start + offset]),
            np.concatenate([self.class_venue, other.class_venue]),
            np.concatenate([self.class_x, other.class_x]),
            np.concatenate([self.class_y, other.class_y]),
            np.concatenate([self.class_radius, other.class_radius]),
        )


def ring_positions(
    index: np.ndarray, count: np.ndarray, circle: np.ndarray, center_x, center_y, distance, rng
):
    """Points evenly spread on a circle of `distance` around a center: `index` of `count` on the
    circle `circle` of the point (numbered from 0), the circles turned at random."""
    turn = rng.random(int(circle.max()) + 1)[circle] * 2 * math.pi
    angle = 2 * math.pi * index / np.maximum(count, 1) + turn
    return center_x + distance * np.cos(angle), center_y + distance * np.sin(angle)


# Source: a lesson lasts 45 minutes (Act 561/2004 Coll., section 26(1)) and is followed by a break
# of at least 10 (decree 48/2005 Coll., section 1(5)): teachers who go from class to class change
# class about every hour.
LESSON_HOURS = 1.0
