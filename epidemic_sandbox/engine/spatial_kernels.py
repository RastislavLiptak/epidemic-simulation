"""Numba kernels for moving people and finding infectious people near them.

All kernels are deterministic for given inputs (randomness is drawn outside), so a run
is reproducible whatever the number of threads.

A kernel that visits every person exists in two compiled versions, one on a single thread and one
on all of them (`sized_kernel`). Starting and joining the threads costs about a millisecond a
call, which is more than the work in a town of a few thousand people.
"""

import functools
import types

import numpy as np
from numba import njit, prange

# Measured on the model with three diseases during an epidemic: a step took 2.2 ms with the
# threads and 1.0 ms without them for 3,000 people, 2.8 and 2.0 for 10,000, 5.3 and 5.1 for
# 30,000, and 14.2 ms and 15.5 for 100,000.
PARALLEL_FROM_PEOPLE = 50_000


def _compile(function, name: str, parallel: bool):
    # A copy under its own name, because Numba names the files of its cache after the function
    # and the two versions must not share one.
    copy = types.FunctionType(
        function.__code__, function.__globals__, name, function.__defaults__, function.__closure__
    )
    copy.__qualname__ = name
    return njit(cache=True, parallel=parallel, nogil=True)(copy)


def sized_kernel(function=None, *, from_people: int = PARALLEL_FROM_PEOPLE):
    """`function` compiled for one thread and for all threads; a call uses the version that suits
    the number of people, the length of its first argument (or of the first array in it, if that
    is a tuple of arrays): the threads from `from_people` on. A kernel that does little for each
    person needs more people to pay for the start of the threads."""

    def build(function):
        serial = _compile(function, f"{function.__name__}_serial", False)
        parallel = _compile(function, f"{function.__name__}_parallel", True)

        @functools.wraps(function)
        def kernel(*arguments):
            first = arguments[0]
            people = (first[0] if isinstance(first, tuple) else first).shape[0]
            return (kernel.parallel if people >= from_people else kernel.serial)(*arguments)

        kernel.serial, kernel.parallel = serial, parallel
        return kernel

    return build if function is None else build(function)


@sized_kernel
def move_toward(x, y, target_x, target_y, home_x, home_y, stay_home, max_distance):
    """Move everybody up to their `max_distance` toward the target (home if they stay in)."""
    for person in prange(x.shape[0]):
        goal_x = home_x[person] if stay_home[person] else target_x[person]
        goal_y = home_y[person] if stay_home[person] else target_y[person]
        delta_x = goal_x - x[person]
        delta_y = goal_y - y[person]
        distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
        if distance <= max_distance[person]:
            x[person] = goal_x
            y[person] = goal_y
        else:
            share = max_distance[person] / distance
            x[person] += delta_x * share
            y[person] += delta_y * share


@njit(cache=True)
def build_grid(x, y, size, cells_per_side):
    """Sort people into grid cells; returns (cell of person, cell starts, people by cell)."""
    count = x.shape[0]
    cell_size = size / cells_per_side
    cell_of = np.empty(count, np.int32)
    cell_start = np.zeros(cells_per_side * cells_per_side + 1, np.int32)
    for person in range(count):
        cell_x = min(int(x[person] / cell_size), cells_per_side - 1)
        cell_y = min(int(y[person] / cell_size), cells_per_side - 1)
        cell = cell_y * cells_per_side + cell_x
        cell_of[person] = cell
        cell_start[cell + 1] += 1
    for cell in range(cell_start.shape[0] - 1):
        cell_start[cell + 1] += cell_start[cell]
    next_slot = cell_start[:-1].copy()
    people_by_cell = np.empty(count, np.int32)
    for person in range(count):
        cell = cell_of[person]
        people_by_cell[next_slot[cell]] = person
        next_slot[cell] += 1
    return cell_of, cell_start, people_by_cell


@sized_kernel
def locate_in_cdf(draws, cdf):
    """Where each of the uniform `draws` falls in the cumulative distribution `cdf`: what
    `np.searchsorted(cdf, draws)` gives, spread over the threads (numpy takes a second for the
    draws of 200,000 people, since every search leaps about in a table too big for the cache)."""
    found = np.empty(draws.shape, np.int64)
    for row in prange(draws.shape[0]):
        for column in range(draws.shape[1]):
            found[row, column] = np.searchsorted(cdf, draws[row, column])
    return found


@njit(cache=True, nogil=True, inline="always")
def proximity(distance_squared, near):
    """How much an infectious person at this squared distance exposes someone, relative to one
    within `near`: fully up to `near`, then falling with the square of the distance (the risk model
    of the NHS COVID-19 app, "inverse square of the distance if greater than 1 m", whose scores
    Ferretti et al. 2024, Nature 626:145, found proportional to the chance of transmission)."""
    near_squared = near * near
    if distance_squared <= near_squared:
        return 1.0
    return near_squared / distance_squared


@sized_kernel
def count_infectious_contacts_by_household(
    x,
    y,
    susceptible,
    infectiousness,
    household,
    at_home,
    venue,
    cell_of,
    cell_start,
    people_by_cell,
    cells_per_side,
    radius,
    near,
):
    """For every susceptible person, the exposure to the infectious people within `radius` in the
    same venue, and the part of it that comes from members of their own household at home.

    The exposure is the infectiousness of each infectious person (0 for people who cannot infect,
    1 for a fully infectious person) weighted by `proximity` with the full distance `near`.
    `venue` is the place everybody is in (see `town.Venues`); people in different venues (behind
    a wall) and people on the way (venue -1) are not in contact. `at_home` marks who is in their
    own home: members of one household count as such only there (out together they meet like
    anybody else). Cells must be at least `radius` wide so that the 3x3 neighbourhood contains
    every contact.
    Returns (all exposure, exposure from the own household at home).
    """
    count = x.shape[0]
    cell_has_infectious = np.zeros(cells_per_side * cells_per_side, np.bool_)
    for person in range(count):
        if infectiousness[person] > 0:
            cell_has_infectious[cell_of[person]] = True
    contacts = np.zeros(count, np.float32)
    at_home_exposure = np.zeros(count, np.float32)
    radius_squared = radius * radius
    for person in prange(count):
        if not susceptible[person] or venue[person] < 0:
            continue
        cell_y = cell_of[person] // cells_per_side
        cell_x = cell_of[person] - cell_y * cells_per_side
        found = np.float32(0.0)
        own = np.float32(0.0)
        for neighbour_y in range(max(cell_y - 1, 0), min(cell_y + 2, cells_per_side)):
            for neighbour_x in range(max(cell_x - 1, 0), min(cell_x + 2, cells_per_side)):
                cell = neighbour_y * cells_per_side + neighbour_x
                if not cell_has_infectious[cell]:
                    continue
                for slot in range(cell_start[cell], cell_start[cell + 1]):
                    other = people_by_cell[slot]
                    if infectiousness[other] > 0 and venue[other] == venue[person]:
                        delta_x = x[other] - x[person]
                        delta_y = y[other] - y[person]
                        squared = delta_x * delta_x + delta_y * delta_y
                        if squared <= radius_squared:
                            weight = np.float32(infectiousness[other] * proximity(squared, near))
                            found += weight
                            if (
                                household[other] == household[person]
                                and at_home[person]
                                and at_home[other]
                            ):
                                own += weight
        contacts[person] = found
        at_home_exposure[person] = own
    return contacts, at_home_exposure


def count_infectious_contacts(
    x,
    y,
    susceptible,
    infectiousness,
    venue,
    cell_of,
    cell_start,
    people_by_cell,
    cells_per_side,
    radius,
):
    """For every susceptible person, the infectiousness summed over people within `radius` in
    the same venue, whatever their distance within it."""
    count = x.shape[0]
    return count_infectious_contacts_by_household(
        x,
        y,
        susceptible,
        infectiousness,
        np.zeros(count, np.int32),
        np.zeros(count, np.bool_),
        venue,
        cell_of,
        cell_start,
        people_by_cell,
        cells_per_side,
        radius,
        radius,
    )[0]


@sized_kernel
def find_pairs(x, y, member, venue, cell_of, cell_start, people_by_cell, cells_per_side, radius):
    """All pairs (first < second) of `member` people within `radius` of each other in the same
    venue (see `count_infectious_contacts_by_household`)."""
    count = x.shape[0]
    radius_squared = radius * radius
    found = np.zeros(count, np.int64)
    for fill in range(2):
        if fill == 1:
            offset = np.zeros(count + 1, np.int64)
            for person in range(count):
                offset[person + 1] = offset[person] + found[person]
            first = np.empty(offset[count], np.int32)
            second = np.empty(offset[count], np.int32)
        for person in prange(count):
            if not member[person] or venue[person] < 0:
                continue
            cell_y = cell_of[person] // cells_per_side
            cell_x = cell_of[person] - cell_y * cells_per_side
            written = 0
            for neighbour_y in range(max(cell_y - 1, 0), min(cell_y + 2, cells_per_side)):
                for neighbour_x in range(max(cell_x - 1, 0), min(cell_x + 2, cells_per_side)):
                    cell = neighbour_y * cells_per_side + neighbour_x
                    for slot in range(cell_start[cell], cell_start[cell + 1]):
                        other = people_by_cell[slot]
                        if other > person and member[other] and venue[other] == venue[person]:
                            delta_x = x[other] - x[person]
                            delta_y = y[other] - y[person]
                            if delta_x * delta_x + delta_y * delta_y <= radius_squared:
                                if fill == 1:
                                    first[offset[person] + written] = person
                                    second[offset[person] + written] = other
                                written += 1
            if fill == 0:
                found[person] = written
    return first, second


@njit(cache=True)
def pick_infectors(
    x,
    y,
    targets,
    draws,
    infectiousness,
    household,
    at_home,
    venue,
    community,
    cell_of,
    cell_start,
    people_by_cell,
    cells_per_side,
    radius,
    near,
):
    """For each person in `targets`, one of the infectious people in contact with them, chosen
    with the chance of their share of the exposure (see `count_infectious_contacts_by_household`:
    weighted by proximity, the own household at home at weight 1, everybody else at `community`);
    -1 if there is none.

    `draws` are uniform random numbers, one per target, so that the kernel stays deterministic.
    """
    infectors = np.full(targets.shape[0], -1, np.int64)
    radius_squared = radius * radius
    for index in range(targets.shape[0]):
        person = targets[index]
        if venue[person] < 0:
            continue
        cell_y = cell_of[person] // cells_per_side
        cell_x = cell_of[person] - cell_y * cells_per_side
        threshold = 0.0
        for sweep in range(2):  # the total first, then the pick
            total = 0.0
            for neighbour_y in range(max(cell_y - 1, 0), min(cell_y + 2, cells_per_side)):
                for neighbour_x in range(max(cell_x - 1, 0), min(cell_x + 2, cells_per_side)):
                    cell = neighbour_y * cells_per_side + neighbour_x
                    for slot in range(cell_start[cell], cell_start[cell + 1]):
                        other = people_by_cell[slot]
                        if infectiousness[other] <= 0 or venue[other] != venue[person]:
                            continue
                        delta_x = x[other] - x[person]
                        delta_y = y[other] - y[person]
                        squared = delta_x * delta_x + delta_y * delta_y
                        if squared > radius_squared:
                            continue
                        weight = infectiousness[other] * proximity(squared, near)
                        if not (
                            household[other] == household[person]
                            and at_home[person]
                            and at_home[other]
                        ):
                            weight *= community
                        total += weight
                        if sweep == 1 and infectors[index] < 0 and total >= threshold:
                            infectors[index] = other
            if sweep == 0:
                if total <= 0:
                    break
                threshold = draws[index] * total
    return infectors
