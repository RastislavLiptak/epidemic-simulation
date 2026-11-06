"""Numba kernels for moving people and finding infectious people near them.

All kernels are deterministic for given inputs (randomness is drawn outside), so a run
is reproducible whatever the number of threads.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
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


@njit(cache=True, parallel=True)
def count_infectious_contacts_by_household(
    x,
    y,
    susceptible,
    infectiousness,
    household,
    venue,
    cell_of,
    cell_start,
    people_by_cell,
    cells_per_side,
    radius,
):
    """For every susceptible person, the infectiousness summed over people within `radius` in the
    same venue, and the part of it that comes from members of their own household.

    `infectiousness` is 0 for people who cannot infect and 1 for a fully infectious person.
    `venue` is the place everybody is in (see `town.Venues`); people in different venues (behind
    a wall) and people on the way (venue -1) are not in contact. Cells must be at least `radius`
    wide so that the 3x3 neighbourhood contains every contact.
    Returns (all contacts, contacts in the own household).
    """
    count = x.shape[0]
    cell_has_infectious = np.zeros(cells_per_side * cells_per_side, np.bool_)
    for person in range(count):
        if infectiousness[person] > 0:
            cell_has_infectious[cell_of[person]] = True
    contacts = np.zeros(count, np.float32)
    at_home = np.zeros(count, np.float32)
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
                        if delta_x * delta_x + delta_y * delta_y <= radius_squared:
                            found += infectiousness[other]
                            if household[other] == household[person]:
                                own += infectiousness[other]
        contacts[person] = found
        at_home[person] = own
    return contacts, at_home


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
    the same venue."""
    return count_infectious_contacts_by_household(
        x,
        y,
        susceptible,
        infectiousness,
        np.zeros(x.shape[0], np.int32),
        venue,
        cell_of,
        cell_start,
        people_by_cell,
        cells_per_side,
        radius,
    )[0]


@njit(cache=True, parallel=True)
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
    venue,
    community,
    cell_of,
    cell_start,
    people_by_cell,
    cells_per_side,
    radius,
):
    """For each person in `targets`, one of the infectious people in contact with them, chosen
    with the chance of their share of the exposure (see `count_infectious_contacts_by_household`:
    the own household at weight 1, everybody else at `community`); -1 if there is none.

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
                        if delta_x * delta_x + delta_y * delta_y > radius_squared:
                            continue
                        weight = infectiousness[other]
                        if household[other] != household[person]:
                            weight *= community
                        total += weight
                        if sweep == 1 and infectors[index] < 0 and total >= threshold:
                            infectors[index] = other
            if sweep == 0:
                if total <= 0:
                    break
                threshold = draws[index] * total
    return infectors
