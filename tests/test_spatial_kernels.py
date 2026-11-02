import numpy as np
import pytest

from epidemic_sandbox.spatial_kernels import (
    build_grid,
    count_infectious_contacts,
    find_pairs,
    move_toward,
)


@pytest.mark.parametrize(
    ("count", "size", "radius"), [(300, 100.0, 7.0), (500, 50.0, 24.0), (200, 100.0, 1.0)]
)
def test_grid_contacts_match_brute_force_with_weights(count, size, radius):
    rng = np.random.default_rng(1)
    x, y = rng.random(count) * size, rng.random(count) * size
    weights = (rng.random(count) < 0.2) * rng.choice([0.5, 1.0, 2.0], count).astype(np.float32)
    susceptible = weights == 0
    cells = max(1, min(int(size // radius), int(np.sqrt(count))))
    grid = build_grid(x, y, size, cells)
    fast = count_infectious_contacts(x, y, susceptible, weights, *grid, cells, radius)
    squared = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    brute = ((squared <= radius**2) * weights[None, :]).sum(axis=1) * susceptible
    assert fast == pytest.approx(brute, abs=1e-4)


def test_grid_lists_every_person_exactly_once():
    rng = np.random.default_rng(2)
    x, y = rng.random(1000) * 80, rng.random(1000) * 80
    cell_of, cell_start, people_by_cell = build_grid(x, y, 80.0, 8)
    assert sorted(people_by_cell) == list(range(1000))
    for cell in range(64):
        members = people_by_cell[cell_start[cell] : cell_start[cell + 1]]
        assert all(cell_of[person] == cell for person in members)


def walk(start, target, home, stay_home, distance):
    arrays = [np.array([value], dtype=float) for value in (*start, *target, *home)]
    x, y, target_x, target_y, home_x, home_y = arrays
    move_toward(
        x, y, target_x, target_y, home_x, home_y, np.array([stay_home]), np.array([distance])
    )
    return x[0], y[0]


def test_people_move_toward_the_target_by_at_most_the_distance():
    assert walk((0, 0), (30, 40), (0, 0), False, 10.0) == pytest.approx((6.0, 8.0))


def test_people_arrive_exactly_when_the_target_is_within_reach():
    assert walk((0, 0), (3, 4), (0, 0), False, 10.0) == (3.0, 4.0)


def test_people_who_stay_home_walk_home_instead():
    assert walk((0, 0), (30, 40), (0, 10), True, 100.0) == (0.0, 10.0)


def test_standing_still_at_the_target_stays_put():
    assert walk((5, 5), (5, 5), (0, 0), False, 10.0) == (5.0, 5.0)


def test_every_person_has_their_own_distance_budget():
    x, y = np.zeros(3), np.zeros(3)
    move_toward(
        x,
        y,
        np.full(3, 100.0),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3, np.bool_),
        np.array([1.0, 10.0, 1000.0]),
    )
    assert x.tolist() == [1.0, 10.0, 100.0]


@pytest.mark.parametrize("radius", [1.0, 6.0, 15.0])
def test_found_pairs_match_brute_force_between_selected_people(radius):
    rng = np.random.default_rng(3)
    count, size = 400, 100.0
    x, y = rng.random(count) * size, rng.random(count) * size
    selected = rng.random(count) < 0.6
    cells = max(1, min(int(size // radius), int(np.sqrt(count))))
    first, second = find_pairs(x, y, selected, *build_grid(x, y, size, cells), cells, radius)
    found = {tuple(sorted(pair)) for pair in zip(first.tolist(), second.tolist(), strict=True)}
    squared = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    near = np.argwhere(
        (squared <= radius**2) & selected[:, None] & selected[None, :] & ~np.eye(count, dtype=bool)
    )
    assert found == {tuple(sorted(pair)) for pair in near.tolist()}
