import numpy as np
import pytest

from epidemic_sandbox.engine.spatial_kernels import (
    build_grid,
    count_infectious_contacts,
    count_infectious_contacts_by_household,
    find_pairs,
    locate_in_cdf,
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
    venue = rng.integers(-1, 3, count)
    cells = max(1, min(int(size // radius), int(np.sqrt(count))))
    grid = build_grid(x, y, size, cells)
    fast = count_infectious_contacts(x, y, susceptible, weights, venue, *grid, cells, radius)
    squared = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    together = (venue[:, None] == venue[None, :]) & (venue[:, None] >= 0)
    brute = ((squared <= radius**2) * together * weights[None, :]).sum(axis=1) * susceptible
    assert fast == pytest.approx(brute, abs=1e-4)


def test_people_in_different_venues_or_on_the_way_are_not_in_contact():
    x, y = np.zeros(4), np.zeros(4)
    weights = np.array([0.0, 1.0, 1.0, 1.0], np.float32)
    venue = np.array([7, 7, 8, -1])  # the same room, behind a wall, on the way
    grid = build_grid(x, y, 10.0, 1)
    fast = count_infectious_contacts(x, y, weights == 0, weights, venue, *grid, 1, 2.0)
    assert fast[0] == pytest.approx(1.0)
    venue[0] = -1
    fast = count_infectious_contacts(x, y, weights == 0, weights, venue, *grid, 1, 2.0)
    assert fast[0] == 0.0


def reference_exposure(x, y, susceptible, weights, household, at_home, venue, radius, near):
    """Every pair, as array expressions: proximity-weighted exposure, and its part from the own
    household at home."""
    squared = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    reach = (squared <= radius**2) & (venue[:, None] == venue[None, :]) & (venue[:, None] >= 0)
    closeness = np.where(squared <= near**2, 1.0, near**2 / np.maximum(squared, 1e-12))
    exposure = reach * closeness * weights[None, :]
    same = (household[:, None] == household[None, :]) & at_home[:, None] & at_home[None, :]
    return exposure.sum(axis=1) * susceptible, (exposure * same).sum(axis=1) * susceptible


@pytest.mark.parametrize("near", [0.5, 1.0, 9.0])
def test_exposure_is_weighted_by_distance_and_the_household_counts_at_home(near):
    rng = np.random.default_rng(3)
    count, size, radius = 400, 60.0, 9.0
    x, y = rng.random(count) * size, rng.random(count) * size
    household = rng.integers(0, 120, count).astype(np.int32)
    weights = (rng.random(count) < 0.25) * rng.choice([0.5, 1.0], count).astype(np.float32)
    susceptible = weights == 0
    venue = np.where(rng.random(count) < 0.5, household, 1000).astype(np.int64)
    at_home = (venue == household) & (rng.random(count) < 0.8)
    cells = int(size // radius)
    grid = build_grid(x, y, size, cells)
    contacts, own = count_infectious_contacts_by_household(
        x, y, susceptible, weights, household, at_home, venue, *grid, cells, radius, near
    )
    expected, expected_own = reference_exposure(
        x, y, susceptible, weights, household, at_home, venue, radius, near
    )
    assert contacts == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert own == pytest.approx(expected_own, rel=1e-5, abs=1e-5)
    assert own.sum() > 0 and own.sum() < contacts.sum()


def test_exposure_falls_with_the_square_of_the_distance_beyond_the_full_distance():
    x = np.array([0.0, 0.5, 2.0, 4.0, 7.0])
    y = np.zeros(5)
    weights = np.array([0.0, 1.0, 1.0, 1.0, 1.0], np.float32)
    grid = build_grid(x + 1.0, y + 1.0, 10.0, 1)
    contacts, _ = count_infectious_contacts_by_household(
        x + 1.0,
        y + 1.0,
        weights == 0,
        weights,
        np.arange(5, dtype=np.int32),
        np.zeros(5, np.bool_),
        np.zeros(5, np.int64),
        *grid,
        1,
        6.0,
        1.0,
    )
    # 0.5 m: fully; 2 m: a quarter; 4 m: a sixteenth; 7 m: beyond the contact distance
    assert contacts[0] == pytest.approx(1.0 + 0.25 + 1 / 16)


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
    venue = rng.integers(-1, 2, count)
    cells = max(1, min(int(size // radius), int(np.sqrt(count))))
    first, second = find_pairs(x, y, selected, venue, *build_grid(x, y, size, cells), cells, radius)
    found = {tuple(sorted(pair)) for pair in zip(first.tolist(), second.tolist(), strict=True)}
    squared = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    together = (venue[:, None] == venue[None, :]) & (venue[:, None] >= 0)
    near = np.argwhere(
        (squared <= radius**2)
        & together
        & selected[:, None]
        & selected[None, :]
        & ~np.eye(count, dtype=bool)
    )
    assert found == {tuple(sorted(pair)) for pair in near.tolist()}


def test_the_kernel_on_one_thread_and_the_kernel_on_all_threads_give_the_same_numbers():
    rng = np.random.default_rng(7)
    count, size, radius = 3000, 200.0, 6.0
    x, y = rng.random(count) * size, rng.random(count) * size
    household = rng.integers(0, 900, count).astype(np.int32)
    weights = (rng.random(count) < 0.2) * rng.random(count).astype(np.float32)
    venue = rng.integers(0, 3, count).astype(np.int64)
    cells = int(size // radius)
    grid = build_grid(x, y, size, cells)
    at_home = rng.random(count) < 0.5
    arguments = (x, y, weights == 0, weights, household, at_home, venue, *grid, cells, radius, 1.0)
    for one, many in zip(
        count_infectious_contacts_by_household.serial(*arguments),
        count_infectious_contacts_by_household.parallel(*arguments),
        strict=True,
    ):
        assert np.array_equal(one, many)
    first, second = find_pairs.serial(x, y, weights == 0, venue, *grid, cells, radius)
    other_first, other_second = find_pairs.parallel(x, y, weights == 0, venue, *grid, cells, radius)
    assert np.array_equal(first, other_first) and np.array_equal(second, other_second)
    moved_one, moved_many = x.copy(), x.copy()
    distance = np.full(count, 3.0)
    stay = np.zeros(count, np.bool_)
    for kernel, moved in ((move_toward.serial, moved_one), (move_toward.parallel, moved_many)):
        kernel(moved, y.copy(), y, x, y, x, stay, distance)
    assert np.array_equal(moved_one, moved_many)


def test_a_small_town_is_computed_on_one_thread_and_a_big_one_on_all():
    from epidemic_sandbox.engine import spatial_kernels

    def count(people):
        return len(people)

    kernel = spatial_kernels.sized_kernel(count)
    kernel.serial = lambda people: "one thread"
    kernel.parallel = lambda people: "all threads"
    small = np.zeros(spatial_kernels.PARALLEL_FROM_PEOPLE - 1)
    big = np.zeros(spatial_kernels.PARALLEL_FROM_PEOPLE)
    assert kernel(small) == "one thread" and kernel(big) == "all threads"


def test_locating_draws_in_a_distribution_is_what_searchsorted_does():
    rng = np.random.default_rng(5)
    cdf = np.cumsum(rng.lognormal(0.0, 1.0, 5000))
    cdf /= cdf[-1]
    draws = np.concatenate([rng.random((3000, 12)), [[0.0] * 12, [1.0] * 12, [cdf[7]] * 12]])
    expected = np.searchsorted(cdf, draws)
    assert np.array_equal(locate_in_cdf.serial(draws, cdf), expected)
    assert np.array_equal(locate_in_cdf.parallel(draws, cdf), expected)
