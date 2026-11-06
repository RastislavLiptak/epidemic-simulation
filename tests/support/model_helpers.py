"""Helpers that arrange a small model for a test: who stands where and in which state."""

import numpy as np
from config_factory import make_model

SPACING = 12.0


def run(model, steps: int):
    for _ in range(steps):
        model.step()
    return model


def arranged_model(
    diseases: dict, positions: dict[int, tuple[float, float]], interactions=(), **world
):
    """Everybody at home on a grid too sparse for contacts; `positions` moves some people.

    Homes are moved too, so nobody walks away during the night hours the tests run in. Everybody
    stands in one open space, one venue, so that only the distances decide who is in contact.
    """
    model = make_model(diseases, interactions, population=100, arena_size=500, **world)
    for index in range(model.population):
        model.x[index], model.y[index] = 5 + SPACING * (index % 8), 5 + SPACING * (index // 8)
    for index, (x, y) in positions.items():
        model.x[index], model.y[index] = x, y
    model.town.home_x[:], model.town.home_y[:] = model.x, model.y
    model._target_x[:], model._target_y[:] = model.x, model.y
    model._venues = lambda home, work: np.where(model._inside, 0, -1)
    return model


def put_into_state(model, person: int, disease: str, state: str) -> None:
    """Put `person` into `state` of `disease` for far longer than any test runs."""
    disease_state = model.disease_states[disease]
    disease_state.state[person] = list(model.diseases[disease].STATE_COLORS).index(state)
    disease_state.timer[person] = 1e6


def make_infectious(model, person: int, disease: str) -> None:
    put_into_state(model, person, disease, model.diseases[disease].INITIAL_CASE_STATE)


def state_of(model, person: int, disease: str) -> str:
    return list(model.diseases[disease].STATE_COLORS)[model.disease_states[disease].state[person]]
