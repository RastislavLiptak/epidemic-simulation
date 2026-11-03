"""Helpers for tests that drive a disease model directly, without a town."""

import numpy as np

ONE_DAY = 1.0
SURE = {"transmission": 1.0}
UNIFORM = {"susceptibility_cv": 0.0}
ADULT = 40


def new_state(model, population=1, cases=(), age=ADULT, seed=0):
    rng = np.random.default_rng(seed)
    ages = np.full(population, age) if np.isscalar(age) else np.asarray(age)
    return model.create_state(population, np.array(cases, dtype=int), rng, ages), rng


def advance(model, disease_state, rng, exposure=0.0):
    model.advance(disease_state, np.full(len(disease_state.state), exposure, np.float32), rng)


def state_names(model, disease_state):
    names = list(model.STATE_COLORS)
    return [names[index] for index in disease_state.state]


def count_in(model, disease_state, state: str) -> int:
    return int(np.count_nonzero(disease_state.state == list(model.STATE_COLORS).index(state)))


def place(model, disease_state, rng, people, state: str) -> None:
    """Put `people` into `state` the way the model does (timer, behaviour), without infecting."""
    model._enter(disease_state, np.asarray(people, dtype=int), state, rng)
