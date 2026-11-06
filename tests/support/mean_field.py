"""Run a disease model in a well-mixed population, to compare it with its textbook equations.

There is no town here: everybody meets everybody equally, so the force of infection on a person
is `contacts` times the infectiousness-weighted share of infectious people (the per-day hazard of
a contact is `-ln(1 - transmission)`). This is exactly the assumption of compartmental ODE models,
which makes the average course of many people comparable with them. Use `duration_shape=1` to
compare with the usual ODEs (exponentially distributed stays) or an Erlang chain of `shape` stages
per state to compare with the gamma stays the models use; set `susceptibility_cv=0` and
`infectiousness_dispersion=0` for a population of identical people.
"""

import math

import numpy as np

AVERAGE_AGE = 40


def run_well_mixed(
    model_class,
    population: int,
    cases: int,
    days: float,
    contacts: float,
    seed: int = 0,
    step_days: float = 1 / 24,
    **parameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts of people in each state (columns, in the order of STATE_COLORS) at every step.

    Returns (times in days, counts) with one row at the start and one after every step. People
    who are dead are counted in the dead state.
    """
    model = model_class(step_days, **parameters)
    rng = np.random.default_rng(seed)
    ages = np.full(population, AVERAGE_AGE)
    state = model.create_state(population, rng.choice(population, cases, replace=False), rng, ages)
    hazard = -math.log(1.0 - model.parameters["transmission"])
    steps = round(days / step_days)
    counts = np.zeros((steps + 1, len(model.STATE_COLORS)))
    counts[0] = np.bincount(state.state, minlength=counts.shape[1])
    for step in range(steps):
        spreading = (model.infectious_weight(state) * state.infectiousness).sum()
        # the exposure is in contacts per day; the model turns it into a chance with `hazard`
        exposure = np.full(population, contacts * spreading / population, np.float32)
        assert hazard > 0 or spreading == 0
        model.advance(state, exposure, rng)
        counts[step + 1] = np.bincount(state.state, minlength=counts.shape[1])
    return np.arange(steps + 1) * step_days, counts


def mean_course(model_class, runs: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """`run_well_mixed` averaged over `runs` seeds."""
    seed = kwargs.pop("seed", 0)
    results = [run_well_mixed(model_class, seed=seed + run, **kwargs) for run in range(runs)]
    return results[0][0], np.mean([counts for _, counts in results], axis=0)


def integrate(derivative, start: np.ndarray, days: float, step_days: float = 1 / 24) -> np.ndarray:
    """Runge-Kutta 4 solution of d(y)/dt = derivative(y), sampled at every `step_days`."""
    steps = round(days / step_days)
    solution = np.zeros((steps + 1, len(start)))
    solution[0] = y = np.asarray(start, float)
    for step in range(steps):
        k1 = derivative(y)
        k2 = derivative(y + step_days / 2 * k1)
        k3 = derivative(y + step_days / 2 * k2)
        k4 = derivative(y + step_days * k3)
        y = y + step_days / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        solution[step + 1] = y
    return solution


def final_size(r0: float, initial_infected_share: float = 0.0) -> float:
    """Share ever infected in an SIR epidemic: the root of z = (1 - i0)(1 - exp(-R0 z)) + i0."""
    z = 0.5
    for _ in range(200):
        z = initial_infected_share + (1 - initial_infected_share) * (1 - math.exp(-r0 * z))
    return z
