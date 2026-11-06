"""Runs computed by worker processes are the runs computed here, only sooner."""

import os

import pytest
from config_factory import APP_CONFIG, LIBRARY
from session_helpers import small_experiment

from epidemic_sandbox.app.simulation import Simulation
from epidemic_sandbox.engine.runs import LocalRuns, PooledRuns
from epidemic_sandbox.engine.workers import WorkerPool

RUNS = 3


@pytest.fixture(scope="module")
def pool():
    workers = WorkerPool(2)
    yield workers
    workers.close()


def started(workers=None, **world) -> Simulation:
    simulation = Simulation(LIBRARY, APP_CONFIG, workers)
    simulation.handle(
        {"type": "start", "experiment": small_experiment(runs=RUNS, seed=5, **world)}
    )
    return simulation


def test_runs_in_worker_processes_give_exactly_the_numbers_of_runs_computed_here(pool):
    here, there = started(), started(pool)
    assert isinstance(here.runs, LocalRuns) and isinstance(there.runs, PooledRuns)
    assert len(there.runs.workers) == 2, "three runs on two workers"
    for simulation in (here, there):
        simulation.handle({"type": "watch", "run": 1})
        simulation.handle({"type": "step_day"})
        simulation.handle({"type": "step_day"})
    first, second = here.frame(), there.frame()
    assert first["day"] == second["day"] == 2
    for run, other in zip(first["runs"], second["runs"], strict=True):
        assert run["counts"] == other["counts"]
        assert run["history"] == other["history"]
    assert first["runs"][1]["x"] == second["runs"][1]["x"]
    assert first["runs"][1]["states"] == second["runs"][1]["states"]
    assert "x" not in second["runs"][0]
    here.close()
    there.close()


def test_a_person_and_the_people_of_a_run_are_described_by_the_worker_that_has_them(pool):
    here, there = started(), started(pool)
    for simulation in (here, there):
        simulation.handle({"type": "step_day"})
    for run in range(RUNS):
        watch = {"type": "watch", "run": run}
        assert there.handle(watch) == here.handle(watch)
        asked = {"type": "inspect", "run": run, "person": 7}
        assert there.handle(asked) == here.handle(asked)
    with pytest.raises(ValueError, match="Unknown person"):
        there.handle({"type": "inspect", "run": 0, "person": 10_000})
    here.close()
    there.close()


def test_stepping_within_a_time_limit_takes_at_least_one_step_and_never_more_than_asked(pool):
    there = started(pool)
    runs = there.runs
    assert runs.step(5, time_limit=0.0) == 1
    assert runs.step(7, time_limit=60.0) == 7
    assert runs.clock.steps == 8
    here = started()
    here.runs.step(8)
    assert there.frame()["runs"][0]["counts"] == here.frame()["runs"][0]["counts"]
    here.close()
    there.close()


def test_a_new_experiment_replaces_the_old_one_on_the_workers_and_a_bad_request_is_refused(pool):
    simulation = started(pool)
    first_key = simulation.runs.key
    simulation.handle({"type": "start", "experiment": small_experiment(runs=2)})
    assert simulation.runs.key != first_key
    with pytest.raises(ValueError, match="Unknown request"):
        pool.ask(simulation.runs.workers[0], "dance", simulation.runs.key)
    assert simulation.handle({"type": "step_day"}) is None, "the worker survived"
    simulation.close()


def test_a_single_run_is_computed_here_even_when_workers_exist(pool):
    simulation = Simulation(LIBRARY, APP_CONFIG, pool)
    simulation.handle({"type": "start", "experiment": small_experiment()})
    assert isinstance(simulation.runs, LocalRuns)
    simulation.close()


def test_the_number_of_workers_is_the_number_of_cores_unless_told_otherwise():
    assert WorkerPool().size == os.cpu_count()
    assert WorkerPool(3).size == 3
