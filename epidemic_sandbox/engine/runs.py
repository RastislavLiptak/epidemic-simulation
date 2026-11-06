"""The runs of an experiment: independent copies of the town, stepped together.

`LocalRuns` computes them in this process, one after another. `PooledRuns` hands them to worker
processes (see `workers`), which compute them at the same time: the runs share nothing, so a
machine with several cores finishes several runs in the time of one. Both give the same numbers
for the same seeds, and both answer the questions the UI asks (counts, history, positions of a
town, a person) without the caller knowing where the runs live.
"""

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from epidemic_sandbox.config.experiment import ExperimentConfig
from epidemic_sandbox.engine.epidemic_model import EpidemicModel
from epidemic_sandbox.engine.workers import Worker, WorkerPool
from epidemic_sandbox.world.schedule import DAYS_PER_WEEK, HOURS_PER_DAY
from epidemic_sandbox.world.town import TownParameters

POSITION_DECIMALS = 4  # positions go to the browser as shares of the arena, to 1/10,000


class LocalRuns:
    """Runs computed one after another in this process."""

    def __init__(self, configs: Sequence[ExperimentConfig], town_parameters: TownParameters):
        self.models = [EpidemicModel(config, town_parameters) for config in configs]

    @property
    def seeds(self) -> list[int]:
        return [model.world["seed"] for model in self.models]

    @property
    def clock(self) -> EpidemicModel:
        """The model that tells the time, which is the same in all runs."""
        return self.models[0]

    def step(self, count: int, time_limit: float | None = None) -> int:
        """Run up to `count` steps of all runs together; fewer after `time_limit` seconds. Returns
        the number of steps done (at least one)."""
        started = perf_counter()
        done = 0
        while done < count and (
            time_limit is None or done == 0 or perf_counter() - started < time_limit
        ):
            for model in self.models:
                model.step()
            done += 1
        return done

    def report(self, history_from: int, watched: int | None, visible: int) -> list[dict]:
        """Per run: the counts of each state, the days of history from `history_from` on and,
        for the run `watched`, where the first `visible` people are and in which state."""
        reports = []
        for index, model in enumerate(self.models):
            data = {"counts": model.state_counts(), "history": model.history[history_from:]}
            if index == watched:
                data.update(_positions(model, visible))
            reports.append(data)
        return reports

    def people(self, run: int, shown: int) -> tuple[list[int], list[int]]:
        town = self.models[run].town
        return town.age[:shown].tolist(), town.role[:shown].tolist()

    def person(self, run: int, person: int) -> dict:
        from epidemic_sandbox.app.person_info import describe_person

        return describe_person(self.models[run], person)

    def close(self) -> None:
        self.models = []


class PooledRuns:
    """Runs computed by worker processes, several at the same time."""

    def __init__(
        self,
        pool: WorkerPool,
        configs: Sequence[ExperimentConfig],
        town_parameters: TownParameters,
    ):
        self.pool = pool
        self.configs = list(configs)
        self.key = pool.new_key()
        self.workers = pool.start(self.key, self.configs, town_parameters)
        self.steps = 0
        self._seconds_per_step = None

    @property
    def seeds(self) -> list[int]:
        return [config.world["seed"] for config in self.configs]

    @property
    def clock(self):
        return _Clock(self.configs[0], self.steps)

    def step(self, count: int, time_limit: float | None = None) -> int:
        """As `LocalRuns.step`. The workers are told how many steps to take, so the number that
        fits in `time_limit` is estimated from the steps before."""
        started = perf_counter()
        done = 0
        while done < count:
            remaining = count - done
            if time_limit is None:
                chunk = remaining
            else:
                left = time_limit - (perf_counter() - started)
                if done and left <= 0:
                    break
                affordable = 1 if not self._seconds_per_step else int(left / self._seconds_per_step)
                chunk = max(1, min(remaining, affordable))
            chunk_started = perf_counter()
            self.pool.ask_all(self.workers, "step", self.key, chunk)
            self._seconds_per_step = (perf_counter() - chunk_started) / chunk
            done += chunk
        self.steps += done
        return done

    def report(self, history_from: int, watched: int | None, visible: int) -> list[dict]:
        answers = self.pool.ask_all(
            self.workers, "report", self.key, history_from, watched, visible
        )
        reports = [None] * len(self.configs)
        for answer in answers:
            for index, data in answer.items():
                reports[index] = data
        return reports

    def people(self, run: int, shown: int) -> tuple[list[int], list[int]]:
        return self.pool.ask(self._owner(run), "people", self.key, run, shown)

    def person(self, run: int, person: int) -> dict:
        return self.pool.ask(self._owner(run), "person", self.key, run, person)

    def close(self) -> None:
        self.pool.ask_all(self.workers, "drop", self.key)

    def _owner(self, run: int) -> Worker:
        return self.workers[run % len(self.workers)]


class _Clock:
    """What the UI needs from a model to tell the time and the size of a step."""

    def __init__(self, config: ExperimentConfig, steps: int):
        self.world = config.world
        self.steps = steps

    @property
    def population(self) -> int:
        return self.world["population"]

    @property
    def step_hours(self) -> float:
        return HOURS_PER_DAY / self.world["steps_per_day"]

    @property
    def day(self) -> float:
        return self.steps / self.world["steps_per_day"]

    @property
    def weekday(self) -> int:
        return (self.steps // self.world["steps_per_day"]) % DAYS_PER_WEEK

    @property
    def hour(self) -> float:
        steps_per_day = self.world["steps_per_day"]
        return (self.steps % steps_per_day) * HOURS_PER_DAY / steps_per_day


def _positions(model: EpidemicModel, visible: int) -> dict:
    """Where the first `visible` people are (shares of the arena) and the state of each."""
    visible = min(visible, model.population)
    arena_size = model.world["arena_size"]
    return {
        "x": np.round(model.x[:visible] / arena_size, POSITION_DECIMALS).tolist(),
        "y": np.round(model.y[:visible] / arena_size, POSITION_DECIMALS).tolist(),
        "states": {
            name: state.state[:visible].tolist() for name, state in model.disease_states.items()
        },
    }
