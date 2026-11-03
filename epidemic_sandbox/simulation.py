from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter

import numpy as np

from epidemic_sandbox.app_config import AppConfig
from epidemic_sandbox.epidemic_model import EpidemicModel
from epidemic_sandbox.experiment_config import ExperimentLibrary
from epidemic_sandbox.parameter_spec import ParameterSpec
from epidemic_sandbox.person_info import describe_person
from epidemic_sandbox.schema import describe_preview, describe_run

DEFAULT_SPEED = 2.0


class Simulation:
    """One user's session: the runs of an experiment, stepped together, driven by UI commands."""

    def __init__(self, library: ExperimentLibrary, app_config: AppConfig):
        self.library = library
        self.frame_budget_seconds = app_config.session.frame_budget_seconds
        self.frame_seconds = app_config.session.frame_seconds
        self.max_displayed = app_config.session.max_displayed_people
        self.max_frame_seconds = app_config.session.max_frame_seconds
        self._speed_spec = ParameterSpec(
            "Speed", "", DEFAULT_SPEED, 0, app_config.session.max_hours_per_second, 0.01
        )
        self.models: list[EpidemicModel] = []
        self.playing = False
        self.speed = DEFAULT_SPEED
        self.run_id = 0
        self.step_ms = 0.0
        self.finished = False
        self.watched: int | None = None
        self._history_sent = 0

    def handle(self, command: Mapping) -> dict | None:
        """Apply one UI command and return a reply message if the command has one.

        Raises ValueError, KeyError or TypeError for a bad command.
        """
        match command["type"]:
            case "preview":
                return self._preview(command["experiment"])
            case "start":
                config = self.library.parse(command["experiment"])
                seed = config.world["seed"]
                self.models = [
                    EpidemicModel(
                        replace(config, world={**config.world, "seed": seed + index}),
                        self.library.town,
                    )
                    for index in range(round(config.world["runs"]))
                ]
                self.playing = False
                self.finished = False
                self.watched = None
                self.run_id += 1
                self._history_sent = 0
                return {
                    "type": "started",
                    "run": describe_run(config, self.models),
                }
            case "play":
                self._require_running()
                self.playing = True
            case "pause":
                self.playing = False
            case "step_day":
                self._require_running()
                self.playing = False
                self._run_steps(self.models[0].world["steps_per_day"])
            case "watch":
                return self._watch(command["run"])
            case "finish":
                self._require_models()
                self._finish()
            case "speed":
                self.speed = self._speed_spec.validate("speed", command["value"])
            case "inspect":
                models = self._require_models()
                run = self._run_index(command["run"])
                return {
                    "type": "person",
                    "run": run,
                    **describe_person(models[run], command["person"]),
                }
            case other:
                raise ValueError(f"Unknown command '{other}'")
        return None

    def steps_per_frame(self) -> int:
        """Steps that make up one frame at the chosen speed (at least one)."""
        return max(1, round(self.speed * self.frame_seconds / self.models[0].step_hours))

    def frame_interval(self) -> float:
        """Seconds between frames while playing, so that simulated time passes at the speed."""
        if self.speed == 0:
            return self.frame_seconds
        return self.steps_per_frame() * self.models[0].step_hours / self.speed

    def advance(self) -> None:
        """Run the steps that belong to one displayed frame."""
        if self.speed:
            self._run_steps(self.steps_per_frame(), self.max_frame_seconds)
            return
        started = perf_counter()
        steps = 0
        while not self.finished and (
            steps == 0 or perf_counter() - started < self.frame_budget_seconds
        ):
            self._run_steps(1)
            steps += 1

    def frame(self) -> dict:
        """JSON-ready snapshot of all runs for the UI; the runs share the clock."""
        models = self._require_models()
        clock = models[0]
        runs = []
        for index, model in enumerate(models):
            data = {"counts": model.state_counts(), "history": model.history[self._history_sent :]}
            if index == self.watched:
                visible = min(self.max_displayed, model.population)
                arena_size = model.world["arena_size"]
                data["x"] = np.round(model.x[:visible] / arena_size, 4).tolist()
                data["y"] = np.round(model.y[:visible] / arena_size, 4).tolist()
                data["states"] = {
                    name: state.state[:visible].tolist()
                    for name, state in model.disease_states.items()
                }
            runs.append(data)
        self._history_sent = len(clock.history)
        return {
            "type": "frame",
            "run_id": self.run_id,
            "day": clock.day,
            "weekday": clock.weekday,
            "hour": clock.hour,
            "playing": self.playing,
            "finished": self.finished,
            "speed": self.speed,
            "step_ms": self.step_ms,
            "runs": runs,
        }

    def _preview(self, definition: Mapping) -> dict:
        try:
            config = self.library.parse(definition, allow_no_diseases=True)
        except (ValueError, KeyError, TypeError) as error:
            return {"type": "preview", "error": str(error)}
        return {"type": "preview", **describe_preview(config, self.library.town)}

    def _watch(self, run: int | None) -> dict:
        """Show the people of one run (positions are only sent for the run being watched)."""
        models = self._require_models()
        if run is None:
            self.watched = None
            return {"type": "watching", "run": None}
        run = self._run_index(run)
        self.watched = run
        shown = min(self.max_displayed, models[run].population)
        town = models[run].town
        return {
            "type": "watching",
            "run": run,
            "ages": town.age[:shown].tolist(),
            "roles": town.role[:shown].tolist(),
        }

    def _run_steps(self, count: int, time_limit: float | None = None) -> None:
        """Run up to `count` steps; fewer after `time_limit` seconds or at the maximum duration."""
        clock = self.models[0]
        limit = clock.world["max_days"] * clock.world["steps_per_day"]
        if limit:
            count = min(count, limit - clock.steps)
        started = perf_counter()
        done = 0
        while done < count and (
            time_limit is None or done == 0 or perf_counter() - started < time_limit
        ):
            for model in self.models:
                model.step()
            done += 1
        if done:
            self.step_ms = (perf_counter() - started) / done * 1000
        if limit and clock.steps >= limit:
            self._finish()

    def _run_index(self, run) -> int:
        if isinstance(run, bool) or not isinstance(run, int) or not 0 <= run < len(self.models):
            raise ValueError(f"Unknown run {run!r}")
        return run

    def _finish(self) -> None:
        self.finished = True
        self.playing = False

    def _require_running(self) -> None:
        self._require_models()
        if self.finished:
            raise ValueError("The experiment is finished; start a new one")

    def _require_models(self) -> list[EpidemicModel]:
        if not self.models:
            raise ValueError("No experiment is running; start one first")
        return self.models
