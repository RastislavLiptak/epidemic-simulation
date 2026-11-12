from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter

from epidemic_sandbox.app.schema import describe_preview, describe_run
from epidemic_sandbox.config.app import AppConfig
from epidemic_sandbox.config.experiment import ExperimentLibrary
from epidemic_sandbox.config.parameter_spec import ParameterSpec
from epidemic_sandbox.engine.runs import LocalRuns, PooledRuns
from epidemic_sandbox.engine.workers import WorkerPool

DEFAULT_SPEED = 2.0
UNLIMITED_STEPS = 10**12  # as many as fit in the time given


class Simulation:
    """One user's session: the runs of an experiment, stepped together, driven by UI commands.

    Several runs are computed by the processes of `workers` at the same time; without workers,
    or with a single run, they are computed in this process.
    """

    def __init__(
        self, library: ExperimentLibrary, app_config: AppConfig, workers: WorkerPool | None = None
    ):
        self.library = library
        self.workers = workers
        self.frame_budget_seconds = app_config.session.frame_budget_seconds
        self.frame_seconds = app_config.session.frame_seconds
        self.max_displayed = app_config.session.max_displayed_people
        self.max_frame_seconds = app_config.session.max_frame_seconds
        self._speed_spec = ParameterSpec(
            "Speed", "", DEFAULT_SPEED, 0, app_config.session.max_hours_per_second, 0.01
        )
        self.runs: LocalRuns | PooledRuns | None = None
        self.playing = False
        self.speed = DEFAULT_SPEED
        self.run_id = 0
        self.step_ms = 0.0
        self.finished = False
        self.watched: int | None = None
        self.definition: Mapping | None = None
        self._run_description: dict | None = None
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
                configs = [
                    replace(config, world={**config.world, "seed": seed + index})
                    for index in range(round(config.world["runs"]))
                ]
                previous = self.runs
                self.runs = self._build_runs(configs)
                if previous is not None:
                    previous.close()
                self.playing = False
                self.finished = False
                self.watched = None
                self.run_id += 1
                self._history_sent = 0
                self.definition = command["experiment"]
                self._run_description = describe_run(config, self.runs.seeds)
                return {"type": "started", "run": self._run_description}
            case "play":
                self._require_running()
                self.playing = True
            case "pause":
                self.playing = False
            case "step_day":
                self._require_running()
                self.playing = False
                self._run_steps(self.runs.clock.world["steps_per_day"])
            case "watch":
                return self._watch(command["run"])
            case "finish":
                self._require_runs()
                self._finish()
            case "speed":
                self.speed = self._speed_spec.validate("speed", command["value"])
            case "inspect":
                runs = self._require_runs()
                run = self._run_index(command["run"])
                return {"type": "person", "run": run, **runs.person(run, command["person"])}
            case other:
                raise ValueError(f"Unknown command '{other}'")
        return None

    def steps_per_frame(self) -> int:
        """Steps that make up one frame at the chosen speed (at least one)."""
        return max(1, round(self.speed * self.frame_seconds / self.runs.clock.step_hours))

    def frame_interval(self) -> float:
        """Seconds between frames while playing, so that simulated time passes at the speed."""
        if self.speed == 0:
            return self.frame_seconds
        return self.steps_per_frame() * self.runs.clock.step_hours / self.speed

    def resume(self) -> dict | None:
        """The message that brings a browser which connects to the running experiment up to date
        (the experiment and the whole history, which the next frame carries), None if there is
        no experiment."""
        if self.runs is None:
            return None
        self.watched = None
        self._history_sent = 0
        return {
            "type": "started",
            "resumed": True,
            "run": self._run_description,
            "experiment": self.definition,
        }

    def advance(self, unpaced: bool = False) -> None:
        """Run the steps that belong to one displayed frame; `unpaced` (nobody is watching) runs
        as many steps as fit in the time of a frame at once, as speed "Max" does."""
        if self.speed and not unpaced:
            self._run_steps(self.steps_per_frame(), self.max_frame_seconds)
            return
        self._run_steps(UNLIMITED_STEPS, self.frame_budget_seconds)

    def frame(self) -> dict:
        """JSON-ready snapshot of all runs for the UI; the runs share the clock."""
        runs = self._require_runs()
        clock = runs.clock
        reports = runs.report(self._history_sent, self.watched, self.max_displayed)
        self._history_sent += max(len(report["history"]) for report in reports)
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
            "runs": reports,
        }

    def _preview(self, definition: Mapping) -> dict:
        try:
            config = self.library.parse(definition, allow_no_diseases=True)
        except (ValueError, KeyError, TypeError) as error:
            return {"type": "preview", "error": str(error)}
        return {"type": "preview", **describe_preview(config, self.library.town)}

    def _watch(self, run: int | None) -> dict:
        """Show the people of one run (positions are only sent for the run being watched)."""
        runs = self._require_runs()
        if run is None:
            self.watched = None
            return {"type": "watching", "run": None}
        run = self._run_index(run)
        self.watched = run
        ids, ages, roles = runs.people(run, self.max_displayed)
        return {"type": "watching", "run": run, "ids": ids, "ages": ages, "roles": roles}

    def _run_steps(self, count: int, time_limit: float | None = None) -> None:
        """Run up to `count` steps; fewer after `time_limit` seconds or at the maximum duration."""
        runs = self._require_runs()
        world = runs.clock.world
        limit = world["max_days"] * world["steps_per_day"]
        if limit:
            count = min(count, limit - runs.clock.steps)
        started = perf_counter()
        done = runs.step(count, time_limit) if count > 0 else 0
        if done:
            self.step_ms = (perf_counter() - started) / done * 1000
        if limit and runs.clock.steps >= limit:
            self._finish()

    def _run_index(self, run) -> int:
        if isinstance(run, bool) or not isinstance(run, int) or not 0 <= run < len(self.runs.seeds):
            raise ValueError(f"Unknown run {run!r}")
        return run

    def _finish(self) -> None:
        if not self.finished:
            self.runs.finish()  # the hours since the last midnight belong to the results
        self.finished = True
        self.playing = False

    def _require_running(self) -> None:
        self._require_runs()
        if self.finished:
            raise ValueError("The experiment is finished; start a new one")

    def _require_runs(self) -> LocalRuns | PooledRuns:
        if self.runs is None:
            raise ValueError("No experiment is running; start one first")
        return self.runs

    def _build_runs(self, configs: list) -> LocalRuns | PooledRuns:
        if self.workers is not None and len(configs) > 1:
            return PooledRuns(self.workers, configs, self.library.town)
        return LocalRuns(configs, self.library.town)

    @property
    def started(self) -> bool:
        return self.runs is not None

    @property
    def models(self) -> list:
        """The models of the runs that are computed in this process (none if they are not)."""
        return self.runs.models if isinstance(self.runs, LocalRuns) else []

    def close(self) -> None:
        if self.runs is not None:
            self.runs.close()
            self.runs = None
