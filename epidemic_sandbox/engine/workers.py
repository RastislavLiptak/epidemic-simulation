"""Worker processes that compute the runs of an experiment at the same time.

Threads cannot do it: a step is mostly many short NumPy calls that hold the interpreter lock, and
with several threads the runs got slower (0.35 times the speed of one thread, measured on 8 runs
of 3,000 people). A process has its own interpreter, so each worker owns some of the runs for the
whole experiment; only small messages travel: how many steps to take, and reports of counts and
positions. The workers outlive the experiments, because starting one costs a few seconds.
"""

import contextlib
import itertools
import multiprocessing
import os
import traceback
from collections.abc import Sequence

from epidemic_sandbox.config.experiment import ExperimentConfig
from epidemic_sandbox.world.town import TownParameters


class Worker:
    """One worker process and the pipe to it."""

    def __init__(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child,), daemon=True)
        self.process.start()
        child.close()

    def send(self, message: tuple) -> None:
        self.connection.send(message)

    def receive(self):
        try:
            status, value = self.connection.recv()
        except EOFError as error:
            raise RuntimeError("A worker process of the runs stopped") from error
        if status == "invalid":
            raise ValueError(value)  # a bad request, as it is reported when the run is here
        if status == "error":
            raise RuntimeError(f"A worker process of the runs failed:\n{value}")
        return value

    def stop(self) -> None:
        with contextlib.suppress(OSError):
            self.connection.send(("stop",))
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()


class WorkerPool:
    """Up to `size` workers (one for each core unless told otherwise), started when an experiment
    first needs them. All the cores are worth using, the efficiency cores of a Mac too: 8 runs of
    3,000 people took 1.56 ms a step on 8 workers and 2.24 ms on 6, as against 8.91 ms in one
    process."""

    def __init__(self, size: int | None = None):
        self.size = size or os.cpu_count() or 1
        self.workers: list[Worker] = []
        self._keys = itertools.count()

    def new_key(self) -> int:
        """The name of an experiment on the workers."""
        return next(self._keys)

    def start(
        self, key: int, configs: Sequence[ExperimentConfig], town_parameters: TownParameters
    ) -> list[Worker]:
        """Build the runs `configs` on the workers; run `i` goes to worker `i % len(workers)`.

        The workers build their towns at the same time, and share the cores between the threads
        of their kernels (a town of many people uses them; a small one does not).
        """
        count = min(len(configs), self.size)
        while len(self.workers) < count:
            self.workers.append(Worker())
        workers = self.workers[:count]
        threads = max(1, self.size // count)
        for position, worker in enumerate(workers):
            mine = [
                (index, config)
                for index, config in enumerate(configs)
                if index % count == position
            ]
            worker.send(("start", key, mine, town_parameters, threads))
        for worker in workers:
            worker.receive()
        return workers

    def ask(self, worker: Worker, command: str, *arguments):
        worker.send((command, *arguments))
        return worker.receive()

    def ask_all(self, workers: Sequence[Worker], command: str, *arguments) -> list:
        """The same request to all workers, which answer while the others still work."""
        for worker in workers:
            worker.send((command, *arguments))
        return [worker.receive() for worker in workers]

    def close(self) -> None:
        for worker in self.workers:
            worker.stop()
        self.workers = []


def serve(connection) -> None:
    """The life of a worker: wait for a request, answer it, until told to stop."""
    from epidemic_sandbox.engine.runs import LocalRuns

    experiments: dict[int, tuple[list[int], LocalRuns]] = {}
    while True:
        try:
            message = connection.recv()
        except EOFError:
            return
        command, *arguments = message
        if command == "stop":
            return
        try:
            answer = _answer(experiments, command, arguments, LocalRuns)
            connection.send(("ok", answer))
        except (ValueError, KeyError, TypeError) as error:
            connection.send(("invalid", str(error)))
        except Exception:  # whatever went wrong is reported to the experiment, not lost here
            connection.send(("error", traceback.format_exc()))


def _answer(experiments: dict, command: str, arguments: list, runs_class):
    if command == "start":
        key, mine, town_parameters, threads = arguments
        import numba

        numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        runs = runs_class([config for _, config in mine], town_parameters)
        experiments[key] = ([index for index, _ in mine], runs)
        return None
    key, *arguments = arguments
    if command == "drop":
        experiments.pop(key, None)
        return None
    indices, runs = experiments[key]
    if command == "step":
        return runs.step(arguments[0])
    if command == "report":
        history_from, watched, visible = arguments
        local = indices.index(watched) if watched in indices else None
        reports = runs.report(history_from, local, visible)
        return dict(zip(indices, reports, strict=True))
    if command not in ("people", "person"):
        raise ValueError(f"Unknown request {command!r}")
    run = indices.index(arguments[0])
    return getattr(runs, command)(run, arguments[1])
