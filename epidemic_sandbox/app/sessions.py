"""Experiments that go on when the browser goes away.

An experiment belongs to a session, named by a token the browser keeps, and not to a WebSocket.
The session computes by itself while it is playing; the browser that watches it comes and goes:
it receives frames while it keeps up, and when it connects again (a locked screen, a closed tab, a
restarted browser) it is sent the whole course so far.
"""

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import monotonic

from starlette.websockets import WebSocket, WebSocketDisconnect

from epidemic_sandbox.app.schema import describe_setup
from epidemic_sandbox.app.simulation import Simulation
from epidemic_sandbox.config.app import AppConfig
from epidemic_sandbox.config.experiment import ExperimentLibrary
from epidemic_sandbox.engine.workers import WorkerPool

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,64}")
OPENED_ELSEWHERE = 4001  # close code: another window took over the experiment
NEXT_FRAME = object()
ATTACH, GONE, MESSAGE = "attach", "gone", "message"
COMMAND_ERRORS = (ValueError, KeyError, TypeError)


class Viewer:
    """A browser connected to a session, and how many frames it has yet to acknowledge."""

    def __init__(self, socket: WebSocket):
        self.socket = socket
        self.unacknowledged = 0
        self.behind_since: float | None = None

    def sent_frame(self, now: float) -> None:
        if self.unacknowledged == 0:
            self.behind_since = now
        self.unacknowledged += 1

    def acknowledged(self, now: float) -> None:
        self.unacknowledged = max(0, self.unacknowledged - 1)
        self.behind_since = now if self.unacknowledged else None


class Session:
    """One experiment, kept on the server whether or not a browser is looking at it."""

    def __init__(
        self,
        token: str,
        library: ExperimentLibrary,
        app_config: AppConfig,
        executor: ThreadPoolExecutor,
        clock: Callable[[], float] = monotonic,
        workers: WorkerPool | None = None,
    ):
        self.token = token
        self.library = library
        self.config = app_config.session
        self.simulation = Simulation(library, app_config, workers)
        self.executor = executor
        self.clock = clock
        self.viewer: Viewer | None = None
        self.last_seen = clock()
        self.unwatched_since: float | None = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    @property
    def idle(self) -> bool:
        """Nobody is watching and nothing is going on."""
        return self.viewer is None and not self.simulation.playing

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    def discard(self) -> None:
        """Stop for good: the runs give up their memory and their worker processes."""
        self.stop()
        self.executor.submit(self.simulation.close)

    async def serve(self, socket: WebSocket) -> None:
        """Let `socket` watch and steer the experiment until the browser goes away."""
        viewer = Viewer(socket)
        self.start()
        await self._inbox.put((ATTACH, viewer, None))
        try:
            while True:
                await self._inbox.put((MESSAGE, viewer, await socket.receive_text()))
        except WebSocketDisconnect:
            await self._inbox.put((GONE, viewer, None))

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        next_frame_at = loop.time()
        while True:
            try:
                item = await self._next_item(next_frame_at - loop.time())
                self._pause_if_unattended()
                if item is NEXT_FRAME:
                    await self._call(self.simulation.advance, not self._watched())
                    interval = self.simulation.frame_interval()
                    next_frame_at = max(next_frame_at + interval, loop.time())
                    if self._watched():
                        await self._send_frame()
                    continue
                kind, viewer, payload = item
                if kind == ATTACH:
                    await self._attach(viewer)
                elif kind == GONE:
                    self._detach(viewer)
                elif viewer is self.viewer:
                    was_playing = self.simulation.playing
                    await self._handle(viewer, payload)
                    if self.simulation.playing and not was_playing:
                        next_frame_at = loop.time()
            except Exception as error:  # the session must outlive a failure of one step
                logger.exception("The experiment of session %s failed", self.token)
                self.simulation.playing = False
                if self.viewer is not None:
                    await self._send(self.viewer, {"type": "error", "message": repr(error)})

    async def _next_item(self, until_frame: float):
        """The next message, or NEXT_FRAME when it is time to compute one.

        While a browser keeps up the run is paced by the playback speed and waits for a browser
        that is behind. A run that is not watched (no browser, or one that has not acknowledged a
        frame for `viewer_stall_seconds`) goes on as fast as it can and sends nothing.
        """
        if not self.simulation.playing:
            return await self._inbox.get()
        if not self._watched():
            await asyncio.sleep(0)
            return self._take_waiting_message()
        viewer = self.viewer
        if viewer.unacknowledged >= self.config.max_unacknowledged_frames:
            wait = viewer.behind_since + self.config.viewer_stall_seconds - self.clock()
        else:
            wait = until_frame
        try:
            return await asyncio.wait_for(self._inbox.get(), max(0.0, wait))
        except TimeoutError:
            return NEXT_FRAME if self._watched() else self._take_waiting_message()

    def _take_waiting_message(self):
        try:
            return self._inbox.get_nowait()
        except asyncio.QueueEmpty:
            return NEXT_FRAME

    def _watched(self) -> bool:
        """A browser is connected and answers; if it is behind, it has not been for too long."""
        viewer = self.viewer
        if viewer is None:
            return False
        if viewer.unacknowledged < self.config.max_unacknowledged_frames:
            return True
        return self.clock() - viewer.behind_since < self.config.viewer_stall_seconds

    async def _attach(self, viewer: Viewer) -> None:
        old, self.viewer = self.viewer, viewer
        self.unwatched_since = None
        self.last_seen = self.clock()
        if old is not None:
            await self._close(old, OPENED_ELSEWHERE)
        await self._send(viewer, {"type": "hello", "setup": describe_setup(self.library)})
        resumed = await self._call(self.simulation.resume)
        if resumed is not None:
            await self._send(viewer, resumed)
            await self._send_frame()

    def _detach(self, viewer: Viewer) -> None:
        if viewer is self.viewer:
            self.viewer = None
            self.last_seen = self.clock()

    async def _handle(self, viewer: Viewer, raw: str) -> None:
        try:
            command = json.loads(raw)
            if command["type"] == "ack":
                viewer.acknowledged(self.clock())
                return
            reply = await self._call(self.simulation.handle, command)
        except COMMAND_ERRORS as error:
            await self._send(viewer, {"type": "error", "message": str(error)})
            return
        if reply is not None:
            await self._send(viewer, reply)
        if self.simulation.started and command["type"] not in ("preview", "inspect"):
            await self._send_frame()

    async def _send_frame(self) -> None:
        viewer = self.viewer
        if viewer is None:
            return
        frame = await self._call(self.simulation.frame)
        viewer.sent_frame(self.clock())
        await self._send(viewer, frame)

    async def _send(self, viewer: Viewer, message: dict) -> None:
        try:
            await viewer.socket.send_json(message)
        except Exception:  # whatever the transport raises, the browser is gone
            self._detach(viewer)

    async def _close(self, viewer: Viewer, code: int) -> None:
        with contextlib.suppress(Exception):
            await viewer.socket.close(code=code)

    def _pause_if_unattended(self) -> None:
        """A run without an end that nobody has watched for long is paused, so that it does not
        fill the memory with its history for ever."""
        simulation = self.simulation
        if not simulation.playing or self._watched():
            self.unwatched_since = None
            return
        now = self.clock()
        if self.unwatched_since is None:
            self.unwatched_since = now
        endless = simulation.runs.clock.world["max_days"] == 0
        if endless and now - self.unwatched_since > self.config.max_unattended_hours * 3600:
            simulation.playing = False

    async def _call(self, function, *arguments):
        """Run `function` on the one thread that computes (see `SessionRegistry`)."""
        return await asyncio.get_running_loop().run_in_executor(
            self.executor, function, *arguments
        )


class SessionRegistry:
    """The sessions of the server by token, and the one thread that computes for all of them.

    The Numba kernels that use several threads must not be started from several Python threads
    at once, so everything that touches a simulation runs on this one thread. Runs of several
    people at once are computed by the processes of `workers`, which are shared by the sessions.
    """

    def __init__(
        self,
        library: ExperimentLibrary,
        app_config: AppConfig,
        clock: Callable[[], float] = monotonic,
        workers: WorkerPool | None = None,
    ):
        self.library = library
        self.app_config = app_config
        self.clock = clock
        self.workers = workers
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
        self.sessions: dict[str, Session] = {}

    def get(self, token: str) -> Session:
        """The session of `token`, a new one if there is none."""
        if token not in self.sessions:
            self.sessions[token] = Session(
                token, self.library, self.app_config, self.executor, self.clock, self.workers
            )
        return self.sessions[token]

    def forget_idle(self) -> list[str]:
        """Drop the sessions that nobody has watched for `kept_idle_hours` and that are not
        playing; returns their tokens."""
        limit = self.app_config.session.kept_idle_hours * 3600
        dropped = [
            token
            for token, session in self.sessions.items()
            if session.idle and self.clock() - session.last_seen > limit
        ]
        for token in dropped:
            self.sessions.pop(token).discard()
        return dropped

    def shutdown(self) -> None:
        for session in self.sessions.values():
            session.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.workers is not None:
            self.workers.close()
