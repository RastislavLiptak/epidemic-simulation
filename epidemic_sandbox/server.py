import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from epidemic_sandbox.app_config import AppConfig, load_app_config
from epidemic_sandbox.experiment_config import ExperimentLibrary, load_library
from epidemic_sandbox.schema import describe_setup, describe_ui
from epidemic_sandbox.simulation import Simulation

WEB_DIRECTORY = Path(__file__).resolve().parent / "web"
CONFIG_DIRECTORY = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_EXPERIMENT_PATH = CONFIG_DIRECTORY / "experiment.toml"
DEFAULT_APP_CONFIG_PATH = CONFIG_DIRECTORY / "app.toml"
# Browsers ask whether a file has changed before using their copy, so that the UI is always current.
REVALIDATE = {"Cache-Control": "no-cache"}
CLIENT_DISCONNECTED = object()
NEXT_FRAME = object()
COMMAND_ERRORS = (ValueError, KeyError, TypeError)
WEBSOCKET_POLICY_VIOLATION = 1008


class RevalidatedStaticFiles(StaticFiles):
    """Files of the UI that browsers do not use without asking whether they have changed."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(REVALIDATE)
        return response


def create_app(library: ExperimentLibrary, app_config: AppConfig) -> Starlette:
    """ASGI app serving the web UI and one simulation session per WebSocket connection."""

    async def simulation_socket(socket: WebSocket) -> None:
        if not _from_this_server(socket):
            await socket.close(code=WEBSOCKET_POLICY_VIOLATION)
            return
        await socket.accept()
        await run_session(socket, library, app_config)

    async def index(request: Request) -> Response:
        return FileResponse(WEB_DIRECTORY / "index.html", headers=REVALIDATE)

    async def ui_constants(request: Request) -> Response:
        return JSONResponse(describe_ui(app_config))

    @asynccontextmanager
    async def lifespan(app):
        warm_up_kernels(library)
        yield

    return Starlette(
        routes=[
            WebSocketRoute("/ws", simulation_socket),
            Route("/api/ui", ui_constants),
            Route("/", index),
            Mount("/ui", RevalidatedStaticFiles(directory=WEB_DIRECTORY)),
        ],
        lifespan=lifespan,
    )


def _from_this_server(socket: WebSocket) -> bool:
    """Browsers say which page opens a WebSocket; refuse pages of other sites (they could drive
    the simulation of someone who has this app open). Clients without an Origin are not browsers."""
    origin = socket.headers.get("origin")
    return origin is None or urlparse(origin).netloc == socket.headers.get("host")


def warm_up_kernels(library: ExperimentLibrary) -> None:
    """Compile (or load from the cache) the Numba kernels before the first user arrives."""
    from epidemic_sandbox.epidemic_model import EpidemicModel

    definition = library.default.to_definition()
    definition["world"].update(population=200, arena_area=250_000)
    EpidemicModel(library.parse(definition), library.town).step()


async def run_session(socket: WebSocket, library: ExperimentLibrary, app_config: AppConfig) -> None:
    """Send frames at a fixed rate while playing and apply UI commands as they arrive.

    The browser acknowledges every frame; while it is `max_unacknowledged_frames` behind, the
    server neither computes nor sends, so a slow browser slows the run down instead of drowning.
    """
    inbox: asyncio.Queue = asyncio.Queue()
    receiver = asyncio.create_task(_forward_messages(socket, inbox))
    send_lock = asyncio.Lock()
    max_behind = app_config.session.max_unacknowledged_frames
    unacknowledged = 0
    simulation = Simulation(library, app_config)
    loop = asyncio.get_running_loop()

    async def send(message: dict) -> None:
        async with send_lock:
            await socket.send_json(message)

    try:
        await send({"type": "hello", "setup": describe_setup(library)})
        next_frame_at = loop.time()
        while True:
            keeping_up = unacknowledged < max_behind
            timeout = (
                max(0.0, next_frame_at - loop.time()) if simulation.playing and keeping_up else None
            )
            try:
                message = await asyncio.wait_for(inbox.get(), timeout)
            except TimeoutError:
                message = NEXT_FRAME
            if message is CLIENT_DISCONNECTED:
                return
            if message is NEXT_FRAME:
                simulation.advance()
                next_frame_at = max(next_frame_at + simulation.frame_interval(), loop.time())
                unacknowledged += 1
                await send(simulation.frame())
                continue
            try:
                command = json.loads(message)
                if command["type"] == "ack":
                    unacknowledged = max(0, unacknowledged - 1)
                    continue
                was_playing = simulation.playing
                reply = simulation.handle(command)
            except COMMAND_ERRORS as error:
                await send({"type": "error", "message": str(error)})
                continue
            if reply is not None:
                await send(reply)
            if simulation.playing and not was_playing:
                next_frame_at = loop.time()
            if simulation.models and command["type"] not in ("preview", "inspect"):
                unacknowledged += 1
                await send(simulation.frame())
    finally:
        receiver.cancel()


async def _forward_messages(socket: WebSocket, inbox: asyncio.Queue) -> None:
    try:
        while True:
            await inbox.put(await socket.receive_text())
    except WebSocketDisconnect:
        await inbox.put(CLIENT_DISCONNECTED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Epidemic sandbox web UI")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG_PATH)
    parser.add_argument("--host", help="overrides the host from the app config")
    parser.add_argument("--port", type=int, help="overrides the port from the app config")
    arguments = parser.parse_args()
    app_config = load_app_config(arguments.app_config)
    library = load_library(arguments.experiment, app_config.limits)
    uvicorn.run(
        create_app(library, app_config),
        host=arguments.host or app_config.server.host,
        port=arguments.port or app_config.server.port,
    )


if __name__ == "__main__":
    main()
