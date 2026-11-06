import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket

from epidemic_sandbox.app.schema import describe_ui
from epidemic_sandbox.app.sessions import TOKEN_PATTERN, SessionRegistry
from epidemic_sandbox.config.app import AppConfig, load_app_config
from epidemic_sandbox.config.experiment import ExperimentLibrary, load_library
from epidemic_sandbox.engine.workers import WorkerPool
from epidemic_sandbox.paths import CONFIG_DIRECTORY, WEB_DIRECTORY

DEFAULT_EXPERIMENT_PATH = CONFIG_DIRECTORY / "experiment.toml"
DEFAULT_APP_CONFIG_PATH = CONFIG_DIRECTORY / "app.toml"
# Browsers ask whether a file has changed before using their copy, so that the UI is always current.
REVALIDATE = {"Cache-Control": "no-cache"}
WEBSOCKET_POLICY_VIOLATION = 1008
SWEEP_SECONDS = 60  # how often sessions that nobody has used for long are looked for


class RevalidatedStaticFiles(StaticFiles):
    """Files of the UI that browsers do not use without asking whether they have changed."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(REVALIDATE)
        return response


def create_app(
    library: ExperimentLibrary, app_config: AppConfig, workers: int | None = None
) -> Starlette:
    """ASGI app serving the web UI and the experiments of the browsers that connect to it.

    Experiments of several runs are computed by `workers` processes (None: one for each core of
    the machine, 0: none, every run is computed by the server itself).
    """
    pool = WorkerPool(workers) if workers != 0 else None
    registry = SessionRegistry(library, app_config, workers=pool)

    async def simulation_socket(socket: WebSocket) -> None:
        token = socket.query_params.get("session", "")
        if not _from_this_server(socket) or not TOKEN_PATTERN.fullmatch(token):
            await socket.close(code=WEBSOCKET_POLICY_VIOLATION)
            return
        await socket.accept()
        await registry.get(token).serve(socket)

    async def index(request: Request) -> Response:
        return FileResponse(WEB_DIRECTORY / "index.html", headers=REVALIDATE)

    async def ui_constants(request: Request) -> Response:
        return JSONResponse(describe_ui(app_config))

    async def forget_idle_sessions() -> None:
        while True:
            await asyncio.sleep(SWEEP_SECONDS)
            registry.forget_idle()

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.get_running_loop().run_in_executor(
            registry.executor, warm_up_kernels, library
        )
        sweeper = asyncio.create_task(forget_idle_sessions())
        try:
            yield
        finally:
            sweeper.cancel()
            registry.shutdown()

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
    from epidemic_sandbox.engine.epidemic_model import EpidemicModel

    definition = library.default.to_definition()
    definition["world"].update(population=200, area_km2=0.25)
    EpidemicModel(library.parse(definition), library.town).step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Epidemic sandbox web UI")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG_PATH)
    parser.add_argument("--host", help="overrides the host from the app config")
    parser.add_argument("--port", type=int, help="overrides the port from the app config")
    parser.add_argument(
        "--workers",
        type=int,
        help="processes that compute the runs of an experiment at the same time (default: one "
        "for each core of the machine; 0: the runs are computed one after another)",
    )
    arguments = parser.parse_args()
    app_config = load_app_config(arguments.app_config)
    library = load_library(arguments.experiment, app_config.limits)
    uvicorn.run(
        create_app(library, app_config, arguments.workers),
        host=arguments.host or app_config.server.host,
        port=arguments.port or app_config.server.port,
    )


if __name__ == "__main__":
    main()
