"""What a browser does over the WebSocket: open a session, start an experiment, acknowledge."""

import uuid

from config_factory import LIBRARY

# The two diseases of an interaction of the library, so that the tests can follow it into a run.
FIRST, SECOND = LIBRARY.interactions[0].source, LIBRARY.interactions[0].target
MAX_MESSAGES = 300


def new_session() -> str:
    """The address of a WebSocket that belongs to a session nobody has used."""
    return f"/ws?session={uuid.uuid4().hex}"


def small_experiment(**world) -> dict:
    """What the browser sends: the world settings and the diseases chosen by name."""
    return {
        "world": {"population": 200, "area_km2": 0.25, **world},
        "diseases": {FIRST: {"initial_infected": 3}, SECOND: {"initial_infected": 2}},
    }


def receive_until(socket, predicate):
    """Wait for a message; frames on the way are acknowledged, as the browser does."""
    for _ in range(MAX_MESSAGES):
        message = socket.receive_json()
        if message["type"] == "frame":
            socket.send_json({"type": "ack"})
        if predicate(message):
            return message
    raise AssertionError("expected message never arrived")


def start(socket, **world):
    socket.send_json({"type": "start", "experiment": small_experiment(**world)})
    started = receive_until(socket, lambda m: m["type"] == "started")
    return started["run"], receive_until(socket, lambda m: m["type"] == "frame")
