"""An experiment belongs to the session of the browser, not to its connection."""

import time
from dataclasses import replace

import pytest
from config_factory import APP_CONFIG, LIBRARY
from session_helpers import new_session, receive_until, small_experiment, start
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from epidemic_sandbox.app.server import create_app
from epidemic_sandbox.app.sessions import OPENED_ELSEWHERE, SessionRegistry

DAYS = 20
SHORT = 4


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(LIBRARY, APP_CONFIG)) as test_client:
        yield test_client


@pytest.fixture
def impatient_client():
    """A server that stops waiting for a browser that does not answer after a third of a second."""
    session = replace(APP_CONFIG.session, viewer_stall_seconds=0.3)
    with TestClient(create_app(LIBRARY, replace(APP_CONFIG, session=session))) as test_client:
        yield test_client


def play_at_full_speed(socket) -> None:
    socket.send_json({"type": "speed", "value": 0})
    socket.send_json({"type": "play"})


def frame_after_play(socket) -> dict:
    """The first frame that shows the run not playing after one that shows it playing."""
    playing = False
    while True:
        frame = receive_until(socket, lambda m: m["type"] == "frame")
        playing = playing or frame["playing"]
        if playing and not frame["playing"]:
            return frame


def test_an_experiment_goes_on_while_the_browser_is_away_and_is_there_when_it_returns(client):
    address = new_session()
    with client.websocket_connect(address) as socket:
        socket.receive_json()
        start(socket, max_days=DAYS)
        play_at_full_speed(socket)
        before = receive_until(socket, lambda m: m["type"] == "frame" and m["day"] >= 1)
    assert before["day"] < DAYS, "the run was over before the browser left"
    time.sleep(1.5)  # nobody is connected
    with client.websocket_connect(address) as socket:
        assert socket.receive_json()["type"] == "hello"
        resumed = receive_until(socket, lambda m: m["type"] == "started")
        frame = receive_until(socket, lambda m: m["type"] == "frame")
    assert resumed["resumed"] and resumed["run"]["population"] == 200
    assert resumed["experiment"] == small_experiment(max_days=DAYS)
    assert frame["day"] > before["day"], "the run stood still while the browser was away"
    days = [record["day"] for record in frame["runs"][0]["history"]]
    assert days == list(range(int(frame["day"]) + 1)), "the history must be complete"


def test_a_run_that_ended_while_the_browser_was_away_is_found_finished(client):
    address = new_session()
    with client.websocket_connect(address) as socket:
        socket.receive_json()
        start(socket, max_days=3)
        play_at_full_speed(socket)
    time.sleep(1.0)
    with client.websocket_connect(address) as socket:
        socket.receive_json()
        receive_until(socket, lambda m: m["type"] == "started")
        frame = receive_until(socket, lambda m: m["type"] == "frame")
    assert frame["finished"] and not frame["playing"] and frame["day"] == 3


def test_a_session_does_not_see_the_experiment_of_another(client):
    with client.websocket_connect(new_session()) as socket:
        socket.receive_json()
        start(socket)
    with client.websocket_connect(new_session()) as socket:
        socket.receive_json()
        socket.send_json({"type": "play"})
        assert "start one first" in socket.receive_json()["message"]


def test_another_window_takes_the_experiment_over_and_the_first_is_told_so(client):
    address = new_session()
    with client.websocket_connect(address) as first:
        first.receive_json()
        start(first)
        with client.websocket_connect(address) as second:
            assert second.receive_json()["type"] == "hello"
            receive_until(second, lambda m: m["type"] == "started")
            with pytest.raises(WebSocketDisconnect) as closed:
                while True:
                    first.receive_json()
    assert closed.value.code == OPENED_ELSEWHERE


@pytest.mark.parametrize("query", ["", "?session=", "?session=short", "?session=has%20space-1234"])
def test_a_connection_needs_a_well_formed_session_name(client, query):
    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(f"/ws{query}"):
        pass
    assert refused.value.code == 1008


def test_a_browser_that_stops_answering_does_not_stop_the_run(impatient_client):
    with impatient_client.websocket_connect(new_session()) as socket:
        socket.receive_json()
        socket.send_json({"type": "start", "experiment": small_experiment(max_days=SHORT)})
        play_at_full_speed(socket)
        time.sleep(3.0)  # frames are never acknowledged, as in a locked screen
        socket.send_json({"type": "pause"})
        last = frame_after_play(socket)
    assert last["finished"] and last["day"] == SHORT


def test_a_browser_that_is_behind_is_still_waited_for_a_while(client):
    with client.websocket_connect(new_session()) as socket:
        socket.receive_json()
        socket.send_json({"type": "start", "experiment": small_experiment()})
        play_at_full_speed(socket)
        time.sleep(0.5)  # far less than viewer_stall_seconds
        socket.send_json({"type": "pause"})
        frames = []
        while not frames or frames[-1]["playing"]:
            message = socket.receive_json()
            if message["type"] == "frame":
                frames.append(message)
    assert len(frames) <= APP_CONFIG.session.max_unacknowledged_frames + 3


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def started_session(registry: SessionRegistry, token: str, **world):
    session = registry.get(token)
    session.simulation.handle({"type": "start", "experiment": small_experiment(**world)})
    return session


def test_sessions_nobody_uses_are_forgotten_but_a_playing_one_is_kept():
    clock = Clock()
    registry = SessionRegistry(LIBRARY, APP_CONFIG, clock)
    try:
        resting = started_session(registry, "resting-session")
        playing = started_session(registry, "playing-session")
        playing.simulation.playing = True
        day = 24 * 3600
        clock.now = APP_CONFIG.session.kept_idle_hours * 3600 - 1
        assert registry.forget_idle() == []
        clock.now += 2
        assert registry.forget_idle() == ["resting-session"]
        assert registry.get("playing-session") is playing
        assert registry.get("resting-session") is not resting, "a new, empty session"
        clock.now += 365 * day
        assert "playing-session" not in registry.forget_idle()
    finally:
        registry.shutdown()


@pytest.mark.parametrize(("max_days", "paused"), [(0, True), (30, False)])
def test_an_endless_run_that_nobody_watches_is_paused_after_a_long_time(max_days, paused):
    clock = Clock()
    registry = SessionRegistry(LIBRARY, APP_CONFIG, clock)
    try:
        session = started_session(registry, "unwatched-session", max_days=max_days)
        session.simulation.playing = True
        session._pause_if_unattended()
        clock.now = APP_CONFIG.session.max_unattended_hours * 3600 + 1
        session._pause_if_unattended()
        assert session.simulation.playing is not paused
    finally:
        registry.shutdown()
