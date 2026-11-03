import re
import time

import pytest
from config_factory import APP_CONFIG, LIBRARY
from starlette.testclient import TestClient

from epidemic_sandbox.server import create_app

# The two diseases of an interaction of the library, so that the tests can follow it into a run.
FIRST, SECOND = LIBRARY.interactions[0].source, LIBRARY.interactions[0].target
BETWEEN_THEM = sum(
    1 for rule in LIBRARY.interactions if {rule.source, rule.target} == {FIRST, SECOND}
)

MAX_MESSAGES = 300
SCRIPTS = [
    "main",
    "arena",
    "person",
    "runs",
    "setup",
    "sidebar",
    "results",
    "statistics",
    "charts",
    "fields",
    "socket",
    "busy",
]


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(LIBRARY, APP_CONFIG)) as test_client:
        yield test_client


def small_experiment(**world) -> dict:
    """What the browser sends: the world settings and the diseases chosen by name."""
    return {
        "world": {"population": 200, "arena_area": 250_000, **world},
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


def test_web_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "Epidemic sandbox" in page.text
    for name in SCRIPTS:
        assert client.get(f"/ui/{name}.js").status_code == 200, name
    assert (
        client.get("/ui/style.css").status_code == 200
        and client.get("/ui/vendor/chart.umd.js").status_code == 200
    )


def test_the_page_loads_its_files_from_where_they_are_served(client):
    page = client.get("/").text
    for path in re.findall(r'(?:src|href)="(ui/[^"]+)"', page):
        assert client.get(f"/{path}").status_code == 200
    assert 'src="ui/main.js"' in page


def test_browsers_must_ask_whether_the_ui_has_changed(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
    answer = client.get("/ui/charts.js")
    assert answer.headers["cache-control"] == "no-cache"
    again = client.get("/ui/charts.js", headers={"If-None-Match": answer.headers["etag"]})
    assert again.status_code == 304


def test_every_script_imported_by_a_script_is_served(client):
    imported = set()
    for name in SCRIPTS:
        imported |= set(re.findall(r'from "\./(\w+\.js)"', client.get(f"/ui/{name}.js").text))
    assert imported <= {f"{name}.js" for name in SCRIPTS}
    for name in imported:
        assert client.get(f"/ui/{name}").status_code == 200


def test_there_is_no_batch_interface_any_more(client):
    assert client.get("/api/batch/x/series.csv").status_code == 404


def test_ui_constants_come_from_the_app_config(client):
    constants = client.get("/api/ui").json()
    assert constants["reconnect_delay_ms"] == APP_CONFIG.ui.reconnect_delay_ms
    assert constants["max_runs"] >= 1


def test_connection_starts_with_the_setup_description(client):
    with client.websocket_connect("/ws") as socket:
        hello = socket.receive_json()
    assert hello["type"] == "hello"
    assert len(hello["setup"]["catalog"]) == len(LIBRARY.catalog)
    assert len(hello["setup"]["interactions"]) == len(LIBRARY.interactions)


def test_preview_answers_with_derived_values_or_an_error(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "preview", "experiment": small_experiment()})
        assert socket.receive_json()["diseases"][FIRST]["mean_contacts"] > 0
        socket.send_json({"type": "preview", "experiment": small_experiment(population=3)})
        assert "error" in socket.receive_json()


def test_start_answers_with_run_description_and_first_frame(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        run, frame = start(socket, runs=2)
    assert [d["name"] for d in run["diseases"]] == [FIRST, SECOND]
    assert (
        run["population"] == 200
        and len(run["interactions"]) == BETWEEN_THEM
        and run["seeds"] == [42, 43]
    )
    assert frame["day"] == 0 and not frame["playing"] and len(frame["runs"]) == 2
    assert "x" not in frame["runs"][0], "nobody is watched until a run is opened"


def test_step_day_and_play_and_pause(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        start(socket)
        socket.send_json({"type": "step_day"})
        assert receive_until(socket, lambda m: m["type"] == "frame")["day"] == 1.0
        socket.send_json({"type": "play"})
        days = [receive_until(socket, lambda m: m["type"] == "frame")["day"] for _ in range(3)]
        assert days == sorted(days) and days[0] < days[-1]
        socket.send_json({"type": "pause"})
        receive_until(socket, lambda m: m["type"] == "frame" and not m["playing"])


def test_inspecting_a_person_answers_without_a_frame(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        start(socket, runs=2)
        socket.send_json({"type": "inspect", "run": 1, "person": 4})
        person = receive_until(socket, lambda m: m["type"] == "person")
    assert person["run"] == 1 and person["id"] == 4 and "timetable" in person


def test_bad_commands_report_an_error_and_keep_the_session_alive(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        for bad in [
            '{"type": "dance"}',
            "not json",
            '{"type": "set_world"}',
            "[1, 2]",
            '{"type": "play"}',
            '{"type": "batch"}',
        ]:
            socket.send_text(bad)
            assert socket.receive_json()["type"] == "error"
        start(socket)


def test_disease_parameters_cannot_be_sent_from_the_browser(client):
    experiment = small_experiment()
    experiment["diseases"][FIRST]["parameters"] = {"transmission": 1.0}
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "start", "experiment": experiment})
        assert "defined in its file" in socket.receive_json()["message"]


def test_invalid_start_is_reported_and_leaves_the_session_usable(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "start", "experiment": small_experiment(population=1)})
        assert "out of range" in socket.receive_json()["message"]
        start(socket)


def test_sessions_are_independent(client):
    with client.websocket_connect("/ws") as first, client.websocket_connect("/ws") as second:
        for socket in (first, second):
            socket.receive_json()
            start(socket)
        first.send_json({"type": "step_day"})
        assert receive_until(first, lambda m: m["type"] == "frame")["day"] == 1.0
        second.send_json({"type": "step_day"})
        assert receive_until(second, lambda m: m["type"] == "frame")["day"] == 1.0


def test_finishing_ends_the_experiment_and_a_maximum_duration_ends_it_by_itself(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        start(socket)
        socket.send_json({"type": "finish"})
        assert receive_until(socket, lambda m: m["type"] == "frame")["finished"]
        socket.send_json({"type": "play"})
        assert "finished" in socket.receive_json()["message"]
        _, first = start(socket, max_days=2)
        assert not first["finished"]
        socket.send_json({"type": "speed", "value": 0})
        socket.send_json({"type": "play"})
        last = receive_until(socket, lambda m: m["type"] == "frame" and m["finished"])
    assert last["day"] == 2 and not last["playing"]


def test_the_people_of_a_run_are_sent_only_after_it_is_opened(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        start(socket, runs=2)
        socket.send_json({"type": "watch", "run": 1})
        watching = receive_until(socket, lambda m: m["type"] == "watching")
        frame = receive_until(socket, lambda m: m["type"] == "frame")
        socket.send_json({"type": "watch", "run": None})
        receive_until(socket, lambda m: m["type"] == "watching")
        closed = receive_until(socket, lambda m: m["type"] == "frame")
    assert len(watching["ages"]) == 200 and len(frame["runs"][1]["x"]) == 200
    assert "x" not in frame["runs"][0] and "x" not in closed["runs"][1]


def test_a_browser_that_does_not_keep_up_is_not_flooded_with_frames(client):
    behind = APP_CONFIG.session.max_unacknowledged_frames
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        socket.send_json({"type": "start", "experiment": small_experiment()})
        socket.send_json({"type": "speed", "value": 0})
        socket.send_json({"type": "play"})
        time.sleep(0.6)  # long enough for dozens of frames at the full frame rate
        socket.send_json({"type": "pause"})  # always answered, even when the browser is behind
        frames = []
        while not frames or frames[-1]["playing"] or len(frames) < 4:
            message = socket.receive_json()
            if message["type"] == "frame":
                frames.append(message)
    assert len(frames) <= behind + 5, "frames kept coming although none were acknowledged"


def test_acknowledging_frames_lets_the_run_continue(client):
    with client.websocket_connect("/ws") as socket:
        socket.receive_json()
        start(socket)
        socket.send_json({"type": "speed", "value": 0})
        socket.send_json({"type": "play"})
        days = [receive_until(socket, lambda m: m["type"] == "frame")["day"] for _ in range(12)]
    assert days[-1] > days[0], "many more frames than the unacknowledged limit arrived"


def test_pages_of_other_sites_cannot_drive_the_simulation(client):
    from starlette.websockets import WebSocketDisconnect

    foreign = {"origin": "http://evil.example"}
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers=foreign):
        pass
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as socket:
        assert socket.receive_json()["type"] == "hello"
