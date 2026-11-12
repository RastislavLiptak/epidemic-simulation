import json

import numpy as np
import pytest
from config_factory import (
    APP_CONFIG,
    TEST_STEPS_PER_DAY,
    definition,
    disease_definition,
    interaction,
    library_with,
)

from epidemic_sandbox.app.simulation import Simulation

pytestmark = pytest.mark.usefixtures("synthetic_in_registry")


def new_simulation(steps_per_day: int = TEST_STEPS_PER_DAY) -> Simulation:
    return Simulation(library_with(steps_per_day), APP_CONFIG)


def start_command(**world) -> dict:
    diseases = {"a": disease_definition("plain", 2), "b": disease_definition("latent", 2)}
    rules = [interaction("a", ["recovered"], "b", "susceptibility", 0.5)]
    return {"type": "start", "experiment": definition(diseases, rules, **world)}


def started_simulation(steps_per_day: int = TEST_STEPS_PER_DAY, **world) -> Simulation:
    simulation = new_simulation(steps_per_day)
    simulation.handle(start_command(**world))
    return simulation


def steps(simulation: Simulation) -> list[int]:
    return [model.steps for model in simulation.models]


def test_commands_need_a_started_experiment():
    for command in [
        {"type": "play"},
        {"type": "step_day"},
        {"type": "inspect", "run": 0, "person": 1},
    ]:
        with pytest.raises(ValueError, match="start one first"):
            new_simulation().handle(command)
    with pytest.raises(ValueError, match="start one first"):
        new_simulation().frame()


def test_start_returns_the_run_description_and_begins_paused():
    simulation = new_simulation()
    reply = simulation.handle(start_command())
    assert reply["type"] == "started" and reply["run"]["population"] == 200
    assert [disease["name"] for disease in reply["run"]["diseases"]] == ["a", "b"]
    assert len(reply["run"]["interactions"]) == 1
    assert reply["run"]["seeds"] == [1]
    assert not simulation.playing and simulation.run_id == 1


def test_preview_does_not_start_anything_and_reports_errors_as_a_reply():
    simulation = new_simulation()
    good = simulation.handle({"type": "preview", "experiment": definition()})
    assert good["type"] == "preview" and good["diseases"]["a"]["mean_contacts"] > 0
    assert not simulation.models
    bad = simulation.handle({"type": "preview", "experiment": definition(population=0)})
    assert bad == {"type": "preview", "error": bad["error"]} and "out of range" in bad["error"]
    assert "error" in simulation.handle({"type": "preview", "experiment": "nonsense"})
    wrong_rule = definition(interactions=[interaction("a", ["recovered"], "zzz")])
    assert "not one of" in simulation.handle({"type": "preview", "experiment": wrong_rule})["error"]


def test_invalid_start_keeps_the_running_experiment():
    simulation = started_simulation()
    models = simulation.models
    with pytest.raises(ValueError):
        simulation.handle(start_command(population=0))
    assert simulation.models is models and simulation.run_id == 1


def test_restarting_creates_new_runs():
    simulation = started_simulation()
    simulation.handle({"type": "step_day"})
    simulation.handle(start_command(population=300))
    assert simulation.models[0].population == 300 and steps(simulation) == [0]
    assert simulation.run_id == 2


def test_several_runs_start_together_with_their_own_seeds_and_towns():
    simulation = started_simulation(runs=3, seed=10)
    assert [model.world["seed"] for model in simulation.models] == [10, 11, 12]
    ages = [model.town.age for model in simulation.models]
    assert not np.array_equal(ages[0], ages[1]) and not np.array_equal(ages[1], ages[2])


def test_all_runs_advance_in_lockstep():
    simulation = started_simulation(runs=3, steps_per_day=12)
    simulation.handle({"type": "step_day"})
    assert steps(simulation) == [12, 12, 12]
    simulation.handle({"type": "play"})
    simulation.advance()
    assert len(set(steps(simulation))) == 1 and steps(simulation)[0] > 12


def test_runs_with_the_same_settings_differ_by_chance_alone():
    simulation = started_simulation(runs=2, population=400)
    for _ in range(10):
        simulation.handle({"type": "step_day"})
    first, second = (model.state_counts() for model in simulation.models)
    assert first != second


def test_play_and_pause():
    simulation = started_simulation()
    simulation.handle({"type": "play"})
    assert simulation.playing
    simulation.advance()
    assert steps(simulation) == [1]
    simulation.handle({"type": "pause"})
    assert not simulation.playing


def test_step_day_advances_one_day_and_pauses():
    simulation = started_simulation(steps_per_day=12)
    simulation.handle({"type": "play"})
    simulation.handle({"type": "step_day"})
    assert steps(simulation) == [12] and not simulation.playing


def test_speed_is_simulated_hours_per_real_second():
    simulation = started_simulation(steps_per_day=24)
    fps = APP_CONFIG.session.frames_per_second
    simulation.handle({"type": "speed", "value": 12 * fps})
    simulation.advance()
    assert steps(simulation) == [12]
    assert simulation.frame_interval() == pytest.approx(12 / (12 * fps))


def test_slow_speeds_still_advance_at_least_one_step_per_frame_and_slow_the_frame_rate():
    simulation = started_simulation(steps_per_day=24)
    simulation.handle({"type": "speed", "value": 2})
    simulation.advance()
    assert steps(simulation) == [1]
    assert simulation.frame_interval() == pytest.approx(0.5)


def test_speed_zero_runs_as_many_steps_as_fit_into_a_frame():
    simulation = started_simulation()
    simulation.handle({"type": "speed", "value": 0})
    simulation.advance()
    assert steps(simulation)[0] > 1
    assert simulation.frame_interval() == pytest.approx(APP_CONFIG.session.frame_seconds)


@pytest.mark.parametrize(
    "value", [-1, APP_CONFIG.session.max_hours_per_second + 1, "4", None, True]
)
def test_invalid_speed_is_rejected(value):
    simulation = started_simulation()
    before = simulation.speed
    with pytest.raises(ValueError):
        simulation.handle({"type": "speed", "value": value})
    assert simulation.speed == before


@pytest.mark.parametrize(
    "command", [{"type": "dance"}, {}, {"type": "set_world"}, {"type": "set_disease"}, "play", None]
)
def test_bad_commands_are_rejected_and_live_settings_no_longer_exist(command):
    with pytest.raises((ValueError, KeyError, TypeError)):
        started_simulation().handle(command)


def test_frame_has_the_counts_of_every_run_but_the_people_of_the_watched_run_only():
    simulation = started_simulation(runs=2)
    frame = simulation.frame()
    json.dumps(frame)
    assert frame["type"] == "frame" and frame["day"] == 0 and not frame["playing"]
    assert (frame["weekday"], frame["hour"]) == (0, 0.0)
    assert len(frame["runs"]) == 2
    for run in frame["runs"]:
        assert run["counts"] == {"a": [198, 2, 0, 0, 0], "b": [198, 0, 2, 0, 0, 0]}
        assert "x" not in run and "states" not in run
    simulation.handle({"type": "watch", "run": 1})
    frame = simulation.frame()
    assert "x" not in frame["runs"][0] and len(frame["runs"][1]["x"]) == 200
    assert all(0 <= value <= 1 for value in frame["runs"][1]["x"] + frame["runs"][1]["y"])
    assert {name: len(states) for name, states in frame["runs"][1]["states"].items()} == {
        "a": 200,
        "b": 200,
    }


def test_watching_describes_the_people_shown_and_stops_with_none():
    simulation = started_simulation()
    simulation.max_displayed = 50
    reply = simulation.handle({"type": "watch", "run": 0})
    assert reply["type"] == "watching" and len(reply["ages"]) == len(reply["roles"]) == 50
    ids = reply["ids"]
    assert len(ids) == len(set(ids)) == 50 and ids == sorted(ids)
    assert max(ids) > 100, "a sample of the town, not its first people"
    model = simulation.models[0]
    assert reply["ages"] == model.town.age[ids].tolist()
    frame = simulation.frame()["runs"][0]
    assert len(frame["x"]) == 50
    arena = model.world["arena_size"]
    assert frame["x"] == np.round(model.x[ids] / arena, 4).tolist()
    assert frame["states"]["a"] == model.disease_states["a"].state[ids].tolist()
    assert simulation.handle({"type": "watch", "run": None}) == {"type": "watching", "run": None}
    assert "x" not in simulation.frame()["runs"][0]


@pytest.mark.parametrize("run", [1, -1, "0", 0.5, True])
def test_watching_an_unknown_run_is_rejected(run):
    simulation = started_simulation()
    with pytest.raises(ValueError, match="Unknown run"):
        simulation.handle({"type": "watch", "run": run})


def test_a_new_experiment_watches_nothing():
    simulation = started_simulation()
    simulation.handle({"type": "watch", "run": 0})
    simulation.handle(start_command())
    assert simulation.watched is None


def test_a_frame_never_takes_longer_to_compute_than_the_limit_allows():
    simulation = started_simulation(population=3000, arena_size=2000)
    simulation.max_frame_seconds = 0.0
    simulation.handle({"type": "speed", "value": 2400})
    simulation.advance()
    assert steps(simulation) == [1], "at least one step, however slow"


def test_frame_clock_follows_the_models():
    simulation = started_simulation()
    simulation.handle({"type": "speed", "value": 30 * 30})
    simulation.advance()
    frame = simulation.frame()
    assert (frame["weekday"], frame["hour"]) == (1, 6.0)


def test_frames_carry_every_new_history_record_exactly_once_for_each_run():
    simulation = started_simulation(steps_per_day=4, runs=2)
    first = simulation.frame()
    assert [[r["day"] for r in run["history"]] for run in first["runs"]] == [[0], [0]]
    assert [run["history"] for run in simulation.frame()["runs"]] == [[], []]
    simulation.handle({"type": "speed", "value": 48 * 30})  # two days per frame
    simulation.advance()
    assert [[r["day"] for r in run["history"]] for run in simulation.frame()["runs"]] == [
        [1, 2],
        [1, 2],
    ]
    simulation.handle(start_command())
    assert [r["day"] for r in simulation.frame()["runs"][0]["history"]] == [0]


def test_inspect_describes_one_person_of_one_run():
    simulation = started_simulation(runs=2)
    reply = simulation.handle({"type": "inspect", "run": 1, "person": 7})
    assert reply["type"] == "person" and reply["run"] == 1 and reply["id"] == 7
    json.dumps(reply)


@pytest.mark.parametrize(
    "command",
    [
        {"type": "inspect", "run": 0, "person": 10_000},
        {"type": "inspect", "run": 0, "person": -1},
        {"type": "inspect", "run": 5, "person": 0},
        {"type": "inspect", "run": -1, "person": 0},
        {"type": "inspect", "run": True, "person": 0},
        {"type": "inspect", "run": 0, "person": "3"},
        {"type": "inspect", "person": 3},
    ],
)
def test_inspecting_an_unknown_person_or_run_is_rejected(command):
    with pytest.raises((ValueError, KeyError, IndexError, TypeError)):
        started_simulation().handle(command)


def test_the_experiment_has_no_end_unless_a_maximum_duration_is_set():
    simulation = started_simulation(steps_per_day=4)
    for _ in range(30):
        simulation.handle({"type": "step_day"})
    assert not simulation.finished and steps(simulation) == [120]
    assert simulation.frame()["finished"] is False


def test_the_experiment_finishes_by_itself_at_the_maximum_duration():
    simulation = started_simulation(steps_per_day=4, max_days=3, runs=2)
    simulation.handle({"type": "speed", "value": 48 * 30})  # two days per frame
    simulation.handle({"type": "play"})
    simulation.advance()
    assert not simulation.finished and steps(simulation) == [8, 8]
    simulation.advance()
    assert simulation.finished and not simulation.playing
    assert steps(simulation) == [12, 12], "never beyond the maximum duration"
    frame = simulation.frame()
    assert frame["finished"] and frame["day"] == 3
    assert [record["day"] for record in frame["runs"][0]["history"]] == [0, 1, 2, 3]


def test_the_maximum_duration_also_stops_a_day_step_and_the_max_speed():
    simulation = started_simulation(steps_per_day=4, max_days=1)
    simulation.handle({"type": "step_day"})
    assert simulation.finished and steps(simulation) == [4]
    simulation = started_simulation(steps_per_day=4, max_days=2)
    simulation.handle({"type": "speed", "value": 0})
    simulation.handle({"type": "play"})
    for _ in range(50):  # frames of the computing budget each, as many as a slow machine needs
        simulation.advance()
        if simulation.finished:
            break
    assert simulation.finished and steps(simulation) == [8]


def test_finish_ends_the_experiment_whenever_the_user_says_so():
    simulation = started_simulation()
    simulation.handle({"type": "play"})
    simulation.handle({"type": "finish"})
    assert simulation.finished and not simulation.playing
    assert simulation.frame()["finished"]


def test_the_hours_after_the_last_midnight_are_recorded_when_the_experiment_is_finished():
    simulation = started_simulation()
    simulation.handle({"type": "step_day"})
    simulation._run_steps(TEST_STEPS_PER_DAY // 4)
    model = simulation.models[0]
    model.disease_states["a"].cumulative_infections += 5  # what the last hours changed
    simulation.handle({"type": "finish"})
    simulation.handle({"type": "finish"})
    days = [record["day"] for record in model.history]
    assert days == [0, 1, 1.25]
    assert model.history[-1]["cumulative_infections"]["a"] == (
        model.disease_states["a"].cumulative_infections
    )


def test_an_experiment_finished_at_midnight_records_no_extra_day():
    simulation = started_simulation()
    simulation.handle({"type": "step_day"})
    simulation.handle({"type": "finish"})
    assert [record["day"] for record in simulation.models[0].history] == [0, 1]


@pytest.mark.parametrize("command", [{"type": "play"}, {"type": "step_day"}])
def test_a_finished_experiment_cannot_continue(command):
    simulation = started_simulation()
    simulation.handle({"type": "finish"})
    with pytest.raises(ValueError, match="finished"):
        simulation.handle(command)
    assert steps(simulation) == [0]


def test_restarting_after_the_finish_begins_a_new_experiment():
    simulation = started_simulation()
    simulation.handle({"type": "finish"})
    simulation.handle(start_command())
    assert not simulation.finished
    simulation.handle({"type": "step_day"})
    assert steps(simulation) == [24]


def test_finish_needs_a_started_experiment():
    with pytest.raises(ValueError, match="start one first"):
        new_simulation().handle({"type": "finish"})


@pytest.mark.parametrize("days", [-1, 1.5, float("inf")])
def test_invalid_maximum_duration_is_rejected(days):
    with pytest.raises(ValueError):
        new_simulation().handle(start_command(max_days=days))
