import json

import pytest
from config_factory import (
    APP_CONFIG,
    LIBRARY,
    disease_definition,
    experiment_config,
    interaction,
)

from epidemic_sandbox.disease_models.disease_model_abstract import AGE_EFFECTS
from epidemic_sandbox.epidemic_model import EpidemicModel
from epidemic_sandbox.schema import describe_preview, describe_run, describe_setup, describe_ui


def definition_from_setup(setup: dict) -> dict:
    """What the UI sends once the user has enabled the diseases of the default experiment."""
    return {
        "world": {control["name"]: control["value"] for control in setup["world"]},
        "diseases": {
            disease["name"]: {"initial_infected": disease["initial_infected"]}
            for disease in setup["catalog"]
            if disease["name"] in LIBRARY.default.diseases
        },
    }


def test_setup_with_default_diseases_enabled_reproduces_the_default_experiment():
    config = LIBRARY.parse(definition_from_setup(describe_setup(LIBRARY)))
    assert config == LIBRARY.default


def test_setup_description_is_json_serializable():
    json.dumps(describe_setup(LIBRARY))


def test_every_control_is_explained_and_has_a_range():
    setup = describe_setup(LIBRARY)
    for control in [*setup["world"], setup["initial_infected"]]:
        assert control["label"] and control["description"]
        assert control["min"] <= control["value"] <= control["max"]


def test_setup_world_controls_are_only_the_ones_the_user_may_set():
    names = [control["name"] for control in describe_setup(LIBRARY)["world"]]
    assert names == [
        "population",
        "arena_area",
        "seed",
        "max_days",
        "runs",
    ]


def test_setup_lists_the_catalog_with_read_only_parameters_and_the_fixed_interactions():
    setup = describe_setup(LIBRARY)
    assert {entry["name"] for entry in setup["catalog"]} == LIBRARY.catalog.keys()
    seir = next(entry for entry in setup["catalog"] if entry["model"] == "seir")
    assert [state["name"] for state in seir["states"]] == [
        "susceptible",
        "exposed",
        "infectious",
        "recovered",
        "dead",
    ]
    assert [state["infectious"] for state in seir["states"]] == [False, False, True, False, False]
    assert [state["in_course"] for state in seir["states"]] == [False, True, True, True, False]
    assert [state["exposable"] for state in seir["states"]] == [True, False, False, True, False]
    assert {"transmission", "contact_distance", "fatality"} <= {
        parameter["name"] for parameter in seir["parameters"]
    }
    for parameter in seir["parameters"]:
        assert parameter["label"] and parameter["description"] and "value" in parameter
        assert "min" not in parameter, "parameters are shown, not edited"
    assert set(seir["age_effects"]) <= set(AGE_EFFECTS)
    assert set(setup["limits"]) == {"name_pattern"}
    assert set(setup["effects"]) == {"susceptibility", "infectiousness"}
    assert len(setup["interactions"]) == 2 and setup["interactions"][0]["states"] == ["recovered"]


def test_the_run_description_lists_diseases_interactions_and_every_run_s_people():
    config = experiment_config(
        {"a": disease_definition("seir", 2, transmission=0.2), "b": disease_definition("sir")},
        [interaction("a", ["recovered"], "b")],
        runs=2,
    )
    models = [EpidemicModel(config, LIBRARY.town) for _ in range(2)]
    run = describe_run(config, models)
    disease = run["diseases"][0]
    assert disease["name"] == "a" and disease["model"] == "seir"
    assert [state["name"] for state in disease["states"]][-1] == "dead"
    assert run["population"] == 200 and run["interactions"][0]["target"] == "b"
    assert run["seeds"] == [config.world["seed"]] * 2
    assert "runs" not in run, "people are described only for the run being watched"
    assert run["role_names"][0] == "homebound"
    json.dumps(run)


def test_preview_reports_contacts_reproduction_number_and_household_risk():
    config = experiment_config(
        {
            "a": disease_definition("sir", transmission=0.5, infectious_days=4.0),
            "b": disease_definition("sir", transmission=1.0),
        }
    )
    preview = describe_preview(config, LIBRARY.town)
    json.dumps(preview)
    a, b = preview["diseases"]["a"], preview["diseases"]["b"]
    assert 0.1 < a["mean_contacts"] < 20
    assert a["mean_field_r0"] == pytest.approx(0.6931 * a["mean_contacts"] * 4.0, rel=1e-3)
    assert 0 < a["household_risk"] < 1
    assert b["mean_field_r0"] is None and b["household_risk"] == 1.0


def test_ui_constants_come_from_the_app_config_and_the_run_limit():
    ui = describe_ui(APP_CONFIG)
    assert ui["toast_ms"] == APP_CONFIG.ui.toast_ms and ui["max_runs"] == 64
    json.dumps(ui)


def test_a_disease_with_a_larger_contact_distance_has_more_contacts():
    config = experiment_config(
        {
            "near": disease_definition("sir", contact_distance=2.0),
            "far": disease_definition("sir", contact_distance=6.0),
        }
    )
    diseases = describe_preview(config, LIBRARY.town)["diseases"]
    assert diseases["far"]["mean_contacts"] > diseases["near"]["mean_contacts"]


def test_setup_description_contains_the_census_age_groups():
    groups = describe_setup(LIBRARY)["age_groups"]
    assert len(groups["labels"]) == len(groups["census_shares"]) == 18
    assert groups["labels"][0] == "0–4" and groups["labels"][-1] == "85+"
    assert sum(groups["census_shares"]) == pytest.approx(1.0)


def test_preview_describes_the_population_of_the_town():
    preview = describe_preview(experiment_config(population=3000, arena_size=2000), LIBRARY.town)
    json.dumps(preview)
    population = preview["population"]
    assert 38 < population["mean_age"] < 46 and 1.8 < population["mean_household_size"] < 2.5
    assert len(population["age_shares"]) == 18 and set(population["role_shares"]) >= {
        "worker",
        "school",
    }
