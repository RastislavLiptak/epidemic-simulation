from pathlib import Path

import pytest
from config_factory import (
    APP_CONFIG,
    CONFIG_DIRECTORY,
    LIBRARY,
    definition,
    disease_definition,
    interaction,
)

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.experiment_config import STEPS_PER_DAY, load_library

SIR = 'model = "sir"\ninitial_infected = 1\n'


def write_library(
    directory: Path, diseases: dict[str, str], enabled=None, world="", extra=""
) -> Path:
    (directory / "diseases").mkdir(parents=True)
    reference = directory.parent / "data" / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    for table in (CONFIG_DIRECTORY.parent / "data" / "reference").glob("*.csv"):
        (reference / table.name).write_text(table.read_text())
    (directory / "town.toml").write_text((CONFIG_DIRECTORY / "town.toml").read_text())
    for name, content in diseases.items():
        (directory / "diseases" / f"{name}.toml").write_text(content)
    names = list(diseases) if enabled is None else enabled
    path = directory / "experiment.toml"
    path.write_text(
        world + "diseases = [" + ", ".join(f'"{name}"' for name in names) + "]\n" + extra
    )
    return path


def load(path: Path):
    return load_library(path, APP_CONFIG.limits)


def test_demo_library_loads_every_model_and_the_default_interactions():
    assert {disease.model for disease in LIBRARY.catalog.values()} == DISEASE_MODELS.keys()
    assert LIBRARY.default.diseases.keys() <= LIBRARY.catalog.keys()
    assert len(LIBRARY.default.interactions) == 2
    assert LIBRARY.default.world["population"] == 3000


def test_catalog_contains_diseases_that_are_not_enabled(tmp_path):
    library = load(write_library(tmp_path / "configs", {"flu": SIR, "cold": SIR}, enabled=["flu"]))
    assert library.catalog.keys() == {"flu", "cold"} and library.default.diseases.keys() == {"flu"}


def test_missing_world_settings_use_defaults(tmp_path):
    assert (
        load(write_library(tmp_path / "configs", {"flu": SIR})).default.world["population"] == 3000
    )


def test_interactions_are_read_from_the_experiment_file(tmp_path):
    extra = (
        '[[interactions]]\nsource = "a"\nstates = ["recovered"]\ntarget = "b"\n'
        'effect = "infectiousness"\nfactor = 0.5\n'
    )
    library = load(write_library(tmp_path / "configs", {"a": SIR, "b": SIR}, extra=extra))
    assert library.default.interactions[0].effect == "infectiousness"


@pytest.mark.parametrize(
    ("diseases", "enabled", "message"),
    [
        ({"flu": SIR}, [], "at least one disease"),
        ({"flu": SIR}, ["gone"], "without a file"),
        ({"flu": 'model = "xyz"\ninitial_infected = 1\n'}, None, "unknown model"),
        ({"flu": SIR + "[parameters]\nbeta = 1\n"}, None, "Unknown parameters"),
        ({"flu": SIR + "[parameters]\ntransmission = 2\n"}, None, "out of range"),
        ({"Flu": SIR}, None, "Disease name"),
        ({"flu-a": SIR}, None, "Disease name"),
    ],
)
def test_invalid_library_is_rejected(tmp_path, diseases, enabled, message):
    with pytest.raises(ValueError, match=message):
        load(write_library(tmp_path / "configs", diseases, enabled))


def test_unknown_world_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown parameters"):
        load(write_library(tmp_path / "configs", {"flu": SIR}, world="mobility = 6\n"))


def test_the_time_step_is_fixed_by_the_application_and_not_part_of_an_experiment():
    config = LIBRARY.parse(definition())
    assert config.world["steps_per_day"] == STEPS_PER_DAY == 48
    assert "steps_per_day" not in config.to_definition()["world"]
    assert LIBRARY.default.world["steps_per_day"] == STEPS_PER_DAY


def test_the_town_is_set_by_its_area_and_its_side_follows_from_it():
    config = LIBRARY.parse(definition(arena_area=2_250_000))
    assert config.world["arena_size"] == 1500
    assert "arena_area" in config.to_definition()["world"]
    assert "arena_size" not in config.to_definition()["world"]
    with pytest.raises(ValueError, match="Unknown parameters"):
        LIBRARY.parse({"world": {"arena_size": 2000}, "diseases": TWO})


def test_a_town_that_is_too_small_is_told_its_least_area_in_square_meters():
    with pytest.raises(ValueError, match=r"area at least 250,000 m²"):
        LIBRARY.parse(definition(population=5000, arena_area=100_000))


def test_definition_round_trips():
    config = LIBRARY.parse(
        definition(
            {
                "a": disease_definition("seir", 3, transmission=0.2),
                "b": disease_definition("sir"),
            },
            [interaction("a", ["recovered"], "b", "infectiousness", 0.4)],
        )
    )
    assert LIBRARY.parse(config.to_definition()) == config
    assert config.diseases["a"].parameters["incubation_days"] == 3.0
    assert config.interactions[0].factor == 0.4


def test_any_number_of_diseases_can_be_chosen():
    many = {f"d{i}": disease_definition("sir") for i in range(25)}
    assert len(LIBRARY.parse(definition(many)).diseases) == 25


TWO = {"a": disease_definition("sir"), "b": disease_definition("seir")}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"world": {"population": 10}}, "out of range"),
        ({"world": {"colour": 1}}, "Unknown parameters"),
        ({"world": {"population": 200.5}}, "whole number"),
        ({"world": {"runs": 65}}, "out of range"),
        ({"world": {"steps_per_day": 48}}, "Unknown parameters"),
        (
            {"world": {"population": 1_000_000, "arena_area": 400_000_000, "runs": 6}},
            "too many together",
        ),
        ({"world": {"space_per_person": 10}}, "Unknown parameters"),
        (
            {"diseases": {"a": disease_definition("sir", contact_distance=50)}},
            "within contact distance",
        ),
        ({"world": {"population": 100_000, "arena_area": 1_000_000}}, "too small"),
        ({"diseases": {}}, "at least one disease"),
        ({"diseases": {"a": disease_definition("sir", 500)}}, "exceeds the population"),
        ({"diseases": {"a": disease_definition("sir", transmission=5)}}, "out of range"),
        ({"diseases": {"a": disease_definition("sir", incubation_days=1)}}, "Unknown parameters"),
        ({"diseases": {"a": disease_definition("nope")}}, "unknown model"),
        ({"diseases": {"Bad Name": disease_definition("sir")}}, "Disease name"),
        ({"diseases": TWO, "interactions": [interaction("a", ["recovered"], "zzz")]}, "not one of"),
        ({"diseases": TWO, "interactions": [interaction("zzz", ["recovered"], "b")]}, "not one of"),
        (
            {"diseases": TWO, "interactions": [interaction("a", ["recovered"], "a")]},
            "two different",
        ),
        ({"diseases": TWO, "interactions": [interaction("a", ["exposed"], "b")]}, "not states of"),
        ({"diseases": TWO, "interactions": [interaction("a", [], "b")]}, "not states of"),
        (
            {"diseases": TWO, "interactions": [interaction("a", ["recovered"], "b", "magic")]},
            "Unknown effect",
        ),
        (
            {"diseases": TWO, "interactions": [interaction("a", ["recovered"], "b", factor=-1)]},
            "out of range",
        ),
        (
            {"diseases": TWO, "interactions": [interaction("a", ["recovered"], "b", factor=1000)]},
            "out of range",
        ),
        (
            {"diseases": TWO, "interactions": [interaction("a", ["recovered"], "b")] * 21},
            "At most 20",
        ),
    ],
)
def test_invalid_definition_is_rejected(change, message):
    base = definition()
    candidate = {**base, **change, "world": {**base["world"], **change.get("world", {})}}
    with pytest.raises(ValueError, match=message):
        LIBRARY.parse(candidate)


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"world": {}},
        {"world": 5, "diseases": {}},
        None,
        [],
        {**definition(), "interactions": "nonsense"},
        {**definition(), "interactions": [5]},
        {**definition(), "interactions": [{"source": "a"}]},
        {**definition(), "interactions": [{**interaction("a", "recovered", "b")}]},
    ],
)
def test_malformed_definition_is_rejected(broken):
    with pytest.raises((ValueError, KeyError, TypeError)):
        LIBRARY.parse(broken)


def test_definition_without_interactions_is_valid():
    candidate = definition()
    del candidate["interactions"]
    assert LIBRARY.parse(candidate).interactions == ()


def test_catalog_diseases_are_enabled_by_name_with_only_the_initial_cases_chosen():
    config = LIBRARY.parse(
        {
            "world": {"population": 500, "arena_area": 640_000},
            "diseases": {"demo_sir": {"initial_infected": 7}},
        }
    )
    assert config.diseases["demo_sir"].initial_infected == 7
    assert config.diseases["demo_sir"].parameters == LIBRARY.catalog["demo_sir"].parameters
    assert config.interactions == (), "interactions need both diseases"


def test_interactions_between_enabled_catalog_diseases_come_from_the_config():
    config = LIBRARY.parse(
        {
            "world": {"population": 500, "arena_area": 640_000},
            "diseases": {"demo_sir": {"initial_infected": 1}, "demo_seir": {"initial_infected": 1}},
        }
    )
    assert [(i.source, i.target) for i in config.interactions] == [("demo_sir", "demo_seir")]


def test_parameters_of_a_catalog_disease_cannot_be_overridden():
    broken = {
        "world": {"population": 500, "arena_area": 640_000},
        "diseases": {"demo_sir": {"initial_infected": 1, "parameters": {"transmission": 1.0}}},
    }
    with pytest.raises(ValueError, match="defined in its file"):
        LIBRARY.parse(broken)


def test_unknown_catalog_disease_is_rejected():
    with pytest.raises(ValueError, match="Unknown disease"):
        LIBRARY.parse({"world": {}, "diseases": {"plague": {"initial_infected": 1}}})


def test_town_preview_may_have_no_diseases_but_an_experiment_may_not():
    definition = {"world": {}, "diseases": {}}
    assert LIBRARY.parse(definition, allow_no_diseases=True).diseases == {}
    with pytest.raises(ValueError, match="at least one disease"):
        LIBRARY.parse(definition)
