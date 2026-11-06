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

from epidemic_sandbox.config.experiment import STEPS_PER_DAY, load_library
from epidemic_sandbox.disease_models import DISEASE_MODELS

INFECTION = 'model = "plain"\ninitial_infected = 1\n'


def write_library(
    directory: Path, diseases: dict[str, str], enabled=None, world="", interactions=None
) -> Path:
    """A library on disk: `enabled` names the diseases of the default experiment (all if None)."""
    (directory / "diseases").mkdir(parents=True)
    reference = directory.parent / "data" / "reference"
    reference.mkdir(parents=True, exist_ok=True)
    for table in (CONFIG_DIRECTORY.parent / "data" / "reference").glob("*.csv"):
        (reference / table.name).write_text(table.read_text())
    (directory / "town.toml").write_text((CONFIG_DIRECTORY / "town.toml").read_text())
    names = list(diseases) if enabled is None else enabled
    for name, content in diseases.items():
        flag = "enabled = true\n" if name in names else ""
        (directory / "diseases" / f"{name}.toml").write_text(flag + content)
    if interactions:
        (directory / "interactions").mkdir()
        for name, content in interactions.items():
            (directory / "interactions" / f"{name}.toml").write_text(content)
    path = directory / "experiment.toml"
    path.write_text(world)
    return path


def load(path: Path):
    return load_library(path, APP_CONFIG.limits)


def test_demo_library_loads_every_model_and_the_default_interactions():
    assert {disease.model for disease in LIBRARY.catalog.values()} == DISEASE_MODELS.keys()
    assert LIBRARY.default.diseases.keys() <= LIBRARY.catalog.keys()
    assert LIBRARY.default.interactions == tuple(
        rule
        for rule in LIBRARY.interactions
        if rule.source in LIBRARY.default.diseases and rule.target in LIBRARY.default.diseases
    )
    assert LIBRARY.default.world["population"] == 3000


@pytest.mark.usefixtures("synthetic_in_registry")
def test_catalog_contains_diseases_that_are_not_enabled(tmp_path):
    library = load(
        write_library(tmp_path / "configs", {"flu": INFECTION, "cold": INFECTION}, enabled=["flu"])
    )
    assert library.catalog.keys() == {"flu", "cold"} and library.default.diseases.keys() == {"flu"}


@pytest.mark.usefixtures("synthetic_in_registry")
def test_missing_world_settings_use_defaults(tmp_path):
    assert (
        load(write_library(tmp_path / "configs", {"flu": INFECTION})).default.world["population"]
        == 3000
    )


@pytest.mark.usefixtures("synthetic_in_registry")
def test_interactions_are_read_from_files_of_their_own(tmp_path):
    rule = (
        'source = "a"\nstates = ["recovered"]\ntarget = "b"\n'
        'effect = "infectiousness"\nfactor = 0.5\n'
    )
    library = load(
        write_library(
            tmp_path / "configs", {"a": INFECTION, "b": INFECTION}, interactions={"a_to_b": rule}
        )
    )
    assert library.default.interactions[0].effect == "infectiousness"


@pytest.mark.usefixtures("synthetic_in_registry")
def test_an_interaction_that_names_a_disease_without_a_file_is_rejected(tmp_path):
    rule = (
        'source = "a"\nstates = ["recovered"]\ntarget = "gone"\n'
        'effect = "susceptibility"\nfactor = 0.5\n'
    )
    with pytest.raises(ValueError, match="without a file"):
        load(
            write_library(tmp_path / "configs", {"a": INFECTION}, interactions={"a_to_gone": rule})
        )


@pytest.mark.usefixtures("synthetic_in_registry")
@pytest.mark.parametrize(
    ("diseases", "enabled", "message"),
    [
        ({"flu": INFECTION}, [], "at least one disease"),
        ({"flu": 'model = "xyz"\ninitial_infected = 1\n'}, None, "unknown model"),
        ({"flu": INFECTION + "[parameters]\nbeta = 1\n"}, None, "Unknown parameters"),
        ({"flu": INFECTION + "[parameters]\ntransmission = 2\n"}, None, "out of range"),
        ({"Flu": INFECTION}, None, "Disease name"),
        ({"flu-a": INFECTION}, None, "Disease name"),
    ],
)
def test_invalid_library_is_rejected(tmp_path, diseases, enabled, message):
    with pytest.raises(ValueError, match=message):
        load(write_library(tmp_path / "configs", diseases, enabled))


@pytest.mark.usefixtures("synthetic_in_registry")
def test_unknown_world_setting_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown parameters"):
        load(write_library(tmp_path / "configs", {"flu": INFECTION}, world="mobility = 6\n"))


@pytest.mark.usefixtures("synthetic_in_registry")
def test_the_time_step_is_fixed_by_the_application_and_not_part_of_an_experiment():
    config = LIBRARY.parse(definition())
    assert config.world["steps_per_day"] == STEPS_PER_DAY == 48
    assert "steps_per_day" not in config.to_definition()["world"]
    assert LIBRARY.default.world["steps_per_day"] == STEPS_PER_DAY


@pytest.mark.usefixtures("synthetic_in_registry")
def test_the_town_is_set_by_its_area_and_its_side_follows_from_it():
    config = LIBRARY.parse(definition(area_km2=2.25))
    assert config.world["arena_size"] == 1500
    assert "area_km2" in config.to_definition()["world"]
    assert "arena_size" not in config.to_definition()["world"]
    with pytest.raises(ValueError, match="Unknown parameters"):
        LIBRARY.parse({"world": {"arena_size": 2000}, "diseases": TWO})


@pytest.mark.usefixtures("synthetic_in_registry")
def test_settings_are_limited_only_by_what_the_simulation_needs():
    # a town of one, a seed beyond a million, 100 runs and an experiment of a thousand years
    # are all accepted: the only ceilings are the room for the people and the memory
    config = LIBRARY.parse(
        definition(population=1, seed=10**12, runs=100, max_days=365_000, area_km2=0.0001)
    )
    assert config.world["seed"] == 10**12 and config.world["runs"] == 100
    with pytest.raises(ValueError, match="at least 0"):
        LIBRARY.parse(definition(seed=-1))


def test_a_town_that_is_too_small_is_told_its_least_area_in_square_kilometers():
    with pytest.raises(ValueError, match=r"area at least 0.25 km²"):
        LIBRARY.parse(definition(population=5000, area_km2=0.1))


@pytest.mark.usefixtures("synthetic_in_registry")
def test_definition_round_trips():
    config = LIBRARY.parse(
        definition(
            {
                "a": disease_definition("latent", 3, transmission=0.2),
                "b": disease_definition("plain"),
            },
            [interaction("a", ["recovered"], "b", "infectiousness", 0.4)],
        )
    )
    assert LIBRARY.parse(config.to_definition()) == config
    assert config.diseases["a"].parameters["incubation_days"] == 3.0
    assert config.interactions[0].factor == 0.4


@pytest.mark.usefixtures("synthetic_in_registry")
def test_any_number_of_diseases_can_be_chosen():
    many = {f"d{i}": disease_definition("plain") for i in range(25)}
    assert len(LIBRARY.parse(definition(many)).diseases) == 25


TWO = {"a": disease_definition("plain"), "b": disease_definition("latent")}


@pytest.mark.usefixtures("synthetic_in_registry")
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"world": {"population": 0}}, "out of range"),
        ({"world": {"colour": 1}}, "Unknown parameters"),
        ({"world": {"population": 200.5}}, "whole number"),
        ({"world": {"runs": 0}}, "out of range"),
        ({"world": {"steps_per_day": 48}}, "Unknown parameters"),
        (
            {"world": {"population": 1_000_000, "area_km2": 400, "runs": 6}},
            "too many together",
        ),
        ({"world": {"space_per_person": 10}}, "Unknown parameters"),
        (
            {"diseases": {"a": disease_definition("plain", contact_distance=50)}},
            "within contact distance",
        ),
        ({"world": {"population": 100_000, "area_km2": 1}}, "too small"),
        ({"diseases": {}}, "at least one disease"),
        ({"diseases": {"a": disease_definition("plain", 500)}}, "exceeds the population"),
        ({"diseases": {"a": disease_definition("plain", transmission=5)}}, "out of range"),
        ({"diseases": {"a": disease_definition("plain", incubation_days=1)}}, "Unknown parameters"),
        ({"diseases": {"a": disease_definition("nope")}}, "unknown model"),
        ({"diseases": {"Bad Name": disease_definition("plain")}}, "Disease name"),
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


@pytest.mark.usefixtures("synthetic_in_registry")
def test_definition_without_interactions_is_valid():
    candidate = definition()
    del candidate["interactions"]
    assert LIBRARY.parse(candidate).interactions == ()


WORLD_OF_500 = {"population": 500, "area_km2": 0.64}
A_TO_B = (
    'source = "a"\nstates = ["recovered"]\ntarget = "b"\neffect = "susceptibility"\nfactor = 0.5\n'
)


@pytest.fixture
def two_diseases(tmp_path, synthetic_in_registry):
    """A library of two diseases, `a` and `b`, with an interaction from `a` to `b`."""
    return load(
        write_library(
            tmp_path / "configs", {"a": INFECTION, "b": INFECTION}, interactions={"a_to_b": A_TO_B}
        )
    )


def test_catalog_diseases_are_enabled_by_name_with_only_the_initial_cases_chosen(two_diseases):
    config = two_diseases.parse({"world": WORLD_OF_500, "diseases": {"a": {"initial_infected": 7}}})
    assert config.diseases["a"].initial_infected == 7
    assert config.diseases["a"].parameters == two_diseases.catalog["a"].parameters
    assert config.interactions == (), "interactions need both diseases"


def test_interactions_between_enabled_catalog_diseases_come_from_the_library(two_diseases):
    config = two_diseases.parse(
        {
            "world": WORLD_OF_500,
            "diseases": {"a": {"initial_infected": 1}, "b": {"initial_infected": 1}},
        }
    )
    assert [(i.source, i.target) for i in config.interactions] == [("a", "b")]


def test_parameters_of_a_catalog_disease_cannot_be_overridden(two_diseases):
    broken = {
        "world": WORLD_OF_500,
        "diseases": {"a": {"initial_infected": 1, "parameters": {"transmission": 1.0}}},
    }
    with pytest.raises(ValueError, match="defined in its file"):
        two_diseases.parse(broken)


def test_unknown_catalog_disease_is_rejected():
    with pytest.raises(ValueError, match="Unknown disease"):
        LIBRARY.parse({"world": {}, "diseases": {"plague": {"initial_infected": 1}}})


def test_town_preview_may_have_no_diseases_but_an_experiment_may_not():
    definition = {"world": {}, "diseases": {}}
    assert LIBRARY.parse(definition, allow_no_diseases=True).diseases == {}
    with pytest.raises(ValueError, match="at least one disease"):
        LIBRARY.parse(definition)
