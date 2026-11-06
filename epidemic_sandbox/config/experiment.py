import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from epidemic_sandbox.config.app import LimitsConfig
from epidemic_sandbox.config.interactions import Interaction
from epidemic_sandbox.config.parameter_spec import ParameterSpec, resolve_parameters
from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.world.town import TownParameters, load_town_parameters

DISEASE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
ARENA_FILL_LIMIT = 0.5
SQUARE_METERS_PER_SQUARE_KILOMETER = 1_000_000

# Time resolution of the simulation, 30 minutes. Measured on the fictional demo diseases that the
# real ones have replaced (3,000 people, 8 seeds each), the share of people ever infected was 79,
# 70, 65, 66 and 66 % at 12, 24, 48, 96 and 144 steps a day and the peak was the same from 48 on:
# a coarser step overestimates the spread, a finer one only costs time in proportion.
STEPS_PER_DAY = 48

WORLD_PARAMETERS = {
    "population": ParameterSpec(
        "Population",
        "Number of people in the town. The cost of a step grows roughly linearly with it.",
        3000,
        1,
        None,
        1,
        control="number",
    ),
    "area_km2": ParameterSpec(
        "Town area (km²)",
        "Area of the square town. A larger town means longer trips to work and school.",
        4,
        0.0001,
        None,
        0.01,
        control="number",
    ),
    "seed": ParameterSpec(
        "Random seed",
        "The same seed and settings always give the same course of the epidemic.",
        42,
        0,
        None,
        1,
        control="number",
    ),
    "max_days": ParameterSpec(
        "Maximum duration (days)",
        "The experiment finishes by itself after this many simulated days. "
        "Leave empty (0) for an experiment without an end.",
        0,
        0,
        None,
        1,
        control="number",
    ),
    "runs": ParameterSpec(
        "Runs",
        "How many independent runs of the experiment to start together (seeds seed, seed+1, …). "
        "Several runs show how much the course of an epidemic depends on chance.",
        1,
        1,
        None,
        1,
        control="number",
    ),
}

INITIAL_INFECTED = ParameterSpec(
    "Initially infected",
    "Number of people infected at the start of the run.",
    5,
    0,
    None,
    1,
    control="number",
)


def arena_size_of(area_km2: float) -> float:
    """Side of the square town in meters, which the simulation uses."""
    return math.sqrt(area_km2 * SQUARE_METERS_PER_SQUARE_KILOMETER)


def check_world(world: Mapping, town: TownParameters) -> None:
    """Raise ValueError when the town cannot hold its people."""
    needed = world["population"] * town.min_area_per_person
    if needed > ARENA_FILL_LIMIT * world["area_km2"] * SQUARE_METERS_PER_SQUARE_KILOMETER:
        least = needed / ARENA_FILL_LIMIT / SQUARE_METERS_PER_SQUARE_KILOMETER
        raise ValueError(
            f"The town is too small for {world['population']:,} people; "
            f"make its area at least {math.ceil(least * 100) / 100:g} km²"
        )


@dataclass(frozen=True)
class DiseaseConfig:
    model: str
    initial_infected: int
    parameters: dict[str, float]
    age_effects: dict[str, tuple] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """World settings, the diseases that run together and how they affect each other."""

    world: dict[str, float]
    diseases: dict[str, DiseaseConfig]
    interactions: tuple[Interaction, ...] = ()

    def to_definition(self) -> dict:
        """The full structure `ExperimentLibrary.parse` accepts, independent of the catalog."""
        return {
            "world": {name: self.world[name] for name in WORLD_PARAMETERS},
            "diseases": {
                name: {
                    "model": disease.model,
                    "initial_infected": disease.initial_infected,
                    "parameters": dict(disease.parameters),
                    "age_effects": {
                        key: [list(row) for row in rows]
                        for key, rows in disease.age_effects.items()
                    },
                }
                for name, disease in self.diseases.items()
            },
            "interactions": [interaction.to_definition() for interaction in self.interactions],
        }


@dataclass(frozen=True)
class ExperimentLibrary:
    """The default experiment, the disease catalog, the fixed interactions and the limits."""

    default: ExperimentConfig
    catalog: dict[str, DiseaseConfig]
    interactions: tuple[Interaction, ...]
    town: TownParameters
    limits: LimitsConfig
    steps_per_day: int = STEPS_PER_DAY

    def parse(self, definition: Mapping, *, allow_no_diseases: bool = False) -> ExperimentConfig:
        """Build a validated experiment.

        `definition` is {"world": {name: value}, "diseases": {disease: {"initial_infected": n}}}:
        the diseases come from the catalog, with the interactions that connect the chosen ones.
        A disease with a "model" and "parameters" of its own is used as given instead, and an
        "interactions" list replaces the catalog's. An experiment without diseases is only
        accepted for a preview of the town (`allow_no_diseases`).
        """
        world = resolve_parameters(WORLD_PARAMETERS, _object(definition, "experiment")["world"])
        world["arena_size"] = arena_size_of(world["area_km2"])
        world["steps_per_day"] = self.steps_per_day
        check_world(world, self.town)
        if world["population"] * world["runs"] > self.limits.max_total_people:
            raise ValueError(
                f"{world['runs']} runs of {world['population']:,} people are too many together; "
                f"the limit is {self.limits.max_total_people:,} people"
            )
        diseases = {
            name: self._disease(name, _object(values, name))
            for name, values in _object(definition["diseases"], "diseases").items()
        }
        if not diseases and not allow_no_diseases:
            raise ValueError("An experiment needs at least one disease")
        for name, disease in diseases.items():
            if not DISEASE_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"Disease name '{name}' must match {DISEASE_NAME_PATTERN.pattern}")
            if disease.initial_infected > world["population"]:
                raise ValueError(f"{name}: initially infected exceeds the population")
            self._check_contact_distance(name, disease)
        if "interactions" in definition:
            raw = definition["interactions"]
            if isinstance(raw, str | Mapping):
                raise TypeError("interactions must be a list")
            interactions = tuple(Interaction.from_definition(item) for item in raw)
        else:
            interactions = tuple(
                rule
                for rule in self.interactions
                if rule.source in diseases and rule.target in diseases
            )
        for interaction in interactions:
            _check_interaction(interaction, diseases)
        return ExperimentConfig(world, diseases, interactions)

    def _disease(self, name: str, values: Mapping) -> DiseaseConfig:
        initial = resolve_parameters(
            {"initial_infected": INITIAL_INFECTED}, {"initial_infected": values["initial_infected"]}
        )
        if "model" in values:
            return _disease_config(
                values["model"],
                {**initial, **_object(values["parameters"], name)},
                values.get("age_effects"),
            )
        if name not in self.catalog:
            raise ValueError(f"Unknown disease '{name}', available: {sorted(self.catalog)}")
        unexpected = sorted(set(values) - {"initial_infected"})
        if unexpected:
            raise ValueError(
                f"{name}: {unexpected} cannot be set, a catalog disease is defined in its file"
            )
        known = self.catalog[name]
        return DiseaseConfig(
            known.model, initial["initial_infected"], known.parameters, known.age_effects
        )

    def _check_contact_distance(self, name: str, disease: DiseaseConfig) -> None:
        smallest_area = min(place.area_mean for place in self.town.places.values())
        contacts = math.pi * disease.parameters["contact_distance"] ** 2 / smallest_area
        if contacts > self.town.max_site_contacts:
            raise ValueError(
                f"{name}: {contacts:.0f} people would be within contact distance at crowded places "
                f"(limit {self.town.max_site_contacts:g})"
            )


def load_library(experiment_path: Path, limits: LimitsConfig) -> ExperimentLibrary:
    """Load the world settings in an experiment file and, next to it, the rest of the library.

    `diseases/` holds one file per disease (those with `enabled = true` start in the default
    experiment), `interactions/` one file per interaction between two diseases, and `town.toml`
    the town. Every disease and interaction is a file of its own, so that none depends on another
    and each can be added or removed without touching the rest, except the interactions that
    name a disease.
    """
    with experiment_path.open("rb") as experiment_file:
        raw = tomllib.load(experiment_file)
    directory = experiment_path.parent
    town = load_town_parameters(directory / "town.toml")
    catalog, enabled_names = {}, []
    for path in sorted((directory / "diseases").glob("*.toml")):
        catalog[path.stem], enabled = _load_disease(path)
        if enabled:
            enabled_names.append(path.stem)
    if not enabled_names:
        raise ValueError(f"{directory / 'diseases'}: at least one disease must set enabled = true")
    interactions = tuple(
        _load_interaction(path, catalog)
        for path in sorted((directory / "interactions").glob("*.toml"))
    )
    library = ExperimentLibrary(None, catalog, interactions, town, limits)
    default = library.parse(
        {
            "world": resolve_parameters(WORLD_PARAMETERS, raw),
            "diseases": {
                name: {"initial_infected": catalog[name].initial_infected} for name in enabled_names
            },
        }
    )
    return ExperimentLibrary(default, catalog, interactions, town, limits)


def _check_interaction(interaction: Interaction, diseases: Mapping[str, DiseaseConfig]) -> None:
    for role, name in (("source", interaction.source), ("target", interaction.target)):
        if name not in diseases:
            raise ValueError(f"Interaction {role} '{name}' is not one of the experiment's diseases")
    if interaction.source == interaction.target:
        raise ValueError("An interaction needs two different diseases")
    states = DISEASE_MODELS[diseases[interaction.source].model].STATE_COLORS
    unknown = [state for state in interaction.states if state not in states]
    if unknown or not interaction.states:
        raise ValueError(
            f"Interaction states {unknown or interaction.states} are not states of "
            f"'{interaction.source}', available: {list(states)}"
        )


def _object(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _load_disease(disease_path: Path) -> tuple[DiseaseConfig, bool]:
    """The disease in a file and whether the default experiment starts with it."""
    if not DISEASE_NAME_PATTERN.fullmatch(disease_path.stem):
        raise ValueError(
            f"Disease name '{disease_path.stem}' must match {DISEASE_NAME_PATTERN.pattern}"
        )
    with disease_path.open("rb") as disease_file:
        raw = tomllib.load(disease_file)
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{disease_path}: enabled must be true or false")
    values = {"initial_infected": raw["initial_infected"], **raw.get("parameters", {})}
    return _disease_config(raw["model"], values, raw.get("age_effects"), disease_path), enabled


def _load_interaction(path: Path, catalog: Mapping[str, DiseaseConfig]) -> Interaction:
    with path.open("rb") as interaction_file:
        interaction = Interaction.from_definition(tomllib.load(interaction_file))
    missing = sorted({interaction.source, interaction.target} - catalog.keys())
    if missing:
        raise ValueError(f"{path}: the interaction names diseases without a file: {missing}")
    return interaction


def _disease_config(
    model_key: str, values: Mapping, age_effects: Mapping | None = None, source: Path | None = None
) -> DiseaseConfig:
    if model_key not in DISEASE_MODELS:
        raise ValueError(
            f"{source}: unknown model '{model_key}', available: {sorted(DISEASE_MODELS)}"
        )
    specs = {"initial_infected": INITIAL_INFECTED, **DISEASE_MODELS[model_key].PARAMETERS}
    resolved = resolve_parameters(specs, values)
    model = DISEASE_MODELS[model_key](
        1.0,
        age_effects=age_effects,
        **{key: value for key, value in resolved.items() if key != "initial_infected"},
    )
    return DiseaseConfig(model_key, resolved.pop("initial_infected"), resolved, model.age_effects)
