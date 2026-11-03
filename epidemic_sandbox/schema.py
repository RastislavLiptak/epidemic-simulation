from epidemic_sandbox.app_config import AppConfig
from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.epidemic_model import EpidemicModel
from epidemic_sandbox.experiment_config import (
    DISEASE_NAME_PATTERN,
    INITIAL_INFECTED,
    WORLD_PARAMETERS,
    DiseaseConfig,
    ExperimentConfig,
    ExperimentLibrary,
)
from epidemic_sandbox.interactions import EFFECTS
from epidemic_sandbox.parameter_spec import ParameterSpec
from epidemic_sandbox.population import ROLES
from epidemic_sandbox.town import TownParameters, mean_contacts, population_statistics


def describe_ui(app_config: AppConfig) -> dict:
    """The constants the browser needs before it connects."""
    return {**vars(app_config.ui), "max_runs": WORLD_PARAMETERS["runs"].maximum}


def describe_setup(library: ExperimentLibrary) -> dict:
    """JSON-ready description of everything the setup screen needs."""
    return {
        "world": [
            _describe_control(name, spec, library.default.world[name])
            for name, spec in WORLD_PARAMETERS.items()
        ],
        "initial_infected": _describe_control(
            "initial_infected", INITIAL_INFECTED, INITIAL_INFECTED.default
        ),
        "catalog": [
            _describe_disease(name, disease)
            for name, disease in library.catalog.items()
        ],
        "interactions": [interaction.to_definition() for interaction in library.interactions],
        "effects": EFFECTS,
        "age_groups": _describe_age_groups(library.town.population),
        "limits": {
            "name_pattern": DISEASE_NAME_PATTERN.pattern,
        },
    }


def describe_run(config: ExperimentConfig, models: list[EpidemicModel]) -> dict:
    """JSON-ready description of started runs: their seeds and the diseases."""
    return {
        "population": config.world["population"],
        "max_days": config.world["max_days"],
        "role_names": list(ROLES),
        "seeds": [model.world["seed"] for model in models],
        "diseases": [
            {
                "name": name,
                "model": disease.model,
                "states": _describe_states(DISEASE_MODELS[disease.model]),
            }
            for name, disease in config.diseases.items()
        ],
        "interactions": [interaction.to_definition() for interaction in config.interactions],
    }


def describe_preview(config: ExperimentConfig, town: TownParameters) -> dict:
    """Quantities derived from the settings that help to judge an experiment before running it."""
    models = {
        name: DISEASE_MODELS[disease.model](
            1 / config.world["steps_per_day"], age_effects=disease.age_effects, **disease.parameters
        )
        for name, disease in config.diseases.items()
    }
    contacts = {
        name: mean_contacts(config.world, town, model.contact_distance)
        for name, model in models.items()
    }
    return {
        "population": population_statistics(config.world, town),
        "diseases": {
            name: {
                "mean_contacts": contacts[name],
                "mean_field_r0": model.mean_field_r0(contacts[name]),
                "household_risk": model.pair_transmission_probability(),
            }
            for name, model in models.items()
        },
    }


def _describe_disease(name: str, disease: DiseaseConfig) -> dict:
    specs = DISEASE_MODELS[disease.model].PARAMETERS
    return {
        "name": name,
        "model": disease.model,
        "initial_infected": disease.initial_infected,
        "states": _describe_states(DISEASE_MODELS[disease.model]),
        "parameters": [
            {
                "name": parameter,
                "label": spec.label,
                "description": spec.description,
                "value": disease.parameters[parameter],
            }
            for parameter, spec in specs.items()
        ],
        "age_effects": {
            key: [list(row) for row in rows] for key, rows in disease.age_effects.items()
        },
    }


def _describe_age_groups(population) -> dict:
    width, shares = population.age_group_width, population.age_shares
    labels = [f"{start * width}–{start * width + width - 1}" for start in range(len(shares) - 1)]
    labels.append(f"{(len(shares) - 1) * width}+")
    return {"labels": labels, "census_shares": shares.tolist()}


def _describe_states(model_class) -> list[dict]:
    """The states with their colors.

    `in_course` marks the states a person passes through when ill and `exposable` those an
    infection can start from, so that the arena can tell a new infection from any other move.
    """
    return [
        {
            "name": state,
            "color": color,
            "infectious": state in model_class.INFECTIOUS_STATES,
            "in_course": state in model_class.course_states(),
            "exposable": state in model_class.exposable_states(),
        }
        for state, color in model_class.STATE_COLORS.items()
    ]


def _describe_control(name: str, spec: ParameterSpec, value: float) -> dict:
    return {
        "name": name,
        "label": spec.label,
        "description": spec.description,
        "min": spec.minimum,
        "max": spec.maximum,
        "step": spec.step,
        "value": value,
        "control": spec.control,
    }
