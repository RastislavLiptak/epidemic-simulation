from dataclasses import replace
from pathlib import Path

from epidemic_sandbox.app_config import load_app_config
from epidemic_sandbox.epidemic_model import EpidemicModel
from epidemic_sandbox.experiment_config import ExperimentConfig, ExperimentLibrary, load_library

CONFIG_DIRECTORY = Path(__file__).resolve().parent.parent / "configs"
APP_CONFIG = load_app_config(CONFIG_DIRECTORY / "app.toml")
LIBRARY = load_library(CONFIG_DIRECTORY / "experiment.toml", APP_CONFIG.limits)

WORLD = {"population": 200, "arena_area": 250_000, "seed": 1, "runs": 1}
TEST_STEPS_PER_DAY = 24  # coarser than in the application, so that tests run fewer steps


def disease_definition(model: str, initial_infected: int = 1, **parameters: float) -> dict:
    return {"model": model, "initial_infected": initial_infected, "parameters": parameters}


def interaction(source, states, target, effect="susceptibility", factor=0.5) -> dict:
    return {
        "source": source,
        "states": list(states),
        "target": target,
        "effect": effect,
        "factor": factor,
    }


def definition(diseases: dict | None = None, interactions=(), **world) -> dict:
    """An experiment definition; a test may give the side of the town, `arena_size`, in meters."""
    if "arena_size" in world:
        world["arena_area"] = round(world.pop("arena_size") ** 2)
    return {
        "world": {**WORLD, **world},
        "diseases": diseases or {"a": disease_definition("sir")},
        "interactions": list(interactions),
    }


def library_with(steps_per_day: int) -> ExperimentLibrary:
    return replace(LIBRARY, steps_per_day=steps_per_day)


def experiment_config(diseases: dict | None = None, interactions=(), **world) -> ExperimentConfig:
    """A parsed experiment; a test may give `steps_per_day`, which no experiment can set."""
    steps_per_day = world.pop("steps_per_day", TEST_STEPS_PER_DAY)
    return library_with(steps_per_day).parse(definition(diseases, interactions, **world))


def make_model(diseases: dict | None = None, interactions=(), town=None, **world) -> EpidemicModel:
    config = experiment_config(diseases, interactions, **world)
    return EpidemicModel(config, town or LIBRARY.town)
