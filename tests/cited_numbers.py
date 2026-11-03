"""Check that the numbers a disease cites a source for are written down once more in its tests.

Changing one of them changes what the disease means (and what its comment claims), so a change has
to be made in the test of the model on purpose, together with its source.
"""

import pytest
from config_factory import LIBRARY

from epidemic_sandbox.disease_models import DISEASE_MODELS


def check_demo_numbers(key: str, shared: dict, demo_only: dict, age_effects: dict) -> None:
    """The demo disease of model `key` has the shared and the demo-only values and age effects."""
    demo = LIBRARY.catalog[f"demo_{key}"]
    for name, value in {**shared, **demo_only}.items():
        assert demo.parameters[name] == pytest.approx(value), name
    for effect, rows in age_effects.items():
        shown = [list(row) for row in demo.age_effects[effect]]
        assert all(row in shown for row in rows), effect


def check_default_numbers(key: str, shared: dict) -> None:
    """The model's own defaults are the values its parameters cite."""
    model = DISEASE_MODELS[key](1.0)
    for name, value in shared.items():
        assert model.parameters[name] == pytest.approx(value), name
