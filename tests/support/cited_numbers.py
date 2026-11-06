"""Check that the numbers a disease cites a source for are written down once more in its tests.

Changing one of them changes what the disease means (and what its comment claims), so a change has
to be made in the test of the model on purpose, together with its source.
"""

import pytest
from config_factory import LIBRARY, disease_of

from epidemic_sandbox.disease_models import DISEASE_MODELS


def check_disease_numbers(
    key: str, shared: dict, disease_only: dict, age_effects: dict, name: str | None = None
) -> None:
    """The disease of model `key` (or the disease `name`) has the shared and its own values and the
    age effects."""
    disease = LIBRARY.catalog[name or disease_of(key)]
    assert disease.model == key
    for name, value in {**shared, **disease_only}.items():
        assert disease.parameters[name] == pytest.approx(value), name
    for effect, rows in age_effects.items():
        shown = [list(row) for row in disease.age_effects[effect]]
        assert all(row in shown for row in rows), effect


def check_default_numbers(key: str, shared: dict) -> None:
    """The model's own defaults are the values its parameters cite."""
    model = DISEASE_MODELS[key](1.0)
    for name, value in shared.items():
        assert model.parameters[name] == pytest.approx(value), name
