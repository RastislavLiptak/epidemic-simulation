import pytest
from synthetic_models import LatentModel, PlainModel, RepeatModel

from epidemic_sandbox.disease_models import DISEASE_MODELS


@pytest.fixture
def synthetic_in_registry(monkeypatch):
    """Let experiments in a test name the models that exist only in tests (see synthetic_models)."""
    monkeypatch.setitem(DISEASE_MODELS, "plain", PlainModel)
    monkeypatch.setitem(DISEASE_MODELS, "latent", LatentModel)
    monkeypatch.setitem(DISEASE_MODELS, "repeat", RepeatModel)
