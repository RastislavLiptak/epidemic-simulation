"""The registry of disease models is the modules of the package, nothing else."""

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from epidemic_sandbox import disease_models
from epidemic_sandbox.disease_models import DISEASE_MODELS, discover_models

MODEL_FILE = textwrap.dedent(
    """
    from synthetic_models import PlainModel


    class {name}Model(PlainModel):
        pass
    """
)


def make_package(tmp_path: Path, monkeypatch, files: dict[str, str]):
    package = tmp_path / "fake_models"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name, content in files.items():
        (package / f"{name}.py").write_text(content)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [name for name in sys.modules if name.split(".")[0] == "fake_models"]:
        monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("fake_models")


def test_every_module_of_the_package_is_a_model_and_nothing_else_is():
    modules = {
        path.stem
        for path in Path(disease_models.__file__).parent.glob("*.py")
        if path.stem not in ("__init__", "disease_model_abstract")
    }
    assert set(DISEASE_MODELS) == modules
    assert list(DISEASE_MODELS) == sorted(DISEASE_MODELS)


def test_a_model_is_found_by_the_name_of_its_module(tmp_path, monkeypatch):
    package = make_package(tmp_path, monkeypatch, {"alpha": MODEL_FILE.format(name="Alpha")})
    assert list(discover_models(package)) == ["alpha"]


@pytest.mark.parametrize(
    "files",
    [
        {"alpha": "x = 1\n"},
        {"alpha": MODEL_FILE.format(name="Beta")},
        {
            "alpha": MODEL_FILE.format(name="Alpha")
            + "\n\nclass AnotherModel(AlphaModel):\n    pass\n"
        },
    ],
    ids=["no model", "named after another module", "two models"],
)
def test_a_module_that_is_not_exactly_one_model_named_after_it_is_refused(
    tmp_path, monkeypatch, files
):
    package = make_package(tmp_path, monkeypatch, files)
    with pytest.raises(ImportError, match="exactly one model class"):
        discover_models(package)
