import importlib
import pkgutil
from types import ModuleType

from epidemic_sandbox.disease_models.disease_model_abstract import DiseaseModelAbstract


def discover_models(package: ModuleType) -> dict[str, type[DiseaseModelAbstract]]:
    """Every disease model in the package's modules, by module name.

    A module holds exactly one model class, `<Name>Model` for the module `<name>`, so that adding
    a model means adding a file and nothing else.
    """
    models = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name == "disease_model_abstract":
            continue
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        classes = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, DiseaseModelAbstract)
            and value.__module__ == module.__name__
        ]
        if len(classes) != 1 or classes[0].__name__ != f"{module_info.name.capitalize()}Model":
            raise ImportError(
                f"{module.__name__} must define exactly one model class, "
                f"{module_info.name.capitalize()}Model"
            )
        models[module_info.name] = classes[0]
    return dict(sorted(models.items()))


DISEASE_MODELS: dict[str, type[DiseaseModelAbstract]] = discover_models(
    importlib.import_module(__name__)
)
