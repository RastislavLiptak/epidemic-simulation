from epidemic_sandbox.disease_models.disease_model_abstract import DiseaseModelAbstract
from epidemic_sandbox.disease_models.seir import SeirModel
from epidemic_sandbox.disease_models.sir import SirModel
from epidemic_sandbox.disease_models.sis import SisModel

DISEASE_MODELS: dict[str, type[DiseaseModelAbstract]] = {
    "sir": SirModel,
    "seir": SeirModel,
    "sis": SisModel,
}
