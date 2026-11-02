from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    INFECTIOUS,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.parameter_spec import ParameterSpec


class SisModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> susceptible again (no immunity), or dead."""

    PARAMETERS = {
        **COMMON_PARAMETERS,
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before becoming susceptible again.",
            5.0,
            0.5,
            60.0,
            0.5,
        ),
    }
    STATE_COLORS = {SUSCEPTIBLE: "#34a853", INFECTIOUS: "#ea4335", DEAD: "#202124"}
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {INFECTIOUS: Phase(SUSCEPTIBLE, "infectious_days", fatal=True)}
