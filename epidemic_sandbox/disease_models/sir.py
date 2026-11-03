from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    IMMUNITY_PARAMETERS,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.parameter_spec import ParameterSpec


class SirModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> recovered (immune for a time) or dead."""

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before recovering.",
            7.0,
            0.5,
            60.0,
            0.5,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        INFECTIOUS: "#ea4335",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
