from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    EXPOSED,
    INFECTIOUS,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.parameter_spec import ParameterSpec


class SeisModel(DiseaseModelAbstract):
    """Susceptible -> exposed (latent) -> infectious -> susceptible again (no immunity), or dead."""

    PARAMETERS = {
        **COMMON_PARAMETERS,
        "incubation_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. "
            "Exposed people do not spread the disease yet. "
            "Default 3 (Source: pooled median incubation of human coronavirus 3.2 days, "
            "rhinovirus 1.9 and RSV 4.4, Lessler et al. 2009, Lancet Infect Dis 9:291).",
            3.0,
            0.5,
            30.0,
            0.5,
        ),
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before becoming susceptible "
            "again; nobody is immune afterwards. "
            "Default 6 (Assumption: a typical span of a few days to a week for respiratory "
            "infections that leave no lasting immunity).",
            6.0,
            0.5,
            60.0,
            0.5,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        INFECTIOUS: "#ea4335",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {
        EXPOSED: Phase(INFECTIOUS, "incubation_days"),
        INFECTIOUS: Phase(SUSCEPTIBLE, "infectious_days", fatal=True),
    }
