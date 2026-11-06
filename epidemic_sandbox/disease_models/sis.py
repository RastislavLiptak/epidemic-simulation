from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    ASYMPTOMATIC_PARAMETERS,
    COMMON_PARAMETERS,
    DEAD,
    DEATH_PARAMETERS,
    DYING,
    INFECTIOUS,
    SUSCEPTIBLE,
    Branch,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.parameter_spec import ParameterSpec


class SisModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> susceptible again (no immunity), or dead.

    A share of the infections (`asymptomatic_share`) never has symptoms: such a person spreads
    the disease for `asymptomatic_days`, behaving as usual, and is susceptible again. The course
    has no latent period, so it splits at the infection.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **ASYMPTOMATIC_PARAMETERS,
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before becoming susceptible again.",
            5.0,
            0.5,
            60.0,
            0.5,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        INFECTIOUS: "#ea4335",
        ASYMPTOMATIC: "#ab47bc",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INFECTION_BRANCH = Branch(ASYMPTOMATIC, "asymptomatic_share", "asymptomatic")
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {
        ASYMPTOMATIC: Phase(SUSCEPTIBLE, "asymptomatic_days"),
        INFECTIOUS: Phase(SUSCEPTIBLE, "infectious_days", fatal=True),
    }
