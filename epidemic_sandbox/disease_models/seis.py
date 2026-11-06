from epidemic_sandbox.config.parameter_spec import ParameterSpec
from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    ASYMPTOMATIC_PARAMETERS,
    COMMON_PARAMETERS,
    DEAD,
    DEATH_PARAMETERS,
    DYING,
    EXPOSED,
    INFECTIOUS,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)


class SeisModel(DiseaseModelAbstract):
    """Susceptible -> exposed (latent) -> infectious -> susceptible again (no immunity), or dead.

    A share of the infections (`asymptomatic_share`) never has symptoms: after the latent period
    such a person spreads the disease for `asymptomatic_days`, behaving as usual, and is
    susceptible again.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **ASYMPTOMATIC_PARAMETERS,
        "incubation_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people do "
            "not spread the disease yet.",
            3.5,
            0.5,
            30.0,
            0.5,
        ),
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before becoming susceptible again; "
            "nobody is immune afterwards.",
            4.0,
            0.5,
            60.0,
            0.5,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        INFECTIOUS: "#ea4335",
        ASYMPTOMATIC: "#ab47bc",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {
        EXPOSED: Phase(
            INFECTIOUS,
            "incubation_days",
            branch_state=ASYMPTOMATIC,
            branch_parameter="asymptomatic_share",
            branch_age_effect="asymptomatic",
        ),
        ASYMPTOMATIC: Phase(SUSCEPTIBLE, "asymptomatic_days"),
        INFECTIOUS: Phase(SUSCEPTIBLE, "infectious_days", fatal=True),
    }
