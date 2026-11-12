from epidemic_sandbox.config.parameter_spec import ParameterSpec
from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    ASYMPTOMATIC_PARAMETERS,
    COMMON_PARAMETERS,
    DEAD,
    DEATH_PARAMETERS,
    DYING,
    IMMUNITY_PARAMETERS,
    INFECTIOUS,
    LIFELONG_DAYS,
    RECOVERED,
    SUSCEPTIBLE,
    Branch,
    DiseaseModelAbstract,
    Phase,
)


class SirsModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> recovered (protected for a while) -> susceptible again, or
    dead.

    For a pool of many types of one virus, such as the rhinoviruses: an infection protects
    against the type that caused it, but there are so many types that a person is soon
    susceptible again to the pool. The protected state stands for the time in which the people
    around still pass on the same type (a household chain) and for the short protection against
    other types; after it, the person is susceptible again. The course has no latent period, so it
    splits at the infection: a share of the infections (`asymptomatic_share`) never has symptoms,
    spreads the disease for `asymptomatic_days`, behaving as usual, and is protected afterwards
    as well.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        **ASYMPTOMATIC_PARAMETERS,
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before being protected for a while.",
            5.0,
            0.5,
            60.0,
            0.1,
        ),
        "immunity_days": ParameterSpec(
            "Protection after an infection (days)",
            "Average time a person who has recovered is protected before becoming susceptible "
            f"again; {LIFELONG_DAYS} days means for life. Default 14 (Estimate for the "
            "rhinoviruses: the protection against other types lasts at most of the order of weeks, "
            "Kamau et al. 2019, J Infect Dis).",
            14.0,
            1.0,
            float(LIFELONG_DAYS),
            1.0,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        INFECTIOUS: "#ea4335",
        ASYMPTOMATIC: "#ab47bc",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INFECTION_BRANCH = Branch(ASYMPTOMATIC, "asymptomatic_share", "asymptomatic")
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days"),
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
