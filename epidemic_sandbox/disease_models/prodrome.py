from epidemic_sandbox.config.parameter_spec import ParameterSpec
from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    ASYMPTOMATIC_PARAMETERS,
    COMMON_PARAMETERS,
    DEAD,
    DEATH_PARAMETERS,
    DYING,
    EXPOSED,
    IMMUNITY_PARAMETERS,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.disease_models.sepir import PRESYMPTOMATIC, WANING

PRODROME = "prodrome"


class ProdromeModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> presymptomatic -> prodrome -> infectious -> waning -> recovered
    or dead.

    SEPIR with a prodrome: between the presymptomatic period and the illness by which the disease
    is recognised (the infectious state, such as the rash of measles) come unspecific symptoms,
    the prodrome (fever, cough, running nose), in which a person spreads the disease and is ill,
    but nobody knows which disease it is. The prodrome has its own chances of staying at home
    and of skipping outings (those of any febrile respiratory illness); the recognised illness
    has the common ones, and a share of those in it can be in hospital (`admission_share`), as a
    law may require. Both the prodrome and the recognised illness are symptomatic: the onset of
    symptoms is the start of the prodrome.

    A share of the infections (`asymptomatic_share`) never has symptoms: after the latent period
    such a person spreads the disease for `asymptomatic_days` and recovers, behaving as usual. A
    share of the people can be immune from the start (`initial_immune`, scaled by the
    "starting_share" age effect).
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        **ASYMPTOMATIC_PARAMETERS,
        "latent_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people do "
            "not spread the disease yet. Default 10 (Derived for measles: transmissible from 4 "
            "days before the rash, which comes 14 days after exposure on average, CDC Pink Book, "
            "chapter 13).",
            10.0,
            0.5,
            30.0,
            0.5,
        ),
        "presymptomatic_days": ParameterSpec(
            "Presymptomatic period (days)",
            "Average number of days a person spreads the disease before the first symptoms. They "
            "feel well and behave as usual. Default 1.5 (Derived for measles: the prodrome begins "
            "11 to 12 days after exposure and transmission 10 days after it, CDC Pink Book).",
            1.5,
            0.1,
            30.0,
            0.1,
        ),
        "prodrome_days": ParameterSpec(
            "Prodrome (days)",
            "Average number of days of unspecific symptoms before the disease is recognised. "
            "Default 2.5 (Derived for measles: the rash 14 days after exposure, the prodrome from "
            "11 to 12 days, CDC Pink Book; it lasts 2 to 4 days there).",
            2.5,
            0.1,
            30.0,
            0.1,
        ),
        "prodrome_infectiousness": ParameterSpec(
            "Prodrome infectiousness (relative)",
            "How much a person spreads per day in the prodrome compared with the recognised "
            "illness. Default 1 (Model: the CDC gives one transmissible period, 4 days before to "
            "4 days after the rash, with no profile).",
            1.0,
            0.0,
            2.0,
            0.05,
        ),
        "prodrome_stay_home": ParameterSpec(
            "Stay-at-home share in the prodrome",
            "Share of the people in the prodrome who stay at home, for a person whose stay-at-home "
            "age effect is 1. Default 0.31 (Source: adults of 16 to 64 years took time off work "
            "in 31 % of PCR-confirmed episodes of influenza A, Fragaszy et al. 2018, Table 2).",
            0.31,
            0.0,
            1.0,
            0.01,
        ),
        "prodrome_limit_outings": ParameterSpec(
            "Fewer outings share in the prodrome",
            "Share of the people in the prodrome who still go to work or school but skip "
            "errands, visits and leisure.",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
        "infectious_days": ParameterSpec(
            "Recognised illness at full infectiousness (days)",
            "Average number of days of the recognised illness at the full rate of spreading; the "
            "waning period follows. Default 2 (Derived for measles: transmissible until 4 days "
            "after the rash, CDC Pink Book, split in half).",
            2.0,
            0.1,
            60.0,
            0.01,
        ),
        "waning_days": ParameterSpec(
            "Waning period (days)",
            "Average number of days a person is still ill and spreads less after the period at "
            "full infectiousness, before recovering. Default 2 (see the recognised illness).",
            2.0,
            0.1,
            60.0,
            0.01,
        ),
        "waning_infectiousness": ParameterSpec(
            "Waning infectiousness (relative)",
            "How much a person spreads per day in the waning period compared with the period at "
            "full infectiousness.",
            0.5,
            0.0,
            1.0,
            0.01,
        ),
        "presymptomatic_infectiousness": ParameterSpec(
            "Presymptomatic infectiousness (relative)",
            "How much a presymptomatic person spreads per day compared with the recognised "
            "illness. Default 1 (Model, as for the prodrome).",
            1.0,
            0.0,
            2.0,
            0.05,
        ),
        "admission_share": ParameterSpec(
            "Admitted to hospital (share)",
            "Share of the people with the recognised illness who are in hospital until they "
            "recover: away from the town, they meet nobody in it. 0 means nursed at home.",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
        "initial_immune": ParameterSpec(
            "Immune at the start (share)",
            "Share of the people, apart from the first cases, who are immune when the experiment "
            "starts, for a person whose starting-share age effect is 1 (see age effects).",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        PRESYMPTOMATIC: "#f57c00",
        PRODROME: "#e8710a",
        INFECTIOUS: "#ea4335",
        WANING: "#f28b82",
        ASYMPTOMATIC: "#ab47bc",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, PRESYMPTOMATIC, PRODROME, INFECTIOUS, WANING})
    SYMPTOMATIC_STATES = frozenset({PRODROME, INFECTIOUS, WANING})
    BEHAVIOUR = {PRODROME: ("prodrome_stay_home", "prodrome_limit_outings")}
    HOSPITAL = {INFECTIOUS: "admission_share", WANING: "admission_share"}
    STATE_INFECTIOUSNESS = {
        PRESYMPTOMATIC: "presymptomatic_infectiousness",
        PRODROME: "prodrome_infectiousness",
        WANING: "waning_infectiousness",
    }
    INFECTION_ENTRY_STATE = EXPOSED
    # Like the seeds of SEPIR: every first case spreads for the whole infectious course.
    INITIAL_CASE_STATE = PRESYMPTOMATIC
    PROTECTION = {RECOVERED: "immunity_protection"}
    STARTING_SHARES = {RECOVERED: "initial_immune"}
    PHASES = {
        EXPOSED: Phase(
            PRESYMPTOMATIC,
            "latent_days",
            branch_state=ASYMPTOMATIC,
            branch_parameter="asymptomatic_share",
            branch_age_effect="asymptomatic",
        ),
        ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days"),
        PRESYMPTOMATIC: Phase(PRODROME, "presymptomatic_days"),
        PRODROME: Phase(INFECTIOUS, "prodrome_days"),
        INFECTIOUS: Phase(WANING, "infectious_days"),
        WANING: Phase(RECOVERED, "waning_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
