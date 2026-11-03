from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    EXPOSED,
    IMMUNITY_PARAMETERS,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.parameter_spec import ParameterSpec

PRESYMPTOMATIC = "presymptomatic"


class SepirModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> presymptomatic -> infectious (symptomatic) -> recovered or dead.

    SEIR with the end of the incubation period split off: after the latent period (exposed, not
    infectious) comes a presymptomatic period in which a person already spreads the disease but
    feels well and goes about the day as usual, then the symptomatic period, in which a person
    may stay at home. The incubation period of the literature is the latent plus the
    presymptomatic period. With the relative infectiousness w of the presymptomatic state, the
    mean durations d_P (presymptomatic) and d_I (symptomatic), a person spreads for
    w d_P + d_I days (weighted), and the share of all transmission that happens before symptoms
    (theta of Fraser et al. 2004) is w d_P / (w d_P + d_I). Isolating the symptomatic can remove at
    most the share 1 - theta, so it alone controls the outbreak only when theta < 1 / R0.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "latent_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people do "
            "not spread the disease yet. Together with the presymptomatic period it makes the "
            "incubation period. Derived: the SARS-CoV-2 incubation period of 5.5 days (Ferretti "
            "et al. 2020, Science) minus the presymptomatic period.",
            3.0,
            0.5,
            30.0,
            0.5,
        ),
        "presymptomatic_days": ParameterSpec(
            "Presymptomatic period (days)",
            "Average number of days a person spreads the disease before showing symptoms. They "
            "feel well and behave as usual. Source: infectiousness of SARS-CoV-2 begins 2.3 days "
            "(0.8-3.0) before symptom onset (He et al. 2020, Nature Medicine).",
            2.5,
            0.5,
            30.0,
            0.5,
        ),
        "infectious_days": ParameterSpec(
            "Symptomatic period (days)",
            "Average number of days a person spreads the disease with symptoms (and may stay at "
            "home) before recovering. Derived: with full infectiousness it makes the share of "
            "transmission before symptoms 45 %, as estimated for SARS-CoV-2 (He et al. 2020: 44 %, "
            "Ganyani et al. 2020: 48 %, Ferretti et al. 2020: 47 %).",
            3.0,
            0.5,
            60.0,
            0.5,
        ),
        "presymptomatic_infectiousness": ParameterSpec(
            "Presymptomatic infectiousness (relative)",
            "How much a presymptomatic person spreads per day compared with a symptomatic person "
            "who goes about as usual: 1 is the same chance of infecting someone within contact "
            "distance, 0 means they do not spread at all, 2 twice as much. The share of "
            "transmission before symptoms is w d_P / (w d_P + d_I) with w this value, d_P the "
            "presymptomatic and d_I the symptomatic period. Assumption: 1, the viral load peaks "
            "shortly before symptoms (He et al. 2020), there is no direct measurement.",
            1.0,
            0.0,
            2.0,
            0.05,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        PRESYMPTOMATIC: "#f57c00",
        INFECTIOUS: "#ea4335",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({PRESYMPTOMATIC, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    STATE_INFECTIOUSNESS = {PRESYMPTOMATIC: "presymptomatic_infectiousness"}
    INFECTION_ENTRY_STATE = EXPOSED
    # Like the seeds of SEIR, who start where the infectious period begins: every first case then
    # spreads for the whole infectious course, presymptomatic part included.
    INITIAL_CASE_STATE = PRESYMPTOMATIC
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(PRESYMPTOMATIC, "latent_days"),
        PRESYMPTOMATIC: Phase(INFECTIOUS, "presymptomatic_days"),
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
