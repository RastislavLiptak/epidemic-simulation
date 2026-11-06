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
from epidemic_sandbox.parameter_spec import ParameterSpec

PRESYMPTOMATIC = "presymptomatic"
WANING = "waning"


class SepirModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> presymptomatic -> infectious -> waning -> recovered or dead.

    SEIR with the end of the incubation period split off: after the latent period (exposed, not
    infectious) comes a presymptomatic period in which a person already spreads the disease but
    feels well and goes about the day as usual, then the symptomatic period, in which a person
    may stay at home. The symptomatic period has two parts: the first at full infectiousness and
    a waning one, in which the person is still ill and spreads less (infectiousness peaks around
    the onset and falls in the following days). The incubation period of the literature is the
    latent plus the presymptomatic period. With the relative infectiousness w of the
    presymptomatic state and v of the waning one, and the mean durations d_P, d_I and d_W, a
    person spreads for w d_P + d_I + v d_W days (weighted), and the share of all transmission
    before symptoms (theta of Fraser et al. 2004) is w d_P / (w d_P + d_I + v d_W) without
    behaviour. Isolating the symptomatic can remove at most the share 1 - theta, so it alone
    controls the outbreak only when theta < 1 / R0.

    A share of the infections (`asymptomatic_share`) never has symptoms: after the latent period
    such a person spreads the disease for `asymptomatic_days` and recovers, behaving as usual.

    A share of the people can be immune from the start (`initial_immune`, scaled by the
    "starting_share" age effect): they begin as recovered, as after a past infection or vaccination.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        **ASYMPTOMATIC_PARAMETERS,
        "latent_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people do "
            "not spread the disease yet. Together with the presymptomatic period it makes the "
            "incubation period. Default 4.1 (Derived: the SARS-CoV-2 incubation period of 5.5 "
            "days, Ferretti et al. 2020, Science, minus the presymptomatic period).",
            4.1,
            0.5,
            30.0,
            0.5,
        ),
        "presymptomatic_days": ParameterSpec(
            "Presymptomatic period (days)",
            "Average number of days a person spreads the disease before showing symptoms. They "
            "feel well and behave as usual. Default 1.4 (Derived: the mean incubation period of "
            "6.9 days minus the mean latent period of 5.5 days, Xin et al. 2022, Clin Infect Dis).",
            1.4,
            0.5,
            30.0,
            0.5,
        ),
        "infectious_days": ParameterSpec(
            "Symptomatic period at full infectiousness (days)",
            "Average number of days after the onset of symptoms in which a person spreads the "
            "disease at the full rate (and may stay at home); the waning period follows. Default "
            "1.75 (Fitted for COVID-19: with the waning period it gives the share of transmission "
            "before the symptoms, Ferretti et al. 2020, and a generation interval within that of "
            "Ganyani et al. 2020).",
            1.75,
            0.1,
            60.0,
            0.01,
        ),
        "waning_days": ParameterSpec(
            "Waning period (days)",
            "Average number of days a person is still ill and spreads less after the period at "
            "full infectiousness, before recovering. Default 6.0 (Derived: no infectious "
            "SARS-CoV-2 was isolated after day 8 of symptoms in mild cases, Wolfel et al. 2020, "
            "Nature, minus the 1.75 days at full infectiousness).",
            6.25,
            0.1,
            60.0,
            0.01,
        ),
        "waning_infectiousness": ParameterSpec(
            "Waning infectiousness (relative)",
            "How much a person spreads per day in the waning period compared with the period at "
            "full infectiousness: 0 means not at all, 1 the same. Default 0.08 (Fitted for "
            "COVID-19, with the period at full infectiousness).",
            0.08,
            0.0,
            1.0,
            0.01,
        ),
        "initial_immune": ParameterSpec(
            "Immune at the start (share)",
            "Share of the people, apart from the first cases, who are immune when the experiment "
            "starts, for a person whose starting-share age effect is 1 (see age effects). They "
            "begin as recovered; 0 means a town where nobody has been exposed or vaccinated.",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
        "presymptomatic_infectiousness": ParameterSpec(
            "Presymptomatic infectiousness (relative)",
            "How much a presymptomatic person spreads per day compared with a symptomatic person "
            "who goes about as usual: 1 is the same chance of infecting someone within contact "
            "distance, 0 means they do not spread at all, 2 twice as much. Default 1 (Source: the "
            "secondary attack rate of presymptomatic cases relative to symptomatic ones is 1.00, "
            "Buitrago-Garcia et al. 2022, PLoS Med).",
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
        WANING: "#f28b82",
        ASYMPTOMATIC: "#ab47bc",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, PRESYMPTOMATIC, INFECTIOUS, WANING})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS, WANING})
    STATE_INFECTIOUSNESS = {
        PRESYMPTOMATIC: "presymptomatic_infectiousness",
        WANING: "waning_infectiousness",
    }
    INFECTION_ENTRY_STATE = EXPOSED
    # Like the seeds of SEIR, who start where the infectious period begins: every first case then
    # spreads for the whole infectious course, presymptomatic part included.
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
        PRESYMPTOMATIC: Phase(INFECTIOUS, "presymptomatic_days"),
        INFECTIOUS: Phase(WANING, "infectious_days"),
        WANING: Phase(RECOVERED, "waning_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
