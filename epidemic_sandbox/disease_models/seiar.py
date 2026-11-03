from dataclasses import replace

from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    EXPOSED,
    FATALITY,
    IMMUNITY_PARAMETERS,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.parameter_spec import ParameterSpec

ASYMPTOMATIC = "asymptomatic"


class SeiarModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> asymptomatic or infectious (symptomatic) -> recovered or dead.

    After the latent period a share of the infected never fall ill: they are less infectious and
    go about their day as usual. The others spread fully from the start of the infectious state,
    react to being ill (stay at home, skip outings) and are the only ones who can die. There is no
    separate pre-symptomatic stage: the people who will fall ill are treated as ill from the
    moment they become infectious.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        "fatality": replace(
            FATALITY,
            label="Fatality (per symptomatic case)",
            description="Chance that a person who falls ill dies of it, for a person of average "
            "age (see age effects). People who never fall ill do not die of the disease, so "
            "the chance per infection is lower by the share of the symptomatic.",
        ),
        **IMMUNITY_PARAMETERS,
        "incubation_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious, for those who "
            "fall ill and for those who do not. Exposed people do not spread the disease yet. "
            "Source: 3.0 days, the latent period in the COVID-19 model of Davies et al. (2020, "
            "Nature Medicine), which splits the incubation of about 5 days into 3 days without "
            "infectiousness and 2 days of infectiousness before symptoms.",
            3.0,
            0.5,
            30.0,
            0.5,
        ),
        "asymptomatic_share": ParameterSpec(
            "Asymptomatic share",
            "Share of the infected who never fall ill (they carry on as usual), for a person whose "
            "branching age effect is 1; the age effects multiply it, up to certainty. Source: "
            "0.70, the 69.0 % of infected close contacts without respiratory symptoms or fever "
            "in Poletti et al. (2021, JAMA Network Open). Counting only people without any "
            "symptom gives 20 % for COVID-19 (Buitrago-Garcia et al. 2020, PLoS Medicine) and a "
            "third for influenza (Carrat et al. 2008, Am J Epidemiol).",
            0.7,
            0.0,
            1.0,
            0.01,
        ),
        "asymptomatic_infectiousness": ParameterSpec(
            "Asymptomatic infectiousness",
            "How infectious a person who never falls ill is compared with a person who falls "
            "ill: 1 means the same chance of infecting a person within contact distance, 0 "
            "that they do not spread at all. It is the infectiousness per contact, so people "
            "who stay at home when ill spread less in the town than this ratio says. "
            "Source: 0.5, as assumed by Davies et al. (2020, Nature Medicine); estimated at 0.55 "
            "(0.46 to 0.62) for COVID-19 by Li et al. (2020, Science).",
            0.5,
            0.0,
            1.0,
            0.05,
        ),
        "infectious_days": ParameterSpec(
            "Symptomatic period (days)",
            "Average number of days a person who falls ill stays infectious before recovering, "
            "counted from the start of infectiousness, so it includes the days just before "
            "symptoms. Source: 5.0 days, 2.1 days before symptoms and 2.9 days of illness in "
            "the COVID-19 model of Davies et al. (2020, Nature Medicine).",
            5.0,
            0.5,
            60.0,
            0.5,
        ),
        "asymptomatic_days": ParameterSpec(
            "Asymptomatic period (days)",
            "Average number of days a person who never falls ill stays infectious before "
            "recovering. Assumption: 5.0 days, the value Davies et al. (2020, Nature Medicine) "
            "assumed for COVID-19 without data, the same as for people who fall ill.",
            5.0,
            0.5,
            60.0,
            0.5,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        ASYMPTOMATIC: "#9c27b0",
        INFECTIOUS: "#ea4335",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    STATE_INFECTIOUSNESS = {ASYMPTOMATIC: "asymptomatic_infectiousness"}
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(
            INFECTIOUS,
            "incubation_days",
            branch_state=ASYMPTOMATIC,
            branch_parameter="asymptomatic_share",
        ),
        ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days"),
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
