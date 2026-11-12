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
    LIFELONG_DAYS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
    Rate,
)
from epidemic_sandbox.disease_models.sepir import PRESYMPTOMATIC, WANING

VACCINATED = "vaccinated"


class SvepirModel(DiseaseModelAbstract):
    """Susceptible -> vaccinated; -> exposed -> presymptomatic -> infectious -> waning ->
    recovered or dead.

    The SEPIR course (a latent period in which a person does not spread yet, a presymptomatic
    period in which they spread but feel well and go about their day, a symptomatic period in
    which they may stay at home, at full and then at waning infectiousness) in a town where
    people are vaccinated. Susceptible people are
    vaccinated at a constant hazard from a start day, and a share of the town is vaccinated from
    the beginning (scaled by the "starting_share" age effect). The vaccine is leaky: a vaccinated
    person can still be infected, at a hazard lowered by the vaccine efficacy, and goes through
    the same course as anybody else. The protection of the vaccinated and of the recovered is
    lost after a while, and they are susceptible again. A share of the infections
    (`asymptomatic_share`) never has symptoms: after the latent period such a person spreads the
    disease for `asymptomatic_days` and recovers, behaving as usual.

    Limits of the model. The campaign is uniform: it does not favour any age (only the starting
    share can), and only the susceptible are vaccinated. The vaccine acts on the chance of
    infection only: breakthrough cases are as infectious and as deadly as others. A vaccine that
    protects some people completely and others not at all (all-or-nothing) is not modelled. The
    waning of the protection is a gamma-distributed stay like every other.
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
            "incubation period. Default 1.0 (Source: influenza virus shedding was detected 1 day "
            "after inoculation on average, Carrat et al. 2008, Am J Epidemiol).",
            1.0,
            0.1,
            30.0,
            0.1,
        ),
        "presymptomatic_days": ParameterSpec(
            "Presymptomatic period (days)",
            "Average number of days a person spreads the disease before showing symptoms. They "
            "feel well and behave as usual. Default 0.5 (Derived: the mean incubation period of "
            "influenza A, 1.52 days from Lessler et al. 2009, Lancet Infect Dis, minus the latent "
            "period).",
            0.5,
            0.1,
            30.0,
            0.1,
        ),
        "infectious_days": ParameterSpec(
            "Symptomatic period at full infectiousness (days)",
            "Average number of days after the onset of symptoms in which a person spreads the "
            "disease at the full rate (and may stay at home); the waning period follows. Default "
            "2.5 (Source: most influenza virus shedding occurred during the first 2-3 days after "
            "the onset, Lau et al. 2010, J Infect Dis).",
            2.5,
            0.1,
            60.0,
            0.01,
        ),
        "waning_days": ParameterSpec(
            "Waning period (days)",
            "Average number of days a person is still ill and spreads less after the period at "
            "full infectiousness, before recovering. Default 1.8 (Derived: influenza virus is shed "
            "for 4.8 days on average, Carrat et al. 2008, of which 0.5 before the symptoms and 2.5 "
            "at full infectiousness).",
            1.8,
            0.1,
            60.0,
            0.01,
        ),
        "waning_stay_home": ParameterSpec(
            "Stay-at-home share in the waning period",
            "Share of the people in the waning period who still stay at home, for a person whose "
            "stay-at-home age effect is 1: those who are off sick for fewer days than they are ill "
            "are back at work or school by then.",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
        "waning_limit_outings": ParameterSpec(
            "Fewer outings share in the waning period",
            "Share of the people in the waning period who still go to work or school but skip "
            "errands, visits and leisure.",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
        "waning_infectiousness": ParameterSpec(
            "Waning infectiousness (relative)",
            "How much a person spreads per day in the waning period compared with the period at "
            "full infectiousness: 0 means not at all, 1 the same. Default 0.5 (Estimate: "
            "infectivity of influenza grows with the viral load to the power 0.136, Tsang et al. "
            "2015, J Infect Dis; a load lower by two orders of magnitude leaves about half).",
            0.5,
            0.0,
            1.0,
            0.01,
        ),
        "presymptomatic_infectiousness": ParameterSpec(
            "Presymptomatic infectiousness (relative)",
            "How much a presymptomatic person spreads per day compared with a symptomatic person "
            "who goes about as usual: 1 is the same chance of infecting someone within contact "
            "distance, 0 means they do not spread at all. Default 0.4 (Fitted for influenza to "
            "the share of transmission before the symptoms).",
            0.4,
            0.0,
            2.0,
            0.05,
        ),
        "vaccination_rate": ParameterSpec(
            "Vaccination rate (per day)",
            "Chance per day, for every susceptible person, of being vaccinated (a constant "
            "hazard): after t days of the campaign a share 1 - exp(-rate * t) of those who "
            "stayed susceptible is vaccinated. 0 means no campaign.",
            0.0,
            0.0,
            1.0,
            0.0005,
        ),
        "vaccination_start_day": ParameterSpec(
            "Campaign start (day)",
            "Day of the experiment on which the vaccination campaign starts.",
            0.0,
            0.0,
            365.0,
            1.0,
        ),
        "initial_coverage": ParameterSpec(
            "Vaccinated at the start (share)",
            "Share of the people, apart from the first cases, who are already vaccinated when "
            "the experiment starts, with the full protection time ahead of them, for a person "
            "whose starting-share age effect is 1 (see age effects).",
            0.0,
            0.0,
            1.0,
            0.001,
        ),
        "vaccine_efficacy": ParameterSpec(
            "Vaccine efficacy",
            "How much the vaccine lowers the chance of infection of a vaccinated person: 1 is "
            "complete protection, 0.7 means they catch the disease at 30 % of the hazard of an "
            "unvaccinated person. Default 0.61 (Source: against influenza A(H1N1)pdm09, Belongia "
            "et al. 2016, Lancet Infect Dis).",
            0.61,
            0.0,
            1.0,
            0.01,
        ),
        "vaccine_days": ParameterSpec(
            "Vaccine protection (days)",
            "Average number of days the protection of a vaccinated person lasts before it is "
            f"gone and they are susceptible again; {LIFELONG_DAYS} days means for life. Default "
            "110 (Derived: the effectiveness against influenza A(H1N1)pdm09 falls by 6-11 "
            "percentage points a month, Ferdinands et al. 2017, Clin Infect Dis; the same total "
            "protection held in full).",
            110.0,
            1.0,
            float(LIFELONG_DAYS),
            1.0,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        VACCINATED: "#4285f4",
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
    BEHAVIOUR = {WANING: ("waning_stay_home", "waning_limit_outings")}
    STATE_INFECTIOUSNESS = {
        PRESYMPTOMATIC: "presymptomatic_infectiousness",
        WANING: "waning_infectiousness",
    }
    INFECTION_ENTRY_STATE = EXPOSED
    # Like the first cases of SEPIR: every first case spreads for the whole infectious course.
    INITIAL_CASE_STATE = PRESYMPTOMATIC
    PROTECTION = {VACCINATED: "vaccine_efficacy", RECOVERED: "immunity_protection"}
    RATES = {SUSCEPTIBLE: Rate(VACCINATED, "vaccination_rate", "vaccination_start_day")}
    STARTING_SHARES = {VACCINATED: "initial_coverage"}
    PHASES = {
        VACCINATED: Phase(SUSCEPTIBLE, "vaccine_days"),
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
