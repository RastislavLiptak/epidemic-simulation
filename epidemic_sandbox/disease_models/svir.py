from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    IMMUNITY_PARAMETERS,
    INFECTIOUS,
    LIFELONG_DAYS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
    Rate,
)
from epidemic_sandbox.parameter_spec import ParameterSpec

VACCINATED = "vaccinated"


class SvirModel(DiseaseModelAbstract):
    """Susceptible -> vaccinated -> (infectious) -> recovered, the SIR model with vaccination.

    Susceptible people are vaccinated at a constant hazard from a start day, and a share of the
    town is vaccinated from the beginning. The vaccine is leaky: a vaccinated person can still
    be infected, at a hazard lowered by the vaccine efficacy. The protection of the vaccinated
    is lost after a while, and they are susceptible again. An infected vaccinated person goes
    through the same infection as anybody else.

    Limits of the model. Vaccination is uniform: it does not favour any age, and only the
    susceptible are vaccinated (the infectious and the recovered are not). The vaccine acts on
    the chance of infection only: real vaccines also make breakthrough cases milder and less
    infectious, but the framework has no mechanism for that, so those cases are as infectious
    and as deadly as unvaccinated ones. A vaccine that protects some people completely and
    others not at all (all-or-nothing) is not modelled. Unlike the textbook SVIR there are no
    births and no deaths other than of the infection (the town does not change), and the waning
    of the protection is a gamma-distributed stay like every other, so it is not a constant
    hazard unless `duration_shape` is 1. The preview reproduction number is that of a town
    without vaccination.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person stays infectious before recovering. "
            "Source: 4.8 days of virus shedding in influenza (Carrat et al. 2008).",
            4.8,
            0.5,
            60.0,
            0.1,
        ),
        "vaccination_rate": ParameterSpec(
            "Vaccination rate (per day)",
            "Chance per day, for every susceptible person, of being vaccinated (a constant "
            "hazard): after t days of the campaign a share 1 - exp(-rate * t) of those who "
            "stayed susceptible is vaccinated, and the share of the whole town vaccinated each "
            "day is the rate times the share of susceptible people. 0 means no campaign. "
            "Source: 0.0025 is the pace of the US 2009 H1N1 campaign, which covered 24 % of the "
            "population in the 116 days after 5 October 2009 (CDC 2010).",
            0.0025,
            0.0,
            1.0,
            0.0005,
        ),
        "vaccination_start_day": ParameterSpec(
            "Campaign start (day)",
            "Day of the experiment on which the vaccination campaign starts. "
            "Assumption: a vaccine that arrives a month after the first cases, as the 2009 "
            "H1N1 vaccine arrived after the start of the autumn wave (CDC 2010).",
            30.0,
            0.0,
            365.0,
            1.0,
        ),
        "initial_coverage": ParameterSpec(
            "Vaccinated at the start (share)",
            "Share of the people, apart from the first cases, who are already vaccinated when "
            "the experiment starts, with the full protection time ahead of them. "
            "Assumption: nobody is vaccinated beforehand.",
            0.0,
            0.0,
            1.0,
            0.01,
        ),
        "vaccine_efficacy": ParameterSpec(
            "Vaccine efficacy",
            "How much the vaccine lowers the chance of infection of a vaccinated person: 1 is "
            "complete protection, 0.7 means they catch the disease at 30 % of the hazard of an "
            "unvaccinated person. Source: 0.61 for influenza A(H1N1)pdm09 (Belongia et al. 2016, "
            "meta-analysis, 95 % CI 0.57-0.65; 0.73 for the monovalent pandemic vaccine).",
            0.61,
            0.0,
            1.0,
            0.01,
        ),
        "vaccine_days": ParameterSpec(
            "Vaccine protection (days)",
            "Average number of days the protection of a vaccinated person lasts before it is "
            f"gone and they are susceptible again; {LIFELONG_DAYS} days means for life. "
            "Derived: the efficacy of influenza vaccines falls by 6-11 % points a month "
            "(Ferdinands et al. 2017), so a 61 % efficacy is gone after about 7 months; a "
            "protection that ends abruptly after half of that gives the same total protection.",
            110.0,
            1.0,
            float(LIFELONG_DAYS),
            1.0,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        VACCINATED: "#4285f4",
        INFECTIOUS: "#ea4335",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {VACCINATED: "vaccine_efficacy", RECOVERED: "immunity_protection"}
    RATES = {SUSCEPTIBLE: Rate(VACCINATED, "vaccination_rate", "vaccination_start_day")}
    STARTING_SHARES = {VACCINATED: "initial_coverage"}
    PHASES = {
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        VACCINATED: Phase(SUSCEPTIBLE, "vaccine_days"),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
