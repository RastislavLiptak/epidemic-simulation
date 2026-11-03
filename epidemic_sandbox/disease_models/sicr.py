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
)
from epidemic_sandbox.parameter_spec import ParameterSpec

CARRIER = "carrier"


class SicrModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> recovered (immune for a time) or dead, or carrier -> recovered.

    The acute illness leaves some survivors as carriers: they feel well and carry on as usual
    but go on spreading the infection, less than the ill, for a long time (typhoid fever,
    hepatitis B). Only the acute illness can kill; the chance of becoming a carrier is scaled by
    the "branching" age effect.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "fatality": ParameterSpec(
            "Fatality (acute illness)",
            "Chance that the acute illness ends in death, for a person whose age effect is 1 "
            "(see age effects). Carriers are well and do not die of the infection. Source: "
            "hospital-based typhoid case fatality 2.0 %, range 0-14.8 % "
            "(Crump et al. 2008 review).",
            0.02,
            0.0,
            1.0,
            0.001,
        ),
        "infectious_days": ParameterSpec(
            "Acute illness (days)",
            "Average number of days a person is acutely ill and fully infectious before recovering "
            "or becoming a carrier. Source: 4 weeks of infectiousness in typhoid (Pitzer et al. "
            "2014).",
            28.0,
            0.5,
            90.0,
            0.5,
        ),
        "carrier_share": ParameterSpec(
            "Carrier share (of survivors)",
            "Share of people surviving the acute illness who become carriers, for a person whose "
            "branching age effect is 1 (see the branching age effect). Source: 3-5 % of "
            "typhoid patients (Gonzalez-Escobedo et al. 2011); in hepatitis B about 90 % of "
            "infants but 5 % of adults.",
            0.03,
            0.0,
            1.0,
            0.005,
        ),
        "carrier_days": ParameterSpec(
            "Carrier period (days)",
            "Average number of days a carrier keeps spreading the infection before clearing it "
            f"and becoming immune; {LIFELONG_DAYS} days means for life. Source: life-long carriage "
            "in typhoid fever (Pitzer et al. 2014) and hepatitis B; weeks to months in cholera and "
            "other Salmonella.",
            float(LIFELONG_DAYS),
            1.0,
            float(LIFELONG_DAYS),
            1.0,
        ),
        "carrier_infectiousness": ParameterSpec(
            "Carrier infectiousness",
            "How much a carrier spreads compared with an acutely ill person (1 is the same). "
            "Source: 0.01-1 in typhoid models, 0.01 best fit, 0.01-0.1 meaning carriers cause "
            "25-75 % of the transmission (Pitzer et al. 2014). Assumption: 0.1, because the "
            "town has no ageing, so carriage of many years never plays out.",
            0.1,
            0.0,
            1.0,
            0.01,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        INFECTIOUS: "#ea4335",
        CARRIER: "#1a73e8",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS, CARRIER})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    STATE_INFECTIOUSNESS = {CARRIER: "carrier_infectiousness"}
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        INFECTIOUS: Phase(
            RECOVERED,
            "infectious_days",
            fatal=True,
            branch_state=CARRIER,
            branch_parameter="carrier_share",
        ),
        CARRIER: Phase(RECOVERED, "carrier_days"),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
