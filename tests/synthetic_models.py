"""Small disease models that exist only in tests, to exercise every mechanism of the base class.

None of them is registered in DISEASE_MODELS (only modules of the package are), so they cannot
leak into the application.
"""

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
    Rate,
)
from epidemic_sandbox.parameter_spec import ParameterSpec

ASYMPTOMATIC = "asymptomatic"
HOSPITAL = "hospital"
VACCINATED = "vaccinated"


def duration(label: str, default: float) -> ParameterSpec:
    return ParameterSpec(label, "Test value.", default, 0.5, 60.0, 0.5)


def share(label: str, default: float) -> ParameterSpec:
    return ParameterSpec(label, "Test value.", default, 0.0, 1.0, 0.05)


class ForkModel(DiseaseModelAbstract):
    """Every mechanism of the base class but vaccination.

    Susceptible -> exposed -> asymptomatic (a share, weaker spreaders, carry on as usual) or
    infectious (symptomatic) -> hospital (a share, confined, not infectious) -> recovered or dead.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "incubation_days": duration("Latent period", 2.0),
        "asymptomatic_days": duration("Asymptomatic period", 4.0),
        "infectious_days": duration("Symptomatic period", 6.0),
        "hospital_days": duration("Hospital stay", 10.0),
        "asymptomatic_share": share("Asymptomatic share", 0.4),
        "hospital_share": share("Hospitalised share", 0.2),
        "asymptomatic_infectiousness": share("Asymptomatic infectiousness", 0.5),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        ASYMPTOMATIC: "#ff9800",
        INFECTIOUS: "#ea4335",
        HOSPITAL: "#7b1fa2",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    CONFINED_STATES = frozenset({HOSPITAL})
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
        INFECTIOUS: Phase(
            RECOVERED, "infectious_days", branch_state=HOSPITAL, branch_parameter="hospital_share"
        ),
        HOSPITAL: Phase(RECOVERED, "hospital_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }


class VaccinationModel(DiseaseModelAbstract):
    """Susceptible -> vaccinated (at a rate from a start day, or from the start) -> susceptible.

    Vaccinated people can still be infected, with the vaccine's protection; the infected are
    infectious and then recovered with partial protection of their own.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "infectious_days": duration("Infectious period", 5.0),
        "vaccination_rate": ParameterSpec("Vaccination rate", "Test value.", 0.1, 0.0, 5.0, 0.01),
        "vaccination_start_day": ParameterSpec("Start day", "Test value.", 0.0, 0.0, 365.0, 1.0),
        "initial_coverage": share("Initial coverage", 0.0),
        "vaccine_protection": share("Vaccine protection", 0.8),
        "vaccine_days": duration("Vaccine protection time", 30.0),
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
    PROTECTION = {VACCINATED: "vaccine_protection", RECOVERED: "immunity_protection"}
    RATES = {SUSCEPTIBLE: Rate(VACCINATED, "vaccination_rate", "vaccination_start_day")}
    STARTING_SHARES = {VACCINATED: "initial_coverage"}
    PHASES = {
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        VACCINATED: Phase(SUSCEPTIBLE, "vaccine_days"),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }


class ChronicModel(DiseaseModelAbstract):
    """Susceptible -> infectious for life; nobody recovers and nobody dies of it."""

    PARAMETERS = {key: spec for key, spec in COMMON_PARAMETERS.items() if key != "fatality"}
    STATE_COLORS = {SUSCEPTIBLE: "#34a853", INFECTIOUS: "#ea4335", DEAD: "#202124"}
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {}
