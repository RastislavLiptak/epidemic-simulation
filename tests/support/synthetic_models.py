"""Small disease models that exist only in tests, to exercise every mechanism of the base class.

None of them is registered in DISEASE_MODELS (only modules of the package are), so they cannot
leak into the application.
"""

from epidemic_sandbox.config.parameter_spec import ParameterSpec
from epidemic_sandbox.disease_models.disease_model_abstract import (
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
    Rate,
)

MILD = "mild"
HOSPITAL = "hospital"
VACCINATED = "vaccinated"


def duration(label: str, default: float) -> ParameterSpec:
    return ParameterSpec(label, "Test value.", default, 0.5, 60.0, 0.5)


def share(label: str, default: float) -> ParameterSpec:
    return ParameterSpec(label, "Test value.", default, 0.0, 1.0, 0.05)


class ForkModel(DiseaseModelAbstract):
    """Every mechanism of the base class but vaccination.

    Susceptible -> exposed -> mild (a share, weaker spreaders without symptoms, who carry on as
    usual) or infectious (symptomatic) -> hospital (a share, confined, not infectious) ->
    recovered or dead.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "incubation_days": duration("Latent period", 2.0),
        "mild_days": duration("Mild period", 4.0),
        "infectious_days": duration("Symptomatic period", 6.0),
        "hospital_days": duration("Hospital stay", 10.0),
        "mild_share": share("Mild share", 0.4),
        "hospital_share": share("Hospitalised share", 0.2),
        "mild_infectiousness": share("Mild infectiousness", 0.5),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        MILD: "#ff9800",
        INFECTIOUS: "#ea4335",
        HOSPITAL: "#7b1fa2",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({MILD, INFECTIOUS})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS})
    CONFINED_STATES = frozenset({HOSPITAL})
    STATE_INFECTIOUSNESS = {MILD: "mild_infectiousness"}
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(
            INFECTIOUS,
            "incubation_days",
            branch_state=MILD,
            branch_parameter="mild_share",
        ),
        MILD: Phase(RECOVERED, "mild_days"),
        INFECTIOUS: Phase(
            RECOVERED, "infectious_days", branch_state=HOSPITAL, branch_parameter="hospital_share"
        ),
        HOSPITAL: Phase(RECOVERED, "hospital_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }


class PlainModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> recovered (immune for a time) or dead, the plainest course.

    The baseline of the tests that check the base class and the town, where the course of the
    disease should not matter.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "infectious_days": duration("Infectious period", 7.0),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        INFECTIOUS: "#ea4335",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }


class LatentModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> infectious -> recovered (immune for a time) or dead."""

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "incubation_days": duration("Latent period", 3.0),
        "infectious_days": duration("Infectious period", 6.0),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        INFECTIOUS: "#ea4335",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(INFECTIOUS, "incubation_days"),
        INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }


class RepeatModel(DiseaseModelAbstract):
    """Susceptible -> infectious -> susceptible again (no immunity), or dead."""

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        "infectious_days": duration("Infectious period", 5.0),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        INFECTIOUS: "#ea4335",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {INFECTIOUS: Phase(SUSCEPTIBLE, "infectious_days", fatal=True)}


class VaccinationModel(DiseaseModelAbstract):
    """Susceptible -> vaccinated (at a rate from a start day, or from the start) -> susceptible.

    Vaccinated people can still be infected, with the vaccine's protection; the infected are
    infectious and then recovered with partial protection of their own.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
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
        DYING: "#5d4037",
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
