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

HOSPITALISED = "hospitalised"


class SeihrModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> infectious -> hospitalised or recovered; hospital ends in death.

    Only the infectious spread the disease. Of them a share (the "branching" age effect carries the
    age profile) is admitted to hospital after the infectious phase, where they are isolated: the
    town keeps them at home and they do not infect. People die only at the end of a hospital stay
    and only the hospitalised die, so `fatality` is the chance to die in hospital; hospital
    capacity and infections in hospital are not modelled.
    """

    PARAMETERS = {
        **{key: spec for key, spec in COMMON_PARAMETERS.items() if key != "fatality"},
        **IMMUNITY_PARAMETERS,
        "fatality": ParameterSpec(
            "Fatality (in hospital)",
            "Chance that a hospitalised person dies at the end of the stay, for a person whose "
            "age effect is 1 (see age effects); nobody dies outside hospital. Derived: infection "
            "fatality ratio divided by the hospitalised share in Ferguson et al. 2020, Table 1, "
            "averaged over the hospitalised of the Czech age structure.",
            0.15,
            0.0,
            1.0,
            0.01,
        ),
        "incubation_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people do "
            "not spread the disease yet. Source: incubation period of 5.1 days (Lauer et al. "
            "2020) minus the half day before symptoms in which people already infect "
            "(Ferguson et al. 2020, Report 9).",
            4.6,
            0.5,
            30.0,
            0.1,
        ),
        "infectious_days": ParameterSpec(
            "Infectious period (days)",
            "Average number of days a person spreads the disease at large, until admission to "
            "hospital or recovery at home (the same for both). Source: the half day before "
            "symptoms plus the mean delay of 5 days from symptoms to hospital admission "
            "(Ferguson et al. 2020, Report 9; 5.6 days in Faes et al. 2020).",
            5.5,
            0.5,
            60.0,
            0.1,
        ),
        "hospitalised": ParameterSpec(
            "Hospitalised share",
            "Share of the infected admitted to hospital after the infectious period, for a "
            "person whose branching age effect is 1; the others recover at home. Hospitalised "
            "people are isolated and do not infect. Derived: proportion hospitalised by age in "
            "Ferguson et al. 2020, Table 1, averaged over the Czech age structure.",
            0.08,
            0.0,
            1.0,
            0.01,
        ),
        "hospital_days": ParameterSpec(
            "Hospital stay (days)",
            "Average number of days a hospitalised person stays before recovering or dying. "
            "Source: mean stay of 10.4 days in Ferguson et al. 2020, Report 9 (11.0 days in "
            "Faes et al. 2020).",
            10.4,
            0.5,
            60.0,
            0.1,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        INFECTIOUS: "#ea4335",
        HOSPITALISED: "#7b1fa2",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    CONFINED_STATES = frozenset({HOSPITALISED})
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(INFECTIOUS, "incubation_days"),
        INFECTIOUS: Phase(
            RECOVERED,
            "infectious_days",
            branch_state=HOSPITALISED,
            branch_parameter="hospitalised",
        ),
        HOSPITALISED: Phase(RECOVERED, "hospital_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
