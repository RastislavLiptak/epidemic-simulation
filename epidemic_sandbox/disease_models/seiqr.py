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

UNDETECTED = "undetected"
QUARANTINED = "quarantined"


class SeiqrModel(DiseaseModelAbstract):
    """Susceptible -> exposed -> infectious -> quarantined or undetected -> recovered or dead.

    SEIR with isolation of detected cases. A person becomes infectious and goes about their days
    until, after the detection delay, the case is either detected (the share `detection_coverage`,
    scaled by the "branching" age effect) and isolated at home until recovery, or not and carries
    on unchanged. A person who is not detected spreads for the whole natural infectious period,
    `detection_delay_days` + `undetected_days`; a detected one for the delay and then, with reduced
    infectiousness, for `quarantine_days` (set it equal to `undetected_days` to isolate until the
    infection would have ended). Death can follow only at the end of the stay in either of the last
    two states, with the same `fatality`. Detection coverage 0 is a plain SEIR.

    Contact tracing (isolating exposed contacts before they spread) is not a separate mechanism:
    it shows as a higher coverage and a shorter delay, as in the models it comes from.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        "incubation_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people "
            "do not spread the disease yet. Source: mean incubation period of SARS, 4.6 days "
            "(Leung et al. 2004, 1,755 Hong Kong patients), as infectiousness began with the "
            "symptoms.",
            4.6,
            0.5,
            30.0,
            0.1,
        ),
        "detection_delay_days": ParameterSpec(
            "Detection delay (days)",
            "Average number of days from becoming infectious until the case is detected and "
            "isolated. Until then the person spreads fully and goes about their days. "
            "Source: mean time from SARS onset to hospital admission, 3 to 5 days in Hong Kong "
            "(Donnelly et al. 2003); 3.43 days used by Hellewell et al. 2020.",
            3.4,
            0.1,
            30.0,
            0.1,
        ),
        "undetected_days": ParameterSpec(
            "Undetected infectious period (days)",
            "Average number of days a case that is not detected keeps spreading after the "
            "detection delay, before recovering. Added to the detection delay it is the natural "
            "length of the infectious period. Derived: 10.8 days, twice the gap between the "
            "mean generation interval of the Singapore cases before 12 March 2003 (10.0 days, "
            "Wallinga and Teunis 2004) and the incubation period (4.6 days), as infectiousness "
            "is constant over the period; "
            "minus the 3.4 days of the delay.",
            7.4,
            0.1,
            60.0,
            0.1,
        ),
        "quarantine_days": ParameterSpec(
            "Isolation (days)",
            "Average number of days a detected person stays isolated at home, spreading with "
            "the reduced infectiousness below. Assumption: isolation lasts until the infection "
            "would have ended, so it equals the undetected infectious period (the constraint of "
            "Chowell et al. 2003).",
            7.4,
            0.5,
            60.0,
            0.1,
        ),
        "detection_coverage": ParameterSpec(
            "Detection coverage",
            "Share of cases that are detected and isolated, for a person whose branching age "
            "effect is 1 (see age effects). Assumption: most SARS cases were symptomatic and met "
            "the case definition, but not all were found in time; no measured value.",
            0.8,
            0.0,
            1.0,
            0.05,
        ),
        "quarantine_infectiousness": ParameterSpec(
            "Infectiousness in isolation",
            "How much an isolated person still spreads compared with a free one, beyond the "
            "contacts lost by staying at home (the household is still met): 0 is perfect "
            "isolation, 1 is no protection inside the home. Assumption, order of magnitude "
            "from Wang et al. 2020 (masks and disinfection in households, odds ratio about "
            "0.2) and Chowell et al. 2003 (0.38 fitted to Hong Kong SARS).",
            0.5,
            0.0,
            1.0,
            0.05,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        INFECTIOUS: "#ea4335",
        UNDETECTED: "#cc79a7",
        QUARANTINED: "#0072b2",
        RECOVERED: "#9aa0a6",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({INFECTIOUS, UNDETECTED, QUARANTINED})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS, UNDETECTED})
    CONFINED_STATES = frozenset({QUARANTINED})
    STATE_INFECTIOUSNESS = {QUARANTINED: "quarantine_infectiousness"}
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(INFECTIOUS, "incubation_days"),
        INFECTIOUS: Phase(
            UNDETECTED,
            "detection_delay_days",
            branch_state=QUARANTINED,
            branch_parameter="detection_coverage",
        ),
        UNDETECTED: Phase(RECOVERED, "undetected_days", fatal=True),
        QUARANTINED: Phase(RECOVERED, "quarantine_days", fatal=True),
        RECOVERED: Phase(SUSCEPTIBLE, "immunity_days"),
    }
