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
    two states, with the same `fatality`. Detection coverage 0 is a plain SEIR. A share of the
    infections (`asymptomatic_share`) never has symptoms: after the incubation period such a
    person spreads the disease for `asymptomatic_days`, is never detected and recovers.

    Contact tracing (isolating exposed contacts before they spread) is not a separate mechanism:
    it shows as a higher coverage and a shorter delay, as in the models it comes from.
    """

    PARAMETERS = {
        **COMMON_PARAMETERS,
        **DEATH_PARAMETERS,
        **IMMUNITY_PARAMETERS,
        **ASYMPTOMATIC_PARAMETERS,
        "incubation_days": ParameterSpec(
            "Latent period (days)",
            "Average number of days between infection and becoming infectious. Exposed people "
            "do not spread the disease yet. Default 4.6 (Source: mean incubation period of SARS, "
            "Leung et al. 2004, 1,755 Hong Kong patients), as infectiousness begins with the "
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
            "Default 3 (Source: diagnosis rate 1/3 per day, Chowell et al. 2003, Table 1; the mean "
            "time from SARS onset to hospital admission was 3 to 5 days in Hong Kong, Donnelly et "
            "al. 2003).",
            3.0,
            0.1,
            30.0,
            0.1,
        ),
        "undetected_days": ParameterSpec(
            "Undetected infectious period (days)",
            "Average number of days a case that is not detected keeps spreading after the "
            "detection delay, before recovering. Added to the detection delay it is the natural "
            "length of the infectious period. Default 7.8 (Derived: the serial interval of SARS "
            "before control was 10.0 days, Lipsitch et al. 2003, Wallinga and Teunis 2004, so the "
            "infectious period is 2 x (10.0 - 4.6) = 10.8 days, minus the 3 days of the delay).",
            7.8,
            0.1,
            60.0,
            0.1,
        ),
        "quarantine_days": ParameterSpec(
            "Isolation (days)",
            "Average number of days a detected person stays isolated at home, spreading with "
            "the reduced infectiousness below. Default 7.8 (Derived: isolation lasts until the "
            "infection would have ended, the constraint 1/gamma2 = 1/gamma1 - 1/alpha of Chowell "
            "et al. 2003).",
            7.8,
            0.5,
            60.0,
            0.1,
        ),
        "detection_coverage": ParameterSpec(
            "Detection coverage",
            "Share of cases that are detected and isolated, for a person whose branching age "
            "effect is 1 (see age effects). Default 0.77 (Derived: the share diagnosed before "
            "they recover in the model of Chowell et al. 2003, alpha / (alpha + gamma1 + delta) "
            "= (1/3) / (1/3 + 1/10.8 + 0.006)).",
            0.77,
            0.0,
            1.0,
            0.05,
        ),
        "quarantine_infectiousness": ParameterSpec(
            "Infectiousness in isolation",
            "How much an isolated person still spreads compared with a free one, beyond the "
            "contacts lost by staying at home (the household is still met): 0 is perfect "
            "isolation, 1 is no protection inside the home. Default 0.38 (Source: the reduced "
            "contact of diagnosed cases with the susceptible, l = 0.38 for Hong Kong and 0.40 for "
            "Singapore, Chowell et al. 2003; there it covers everything that isolation removes, "
            "so here isolation is slightly stronger than the fitted one).",
            0.38,
            0.0,
            1.0,
            0.01,
        ),
    }
    STATE_COLORS = {
        SUSCEPTIBLE: "#34a853",
        EXPOSED: "#fbbc04",
        INFECTIOUS: "#ea4335",
        UNDETECTED: "#cc79a7",
        QUARANTINED: "#0072b2",
        ASYMPTOMATIC: "#ab47bc",
        RECOVERED: "#9aa0a6",
        DYING: "#5d4037",
        DEAD: "#202124",
    }
    INFECTIOUS_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS, UNDETECTED, QUARANTINED})
    SYMPTOMATIC_STATES = frozenset({INFECTIOUS, UNDETECTED})
    CONFINED_STATES = frozenset({QUARANTINED})
    STATE_INFECTIOUSNESS = {QUARANTINED: "quarantine_infectiousness"}
    INFECTION_ENTRY_STATE = EXPOSED
    INITIAL_CASE_STATE = INFECTIOUS
    PROTECTION = {RECOVERED: "immunity_protection"}
    PHASES = {
        EXPOSED: Phase(
            INFECTIOUS,
            "incubation_days",
            branch_state=ASYMPTOMATIC,
            branch_parameter="asymptomatic_share",
            branch_age_effect="asymptomatic",
        ),
        ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days"),
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
