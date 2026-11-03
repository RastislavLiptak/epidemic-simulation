from dataclasses import replace

from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    DURATION_SHAPE,
    INFECTIOUS,
    LIMIT_OUTINGS,
    STAY_HOME,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
)


class SiModel(DiseaseModelAbstract):
    """Susceptible -> infectious for life: nobody recovers, nobody is immune, nobody dies of it.

    Without a state to leave there is no timed phase, so the infection spreads until everybody has
    it (logistic growth). The model has no parameter of its own; the common ones keep their
    meaning except where the descriptions below say otherwise.
    """

    PARAMETERS = {
        **{key: spec for key, spec in COMMON_PARAMETERS.items() if key != "fatality"},
        "duration_shape": replace(
            DURATION_SHAPE,
            description="Has no effect in this model: it has no timed phase, because an infected "
            "person stays infectious for life.",
        ),
        "stay_home": replace(
            STAY_HOME,
            description="Share of infected people who stay at home for the rest of their lives, "
            "like a permanent isolation of carriers. Leave at 0 for a lifelong infection that "
            "does not make people ill enough to change their days.",
        ),
        "limit_outings": replace(
            LIMIT_OUTINGS,
            description="Share of infected people who go to work or school but skip errands, "
            "visits and leisure for the rest of their lives. Leave at 0 for a lifelong infection "
            "that does not change people's days.",
        ),
    }
    STATE_COLORS = {SUSCEPTIBLE: "#34a853", INFECTIOUS: "#ea4335", DEAD: "#202124"}
    INFECTIOUS_STATES = frozenset({INFECTIOUS})
    INFECTION_ENTRY_STATE = INFECTIOUS
    INITIAL_CASE_STATE = INFECTIOUS
    PHASES = {}
