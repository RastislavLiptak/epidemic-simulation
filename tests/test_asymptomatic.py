"""Infections without symptoms: the course every registered model has (see the base class)."""

import numpy as np
import pytest
from disease_helpers import ONE_DAY, count_in, new_state, place

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    ASYMPTOMATIC_PARAMETERS,
    COMMON_PARAMETERS,
    DEAD,
    DEATH_PARAMETERS,
    DYING,
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)
from epidemic_sandbox.disease_models.sepir import PRESYMPTOMATIC, SepirModel
from epidemic_sandbox.disease_models.sis import SisModel
from epidemic_sandbox.measure import summarise

KEYS = sorted(DISEASE_MODELS)
SHARP = {"duration_shape": 50.0}


def short_sepir(**parameters) -> SepirModel:
    """SEPIR whose latent period ends within one day step."""
    return SepirModel(ONE_DAY, latent_days=0.5, **SHARP, **parameters)


@pytest.mark.parametrize("key", KEYS)
def test_every_model_has_the_course_without_symptoms_and_takes_it_only_when_asked(key):
    model_class = DISEASE_MODELS[key]
    assert ASYMPTOMATIC in model_class.STATE_COLORS
    assert ASYMPTOMATIC_PARAMETERS.keys() <= model_class.PARAMETERS.keys()
    assert model_class.PARAMETERS["asymptomatic_share"].default == 0.0
    assert ASYMPTOMATIC in model_class.INFECTIOUS_STATES
    model = model_class(ONE_DAY)
    assert all(ASYMPTOMATIC not in [s for s, _, _ in c.stays] for c in model.courses())


@pytest.mark.parametrize("share", [0.0, 0.3, 1.0])
def test_a_share_of_the_infections_leaves_the_latent_period_without_symptoms(share):
    model = short_sepir(asymptomatic_share=share)
    disease_state, rng = new_state(model, 40_000)
    place(model, disease_state, rng, np.arange(40_000), EXPOSED)
    model.finish_phases(disease_state, rng)
    without = count_in(model, disease_state, ASYMPTOMATIC) / 40_000
    assert without == pytest.approx(share, abs=0.01)
    assert count_in(model, disease_state, PRESYMPTOMATIC) / 40_000 == pytest.approx(
        1 - share, abs=0.01
    )


def test_the_age_effect_scales_the_share_without_symptoms():
    effects = {"asymptomatic": [[0, 19, 2.0], [60, 120, 0.5]]}
    model = SepirModel(
        ONE_DAY, latent_days=0.5, asymptomatic_share=0.3, age_effects=effects, **SHARP
    )
    ages = np.repeat([10, 40, 70], 20_000)
    disease_state, rng = new_state(model, len(ages), age=ages)
    place(model, disease_state, rng, np.arange(len(ages)), EXPOSED)
    model.finish_phases(disease_state, rng)
    taken = (disease_state.state == model.state_index(ASYMPTOMATIC)).reshape(3, -1).mean(axis=1)
    assert taken == pytest.approx([0.6, 0.3, 0.15], abs=0.015)


def test_without_a_latent_period_the_course_splits_at_the_infection():
    model = SisModel(ONE_DAY, transmission=1.0, asymptomatic_share=0.44)
    disease_state, rng = new_state(model, 40_000)
    infected = np.arange(40_000)
    model.apply_infections(disease_state, infected, rng)
    assert count_in(model, disease_state, ASYMPTOMATIC) / 40_000 == pytest.approx(0.44, abs=0.01)
    assert count_in(model, disease_state, INFECTIOUS) / 40_000 == pytest.approx(0.56, abs=0.01)
    assert disease_state.ever_infected.all()


def test_people_without_symptoms_never_stay_at_home_nor_skip_outings():
    model = short_sepir(asymptomatic_share=1.0, stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, 1000)
    place(model, disease_state, rng, np.arange(1000), EXPOSED)
    model.finish_phases(disease_state, rng)
    assert count_in(model, disease_state, ASYMPTOMATIC) == 1000
    assert not disease_state.stays_home.any() and not disease_state.limits_outings.any()


@pytest.mark.parametrize(("fatality", "share", "expected"), [(0.2, 0.5, 0.2), (0.8, 0.5, 0.5)])
def test_the_deaths_fall_on_the_ill_and_keep_the_rate_per_infection(fatality, share, expected):
    model = SepirModel(
        ONE_DAY,
        latent_days=0.5,
        presymptomatic_days=0.5,
        infectious_days=0.5,
        waning_days=0.5,
        asymptomatic_days=0.5,
        death_delay_days=0.0,
        fatality=fatality,
        asymptomatic_share=share,
        immunity_days=36500,
        **SHARP,
    )
    disease_state, rng = new_state(model, 40_000)
    model.apply_infections(disease_state, np.arange(40_000), rng)
    went_without = np.zeros(40_000, np.bool_)
    for _ in range(8):
        model.finish_phases(disease_state, rng)
        went_without |= disease_state.state == model.state_index(ASYMPTOMATIC)
    dead = disease_state.state == model.dead_index
    assert dead.mean() == pytest.approx(expected, abs=0.01)  # capped where the ill cannot all die
    assert not (dead & went_without).any()
    assert went_without.mean() == pytest.approx(share, abs=0.01)


def test_an_infection_without_symptoms_spreads_the_given_share_of_one_with_symptoms():
    model = SepirModel(ONE_DAY, asymptomatic_share=0.25, asymptomatic_infectiousness=0.32)
    p = model.parameters
    with_symptoms = (
        p["presymptomatic_infectiousness"] * p["presymptomatic_days"]
        + p["infectious_days"]
        + p["waning_infectiousness"] * p["waning_days"]
    )
    per_day = model.state_infectiousness(ASYMPTOMATIC)
    assert per_day * p["asymptomatic_days"] == pytest.approx(0.32 * with_symptoms)
    assert model.infectious_days() == pytest.approx(
        0.75 * with_symptoms + 0.25 * 0.32 * with_symptoms
    )
    courses = model.courses()
    without = sum(c.probability for c in courses if ASYMPTOMATIC in [s for s, _, _ in c.stays])
    assert without == pytest.approx(0.25)
    died = sum(c.probability for c in courses if c.end == DEAD)
    assert died == pytest.approx(p["fatality"])  # per infection, all of it among the ill


def test_the_infectiousness_without_symptoms_does_not_depend_on_their_share():
    weights = [
        SepirModel(ONE_DAY, asymptomatic_share=share).state_infectiousness(ASYMPTOMATIC)
        for share in (0.0, 0.5, 1.0)
    ]
    assert weights == pytest.approx([weights[0]] * 3)


def test_the_age_effect_without_symptoms_needs_the_course():
    class NoCourse(DiseaseModelAbstract):
        PARAMETERS = {
            **COMMON_PARAMETERS,
            **DEATH_PARAMETERS,
            "infectious_days": SisModel.PARAMETERS["infectious_days"],
        }
        STATE_COLORS = {SUSCEPTIBLE: "#000", INFECTIOUS: "#111", DYING: "#222", DEAD: "#333"}
        INFECTIOUS_STATES = frozenset({INFECTIOUS})
        INFECTION_ENTRY_STATE = INFECTIOUS
        INITIAL_CASE_STATE = INFECTIOUS
        PHASES = {INFECTIOUS: Phase(SUSCEPTIBLE, "infectious_days", fatal=True)}

    with pytest.raises(ValueError, match="asymptomatic"):
        NoCourse(ONE_DAY, age_effects={"asymptomatic": [[0, 18, 2.0]]})


def declare(**changes) -> type:
    """A small SEIR with the course without symptoms, changed by `changes`."""
    attributes = {
        "PARAMETERS": {
            **COMMON_PARAMETERS,
            **DEATH_PARAMETERS,
            **ASYMPTOMATIC_PARAMETERS,
            "latent_days": SepirModel.PARAMETERS["latent_days"],
            "infectious_days": SepirModel.PARAMETERS["infectious_days"],
        },
        "STATE_COLORS": {
            SUSCEPTIBLE: "#000",
            EXPOSED: "#111",
            ASYMPTOMATIC: "#222",
            INFECTIOUS: "#333",
            RECOVERED: "#444",
            DYING: "#555",
            DEAD: "#666",
        },
        "INFECTIOUS_STATES": frozenset({ASYMPTOMATIC, INFECTIOUS}),
        "SYMPTOMATIC_STATES": frozenset({INFECTIOUS}),
        "INFECTION_ENTRY_STATE": EXPOSED,
        "INITIAL_CASE_STATE": INFECTIOUS,
        "PHASES": {
            EXPOSED: Phase(
                INFECTIOUS,
                "latent_days",
                branch_state=ASYMPTOMATIC,
                branch_parameter="asymptomatic_share",
                branch_age_effect="asymptomatic",
            ),
            ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days"),
            INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
        },
    }
    attributes.update(changes)
    return type("Declared", (DiseaseModelAbstract,), attributes)


def test_a_correct_declaration_of_the_course_without_symptoms_is_accepted():
    declare()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"SYMPTOMATIC_STATES": frozenset({INFECTIOUS, ASYMPTOMATIC})}, "symptomatic"),
        ({"INFECTIOUS_STATES": frozenset({INFECTIOUS})}, "infectious"),
        (
            {
                "PARAMETERS": {
                    **COMMON_PARAMETERS,
                    **DEATH_PARAMETERS,
                    "latent_days": SepirModel.PARAMETERS["latent_days"],
                    "infectious_days": SepirModel.PARAMETERS["infectious_days"],
                }
            },
            "unknown parameter",
        ),
        ({"STATE_INFECTIOUSNESS": {ASYMPTOMATIC: "asymptomatic_infectiousness"}}, "follows from"),
        (
            {
                "PHASES": {
                    EXPOSED: Phase(
                        INFECTIOUS,
                        "latent_days",
                        branch_state=ASYMPTOMATIC,
                        branch_parameter="asymptomatic_share",
                        branch_age_effect="asymptomatic",
                    ),
                    ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days", fatal=True),
                    INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
                }
            },
            "does not kill",
        ),
        (
            {
                "PHASES": {
                    EXPOSED: Phase(
                        INFECTIOUS,
                        "latent_days",
                        branch_state=ASYMPTOMATIC,
                        branch_parameter="asymptomatic_share",
                    ),
                    ASYMPTOMATIC: Phase(RECOVERED, "asymptomatic_days"),
                    INFECTIOUS: Phase(RECOVERED, "infectious_days", fatal=True),
                }
            },
            "one branch only",
        ),
        ({"INITIAL_CASE_STATE": ASYMPTOMATIC}, "branch only"),
    ],
)
def test_a_wrong_declaration_of_the_course_without_symptoms_is_rejected(changes, message):
    with pytest.raises(TypeError, match=message):
        declare(**changes)


def test_the_share_before_symptoms_counts_only_infectors_who_had_symptoms():
    # two first cases, 0 with symptoms (onset at step 15) and 1 without; 0 infects 2 before its
    # onset and 3 after it, 1 infects 4 (who has no symptoms either) and 5
    record = {
        "step": np.array([0, 0, 10, 20, 30, 31]),
        "person": np.arange(6),
        "infector": np.array([-1, -1, 0, 0, 1, 1]),
        "infector_record": np.array([-1, -1, 0, 0, 1, 1]),
        "setting": np.array([0, 0, 2, 2, 2, 2]),
        "infector_symptomatic": np.array([-1, -1, 0, 1, 0, 0]),
        "onset_step": np.array([15, -1, -1, -1, -1, -1]),
        "death_step": np.full(6, -1),
        "asymptomatic": np.array([0, 1, 0, 0, 1, 0]),
    }
    run = {
        "record": record,
        "settings": ("seed", "outside", "household"),
        "age": np.full(6, 40),
        "population": 1000,
    }
    result = summarise([run], steps_per_day=1)
    # of the infections by 0, one of two before the onset; those by 1 do not count
    assert result["share_before_symptoms"] == pytest.approx(0.5)
    assert result["share_without_symptoms"] == pytest.approx(0.25)


def test_the_trace_marks_the_infections_without_symptoms():
    from epidemic_sandbox.tracing import DiseaseTrace

    assert "asymptomatic" in DiseaseTrace().arrays()
