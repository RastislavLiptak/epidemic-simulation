"""The mechanisms of the base class that models are declared with, on test-only models."""

import math

import numpy as np
import pytest
from disease_helpers import (
    ONE_DAY,
    SURE,
    UNIFORM,
    advance,
    count_in,
    new_state,
    place,
    state_names,
)
from synthetic_models import (
    ASYMPTOMATIC,
    HOSPITAL,
    VACCINATED,
    ChronicModel,
    ForkModel,
    PlainModel,
    RepeatModel,
    VaccinationModel,
)

from epidemic_sandbox.disease_models.disease_model_abstract import (
    COMMON_PARAMETERS,
    DEAD,
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
    Rate,
)

SHARP = {"duration_shape": 50.0}


def short_fork(**parameters) -> ForkModel:
    """Fork model whose phases all end within one day step."""
    return ForkModel(
        ONE_DAY,
        incubation_days=0.5,
        asymptomatic_days=0.5,
        infectious_days=0.5,
        hospital_days=0.5,
        **SHARP,
        **parameters,
    )


# --- branching ---------------------------------------------------------------------------


@pytest.mark.parametrize("share", [0.0, 0.25, 1.0])
def test_a_branch_is_taken_with_the_configured_chance(share):
    model = short_fork(asymptomatic_share=share)
    disease_state, rng = new_state(model, population=40_000)
    place(model, disease_state, rng, np.arange(40_000), EXPOSED)
    advance(model, disease_state, rng)
    taken = count_in(model, disease_state, ASYMPTOMATIC) / 40_000
    assert taken == pytest.approx(share, abs=0.01)
    assert (
        count_in(model, disease_state, ASYMPTOMATIC) + count_in(model, disease_state, INFECTIOUS)
        == 40_000
    ), "everybody takes one of the two ways"


def test_the_chance_of_a_branch_depends_on_age_through_the_branching_effect():
    effects = {"branching": [[0, 15, 0.0], [65, 120, 2.0]]}
    model = short_fork(asymptomatic_share=0.3, age_effects=effects)
    ages = np.repeat([5, 40, 80], 100_000)
    disease_state, rng = new_state(model, population=300_000, age=ages)
    place(model, disease_state, rng, np.arange(300_000), EXPOSED)
    advance(model, disease_state, rng)
    taken = (disease_state.state == list(model.STATE_COLORS).index(ASYMPTOMATIC)).reshape(3, -1)
    assert taken.mean(axis=1) == pytest.approx([0.0, 0.3, 0.6], abs=0.01)


def test_a_branching_effect_that_pushes_the_chance_above_one_means_certainty():
    effects = {"branching": [[0, 120, 5.0]]}
    model = short_fork(asymptomatic_share=0.5, age_effects=effects)
    disease_state, rng = new_state(model, population=1000)
    place(model, disease_state, rng, np.arange(1000), EXPOSED)
    advance(model, disease_state, rng)
    assert count_in(model, disease_state, ASYMPTOMATIC) == 1000


def test_death_is_settled_before_the_branch_among_the_survivors():
    class Both(ForkModel):
        PHASES = {
            **ForkModel.PHASES,
            INFECTIOUS: Phase(
                RECOVERED,
                "infectious_days",
                fatal=True,
                branch_state=HOSPITAL,
                branch_parameter="hospital_share",
            ),
        }
        PARAMETERS = ForkModel.PARAMETERS

    model = Both(
        ONE_DAY, infectious_days=0.5, hospital_share=0.5, fatality=0.2, hospital_days=60.0, **SHARP
    )
    disease_state, rng = new_state(model, population=60_000)
    place(model, disease_state, rng, np.arange(60_000), INFECTIOUS)
    advance(model, disease_state, rng)
    assert disease_state.deaths / 60_000 == pytest.approx(0.2, abs=0.01)
    assert count_in(model, disease_state, HOSPITAL) / 60_000 == pytest.approx(0.4, abs=0.01)
    assert count_in(model, disease_state, RECOVERED) / 60_000 == pytest.approx(0.4, abs=0.01)


def test_the_hospital_branch_is_confined_and_ends_in_recovery_or_death():
    model = short_fork(hospital_share=1.0, fatality=0.5)
    disease_state, rng = new_state(model, population=20_000)
    place(model, disease_state, rng, np.arange(20_000), INFECTIOUS)
    advance(model, disease_state, rng)
    assert count_in(model, disease_state, HOSPITAL) == 20_000
    advance(model, disease_state, rng)
    assert disease_state.deaths / 20_000 == pytest.approx(0.5, abs=0.015)
    assert count_in(model, disease_state, RECOVERED) + disease_state.deaths == 20_000


def test_age_effect_on_branching_needs_a_model_with_a_branch():
    with pytest.raises(ValueError, match="branching"):
        PlainModel(ONE_DAY, age_effects={"branching": [[0, 120, 2.0]]})


def test_age_effect_on_fatality_needs_a_model_where_somebody_can_die_of_it():
    with pytest.raises(ValueError, match="fatality"):
        ChronicModel(ONE_DAY, age_effects={"fatality": [[0, 120, 2.0]]})


# --- infectiousness of the states ---------------------------------------------------------


def test_states_spread_with_their_own_relative_infectiousness():
    model = ForkModel(ONE_DAY, asymptomatic_infectiousness=0.25)
    disease_state, rng = new_state(model, population=len(model.STATE_COLORS))
    disease_state.state[:] = np.arange(len(model.STATE_COLORS))
    assert dict(
        zip(model.STATE_COLORS, model.infectious_weight(disease_state).tolist(), strict=True)
    ) == {
        SUSCEPTIBLE: 0.0,
        EXPOSED: 0.0,
        ASYMPTOMATIC: 0.25,
        INFECTIOUS: 1.0,
        HOSPITAL: 0.0,
        RECOVERED: 0.0,
        DEAD: 0.0,
    }


def test_infectious_states_without_a_declared_weight_spread_fully():
    model = RepeatModel(ONE_DAY)
    disease_state, _ = new_state(model, population=3, cases=[1])
    assert model.infectious_weight(disease_state).tolist() == [0.0, 1.0, 0.0]


# --- behaviour of the ill -----------------------------------------------------------------


def test_only_symptomatic_states_make_people_stay_home_or_skip_outings():
    model = ForkModel(ONE_DAY, stay_home=1.0, limit_outings=1.0)
    disease_state, rng = new_state(model, population=10)
    place(model, disease_state, rng, [0], INFECTIOUS)
    place(model, disease_state, rng, [1], ASYMPTOMATIC)
    place(model, disease_state, rng, [2], EXPOSED)
    assert disease_state.stays_home[:3].tolist() == [True, False, False]
    assert disease_state.limits_outings[:3].tolist() == [True, False, False]


def test_confined_states_keep_people_at_home_whatever_the_settings():
    model = ForkModel(ONE_DAY, stay_home=0.0, limit_outings=0.0)
    disease_state, rng = new_state(model, population=10)
    place(model, disease_state, rng, [4], HOSPITAL)
    assert disease_state.stays_home[4] and not disease_state.stays_home[5]


def test_people_leave_home_when_they_leave_a_confined_state():
    model = short_fork(hospital_share=1.0, stay_home=0.0)
    disease_state, rng = new_state(model, population=10)
    place(model, disease_state, rng, [4], HOSPITAL)
    for _ in range(3):
        advance(model, disease_state, rng)
    assert state_names(model, disease_state)[4] == RECOVERED
    assert not disease_state.stays_home[4]


# --- protection ---------------------------------------------------------------------------


def test_every_protected_state_reduces_the_hazard_by_its_own_protection():
    model = VaccinationModel(
        ONE_DAY,
        transmission=0.5,
        vaccination_rate=0.0,
        vaccine_protection=0.75,
        immunity_protection=0.5,
        **UNIFORM,
    )
    states = list(model.STATE_COLORS)
    for state, protection in ((VACCINATED, 0.75), (RECOVERED, 0.5), (SUSCEPTIBLE, 0.0)):
        disease_state, rng = new_state(model, population=200_000)
        disease_state.state[:] = states.index(state)
        advance(model, disease_state, rng, exposure=1)
        expected = 1 - 0.5 ** (1 - protection)
        assert disease_state.cumulative_infections / 200_000 == pytest.approx(expected, abs=0.006)


def test_a_state_with_complete_protection_cannot_be_infected():
    model = VaccinationModel(ONE_DAY, **SURE, vaccination_rate=0.0, vaccine_protection=1.0)
    disease_state, rng = new_state(model, population=100)
    disease_state.state[:] = list(model.STATE_COLORS).index(VACCINATED)
    assert not model.exposable_mask(disease_state).any()
    advance(model, disease_state, rng, exposure=10)
    assert disease_state.cumulative_infections == 0


def test_only_partially_protected_people_can_be_exposed():
    model = VaccinationModel(ONE_DAY, vaccination_rate=0.0, vaccine_protection=0.5)
    disease_state, _ = new_state(model, population=5)
    disease_state.state[:] = [0, 1, 2, 3, 4]  # susceptible, vaccinated, infectious, recovered, dead
    assert model.exposable_mask(disease_state).tolist() == [True, True, False, False, False]
    complete = VaccinationModel(
        ONE_DAY, vaccination_rate=0.0, vaccine_protection=1.0, immunity_protection=0.5
    )
    assert complete.exposable_mask(disease_state).tolist() == [True, False, False, True, False]


# --- spontaneous transitions (vaccination) --------------------------------------------------


@pytest.mark.parametrize("step_days", [1.0, 1 / 24])
def test_a_rate_moves_people_at_the_chance_given_by_a_constant_hazard(step_days):
    model = VaccinationModel(step_days, vaccination_rate=0.5, vaccine_days=60.0)
    disease_state, rng = new_state(model, population=600_000)
    for _ in range(round(1 / step_days)):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) / 600_000 == pytest.approx(
        1 - math.exp(-0.5), abs=0.004
    )


def test_a_rate_starts_on_its_start_day():
    model = VaccinationModel(1 / 4, vaccination_rate=2.0, vaccination_start_day=3.0)
    disease_state, rng = new_state(model, population=10_000)
    for _ in range(12):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) == 0, "three days, nothing yet"
    for _ in range(4):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, VACCINATED) > 5000


def test_a_zero_rate_moves_nobody():
    model = VaccinationModel(ONE_DAY, vaccination_rate=0.0)
    disease_state, rng = new_state(model, population=1000)
    for _ in range(20):
        advance(model, disease_state, rng)
    assert not disease_state.state.any()


def test_a_rate_moves_only_the_people_in_its_state():
    model = VaccinationModel(ONE_DAY, vaccination_rate=5.0, infectious_days=60.0)
    disease_state, rng = new_state(model, population=1000, cases=np.arange(100))
    advance(model, disease_state, rng)
    assert count_in(model, disease_state, INFECTIOUS) == 100
    assert count_in(model, disease_state, VACCINATED) > 800


def test_a_person_infected_and_vaccinated_in_the_same_step_is_infected():
    model = VaccinationModel(ONE_DAY, **SURE, vaccination_rate=5.0, vaccination_start_day=0.0)
    disease_state, rng = new_state(model, population=2000)
    advance(model, disease_state, rng, exposure=1)
    assert count_in(model, disease_state, INFECTIOUS) == 2000
    assert disease_state.cumulative_infections == 2000


def test_vaccine_protection_wanes_after_the_configured_time():
    model = VaccinationModel(
        ONE_DAY, vaccination_rate=0.0, vaccine_days=10.0, **SHARP, initial_coverage=1.0
    )
    disease_state, rng = new_state(model, population=500)
    assert count_in(model, disease_state, VACCINATED) == 500
    for _ in range(20):
        advance(model, disease_state, rng)
    assert count_in(model, disease_state, SUSCEPTIBLE) == 500


# --- starting shares ---------------------------------------------------------------------


def test_a_share_of_the_people_starts_in_the_given_state_and_the_cases_keep_theirs():
    model = VaccinationModel(ONE_DAY, initial_coverage=0.3, vaccination_rate=0.0)
    disease_state, _ = new_state(model, population=100_000, cases=np.arange(1000))
    states = list(model.STATE_COLORS)
    assert (disease_state.state[:1000] == states.index(INFECTIOUS)).all()
    vaccinated = disease_state.state[1000:] == states.index(VACCINATED)
    assert vaccinated.mean() == pytest.approx(0.3, abs=0.005)
    assert not disease_state.ever_infected[1000:].any()
    assert (disease_state.timer[1000:][vaccinated] > 0).all(), "their protection runs out"


def test_no_starting_share_means_everybody_but_the_cases_starts_susceptible():
    model = VaccinationModel(ONE_DAY, initial_coverage=0.0)
    disease_state, _ = new_state(model, population=1000, cases=[1, 2])
    assert np.count_nonzero(disease_state.state) == 2


# --- the end of the epidemic -------------------------------------------------------------


def test_an_infection_without_an_end_is_never_quiescent():
    model = ChronicModel(ONE_DAY)
    disease_state, _ = new_state(model, population=10)
    assert model.is_quiescent(disease_state)
    disease_state, _ = new_state(model, population=10, cases=[3])
    assert not model.is_quiescent(disease_state)


def test_a_running_vaccination_keeps_the_disease_from_being_quiescent():
    running = VaccinationModel(ONE_DAY, vaccination_rate=0.1)
    stopped = VaccinationModel(ONE_DAY, vaccination_rate=0.0)
    for model, expected in ((running, False), (stopped, True)):
        disease_state, _ = new_state(model, population=10)
        assert model.is_quiescent(disease_state) is expected


# --- what a declared course means for the preview numbers --------------------------------------


def test_courses_list_every_way_through_the_disease_with_its_probability():
    model = ForkModel(
        ONE_DAY, asymptomatic_share=0.4, hospital_share=0.2, fatality=0.1, immunity_days=36500
    )
    courses = {
        tuple(state for state, _, _ in course.stays): course.probability
        for course in model.courses()
    }
    assert sum(courses.values()) == pytest.approx(1.0)
    assert courses[(EXPOSED, ASYMPTOMATIC, RECOVERED)] == pytest.approx(0.4)
    assert courses[(EXPOSED, INFECTIOUS, RECOVERED)] == pytest.approx(0.6 * 0.8)
    assert courses[(EXPOSED, INFECTIOUS, HOSPITAL, RECOVERED)] == pytest.approx(0.6 * 0.2 * 0.9)
    assert courses[(EXPOSED, INFECTIOUS, HOSPITAL)] == pytest.approx(0.6 * 0.2 * 0.1)


def test_infectious_days_weigh_each_state_by_its_infectiousness_and_the_chance_of_reaching_it():
    model = ForkModel(
        ONE_DAY,
        asymptomatic_share=0.4,
        asymptomatic_days=4.0,
        asymptomatic_infectiousness=0.5,
        infectious_days=6.0,
        hospital_share=0.2,
    )
    assert model.infectious_days() == pytest.approx(0.4 * 0.5 * 4.0 + 0.6 * 6.0)


def test_reproduction_number_follows_the_weighted_infectious_days():
    model = ForkModel(ONE_DAY, transmission=0.5)
    assert model.mean_field_r0(3.0) == pytest.approx(-math.log(0.5) * 3.0 * model.infectious_days())


def test_household_risk_of_a_forked_course_mixes_its_ways():
    model = ForkModel(
        ONE_DAY,
        transmission=0.3,
        asymptomatic_share=0.4,
        asymptomatic_days=4.0,
        asymptomatic_infectiousness=0.5,
        infectious_days=6.0,
        hospital_share=0.0,
        duration_shape=4.0,
    )
    hazard = -math.log(0.7)
    survive_asymptomatic = (1 + hazard * 0.5 * 4.0 / 4.0) ** -4.0
    survive_symptomatic = (1 + hazard * 6.0 / 4.0) ** -4.0
    assert model.pair_transmission_probability() == pytest.approx(
        1 - (0.4 * survive_asymptomatic + 0.6 * survive_symptomatic)
    )


def test_a_lifelong_infection_has_no_bounded_reproduction_number_and_certain_household_risk():
    model = ChronicModel(ONE_DAY, transmission=0.2)
    assert model.mean_field_r0(2.0) is None
    assert model.pair_transmission_probability() == 1.0
    assert ChronicModel(ONE_DAY, transmission=0.0).pair_transmission_probability() == 0.0


# --- declarations that cannot work are refused when the class is defined ----------------------


def declaration(**changes):
    """Define a variant of the fork model; the changes replace class attributes."""
    return type("Variant", (ForkModel,), changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "PHASES": {
                    **ForkModel.PHASES,
                    EXPOSED: Phase(INFECTIOUS, "incubation_days", False, ASYMPTOMATIC),
                }
            },
            "branch",
        ),
        (
            {
                "PHASES": {
                    **ForkModel.PHASES,
                    EXPOSED: Phase(
                        INFECTIOUS, "incubation_days", False, "nowhere", "hospital_share"
                    ),
                }
            },
            "unknown state",
        ),
        (
            {
                "PHASES": {
                    **ForkModel.PHASES,
                    EXPOSED: Phase(INFECTIOUS, "incubation_days", False, ASYMPTOMATIC, "nope"),
                }
            },
            "unknown parameter",
        ),
        ({"RATES": {RECOVERED: Rate(SUSCEPTIBLE, "hospital_days")}}, "only the susceptible"),
        ({"RATES": {SUSCEPTIBLE: Rate(DEAD, "hospital_days")}}, "only a fatal phase kills"),
        (
            {"PHASES": {**ForkModel.PHASES, HOSPITAL: Phase(DEAD, "hospital_days")}},
            "only a fatal phase kills",
        ),
        (
            {
                "PHASES": {
                    **ForkModel.PHASES,
                    EXPOSED: Phase(INFECTIOUS, "incubation_days", False, DEAD, "hospital_share"),
                }
            },
            "only a fatal phase kills",
        ),
        (
            {
                "PARAMETERS": {
                    **ForkModel.PARAMETERS,
                    "spare": ForkModel.PARAMETERS["hospital_days"],
                }
            },
            "not used",
        ),
        ({"RATES": {SUSCEPTIBLE: Rate("nowhere", "hospital_days")}}, "unknown state"),
        ({"RATES": {SUSCEPTIBLE: Rate(EXPOSED, "nope")}}, "unknown parameter"),
        ({"RATES": {SUSCEPTIBLE: Rate(EXPOSED, "hospital_days", "nope")}}, "unknown parameter"),
        ({"PROTECTION": {RECOVERED: "nope"}}, "unknown parameter"),
        ({"PROTECTION": {SUSCEPTIBLE: "immunity_protection"}}, "susceptible"),
        ({"PROTECTION": {"nowhere": "immunity_protection"}}, "unknown state"),
        ({"STATE_INFECTIOUSNESS": {EXPOSED: "asymptomatic_infectiousness"}}, "not infectious"),
        ({"STATE_INFECTIOUSNESS": {ASYMPTOMATIC: "nope"}}, "unknown parameter"),
        ({"SYMPTOMATIC_STATES": frozenset({"nowhere"})}, "unknown state"),
        ({"CONFINED_STATES": frozenset({"nowhere"})}, "unknown state"),
        ({"STARTING_SHARES": {"nowhere": "hospital_share"}}, "unknown state"),
        ({"STARTING_SHARES": {RECOVERED: "nope"}}, "unknown parameter"),
        (
            {"PHASES": {**ForkModel.PHASES, SUSCEPTIBLE: Phase(EXPOSED, "incubation_days")}},
            "susceptible",
        ),
        (
            {"PHASES": {**ForkModel.PHASES, ASYMPTOMATIC: Phase(EXPOSED, "asymptomatic_days")}},
            "cycle",
        ),
        (
            {
                "PHASES": {
                    key: ForkModel.PHASES[key]
                    for key in (EXPOSED, ASYMPTOMATIC, HOSPITAL, RECOVERED)
                }
            },
            "unreachable",
        ),
    ],
)
def test_impossible_declarations_are_refused(changes, message):
    with pytest.raises(TypeError, match=message):
        declaration(**changes)


def test_a_person_keeps_the_decision_to_stay_home_between_consecutive_symptomatic_states():
    class TwoStages(ForkModel):
        SYMPTOMATIC_STATES = frozenset({ASYMPTOMATIC, INFECTIOUS})

    model = TwoStages(ONE_DAY, stay_home=0.5, limit_outings=0.3)
    disease_state, rng = new_state(model, population=40_000)
    place(model, disease_state, rng, np.arange(40_000), ASYMPTOMATIC)
    home, outings = disease_state.stays_home.copy(), disease_state.limits_outings.copy()
    assert home.mean() == pytest.approx(0.5, abs=0.01)
    place(model, disease_state, rng, np.arange(40_000), INFECTIOUS)
    assert (disease_state.stays_home == home).all()
    assert (disease_state.limits_outings == outings).all()
    place(model, disease_state, rng, np.arange(40_000), EXPOSED)
    assert not disease_state.stays_home.any()
    place(model, disease_state, rng, np.arange(40_000), INFECTIOUS)
    assert disease_state.stays_home.mean() == pytest.approx(0.5, abs=0.01), "a new draw"


def test_a_model_with_a_fatal_phase_must_have_a_fatality_parameter():
    with pytest.raises(TypeError, match="fatality"):

        class NoFatality(DiseaseModelAbstract):
            PARAMETERS = {k: v for k, v in COMMON_PARAMETERS.items() if k != "fatality"}
            STATE_COLORS = {SUSCEPTIBLE: "#000001", INFECTIOUS: "#000002", DEAD: "#000003"}
            INFECTIOUS_STATES = frozenset({INFECTIOUS})
            INFECTION_ENTRY_STATE = INFECTIOUS
            INITIAL_CASE_STATE = INFECTIOUS
            PHASES = {INFECTIOUS: Phase(SUSCEPTIBLE, "transmission", fatal=True)}


def test_starting_shares_that_add_up_to_more_than_everybody_are_refused():
    class Greedy(VaccinationModel):
        STARTING_SHARES = {VACCINATED: "initial_coverage", RECOVERED: "vaccine_protection"}

    with pytest.raises(ValueError, match="more than everybody"):
        Greedy(ONE_DAY, initial_coverage=0.6, vaccine_protection=0.6)
    Greedy(ONE_DAY, initial_coverage=0.5, vaccine_protection=0.5)


def test_the_course_states_are_those_between_infection_and_the_end_of_the_illness():
    assert ForkModel.course_states() == {EXPOSED, ASYMPTOMATIC, INFECTIOUS, HOSPITAL, RECOVERED}
    assert VaccinationModel.course_states() == {INFECTIOUS, RECOVERED}, "not the vaccinated"
    assert ChronicModel.course_states() == {INFECTIOUS}


def test_infection_can_start_from_the_susceptible_and_the_protected_states():
    assert ForkModel.exposable_states() == {SUSCEPTIBLE, RECOVERED}
    assert VaccinationModel.exposable_states() == {SUSCEPTIBLE, VACCINATED, RECOVERED}
    assert ChronicModel.exposable_states() == {SUSCEPTIBLE}


def test_protection_of_a_state_is_the_value_of_its_parameter_or_zero():
    model = VaccinationModel(ONE_DAY, vaccine_protection=0.7, immunity_protection=0.4)
    assert [model.protection_of(state) for state in (SUSCEPTIBLE, VACCINATED, RECOVERED)] == [
        0.0,
        0.7,
        0.4,
    ]
