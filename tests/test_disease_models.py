import math

import numpy as np
import pytest
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, new_state, place, state_names
from synthetic_models import LatentModel, PlainModel, RepeatModel

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import (
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
    DiseaseModelAbstract,
    Phase,
)


def test_registry_models_have_consistent_definitions():
    for model_class in DISEASE_MODELS.values():
        assert model_class.initial_state() == SUSCEPTIBLE
        assert {"transmission", "duration_shape", "stay_home"} <= model_class.PARAMETERS.keys()


def test_inconsistent_model_declaration_is_rejected():
    with pytest.raises(TypeError, match="unknown state"):

        class Broken(DiseaseModelAbstract):
            PARAMETERS = PlainModel.PARAMETERS
            STATE_COLORS = PlainModel.STATE_COLORS
            INFECTIOUS_STATES = PlainModel.INFECTIOUS_STATES
            INFECTION_ENTRY_STATE = INFECTIOUS
            INITIAL_CASE_STATE = INFECTIOUS
            PHASES = {INFECTIOUS: Phase("nowhere", "infectious_days")}


def test_defaults_are_used_for_missing_parameters():
    model = PlainModel(ONE_DAY)
    assert model.parameters == {name: spec.default for name, spec in PlainModel.PARAMETERS.items()}


def test_unknown_parameter_and_out_of_range_value_are_rejected():
    with pytest.raises(ValueError, match="Unknown parameters"):
        PlainModel(ONE_DAY, incubation_days=3.0)
    with pytest.raises(ValueError, match="out of range"):
        PlainModel(ONE_DAY, transmission=1.5)


def test_initial_cases_start_in_the_initial_case_state():
    model = LatentModel(ONE_DAY)
    disease_state, _ = new_state(model, population=10, cases=[2, 5])
    assert state_names(model, disease_state).count(INFECTIOUS) == 2
    assert disease_state.ever_infected.sum() == 2 and disease_state.cumulative_infections == 2


@pytest.mark.parametrize("model_class", DISEASE_MODELS.values())
def test_nobody_is_infected_without_exposure(model_class):
    model = model_class(ONE_DAY, **SURE)
    disease_state, rng = new_state(model, population=500)
    for _ in range(10):
        advance(model, disease_state, rng, exposure=0)
    assert not disease_state.state.any()


def test_certain_transmission_infects_everybody_who_is_exposed():
    model = PlainModel(ONE_DAY, **SURE)
    disease_state, rng = new_state(model, population=100)
    exposure = np.zeros(100, np.float32)
    exposure[:40] = 0.5
    model.advance(disease_state, exposure, rng)
    assert state_names(model, disease_state).count(INFECTIOUS) == 40
    assert disease_state.cumulative_infections == 40


def test_infection_enters_the_exposed_state_in_seir():
    model = LatentModel(ONE_DAY, **SURE)
    disease_state, rng = new_state(model)
    advance(model, disease_state, rng, exposure=1)
    assert state_names(model, disease_state) == [EXPOSED]


def test_seir_person_goes_through_all_states_in_order_and_stays_recovered():
    model = LatentModel(
        ONE_DAY, **SURE, incubation_days=2.0, infectious_days=2.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model)
    advance(model, disease_state, rng, exposure=1)
    visited = [EXPOSED]
    for _ in range(40):
        advance(model, disease_state, rng)
        current = state_names(model, disease_state)[0]
        if current != visited[-1]:
            visited.append(current)
    assert visited == [EXPOSED, INFECTIOUS, RECOVERED]


def test_sir_recovered_people_cannot_be_infected_again():
    model = PlainModel(ONE_DAY, **SURE, infectious_days=0.5, duration_shape=50.0)
    disease_state, rng = new_state(model, cases=[0])
    for _ in range(10):
        advance(model, disease_state, rng)
    assert state_names(model, disease_state) == [RECOVERED]
    advance(model, disease_state, rng, exposure=5)
    assert state_names(model, disease_state) == [RECOVERED]


def test_sis_people_can_be_infected_again_and_reinfections_are_counted():
    model = RepeatModel(ONE_DAY, **SURE, infectious_days=0.5, duration_shape=50.0)
    disease_state, rng = new_state(model, cases=[0])
    for _ in range(10):
        advance(model, disease_state, rng)
    assert state_names(model, disease_state) == [SUSCEPTIBLE]
    advance(model, disease_state, rng, exposure=1)
    assert state_names(model, disease_state) == [INFECTIOUS]
    assert disease_state.cumulative_infections == 2 and disease_state.ever_infected.sum() == 1


def test_infection_chance_is_per_day_of_contact_whatever_the_step_length():
    hour = 1 / 24
    model = PlainModel(hour, transmission=0.5, **UNIFORM)
    disease_state, rng = new_state(model, population=200_000)
    advance(model, disease_state, rng, exposure=1)
    assert disease_state.cumulative_infections / 200_000 == pytest.approx(1 - 0.5**hour, abs=0.002)


def test_exposure_scales_the_infection_hazard():
    hour = 1 / 24
    model = PlainModel(hour, transmission=0.5, **UNIFORM)
    disease_state, rng = new_state(model, population=200_000)
    advance(model, disease_state, rng, exposure=2.5)
    assert disease_state.cumulative_infections / 200_000 == pytest.approx(
        1 - 0.5 ** (2.5 * hour), abs=0.003
    )


@pytest.mark.parametrize("shape", [1.0, 4.0, 25.0])
def test_phase_lengths_are_gamma_distributed_with_the_given_mean_and_shape(shape):
    model = PlainModel(ONE_DAY, infectious_days=6.0, duration_shape=shape)
    disease_state, _ = new_state(model, population=200_000, cases=np.arange(200_000))
    timers = disease_state.timer.astype(float)
    assert timers.mean() == pytest.approx(6.0, rel=0.02)
    assert timers.std() / timers.mean() == pytest.approx(1 / np.sqrt(shape), rel=0.03)


@pytest.mark.parametrize("step_days", [1.0, 1 / 24])
def test_mean_time_to_recovery_is_the_mean_phase_length_plus_half_a_step(step_days):
    model = PlainModel(step_days, infectious_days=3.0, duration_shape=4.0)
    disease_state, rng = new_state(model, population=20_000, cases=np.arange(20_000))
    recovered_at = np.full(20_000, np.nan)
    step = 0
    while np.isnan(recovered_at).any():
        advance(model, disease_state, rng)
        step += 1
        recovered_at[np.isnan(recovered_at) & (disease_state.state == 2)] = step * step_days
    assert recovered_at.mean() == pytest.approx(3.0 + step_days / 2, abs=0.05)


@pytest.mark.parametrize(("share", "expected"), [(0.0, 0.0), (0.3, 0.3), (1.0, 1.0)])
def test_infectious_people_stay_home_with_the_configured_probability(share, expected):
    model = PlainModel(ONE_DAY, stay_home=share)
    disease_state, _ = new_state(model, population=50_000, cases=np.arange(25_000))
    assert disease_state.stays_home[:25_000].mean() == pytest.approx(expected, abs=0.01)
    assert not disease_state.stays_home[25_000:].any()


def test_people_stop_staying_home_when_they_stop_being_infectious():
    model = PlainModel(ONE_DAY, stay_home=1.0, infectious_days=0.5, duration_shape=50.0)
    disease_state, rng = new_state(model, cases=[0])
    assert disease_state.stays_home[0]
    for _ in range(10):
        advance(model, disease_state, rng)
    assert not disease_state.stays_home[0]


def test_mean_field_r0_from_contacts_and_infectious_time():
    model = LatentModel(ONE_DAY, transmission=0.5, incubation_days=3.0, infectious_days=6.0)
    assert model.infectious_days() == 6.0
    assert model.mean_field_r0(2.0) == pytest.approx(-math.log(0.5) * 2.0 * 6.0)
    assert LatentModel(ONE_DAY, **SURE).mean_field_r0(2.0) is None


@pytest.mark.parametrize("shape", [1.0, 4.0, 30.0])
def test_pair_transmission_probability_matches_sampled_infectious_periods(shape):
    model = PlainModel(ONE_DAY, transmission=0.3, infectious_days=5.0, duration_shape=shape)
    periods = np.random.default_rng(0).gamma(shape, 5.0 / shape, 400_000)
    hazard = -math.log(0.7)
    assert model.pair_transmission_probability() == pytest.approx(
        np.mean(1 - np.exp(-hazard * periods)), abs=0.003
    )


def test_pair_transmission_probability_edge_cases():
    assert PlainModel(ONE_DAY, **SURE).pair_transmission_probability() == 1.0
    assert PlainModel(ONE_DAY, transmission=0.0).pair_transmission_probability() == 0.0


def test_quiescence_means_nobody_is_in_a_timed_phase():
    model = PlainModel(ONE_DAY)
    disease_state, _ = new_state(model, population=10)
    assert model.is_quiescent(disease_state)
    disease_state, _ = new_state(model, population=10, cases=[1])
    assert not model.is_quiescent(disease_state)


def test_certain_transmission_still_infects_people_with_any_susceptibility():
    model = PlainModel(ONE_DAY, **SURE, susceptibility_cv=2.0)
    disease_state, rng = new_state(model, population=1000)
    advance(model, disease_state, rng, exposure=0.01)
    assert disease_state.cumulative_infections == 1000


def test_susceptibility_varies_between_people_around_one():
    model = PlainModel(ONE_DAY, susceptibility_cv=0.5)
    disease_state, _ = new_state(model, population=200_000)
    values = disease_state.susceptibility.astype(float)
    assert values.mean() == pytest.approx(1.0, rel=0.02)
    assert values.std() / values.mean() == pytest.approx(0.5, rel=0.05)


def test_infectiousness_follows_a_gamma_with_mean_one_and_the_given_dispersion():
    model = PlainModel(ONE_DAY, infectiousness_dispersion=0.25)
    disease_state, _ = new_state(model, population=200_000)
    values = disease_state.infectiousness.astype(float)
    assert values.mean() == pytest.approx(1.0, rel=0.02)
    assert values.var() == pytest.approx(1 / 0.25, rel=0.05)
    top_fifth = np.sort(values)[-len(values) // 5 :]
    assert np.median(values) < 0.2 and top_fifth.sum() / values.sum() > 0.7, "superspreaders"


def test_everybody_spreads_equally_without_dispersion():
    disease_state, _ = new_state(PlainModel(ONE_DAY), population=100)
    assert set(disease_state.infectiousness.tolist()) == {1.0}


def test_age_effect_scales_susceptibility_of_the_age_range():
    effects = {"susceptibility": [[0, 15, 0.5], [65, 120, 2.0]]}
    model = PlainModel(ONE_DAY, age_effects=effects, **UNIFORM)
    disease_state, _ = new_state(model, population=3, age=[10, 40, 70])
    assert disease_state.susceptibility.tolist() == [0.5, 1.0, 2.0]


def test_invalid_age_effects_are_rejected():
    with pytest.raises(ValueError, match="Unknown age effects"):
        PlainModel(ONE_DAY, age_effects={"mood": []})
    with pytest.raises(ValueError, match="Invalid age ranges"):
        PlainModel(ONE_DAY, age_effects={"fatality": [[50, 40, 1.0]]})
    with pytest.raises(ValueError, match="Invalid age ranges"):
        PlainModel(ONE_DAY, age_effects={"fatality": [[0, 40, -1.0]]})


def finish_infection(model, disease_state, rng, days=20):
    for _ in range(days):
        advance(model, disease_state, rng)


def test_nobody_dies_without_fatality():
    model = PlainModel(ONE_DAY, infectious_days=2.0)
    disease_state, rng = new_state(model, population=2000, cases=np.arange(2000))
    finish_infection(model, disease_state, rng)
    assert not (disease_state.state == model.dead_index).any()


def test_fatality_is_the_share_of_infections_ending_in_death():
    model = PlainModel(ONE_DAY, infectious_days=2.0, fatality=0.2)
    disease_state, rng = new_state(model, population=50_000, cases=np.arange(50_000))
    finish_infection(model, disease_state, rng)
    assert (disease_state.state == model.dead_index).mean() == pytest.approx(0.2, abs=0.01)


def test_fatality_depends_on_age():
    effects = {"fatality": [[0, 15, 0.0], [65, 120, 5.0]]}
    model = PlainModel(ONE_DAY, infectious_days=2.0, fatality=0.1, age_effects=effects)
    ages = np.repeat([5, 40, 80], 20_000)
    disease_state, rng = new_state(model, population=60_000, cases=np.arange(60_000), age=ages)
    finish_infection(model, disease_state, rng)
    dead = (disease_state.state == model.dead_index).reshape(3, 20_000).mean(axis=1)
    assert dead[0] == 0.0
    assert dead[1] == pytest.approx(0.1, abs=0.01)
    assert dead[2] == pytest.approx(0.5, abs=0.015)


def test_the_dead_stay_dead_and_cannot_be_infected():
    model = PlainModel(ONE_DAY, **SURE, infectious_days=0.5, fatality=1.0, duration_shape=50.0)
    disease_state, rng = new_state(model, cases=[0])
    finish_infection(model, disease_state, rng, days=5)
    assert state_names(model, disease_state) == ["dead"]
    advance(model, disease_state, rng, exposure=10)
    assert state_names(model, disease_state) == ["dead"]
    assert disease_state.cumulative_infections == 1


def test_immunity_wanes_after_the_configured_time():
    model = PlainModel(
        ONE_DAY, infectious_days=1.0, immunity_days=30.0, duration_shape=50.0, **SURE
    )
    disease_state, rng = new_state(model, population=500, cases=np.arange(500))
    finish_infection(model, disease_state, rng, days=15)
    assert (disease_state.state == model._index[RECOVERED]).all(), "recovered and still immune"
    finish_infection(model, disease_state, rng, days=40)
    assert not disease_state.state.any(), "everybody is susceptible again"


def test_lifelong_immunity_never_wanes():
    model = PlainModel(ONE_DAY, infectious_days=1.0, duration_shape=50.0)
    disease_state, rng = new_state(model, cases=[0])
    finish_infection(model, disease_state, rng, days=2000)
    assert state_names(model, disease_state) == [RECOVERED]


def test_partial_protection_reduces_the_hazard_of_immune_people():
    model = PlainModel(
        ONE_DAY, infectious_days=1.0, immunity_protection=0.75, transmission=0.5, **UNIFORM
    )
    immune = np.full(200_000, model._index[RECOVERED], np.int8)
    disease_state, rng = new_state(model, population=200_000)
    disease_state.state[:] = immune
    advance(model, disease_state, rng, exposure=1)
    assert disease_state.cumulative_infections / 200_000 == pytest.approx(1 - 0.5**0.25, abs=0.01)
    assert (disease_state.state == 1).sum() == disease_state.cumulative_infections


def test_complete_protection_means_no_reinfection():
    model = PlainModel(ONE_DAY, **SURE, immunity_protection=1.0)
    disease_state, rng = new_state(model, population=100)
    disease_state.state[:] = model._index[RECOVERED]
    advance(model, disease_state, rng, exposure=10)
    assert disease_state.cumulative_infections == 0


def test_stay_at_home_and_fewer_outings_are_drawn_independently():
    model = PlainModel(ONE_DAY, stay_home=0.5, limit_outings=0.2)
    disease_state, _ = new_state(model, population=100_000, cases=np.arange(100_000))
    assert disease_state.stays_home.mean() == pytest.approx(0.5, abs=0.01)
    assert disease_state.limits_outings.mean() == pytest.approx(0.2, abs=0.01)


def test_burying_makes_people_dead_for_this_disease_without_counting_a_death_of_it():
    model = PlainModel(ONE_DAY, stay_home=1.0, limit_outings=1.0)
    disease_state, _ = new_state(model, population=5, cases=[0, 1])
    model.bury(disease_state, np.array([1, 3]))
    assert state_names(model, disease_state) == [
        "infectious",
        "dead",
        "susceptible",
        "dead",
        "susceptible",
    ]
    assert not disease_state.stays_home[1] and not disease_state.limits_outings[1]
    assert disease_state.deaths == 0


def test_deaths_of_the_disease_are_counted():
    model = PlainModel(ONE_DAY, infectious_days=2.0, fatality=1.0)
    disease_state, rng = new_state(model, population=300, cases=np.arange(300))
    finish_infection(model, disease_state, rng)
    assert disease_state.deaths == 300


@pytest.mark.parametrize("shape", [1.0, 2.0])
def test_lifelong_immunity_never_wanes_whatever_the_regularity(shape):
    model = PlainModel(ONE_DAY, duration_shape=shape)
    disease_state, rng = new_state(model, population=100_000)
    place(model, disease_state, rng, np.arange(100_000), RECOVERED)
    for _ in range(1000):
        advance(model, disease_state, rng)
    assert (disease_state.state == list(model.STATE_COLORS).index(RECOVERED)).all()


def test_seir_people_die_only_at_the_end_of_the_infectious_period():
    model = LatentModel(
        ONE_DAY, fatality=1.0, incubation_days=3.0, infectious_days=4.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model, population=200)
    place(model, disease_state, rng, np.arange(200), EXPOSED)
    dead_by_day = []
    for _ in range(12):
        advance(model, disease_state, rng)
        dead_by_day.append(int((disease_state.state == model.dead_index).sum()))
    assert dead_by_day[:4] == [0] * 4, "nobody dies while exposed or early in the illness"
    assert dead_by_day[-1] == 200


def test_sir_deaths_happen_only_when_the_infection_ends_not_when_immunity_ends():
    model = PlainModel(
        ONE_DAY, fatality=0.5, infectious_days=2.0, immunity_days=10.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model, population=5000, cases=np.arange(5000))
    for _ in range(6):
        advance(model, disease_state, rng)
    deaths_after_infection = disease_state.deaths
    assert deaths_after_infection == pytest.approx(2500, abs=4 * 35)
    for _ in range(60):
        advance(model, disease_state, rng)
    assert disease_state.deaths == deaths_after_infection
