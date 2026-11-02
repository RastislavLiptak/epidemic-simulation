import dataclasses
import math
from datetime import date

import numba
import numpy as np
import pytest
from config_factory import (
    LIBRARY,
    disease_definition,
    interaction,
    make_model,
)

SURE = {"transmission": 1.0, "infectious_days": 60.0}
UNIFORM = {"susceptibility_cv": 0.0}
SPACING = 12.0


def three_diseases() -> dict:
    return {
        "a": disease_definition("sir", 3, transmission=0.5),
        "b": disease_definition("seir", 3, transmission=0.5),
        "c": disease_definition("sis", 3, transmission=0.5),
    }


def run(model, steps: int):
    for _ in range(steps):
        model.step()
    return model


def arranged_model(
    diseases: dict, positions: dict[int, tuple[float, float]], interactions=(), **world
):
    """Everybody at home on a grid too sparse for contacts; `positions` moves some people.

    Homes are moved too, so nobody walks away during the night hours the tests run in.
    """
    model = make_model(diseases, interactions, population=100, arena_size=500, **world)
    for index in range(model.population):
        model.x[index], model.y[index] = 5 + SPACING * (index % 8), 5 + SPACING * (index // 8)
    for index, (x, y) in positions.items():
        model.x[index], model.y[index] = x, y
    model.town.home_x[:], model.town.home_y[:] = model.x, model.y
    model._target_x[:], model._target_y[:] = model.x, model.y
    return model


def put_into_state(model, person: int, disease: str, state: str) -> None:
    """Put `person` into `state` of `disease` for far longer than any test runs."""
    disease_state = model.disease_states[disease]
    disease_state.state[person] = list(model.diseases[disease].STATE_COLORS).index(state)
    disease_state.timer[person] = 1e6


def make_infectious(model, person: int, disease: str) -> None:
    put_into_state(model, person, disease, model.diseases[disease].INITIAL_CASE_STATE)


def state_of(model, person: int, disease: str) -> str:
    return list(model.diseases[disease].STATE_COLORS)[model.disease_states[disease].state[person]]


def test_population_and_initial_cases():
    model = make_model(three_diseases())
    assert model.population == len(model.x) == 200
    assert [sum(counts[1:]) for counts in model.state_counts().values()] == [3, 3, 3]


def test_everybody_starts_at_home():
    model = make_model(three_diseases())
    assert np.array_equal(model.x, model.town.home_x) and np.array_equal(model.y, model.town.home_y)


def test_state_counts_sum_to_population_and_states_are_valid():
    model = run(make_model(three_diseases()), 5 * 24)
    for name, counts in model.state_counts().items():
        assert sum(counts) == 200 and len(counts) == len(model.diseases[name].STATE_COLORS)
        assert model.disease_states[name].state.max() < len(counts)


def test_people_stay_inside_the_arena():
    model = make_model(three_diseases(), arena_size=300)
    for _ in range(24 * 3):
        model.step()
        assert (
            model.x.min() >= 0
            and model.x.max() <= 300
            and model.y.min() >= 0
            and model.y.max() <= 300
        )


def test_days_weekdays_and_hours_are_counted_from_monday_midnight():
    model = make_model(three_diseases(), steps_per_day=8)
    assert (model.weekday, model.hour, model.day) == (0, 0.0, 0.0)
    run(model, 8 * 5 + 3)
    assert (model.weekday, model.hour) == (5, 9.0)
    assert model.steps == 43 and model.day == pytest.approx(43 / 8)
    assert [record["day"] for record in model.history] == [0, 1, 2, 3, 4, 5]


def test_each_run_has_its_own_town_and_stays_reproducible_from_the_seed():
    first, second = make_model(seed=1), make_model(seed=2)
    assert not np.array_equal(first.town.age, second.town.age)
    assert np.array_equal(first.town.age, make_model(seed=1).town.age)


def distances(model, x, y, people):
    return np.hypot(model.x - x, model.y - y)[people]


def test_most_workers_are_at_work_at_midday_on_a_weekday():
    model = make_model(three_diseases(), population=1000, arena_size=1500)
    town = model.town
    run(model, 14)  # 14:00 on Monday
    works = np.isin(town.role, (4,)) & ~town.at_school
    present = distances(model, town.work_x, town.work_y, works) < 1e-6
    assert present.mean() > 0.7, "the rest are absent, at lunch, on the way or late"
    assert present.mean() < 0.99, "people do not all behave alike"


def test_workers_do_not_all_arrive_at_the_same_time():
    model = make_model(three_diseases(), population=1500, arena_size=1500)
    town = model.town
    works = np.flatnonzero((town.role == 4) & ~town.at_school)
    arrival = np.full(len(works), np.nan)
    for hour in range(4, 12):
        model.step()
        here = distances(model, town.work_x, town.work_y, works) < 1e-6
        arrival[np.isnan(arrival) & here] = hour
    assert len(np.unique(arrival[~np.isnan(arrival)])) >= 4


@pytest.mark.parametrize("steps_per_day", [4, 24, 96])
def test_everybody_who_is_done_for_the_day_is_home_at_night_whatever_the_time_resolution(
    steps_per_day,
):
    model = make_model(three_diseases(), steps_per_day=steps_per_day, arena_size=2000)
    run(model, math.ceil((24 + 4) * steps_per_day / 24) + 1)  # Tuesday morning
    idle = np.array([model.current_code(person) == 0 for person in range(model.population)])
    assert idle.sum() > 20
    assert distances(model, model.town.home_x, model.town.home_y, idle).max() < 1e-6


def test_only_a_few_go_to_work_at_the_weekend_and_nobody_to_school():
    model = make_model(three_diseases(), population=2000, arena_size=2000)
    town = model.town
    run(model, 5 * 24 + 12)  # Saturday noon
    works_elsewhere = np.hypot(town.work_x - town.home_x, town.work_y - town.home_y) > 50
    at_work = np.hypot(model.x - town.work_x, model.y - town.work_y) < 1
    workers = (town.role == 4) & ~town.at_school & works_elsewhere
    assert 0.02 < at_work[workers].mean() < 0.3
    assert not (at_work & town.at_school & works_elsewhere).any()


def test_nobody_goes_to_school_in_the_christmas_holidays():
    model = make_model(three_diseases(), population=1000, arena_size=1500)
    town = model.town
    calendar = model.town_parameters.schedule.calendar
    days_to_christmas_eve_eve = (date(2018, 12, 27) - calendar.first_day).days
    assert calendar.closed_places(days_to_christmas_eve_eve)
    model.steps = days_to_christmas_eve_eve * model.world["steps_per_day"]
    run(model, 12)
    pupils = np.isin(town.role, (1, 2)) & (town.home_x != town.work_x)
    at_school = np.hypot(model.x - town.work_x, model.y - town.work_y) < 1
    assert not at_school[pupils].any()


def test_people_travel_no_faster_than_their_personal_speed():
    model = make_model(three_diseases(), population=1000, arena_size=1500)
    run(model, 6)
    maximum = model.town.speed_kmh * 1000 * model.step_hours
    for _ in range(4):
        before = (model.x.copy(), model.y.copy())
        model.step()
        walked = np.hypot(model.x - before[0], model.y - before[1])
        assert (walked <= maximum + 1e-6).all()
    assert walked.max() > 100, "people do move in the morning"


def test_people_move_continuously_not_in_jumps():
    model = make_model(three_diseases(), population=1000, arena_size=1500, steps_per_day=96)
    run(model, 6 * 4)
    moving = []
    for _ in range(8 * 4):
        before = (model.x.copy(), model.y.copy())
        model.step()
        moving.append((np.hypot(model.x - before[0], model.y - before[1]) > 1e-9).mean())
    assert 0.02 < np.mean(moving) < 0.7, "a share of people is always on the way"
    assert min(moving) > 0.002, "no go and stop waves: somebody is moving in every step"


def test_infectious_people_who_stay_home_do_not_go_to_work():
    diseases = {
        "a": disease_definition("sir", 150, transmission=0.0, infectious_days=60, stay_home=1.0)
    }
    model = make_model(diseases, arena_size=2000)
    town = model.town
    run(model, 12)
    infectious = model.disease_states["a"].state == 1
    away = np.hypot(town.work_x - town.home_x, town.work_y - town.home_y) > 50
    candidates = infectious & away & (town.role == 4)
    assert candidates.sum() > 20
    assert distances(model, town.home_x, town.home_y, candidates).max() < 1e-6


def test_infectious_people_with_fewer_outings_still_work_but_skip_errands():
    def share_out(limit):
        diseases = {
            "a": disease_definition(
                "sir", 200, transmission=0.0, infectious_days=60, limit_outings=limit
            )
        }
        model = make_model(diseases, population=600, arena_size=1500)
        run(model, 5 * 24 + 13)  # Saturday afternoon
        outing = np.array([model.current_code(person) >= 2 for person in range(600)])
        infectious = model.disease_states["a"].state == 1
        away = np.hypot(model.x - model.town.home_x, model.y - model.town.home_y) > 1
        return (outing & infectious & away).sum(), (outing & infectious).sum()

    away_free, planned_free = share_out(0.0)
    away_limited, _ = share_out(1.0)
    assert planned_free > 10 and away_free > 5
    assert away_limited == 0


def test_staying_at_home_slows_the_spread():
    def attack_rate(stay_home):
        diseases = {"a": disease_definition("sir", 30, transmission=0.3, stay_home=stay_home)}
        model = run(make_model(diseases, population=1500, arena_size=1500), 24 * 30)
        return model.history[-1]["ever_infected"]["a"]

    assert attack_rate(1.0) < attack_rate(0.0)


def test_infection_spreads_only_within_contact_distance():
    model = arranged_model({"a": disease_definition("sir", 0, **SURE)}, {1: (5, 6.5)})
    make_infectious(model, 0, "a")
    model.step()
    assert state_of(model, 1, "a") == "infectious"
    assert model.state_counts()["a"] == [98, 2, 0, 0]


def test_far_people_are_not_infected():
    model = arranged_model({"a": disease_definition("sir", 0, **SURE)}, {})
    make_infectious(model, 0, "a")
    model.step()
    assert model.state_counts()["a"] == [99, 1, 0, 0]


def test_every_disease_has_its_own_contact_distance():
    diseases = {
        "near": disease_definition("sir", 0, contact_distance=2.0, **SURE),
        "far": disease_definition("sir", 0, contact_distance=6.0, **SURE),
    }
    model = arranged_model(diseases, {1: (5, 8.5)})
    make_infectious(model, 0, "near")
    make_infectious(model, 0, "far")
    model.step()
    assert state_of(model, 1, "near") == "susceptible"
    assert state_of(model, 1, "far") == "infectious"


def test_step_is_synchronous():
    positions = {0: (50, 50), 1: (51.5, 50), 2: (53, 50)}
    model = arranged_model({"a": disease_definition("sir", 0, **SURE)}, positions)
    make_infectious(model, 0, "a")
    model.step()
    assert [state_of(model, person, "a") for person in range(3)] == [
        "infectious",
        "infectious",
        "susceptible",
    ]
    model.step()
    assert state_of(model, 2, "a") == "infectious"


def test_diseases_spread_independently_in_the_same_people():
    diseases = {
        "a": disease_definition("sir", 0, **SURE),
        "b": disease_definition("sir", 0, **SURE),
    }
    model = arranged_model(diseases, {1: (5, 6.5)})
    make_infectious(model, 0, "a")
    model.step()
    assert (state_of(model, 1, "a"), state_of(model, 1, "b")) == ("infectious", "susceptible")


def test_one_person_can_carry_several_diseases():
    diseases = {
        "a": disease_definition("sir", 0, **SURE),
        "b": disease_definition("seir", 0, **SURE),
    }
    model = arranged_model(diseases, {1: (5, 6.5)})
    make_infectious(model, 0, "a")
    make_infectious(model, 1, "b")
    model.step()
    assert (state_of(model, 0, "a"), state_of(model, 0, "b")) == ("infectious", "exposed")
    assert (state_of(model, 1, "a"), state_of(model, 1, "b")) == ("infectious", "infectious")


def test_a_more_infectious_person_infects_more_often():
    def catch_rate(boost):
        caught = 0
        for seed in range(1500):
            model = arranged_model(
                {
                    "a": disease_definition(
                        "sir", 0, transmission=0.5, infectious_days=60, **UNIFORM
                    )
                },
                {1: (5, 6.5)},
                seed=seed,
            )
            make_infectious(model, 0, "a")
            model.disease_states["a"].infectiousness[0] = boost
            model.step()
            caught += state_of(model, 1, "a") == "infectious"
        return caught / 1500

    assert catch_rate(3.0) == pytest.approx(1 - 0.5 ** (3 / 24), abs=0.02)
    assert catch_rate(0.0) == 0


def test_the_dead_do_not_move_infect_or_get_infected():
    diseases = {"a": disease_definition("sir", 0, **SURE, fatality=1.0)}
    model = arranged_model(diseases, {1: (5, 6.5)})
    put_into_state(model, 1, "a", "dead")
    make_infectious(model, 0, "a")
    model.step()
    assert state_of(model, 1, "a") == "dead"
    put_into_state(model, 0, "a", "dead")
    put_into_state(model, 1, "a", "infectious")
    put_into_state(model, 2, "a", "susceptible")
    model.step()
    assert state_of(model, 0, "a") == "dead"


def test_dead_people_stay_where_they_died():
    diseases = {
        "a": disease_definition("sir", 600, transmission=0.0, fatality=1.0, infectious_days=1.0)
    }
    model = make_model(diseases, population=600, arena_size=1500, seed=3)
    run(model, 24 * 2)
    died_away = np.hypot(model.x - model.town.home_x, model.y - model.town.home_y) > 1
    run(model, 24 * 4)  # everybody has died by now
    dead = model.disease_states["a"].state == model.diseases["a"].dead_index
    assert dead.all()
    assert died_away.sum() > 5, "some are on their way or elsewhere when they die"
    before = (model.x.copy(), model.y.copy())
    run(model, 24)
    assert np.array_equal(model.x, before[0]) and np.array_equal(model.y, before[1])


def test_an_epidemic_with_fatality_kills_the_expected_share():
    diseases = {
        "a": disease_definition("sir", 300, transmission=0.0, fatality=0.3, infectious_days=3.0)
    }
    model = run(make_model(diseases, population=1500, arena_size=2000), 24 * 20)
    counts = model.state_counts()["a"]
    assert counts[3] / 300 == pytest.approx(0.3, abs=0.08)
    assert counts[1] == 0, "everybody has either recovered or died"


def test_immunity_that_fades_lets_the_disease_return():
    def infections(immunity_days):
        diseases = {
            "a": disease_definition(
                "sir", 40, transmission=0.4, infectious_days=6.0, immunity_days=immunity_days
            )
        }
        record = run(make_model(diseases, population=1500, arena_size=1500), 24 * 120).history[-1]
        return record["cumulative_infections"]["a"], record["ever_infected"]["a"]

    fading, fading_people = infections(20.0)
    lifelong, lifelong_people = infections(36500.0)
    assert lifelong == lifelong_people, "nobody is infected twice"
    assert fading > 2 * fading_people, "many are infected again"


def test_town_too_small_for_the_population_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        make_model(population=2000, arena_size=300)


# --- interactions -----------------------------------------------------------------------------

TWO_SURE = {"a": disease_definition("sir", 0, **SURE), "b": disease_definition("sir", 0, **SURE)}


def spread_to_neighbour(rule, source_state, interactions_target="b", repetitions=1):
    """After one step, how many of `repetitions` neighbours of an infectious person caught `b`."""
    caught = 0
    for repetition in range(repetitions):
        model = arranged_model(
            {
                "a": disease_definition("sir", 0, **SURE),
                "b": disease_definition("sir", 0, transmission=0.5, infectious_days=60, **UNIFORM),
            },
            {1: (5, 6.5)},
            [rule],
            seed=repetition,
        )
        make_infectious(model, 0, "b")
        if source_state is not None:
            put_into_state(model, 1, "a", source_state)
        model.step()
        caught += state_of(model, 1, interactions_target) == "infectious"
    return caught


def test_interaction_with_factor_zero_makes_the_target_disease_impossible():
    rule = interaction("a", ["recovered"], "b", "susceptibility", 0.0)
    assert spread_to_neighbour(rule, "recovered", repetitions=300) == 0


def test_interaction_does_nothing_outside_its_states():
    rule = interaction("a", ["recovered"], "b", "susceptibility", 0.0)
    assert spread_to_neighbour(rule, "susceptible", repetitions=300) > 0
    assert spread_to_neighbour(rule, "infectious", repetitions=300) > 0


def test_susceptibility_factor_scales_the_chance_of_catching_the_disease():
    def catch_rate(factor):
        rule = interaction("a", ["infectious"], "b", "susceptibility", factor)
        return spread_to_neighbour(rule, "infectious", repetitions=1500) / 1500

    base_chance = 1 - 0.5 ** (1 / 24)
    assert catch_rate(1.0) == pytest.approx(base_chance, abs=0.012)
    assert catch_rate(3.0) == pytest.approx(1 - 0.5 ** (3 / 24), abs=0.018)
    assert catch_rate(0.3) == pytest.approx(1 - 0.5 ** (0.3 / 24), abs=0.01)


def test_infectiousness_factor_scales_how_much_a_sick_person_infects():
    def catch_rate(factor):
        caught = 0
        for seed in range(1500):
            rule = interaction("a", ["infectious"], "b", "infectiousness", factor)
            model = arranged_model(
                {
                    "a": disease_definition("sir", 0, **SURE),
                    "b": disease_definition(
                        "sir", 0, transmission=0.5, infectious_days=60, **UNIFORM
                    ),
                },
                {1: (5, 6.5)},
                [rule],
                seed=seed,
            )
            make_infectious(model, 0, "b")
            put_into_state(model, 0, "a", "infectious")
            model.step()
            caught += state_of(model, 1, "b") == "infectious"
        return caught / 1500

    assert catch_rate(0.0) == 0
    assert catch_rate(2.0) == pytest.approx(1 - 0.5 ** (2 / 24), abs=0.015)


def test_infectiousness_interaction_only_acts_on_people_in_the_source_state():
    model = arranged_model(
        {"a": disease_definition("sir", 0, **SURE), "b": disease_definition("sir", 0, **SURE)},
        {1: (5, 6.5)},
        [interaction("a", ["infectious"], "b", "infectiousness", 0.0)],
    )
    make_infectious(model, 0, "b")
    model.step()
    assert state_of(model, 1, "b") == "infectious"


def test_several_interactions_multiply():
    rules = [
        interaction("a", ["infectious"], "b", "susceptibility", 0.5),
        interaction("c", ["infectious"], "b", "susceptibility", 0.2),
    ]
    diseases = {**TWO_SURE, "c": disease_definition("sir", 0, **SURE)}
    model = arranged_model(diseases, {}, rules)
    put_into_state(model, 5, "a", "infectious")
    put_into_state(model, 5, "c", "infectious")
    put_into_state(model, 6, "a", "infectious")
    susceptibility = model._relative_susceptibility("b")
    assert susceptibility[5] == pytest.approx(0.1) and susceptibility[6] == pytest.approx(0.5)
    assert susceptibility[7] == 1.0


def test_interactions_see_the_states_at_the_start_of_the_step():
    rule = interaction("a", ["infectious"], "b", "susceptibility", 0.0)
    model = arranged_model(
        {"a": disease_definition("sir", 0, **SURE), "b": disease_definition("sir", 0, **SURE)},
        {1: (5, 6.5)},
        [rule],
    )
    make_infectious(model, 0, "b")
    model.step()
    assert state_of(model, 1, "b") == "infectious"
    assert state_of(model, 1, "a") == "susceptible"


def test_cross_immunity_reduces_the_epidemic_of_the_protected_disease():
    def attack_rate(interactions):
        diseases = {
            "a": disease_definition("sir", 40, transmission=0.4, **UNIFORM),
            "b": disease_definition("sir", 40, transmission=0.4, **UNIFORM),
        }
        model = run(make_model(diseases, interactions, population=1500, arena_size=1500), 24 * 60)
        return model.history[-1]["ever_infected"]

    protected = attack_rate([interaction("a", ["recovered"], "b", "susceptibility", 0.1)])
    free = attack_rate([])
    assert protected["b"] < free["b"]


def test_interaction_effect_is_directional():
    def totals(interactions):
        diseases = {
            "a": disease_definition("sir", 40, transmission=0.4, **UNIFORM),
            "b": disease_definition("sir", 40, transmission=0.4, **UNIFORM),
        }
        model = run(make_model(diseases, interactions, population=1500, arena_size=1500), 24 * 60)
        return model.history[-1]["ever_infected"]

    boosted = totals([interaction("a", ["infectious", "recovered"], "b", "susceptibility", 5.0)])
    assert boosted["b"] > totals([])["b"]


# --- reproducibility and correctness against theory -------------------------------------------


def snapshot(model):
    return [
        model.x.copy(),
        model.y.copy(),
        *(state.state.copy() for state in model.disease_states.values()),
    ]


def same(first, second):
    return all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_same_seed_gives_same_course_and_other_seed_another():
    interactions = [interaction("a", ["recovered"], "b", "susceptibility", 0.3)]
    first = run(make_model(three_diseases(), interactions), 100)
    second = run(make_model(three_diseases(), interactions), 100)
    other = run(make_model(three_diseases(), interactions, seed=2), 100)
    assert same(snapshot(first), snapshot(second)) and not same(snapshot(first), snapshot(other))


def test_course_does_not_depend_on_the_number_of_threads():
    interactions = [interaction("a", ["recovered"], "b", "susceptibility", 0.3)]
    kwargs = {"population": 3000, "arena_size": 1500}
    threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single = run(make_model(three_diseases(), interactions, **kwargs), 80)
        numba.set_num_threads(threads)
        parallel = run(make_model(three_diseases(), interactions, **kwargs), 80)
    finally:
        numba.set_num_threads(threads)
    assert same(snapshot(single), snapshot(parallel))


def isolated_pairs_town():
    """A town of two-person households whose members stay at home all the time."""
    nobody = np.zeros_like(LIBRARY.town.population.employment)
    population = dataclasses.replace(
        LIBRARY.town.population,
        fingerprint="isolated pairs",
        household_sizes=np.array([2]),
        household_weights=np.array([1.0]),
        kinds={2: (("couple",), np.array([1.0]))},
        kind_names=("couple",),
        kind_shares=np.array([1.0]),
        preschool=nobody,
        school=nobody,
        university=nobody,
        employment=nobody,
        employment_bands=np.zeros((0, 2), int),
        care_home_rates=nobody,
    )
    schedule = LIBRARY.town.schedule
    quiet = tuple(
        dataclasses.replace(activity, probability=np.zeros_like(activity.probability))
        for activity in schedule.activities
    )
    return dataclasses.replace(
        LIBRARY.town,
        fingerprint="isolated pairs",
        population=population,
        schedule=dataclasses.replace(schedule, activities=quiet),
    )


def test_transmission_within_households_matches_the_exact_probability():
    diseases = {
        "a": disease_definition(
            "sir", 2000, transmission=0.3, infectious_days=5.0, duration_shape=4.0, **UNIFORM
        )
    }
    model = make_model(
        diseases,
        population=20_000,
        arena_size=8000,
        steps_per_day=12,
        town=isolated_pairs_town(),
    )
    initial = model.disease_states["a"].ever_infected.copy()
    run(model, 12 * 40)
    ever = model.disease_states["a"].ever_infected
    partner_of = np.arange(20_000) ^ 1
    partners_at_risk = initial[partner_of] & ~initial
    expected = model.diseases["a"].pair_transmission_probability()
    assert partners_at_risk.sum() > 1500
    assert ever[partners_at_risk].mean() == pytest.approx(expected, abs=0.035)
    untouched_households = ~initial & ~initial[partner_of]
    assert ever[untouched_households].mean() < 0.02


def test_a_short_outing_is_not_skipped_when_it_fits_into_one_step():
    from epidemic_sandbox.schedule import FIRST_ACTIVITY, HOME

    model = make_model(three_diseases(), population=300, arena_size=900, steps_per_day=48)
    model.step()  # plans Monday
    town, schedule = model.town, model.schedule
    person = 7
    shop = model.town_parameters.schedule.activity_index("shopping")
    schedule.times[person] = np.inf
    schedule.codes[person] = HOME
    schedule.times[person, :2] = [15.56, 15.81]  # 15 minutes, inside the step 15:30-16:00
    schedule.codes[person, :2] = [FIRST_ACTIVITY + shop, HOME]
    schedule.choice[person, shop] = 0
    model._pointer[person] = 0
    town.speed_kmh[person] = 1000.0
    radius = town.activity_radius[shop][person, 0]
    destination = (
        town.activity_x[shop][person, 0] + radius * schedule.spot_x[person],
        town.activity_y[shop][person, 0] + radius * schedule.spot_y[person],
    )
    seen = []
    while model.hour < 17:
        model.step()
        seen.append(
            bool(
                np.hypot(model.x[person] - destination[0], model.y[person] - destination[1]) < 1e-6
            )
        )
    assert any(seen), "the person never was at the shop"
    assert (model.x[person], model.y[person]) == (town.home_x[person], town.home_y[person])


def killing_and_harmless():
    return {
        "killer": disease_definition(
            "sir", 0, transmission=1.0, infectious_days=1.0, fatality=1.0, duration_shape=50.0
        ),
        "other": disease_definition("sir", 0, transmission=1.0, infectious_days=60.0),
    }


def about_to_die(model, people):
    """Infectious with the killer disease and out of time: they die in the next step."""
    for person in people:
        make_infectious(model, person, "killer")
        model.disease_states["killer"].timer[person] = 0.01


def test_somebody_who_died_of_one_disease_is_dead_for_every_disease():
    model = arranged_model(killing_and_harmless(), {})
    make_infectious(model, 3, "other")
    about_to_die(model, [3])
    model.step()
    assert state_of(model, 3, "killer") == "dead"
    assert state_of(model, 3, "other") == "dead", "an infectious person is not infectious any more"
    assert model.state_counts()["other"][1] == 0
    assert not model.disease_states["other"].stays_home[3]


def test_a_person_dead_of_one_disease_cannot_catch_or_spread_another():
    model = arranged_model(killing_and_harmless(), {1: (5, 6.5), 9: (50, 50), 10: (51, 50)})
    model.disease_states["killer"].state[1] = model.diseases["killer"].dead_index
    model.disease_states["killer"].state[9] = model.diseases["killer"].dead_index
    model._bury_the_dead()
    make_infectious(model, 0, "other")
    make_infectious(model, 10, "other")
    model.step()
    assert state_of(model, 1, "other") == "dead", "a dead neighbour is not infected"
    assert state_of(model, 9, "other") == "dead"
    assert state_of(model, 10, "other") == "infectious"


def test_deaths_are_counted_for_the_disease_that_caused_them():
    model = arranged_model(killing_and_harmless(), {})
    about_to_die(model, [1, 2, 3])
    model.step()
    assert model.disease_states["killer"].deaths == 3
    assert model.disease_states["other"].deaths == 0
    counts = model.state_counts()
    assert counts["killer"][-1] == 3
    assert counts["other"][-1] == 0, "the victims of another disease are not its dead"
    run(model, 23)
    assert model.history[-1]["deaths"] == {"killer": 3, "other": 0}
    assert model.history[-1]["counts"]["other"][-1] == 0


def two_killers():
    return {
        name: disease_definition(
            "sir", 0, transmission=1.0, infectious_days=1.0, fatality=1.0, duration_shape=50.0
        )
        for name in ("x", "y")
    }


@pytest.mark.parametrize("seed", range(6))
def test_a_person_killed_by_two_diseases_in_one_step_dies_once(seed):
    model = arranged_model(two_killers(), {}, seed=seed)
    for disease in ("x", "y"):
        make_infectious(model, 3, disease)
        model.disease_states[disease].timer[3] = 0.01
    model.step()
    deaths = {name: state.deaths for name, state in model.disease_states.items()}
    assert sum(deaths.values()) == 1, f"one person died, but deaths are {deaths}"
    assert state_of(model, 3, "x") == state_of(model, 3, "y") == "dead"


@pytest.mark.parametrize("seed", range(6))
def test_somebody_who_dies_in_a_step_is_not_infected_by_another_disease_in_it(seed):
    model = arranged_model(killing_and_harmless(), {1: (5, 6.5)}, seed=seed)
    make_infectious(model, 0, "other")
    about_to_die(model, [1])
    model.step()
    other = model.disease_states["other"]
    assert state_of(model, 1, "other") == "dead"
    assert other.cumulative_infections == 0 and not other.ever_infected[1]


def reinfection_model(protection: float):
    disease = disease_definition(
        "sir",
        0,
        transmission=1.0,
        infectious_days=60.0,
        immunity_days=36500,
        immunity_protection=protection,
        duration_shape=50.0,
    )
    model = arranged_model({"a": disease}, {1: (5, 6.5)})
    make_infectious(model, 0, "a")
    put_into_state(model, 1, "a", "recovered")
    return model


def test_a_person_with_no_protection_left_is_reinfected_by_a_close_infectious_person():
    model = run(reinfection_model(0.0), 24)
    assert state_of(model, 1, "a") == "infectious"
    assert model.disease_states["a"].cumulative_infections >= 1


def test_complete_protection_keeps_a_recovered_person_healthy():
    model = run(reinfection_model(1.0), 24)
    assert state_of(model, 1, "a") == "recovered"
