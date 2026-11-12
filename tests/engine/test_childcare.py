"""Who looks after the small children: an adult of the household who stays in, the grandparents,
or the guardian; children join the guardian's outings when they are fetched."""

import numpy as np
from config_factory import disease_definition, make_model

from epidemic_sandbox.world.schedule import (
    FIRST_ACTIVITY,
    WORK,
    WORK_EVENTS,
    _accompany,
    draw_day_schedule,
)

DISEASE = {"a": disease_definition("sirs", 0)}


def town_model(population=4000, seed=1):
    return make_model(DISEASE, population=population, area_km2=2.0, seed=seed)


def day(model, rng_seed=0, weekend=False, closed=()):
    params = model.town_parameters.schedule
    return draw_day_schedule(
        model.town.role, model._traits, params, weekend, np.random.default_rng(rng_seed), closed
    )


def outings_from(schedule, person) -> list[float]:
    times, codes = schedule.times[person], schedule.codes[person]
    return [t for t, c in zip(times, codes, strict=True) if np.isfinite(t) and c >= FIRST_ACTIVITY]


def test_a_child_joins_an_outing_of_its_guardian_that_began_before_it_was_fetched():
    times = np.full((2, WORK_EVENTS + 3), np.inf)
    codes = np.zeros((2, WORK_EVENTS + 3), np.int8)
    times[0, WORK_EVENTS : WORK_EVENTS + 2] = [15.0, 17.0]  # the guardian shops 15-17
    codes[0, WORK_EVENTS] = FIRST_ACTIVITY
    _accompany(times, codes, np.array([1]), np.array([0]), np.array([16.0]))
    assert times[1, WORK_EVENTS : WORK_EVENTS + 2].tolist() == [16.0, 17.0]
    _accompany(times, codes, np.array([1]), np.array([0]), np.array([16.9]))
    assert np.isinf(times[1, WORK_EVENTS : WORK_EVENTS + 2]).all(), "too little of it left"


def test_the_adult_who_looks_after_a_child_at_home_goes_out_only_when_the_guardian_is_back():
    model = town_model()
    town = model.town
    schedule = day(model, closed=(1, 2))  # kindergartens and schools closed: the children at home
    care = model.town_parameters.schedule.childcare
    wards = np.flatnonzero((town.guardian >= 0) & (town.age < care.alone_all_day_from))
    attends = schedule.attends
    checked = 0
    for child in wards:
        guardian = town.guardian[child]
        if not attends[guardian] or schedule.cared[child]:
            continue
        codes, times = schedule.codes[guardian], schedule.times[guardian]
        last_work = np.flatnonzero((codes == WORK) & np.isfinite(times)).max()
        back = times[last_work + 1]  # from work home
        members = np.flatnonzero(town.household == town.household[child])
        free = [m for m in members if town.age[m] >= 18 and not attends[m] and m != guardian]
        assert free, "the guardian works only when somebody is at home with the child"
        assert any(min(outings_from(schedule, m), default=np.inf) >= back for m in free)
        checked += 1
    assert checked > 5


def test_children_with_grandparent_care_are_as_many_as_the_survey_says_where_the_town_allows():
    model = town_model(population=8000)
    town, traits = model.town, model._traits
    care = model.town_parameters.schedule.childcare
    wards = np.flatnonzero(town.guardian >= 0)
    has_grandparents = (town.visit_kin[wards] == 1).any(axis=1)
    cared = traits.care_hours[wards] > 0
    assert (traits.care_host[wards][cared] >= 0).all()
    assert not (cared & ~has_grandparents).any(), "only children with grandparents in the town"
    for start, end, share in care.grandparent_care:
        band = (town.age[wards] >= start) & (town.age[wards] < end) & has_grandparents
        if band.sum() > 50:
            ages = (town.age[wards] >= start) & (town.age[wards] < end)
            expected = min(1.0, share / has_grandparents[ages].mean())
            assert abs(cared[band].mean() - expected) < 0.1
    hours = traits.care_hours[wards][cared]
    assert 6.0 < hours.mean() < 11.0


def test_a_child_at_its_grandparents_goes_there_after_its_place_and_one_of_them_stays_in():
    model = town_model(population=8000)
    town, traits = model.town, model._traits
    visit = int(model._visits[0])
    found = 0
    for seed in range(4):
        schedule = day(model, rng_seed=seed)
        for child in np.flatnonzero(schedule.cared)[:40]:
            codes, times = schedule.codes[child], schedule.times[child]
            at_grandparents = np.flatnonzero(codes == FIRST_ACTIVITY + visit)
            assert len(at_grandparents)
            assert schedule.choice[child, visit] == traits.care_choice[child]
            begins = times[at_grandparents[0]]
            if schedule.attends[child]:
                end_of_place = times[np.flatnonzero((codes == WORK) & np.isfinite(times))[0]]
                assert begins > end_of_place
            host = traits.care_host[child]
            members = np.flatnonzero(town.household == host)
            free = members[~schedule.attends[members] & (town.age[members] >= 18)]
            assert len(free), "somebody of the grandparents' household is free"
            found += 1
    assert found > 20
