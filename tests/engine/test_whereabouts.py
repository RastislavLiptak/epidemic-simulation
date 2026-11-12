"""Where people are on weekdays against the national travel survey (Cesko v pohybu, Kouril,
Simecek, Dytrt 2022, one workday of everybody 6+ in 9,000 households, 2017-2019): days without a
trip, who is at home at 10 h, and the time on the way."""

import numpy as np
import pytest
from config_factory import disease_definition, make_model

from epidemic_sandbox.world.population import RESIDENT, ROLES
from epidemic_sandbox.world.schedule import HOME

# Tables 34-39: "necestuje" (no trip that day) and "doma" at its minimum, at 10 h, plus "necestuje"
NO_TRIP = {"under 18": 0.133, "active 18-64": 0.099, "inactive 18-64": 0.254, "65+": 0.301}
AT_HOME_AT_10 = {"under 18": 0.207, "active 18-64": 0.229, "inactive 18-64": 0.569, "65+": 0.622}
# Tables 32 and 28: minutes on the way of a day of a person who travelled, times the share of
# people who travelled
AGE_BANDS = [(6, 19), (19, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 128)]
ON_THE_WAY = [66 * 0.837, 88 * 0.845, 81 * 0.858, 89 * 0.876, 83 * 0.854, 75 * 0.769, 70 * 0.666]
WEEKS = (5, 12, 20, 30, 36, 44)  # the school year with a week of the summer holidays


@pytest.fixture(scope="module")
def observed():
    diseases = {"a": disease_definition("sirs", 0)}
    model = make_model(diseases, population=6000, area_km2=8.0, seed=4, steps_per_day=48)
    town = model.town
    private = town.role != RESIDENT
    active = np.isin(town.role, [ROLES.index("worker"), ROLES.index("remote")])
    groups = {
        "under 18": private & (town.age >= 6) & (town.age < 18),
        "active 18-64": private & (town.age >= 18) & (town.age < 65) & active,
        "inactive 18-64": private & (town.age >= 18) & (town.age < 65) & ~active,
        "65+": private & (town.age >= 65),
    }
    bands = [private & (town.age >= low) & (town.age < high) for low, high in AGE_BANDS]
    counts = {name: np.zeros(4) for name in groups}  # no trip, days, at home at 10, samples
    way, band_days = np.zeros(len(bands)), np.zeros(len(bands))
    first_vehicle = model._vehicles.first_venue
    for week in WEEKS:
        model.steps = week * 7 * 48
        for day in range(7):
            left = np.zeros(model.population, np.bool_)
            for step in range(48):
                model.step()
                inside = (model.venue >= 0) & (model.venue < first_vehicle)
                left |= model._code != HOME
                for index, members in enumerate(bands):
                    way[index] += (~inside & members).sum() * 0.5 * 60
                if day < 5 and step in (20, 21):  # 10:00-11:00
                    for name, members in groups.items():
                        counts[name][2] += (inside & (model._code == HOME) & members).sum()
                        counts[name][3] += members.sum()
            for index, members in enumerate(bands):
                band_days[index] += members.sum()
            if day < 5:
                for name, members in groups.items():
                    counts[name][0] += (~left & members).sum()
                    counts[name][1] += members.sum()
    return counts, way / band_days


def test_the_time_on_the_way_is_that_of_the_survey(observed):
    _, on_the_way = observed
    assert on_the_way == pytest.approx(ON_THE_WAY, rel=0.2)


def test_days_without_a_trip_are_as_common_as_in_the_survey(observed):
    counts, _ = observed
    for name, share in NO_TRIP.items():
        assert counts[name][0] / counts[name][1] == pytest.approx(share, abs=0.06), name


def test_the_old_and_those_without_work_are_at_home_in_the_morning_more_than_the_others(observed):
    counts, _ = observed
    at_home = {name: counts[name][2] / counts[name][3] for name in counts}
    assert min(at_home["65+"], at_home["inactive 18-64"]) > max(
        at_home["active 18-64"], at_home["under 18"]
    )
    # The model has people at home at 10 h more than the survey: the old and those without work
    # go out in the morning in the survey and are 10 % on the way at 10 h there; the gap is a
    # known one (see the report of [activities] in town.toml), this keeps it from growing.
    for name, share in AT_HOME_AT_10.items():
        assert at_home[name] == pytest.approx(share, abs=0.17), name
