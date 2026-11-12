"""The measurements diseases are fitted with (analysis.measure) against known answers."""

import numpy as np
import pytest
from config_factory import LIBRARY

from epidemic_sandbox.analysis import calibrate, measure
from epidemic_sandbox.analysis.calibrate import load_targets
from epidemic_sandbox.analysis.measure import Experiment, household_attack_rate, negative_binomial_k


@pytest.mark.parametrize("k", [0.1, 0.5, 2.0])
def test_the_dispersion_k_is_recovered_from_negative_binomial_counts(k):
    mean = 2.5
    counts = np.random.default_rng(1).negative_binomial(k, k / (k + mean), 20_000)
    assert negative_binomial_k(counts) == pytest.approx(k, rel=0.1)


@pytest.mark.parametrize("rate", [0.1, 0.3])
def test_the_growth_rate_of_exponentially_growing_infections_is_recovered(rate):
    # large counts, so that the noise of a day does not hide the growth
    steps_per_day, population = 48, 300_000
    rng = np.random.default_rng(2)
    days = np.arange(round(np.log(400) / rate))  # up to 20,000 a day
    daily = rng.poisson(50 * np.exp(rate * days))
    steps = np.concatenate(
        [
            day * steps_per_day + rng.integers(0, steps_per_day, n)
            for day, n in zip(days, daily, strict=True)
        ]
    )
    record = {"step": np.sort(steps)}
    assert measure._growth_rate(record, population, steps_per_day) == pytest.approx(rate, rel=0.1)


def test_the_household_attack_rate_counts_the_other_members_and_not_the_first_cases_again():
    # the cold passes to almost everybody at home and leaves no immunity, so the first cases are
    # infected again by the members they infected: they are not secondary cases
    experiment = Experiment("common_cold", {"transmission": 0.99})
    result = household_attack_rate(LIBRARY, experiment, [1], index_cases=40, days=14, workers=1)
    # not everybody at home is infected even when every shared day infects: a child who sleeps and
    # plays in a room of their own, a member who is away for the whole follow-up (measured: 0.77)
    assert 0.6 < result["household_attack_rate"] <= 1.0


def test_a_town_of_another_size_has_the_density_of_the_default_town():
    town = Experiment("covid19", {}).of_size(2000)
    assert town.area_km2 == 4 * 2000 / 3000
    town.config(LIBRARY, seed=1, initial_infected=1)  # accepted as an experiment


def test_targets_need_a_value_in_their_range_and_a_source_and_may_give_a_follow_up(tmp_path):
    good = tmp_path / "good.toml"
    good.write_text(
        '[targets.household_attack_rate]\nvalue = 0.3\nlow = 0.2\nhigh = 0.4\nsource = "A"\n'
        "days = 21\n"
    )
    assert load_targets(good)["household_attack_rate"]["days"] == 21
    bad = tmp_path / "bad.toml"
    bad.write_text('[targets.r]\nvalue = 3\nlow = 1\nhigh = 2\nsource = "A"\n')
    with pytest.raises(ValueError, match="outside its range"):
        load_targets(bad)


def test_the_calibration_command_explains_its_options(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["calibrate", "--help"])
    with pytest.raises(SystemExit) as finished:
        calibrate.main()
    assert finished.value.code == 0
    assert "--endemic-initial" in capsys.readouterr().out


def test_the_growth_rate_is_fitted_between_the_shares_of_the_town_of_the_window():
    steps_per_day, population = 48, 100_000
    rng = np.random.default_rng(4)
    # a burst of first infections (0.15 % of the town) that a fit from the first infection would
    # include, then growth at 0.2 a day
    daily = np.concatenate([[150, 0, 0], rng.poisson(10 * np.exp(0.2 * np.arange(45)))])
    steps = np.concatenate(
        [day * steps_per_day + rng.integers(1, steps_per_day, n) for day, n in enumerate(daily)]
    )
    record = {"step": np.sort(steps)}
    assert measure._growth_rate(record, population, steps_per_day) == pytest.approx(0.2, rel=0.1)
    small = {"step": np.sort(steps)[: int(0.002 * population)]}
    assert np.isnan(measure._growth_rate(small, population, steps_per_day)), "not there yet"


def test_infections_of_a_person_within_the_episode_window_count_once():
    record = {
        "person": np.array([1, 1, 1, 2, 2]),
        "step": np.array([0, 100, 1000, 50, 60]),
    }
    assert measure._new_episodes(record, 0).tolist() == [True] * 5
    # a window of 200 steps: the second infection of person 1 and of person 2 belong to the first
    assert measure._new_episodes(record, 200).tolist() == [True, False, True, True, False]


def test_the_hourly_risk_is_the_hazard_of_an_hour_at_the_mean_infectiousness_of_the_illness():
    model = Experiment("influenza", {}).model(LIBRARY)
    p = model.parameters
    weighted = (
        p["presymptomatic_infectiousness"] * p["presymptomatic_days"]
        + p["infectious_days"]
        + p["waning_infectiousness"] * p["waning_days"]
    )
    days = p["presymptomatic_days"] + p["infectious_days"] + p["waning_days"]
    hazard = -np.log(1 - p["transmission"]) / 24 * (1 + p["room_exposure"])  # close range and air
    expected = 1 - np.exp(-hazard * weighted / days)
    assert measure.hourly_risk(model) == pytest.approx(expected, rel=1e-6)


def test_the_household_measurement_takes_the_first_cases_and_members_of_the_ages_of_a_study():
    experiment = Experiment("common_cold", {"transmission": 0.99})
    children = household_attack_rate(
        LIBRARY, experiment, [1], index_cases=30, days=10, workers=1, index_ages=(5, 16)
    )
    adults_only = household_attack_rate(
        LIBRARY, experiment, [1], index_cases=30, days=10, workers=1, contact_ages=(18, 120)
    )
    assert children["contacts"] > 0 and adults_only["contacts"] > 0
    assert 0.6 < children["household_attack_rate"] <= 1.0  # see the test above (measured: 0.74)
    assert children["household_direct_rate"] <= children["household_attack_rate"]


def test_targets_may_give_the_ages_of_a_household_study_and_the_first_cases_of_an_epidemic(
    tmp_path,
):
    path = tmp_path / "ages.toml"
    path.write_text(
        '[targets.household_attack_rate]\nvalue = 0.3\nlow = 0.2\nhigh = 0.4\nsource = "A"\n'
        "index_ages = [4, 13]\ncontact_ages = [0, 14]\n"
        '[targets.reproduction_number]\nvalue = 15\nlow = 12\nhigh = 18\nsource = "B"\n'
        "initial = 1\n"
        '[targets.incidence_per_person_year]\nvalue = 2.7\nlow = 1.2\nhigh = 3.9\nsource = "C"\n'
        "episode_days = 12\n"
    )
    targets = load_targets(path)
    assert targets["household_attack_rate"]["index_ages"] == [4, 13]
    bad = tmp_path / "bad_ages.toml"
    bad.write_text(
        '[targets.household_attack_rate]\nvalue = 0.3\nlow = 0.2\nhigh = 0.4\nsource = "A"\n'
        "index_ages = [13, 4]\n"
    )
    with pytest.raises(ValueError, match="index_ages"):
        load_targets(bad)


def test_a_knob_is_a_parameter_or_the_multiplier_of_an_age_range():
    experiment = Experiment("influenza", {})
    knob = calibrate.Knob.parse("age.susceptibility.0-5")
    assert (knob.effect, knob.low, knob.high) == ("susceptibility", 0.0, 5.0)
    assert knob.value(LIBRARY, experiment) == pytest.approx(
        LIBRARY.catalog["influenza"].age_effects["susceptibility"][0][2]
    )
    changed = knob.set(LIBRARY, experiment, 1.5)
    assert knob.value(LIBRARY, changed) == pytest.approx(1.5)
    assert calibrate.Knob.parse("transmission").set(LIBRARY, experiment, 0.2).overrides == {
        "transmission": 0.2
    }


def test_the_fit_finds_the_value_that_meets_the_target_with_the_same_seeds(monkeypatch):
    calls = []

    def measured(library, experiment, name, targets, conditions, settings):
        calls.append(settings.seeds)
        return 3.0 * experiment.overrides["transmission"], 0.001  # monotone in the knob

    monkeypatch.setattr(calibrate, "_measure_one", measured)
    targets = {"household_attack_rate": {"value": 0.3, "low": 0.2, "high": 0.4, "source": "A"}}
    settings = calibrate.Settings(seeds=[1, 2, 3])
    fitted = calibrate.fit(
        LIBRARY,
        Experiment("covid19", {}),
        [(calibrate.Knob.parse("transmission"), "household_attack_rate")],
        targets,
        settings,
        rounds=2,
        steps=40,
        report=lambda text: None,
    )
    assert 3.0 * fitted.overrides["transmission"] == pytest.approx(0.3, abs=0.002)
    assert all(seeds == [1, 2, 3] for seeds in calls), "common random numbers"


def test_the_reproduction_number_counts_the_offspring_of_the_generations_from_the_first_one_given():
    # a chain: first case 0 (generation 0) infects 1 and 2 (generation 1); 1 infects 3, 4, 5 and 2
    # infects 6 (generation 2); of generation 2 only 3 infects (7, 8; generation 3); the last
    # generation counted is the second, as the town is tiny
    infector = np.array([-1, 0, 0, 1, 1, 1, 2, 3, 3])
    count = len(infector)
    record = {
        "step": np.arange(count) * 10,
        "person": np.arange(count),
        "infector": infector,
        "infector_record": infector,
        "setting": np.where(infector < 0, 0, 2),
        "infector_symptomatic": np.where(infector < 0, -1, 1),
        "onset_step": np.full(count, -1),
        "death_step": np.full(count, -1),
        "asymptomatic": np.zeros(count, np.int64),
    }
    run = {
        "record": record,
        "settings": ("seed", "outside", "household"),
        "age": np.full(count, 40),
        "population": 1_000_000,
    }
    from_second = measure.summarise([run], 1, first_generation=2)
    from_first = measure.summarise([run], 1, first_generation=1)
    # generation 2 (3, 4, 5, 6): only 3 infects, two people
    assert from_second["reproduction_number"] == pytest.approx(2 / 4)
    # generations 1 and 2: 3 + 1 + 2 + 0 + 0 + 0 infected by six people
    assert from_first["reproduction_number"] == pytest.approx(6 / 6)
