"""The measurements diseases are fitted with (epidemic_sandbox.measure) against known answers."""

import numpy as np
import pytest
from config_factory import LIBRARY

from epidemic_sandbox import calibrate, measure
from epidemic_sandbox.calibrate import load_targets
from epidemic_sandbox.measure import Experiment, household_attack_rate, negative_binomial_k


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
    assert 0.8 < result["household_attack_rate"] <= 1.0


def test_a_town_of_another_size_has_the_density_of_the_default_town_and_a_whole_area():
    town = Experiment("covid19", {}).of_size(2000)
    assert town.arena_area == round(4_000_000 * 2000 / 3000)
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
