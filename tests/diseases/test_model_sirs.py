"""The common cold (SIRS, rhinoviruses) against its definition and the numbers its comments cite.

SIRS is the infection without a latent period that protects for a short time: the course splits
at the infection into the courses with and without symptoms, and both end in the protected state,
which is left for the susceptible one after `immunity_days`.
"""

import numpy as np
import pytest
from cited_numbers import check_disease_numbers
from config_factory import LIBRARY
from disease_helpers import ONE_DAY, SURE, UNIFORM, advance, new_state, state_names

from epidemic_sandbox.disease_models.disease_model_abstract import (
    ASYMPTOMATIC,
    INFECTIOUS,
    RECOVERED,
    SUSCEPTIBLE,
)
from epidemic_sandbox.disease_models.sirs import SirsModel

# (values both the disease and the model default have, the disease's own values, age effects)
SOURCED = (
    {"immunity_days": 14.0},  # Kamau et al. 2019
    {  # Peltola et al. 2008, Lessler et al. 2009, Fragaszy et al. 2018, Van Kerckhove et al. 2013
        "transmission": 0.3554,
        "room_exposure": 0.0719,  # Fitted (Dick et al. 1987: aerosol route)
        "community_factor": 1.0,
        "near_distance": 1.0,
        "asleep_emission": 0.2,
        "outside_infections": 0.234,
        "contact_distance": 2.0,
        "infectious_days": 5.4,
        "duration_shape": 3.2,
        "fatality": 0.0,
        "stay_home": 0.05,  # Fragaszy et al. 2018, acute respiratory illness
        "limit_outings": 0.15,
        "asymptomatic_share": 0.44,  # Byington et al. 2015
        "asymptomatic_infectiousness": 0.52,  # Adler et al. 2018
        "asymptomatic_days": 5.4,
        "immunity_protection": 1.0,
        "susceptibility_cv": 0.0,
        "infectiousness_dispersion": 0.0,
    },
    {
        "susceptibility": [[5, 18, 0.498]],
        "stay_home": [[0, 16, 1.273]],
        "outside_infections": [[0, 18, 0.0], [65, 120, 0.0]],
    },
)


def test_the_disease_has_the_numbers_its_comment_cites():
    check_disease_numbers("sirs", *SOURCED)


def test_the_model_defaults_are_the_cited_numbers():
    model = SirsModel(ONE_DAY)
    for name, value in SOURCED[0].items():
        assert model.parameters[name] == pytest.approx(value), name


def test_a_cold_lowers_the_chance_of_catching_influenza_by_the_odds_ratio_of_co_detection():
    rules = [rule for rule in LIBRARY.interactions if rule.source == "common_cold"]
    assert [(rule.states, rule.target, rule.effect) for rule in rules] == [
        (("infectious",), "influenza", "susceptibility")
    ]
    assert rules[0].factor == pytest.approx(0.27)  # Nickbakhsh et al. 2019, PNAS


def test_an_infection_protects_for_a_while_then_the_person_is_susceptible_again():
    model = SirsModel(
        ONE_DAY, **SURE, **UNIFORM, infectious_days=2.0, immunity_days=10.0, duration_shape=50.0
    )
    disease_state, rng = new_state(model, population=1, cases=[0])
    seen = [state_names(model, disease_state)[0]]
    for _ in range(30):
        advance(model, disease_state, rng, exposure=1.0)
        seen.append(state_names(model, disease_state)[0])
    assert seen[0] == INFECTIOUS
    first_protected = seen.index(RECOVERED)
    assert first_protected <= 3, "recovered after the infectious days"
    again = seen.index(INFECTIOUS, first_protected)
    assert again - first_protected >= 9, "nobody is infected while protected, however exposed"
    assert again - first_protected <= 12, "infected again once the protection is over"


def test_both_courses_end_in_the_protected_state():
    model = SirsModel(ONE_DAY, **SURE, **UNIFORM, asymptomatic_share=0.5, duration_shape=50.0)
    disease_state, rng = new_state(model, population=2000)
    advance(model, disease_state, rng, exposure=1.0)
    names = np.array(state_names(model, disease_state))
    assert set(names) == {INFECTIOUS, ASYMPTOMATIC}
    assert (names == ASYMPTOMATIC).mean() == pytest.approx(0.5, abs=0.04)
    for _ in range(int(model.parameters["asymptomatic_days"]) + 10):
        advance(model, disease_state, rng)
    names = set(state_names(model, disease_state))
    assert names <= {RECOVERED, SUSCEPTIBLE} and RECOVERED in names
