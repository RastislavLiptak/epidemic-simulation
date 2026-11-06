"""The common cold (SIS, rhinoviruses) against the numbers its comments cite, and its interaction.

The SIS model itself is checked against its equations in the tests of the contract and of the
disease core; here are only the values that the disease file takes from sources.
"""

import pytest
from cited_numbers import check_disease_numbers
from config_factory import LIBRARY

# (values both the disease and the model default have, the disease's own values, age effects)
SOURCED = (
    {},
    {  # Peltola et al. 2008, Lessler et al. 2009, Fragaszy et al. 2018, Van Kerckhove et al. 2013
        "transmission": 0.108,
        "community_factor": 3.65,
        "outside_infections": 0.145,
        "contact_distance": 2.0,
        "infectious_days": 5.4,
        "duration_shape": 3.2,
        "fatality": 0.0,
        "stay_home": 0.12,  # Dicpinigaitis et al. 2015
        "limit_outings": 0.16,
        "asymptomatic_share": 0.44,  # Byington et al. 2015
        "asymptomatic_infectiousness": 0.52,  # Adler et al. 2018
        "asymptomatic_days": 5.4,
        "susceptibility_cv": 0.0,
        "infectiousness_dispersion": 0.0,
    },
    {"susceptibility": [[5, 18, 0.55]]},
)


def test_the_disease_has_the_numbers_its_comment_cites():
    check_disease_numbers("sis", *SOURCED)


def test_a_cold_lowers_the_chance_of_catching_influenza_by_the_odds_ratio_of_co_detection():
    rules = [rule for rule in LIBRARY.interactions if rule.source == "common_cold"]
    assert [(rule.states, rule.target, rule.effect) for rule in rules] == [
        (("infectious",), "influenza", "susceptibility")
    ]
    assert rules[0].factor == pytest.approx(0.27)  # Nickbakhsh et al. 2019, PNAS
