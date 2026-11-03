"""The numbers of the demo diseases and model defaults that cite a source, written down once more.

Changing one of them changes what the disease means (and what its comment claims), so a change has
to be made here on purpose, together with its source.
"""

import pytest
from config_factory import LIBRARY

from epidemic_sandbox.disease_models import DISEASE_MODELS

# key: (values both the demo disease and the model default have, demo-only values, age effects)
SOURCED = {
    "sicr": (
        {
            "infectious_days": 28.0,  # Pitzer et al. 2014, Table 1
            "carrier_share": 0.03,  # Gonzalez-Escobedo et al. 2011: 3-5 % of typhoid patients
            "carrier_days": 36500,  # Pitzer et al. 2014: carriage for life
            "carrier_infectiousness": 0.1,
            "fatality": 0.02,  # Crump et al. 2008
        },
        {"transmission": 0.1, "immunity_days": 730},  # Pitzer et al. 2014: 104 weeks
        {"branching": [[0, 15, 0.1], [15, 50, 1.0], [50, 120, 3.4]]},  # Pitzer: 0.3-10.1 %
    ),
    "seiar": (
        {  # Davies et al. 2020 (Supplementary Table 1), Li et al. 2020, Poletti et al. 2021
            "incubation_days": 3.0,
            "infectious_days": 5.0,
            "asymptomatic_days": 5.0,
            "duration_shape": 4.0,
            "asymptomatic_infectiousness": 0.5,
            "asymptomatic_share": 0.7,
        },
        {"transmission": 0.5, "contact_distance": 3.0, "fatality": 0.007},
        {
            "branching": [
                [0, 20, 1.17],
                [20, 40, 1.11],
                [40, 60, 0.99],
                [60, 80, 0.92],
                [80, 120, 0.51],
            ],
            "susceptibility": [[0, 20, 0.5], [20, 120, 1.0]],
        },
    ),
    "si": ({}, {"transmission": 0.1, "contact_distance": 2.0}, {}),  # compressed time scale
    "sir": ({}, {"immunity_protection": 0.9, "transmission": 0.55}, {}),
    "sepir": (
        {"latent_days": 3.0, "presymptomatic_days": 2.5, "infectious_days": 3.0},
        {"fatality": 0.0116, "transmission": 0.4},  # Levin et al. 2020
        {
            "fatality": [
                [0, 20, 0.002],
                [20, 50, 0.05],
                [50, 65, 0.5],
                [65, 80, 2.7],
                [80, 120, 10.8],
            ]
        },
    ),
    "seihr": (
        {  # Ferguson et al. 2020 (Report 9), Faes et al. 2020
            "incubation_days": 4.6,
            "infectious_days": 5.5,
            "hospital_days": 10.4,
            "hospitalised": 0.08,
            "fatality": 0.15,
        },
        {"transmission": 0.55},
        {"branching": [[0, 10, 0.0125], [80, 120, 3.4125]]},
    ),
    "svir": (
        {  # Carrat et al. 2008, Belongia et al. 2016, CDC 2010
            "infectious_days": 4.8,
            "vaccine_efficacy": 0.61,
            "vaccination_rate": 0.0025,
            "vaccine_days": 110,
        },
        {"initial_coverage": 0.3, "vaccination_start_day": 0},
        {"susceptibility": [[0, 65, 1.0], [65, 120, 0.66]]},  # Hancock et al. 2009
    ),
    "seiqr": (
        {  # Leung et al. 2004, Donnelly et al. 2003, Wallinga and Teunis 2004
            "incubation_days": 4.6,
            "detection_delay_days": 3.4,
            "undetected_days": 7.4,
            "quarantine_days": 7.4,
            "detection_coverage": 0.8,
            "quarantine_infectiousness": 0.5,
        },
        {"fatality": 0.1, "transmission": 0.8},
        {"fatality": [[0, 25, 0.05], [25, 45, 0.6], [45, 65, 1.5], [65, 120, 5.0]]},  # WHO 2003
    ),
}


@pytest.mark.parametrize("key", sorted(SOURCED))
def test_the_demo_disease_has_the_numbers_its_comment_cites(key):
    shared, demo_only, age_effects = SOURCED[key]
    demo = LIBRARY.catalog[f"demo_{key}"]
    for name, value in {**shared, **demo_only}.items():
        assert demo.parameters[name] == pytest.approx(value), name
    for effect, rows in age_effects.items():
        shown = [list(row) for row in demo.age_effects[effect]]
        assert all(row in shown for row in rows), effect


@pytest.mark.parametrize("key", sorted(SOURCED))
def test_the_model_defaults_are_the_cited_numbers(key):
    shared, _, _ = SOURCED[key]
    model = DISEASE_MODELS[key](1.0)
    for name, value in shared.items():
        assert model.parameters[name] == pytest.approx(value), name
