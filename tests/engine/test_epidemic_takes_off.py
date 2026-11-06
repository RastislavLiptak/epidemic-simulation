"""The diseases that were real epidemics take off in the town, as they did.

COVID-19, SARS and influenza spread through populations when nothing stopped them. Their
household transmission alone (the secondary attack rate in households) does not make an epidemic
in the town: what passes between households is the community factor of each disease. If a change
of the town or of the diseases makes them die out again, this test says so.
"""

import pytest
from config_factory import LIBRARY

from epidemic_sandbox.engine.epidemic_model import EpidemicModel

DAYS = 100
SEEDS = 60  # first cases, the size of an outbreak that has been noticed


@pytest.mark.parametrize(
    ("name", "overrides", "minimum"),
    [
        ("covid19", {}, 300),
        ("sars2003", {"detection_coverage": 0.0}, 300),
        ("influenza", {"initial_coverage": 0.0}, 300),
    ],
)
def test_the_epidemic_takes_off_in_the_town(name, overrides, minimum):
    disease = LIBRARY.catalog[name]
    config = LIBRARY.parse(
        {
            "world": {"population": 3000, "area_km2": 4, "seed": 2, "runs": 1},
            "diseases": {
                name: {
                    "model": disease.model,
                    "initial_infected": SEEDS,
                    "parameters": {**disease.parameters, **overrides},
                    "age_effects": {
                        k: [list(r) for r in v] for k, v in disease.age_effects.items()
                    },
                }
            },
            "interactions": [],
        }
    )
    model = EpidemicModel(config, LIBRARY.town)
    for _ in range(DAYS * model.world["steps_per_day"]):
        model.step()
    infected = model.history[-1]["cumulative_infections"][name]
    assert infected > minimum, f"{name}: only {infected} infections in {DAYS} days"
