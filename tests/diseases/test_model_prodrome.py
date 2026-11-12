"""Measles (the prodrome model) against its definition and the numbers its comments cite.

The model is SEPIR with an unspecific illness, the prodrome, before the illness by which the
disease is recognised: E -> P -> prodrome -> infectious -> waning -> recovered (or dead). The
prodrome and the recognised illness are symptomatic, each with its own behaviour, and a share of
those with the recognised illness is in hospital.
"""

import numpy as np
import pytest
from cited_numbers import check_disease_numbers
from disease_helpers import ONE_DAY, SURE, advance, new_state, place, state_names

from epidemic_sandbox.disease_models.disease_model_abstract import (
    EXPOSED,
    INFECTIOUS,
    RECOVERED,
)
from epidemic_sandbox.disease_models.prodrome import PRODROME, ProdromeModel
from epidemic_sandbox.disease_models.sepir import PRESYMPTOMATIC, WANING

ORDER = list(ProdromeModel.STATE_COLORS)
DAYS = {
    "latent_days": 3.0,
    "presymptomatic_days": 1.0,
    "prodrome_days": 2.0,
    "infectious_days": 2.0,
    "waning_days": 2.0,
}

# (values both the disease and the model default have, the disease's own values, age effects)
MEASLES = (
    {  # CDC Pink Book, chapter 13
        "latent_days": 10.0,
        "presymptomatic_days": 1.5,
        "prodrome_days": 2.5,
        "infectious_days": 2.0,
        "waning_days": 2.0,
        "prodrome_stay_home": 0.31,  # Fragaszy et al. 2018
    },
    {
        "waning_infectiousness": 0.5,
        "presymptomatic_infectiousness": 1.0,
        "prodrome_infectiousness": 1.0,
        "transmission": 0.0212,
        "contact_distance": 2.0,
        "near_distance": 1.0,
        "community_factor": 1.0,
        "room_exposure": 150.0,  # fitted to R0 12-18, about 90 quanta/h (Riley et al. 1978)
        "asleep_emission": 0.2,
        "duration_shape": 23.0,  # Lessler et al. 2009
        "fatality": 0.000726,  # ECDC 2019 and 2024
        "death_delay_days": 8.0,
        "stay_home": 1.0,  # isolation for 7 days from the rash
        "prodrome_limit_outings": 0.33,
        "admission_share": 1.0,  # decree 306/2012 Coll.
        "initial_immune": 0.93,  # Tomaskova et al. 2018
        "asymptomatic_share": 0.0,
    },
    {"stay_home": [[0, 16, 1.806]]},
)


def test_measles_has_the_numbers_its_comment_cites():
    check_disease_numbers("prodrome", *MEASLES, name="measles")


def test_the_model_defaults_are_the_cited_numbers():
    model = ProdromeModel(ONE_DAY)
    for name, value in MEASLES[0].items():
        assert model.parameters[name] == pytest.approx(value), name


def test_a_person_goes_through_the_prodrome_to_the_recognised_illness():
    model = ProdromeModel(ONE_DAY, **SURE, **DAYS, duration_shape=50.0)
    disease_state, rng = new_state(model)
    advance(model, disease_state, rng, exposure=1)
    visited = [state_names(model, disease_state)[0]]
    for _ in range(30):
        advance(model, disease_state, rng)
        current = state_names(model, disease_state)[0]
        if current != visited[-1]:
            visited.append(current)
    assert visited == [EXPOSED, PRESYMPTOMATIC, PRODROME, INFECTIOUS, WANING, RECOVERED]


def test_the_prodrome_and_the_recognised_illness_have_their_own_behaviour():
    model = ProdromeModel(
        ONE_DAY, **DAYS, prodrome_stay_home=0.3, stay_home=0.8, duration_shape=50.0
    )
    population = 20_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), PRODROME)
    in_prodrome = disease_state.stays_home.copy()
    assert in_prodrome.mean() == pytest.approx(0.3, abs=0.015)
    for _ in range(3):
        advance(model, disease_state, rng)
    assert (disease_state.state == ORDER.index(INFECTIOUS)).mean() > 0.99
    assert disease_state.stays_home.mean() == pytest.approx(0.8, abs=0.015)
    # one draw per illness: who stayed at home in the prodrome stays at home with the rash
    assert disease_state.stays_home[in_prodrome].all()


def test_the_admitted_are_away_from_the_town_and_spread_nothing_in_it():
    model = ProdromeModel(ONE_DAY, **DAYS, admission_share=0.6, duration_shape=50.0)
    population = 20_000
    disease_state, rng = new_state(model, population=population)
    place(model, disease_state, rng, np.arange(population), INFECTIOUS)
    admitted = model.admitted(disease_state)
    assert admitted.mean() == pytest.approx(0.6, abs=0.015)
    assert (model.away_mask(disease_state) == admitted).all()
    assert not model.infectious_weight(disease_state)[admitted].any()
    assert (model.infectious_weight(disease_state)[~admitted] == 1).all()
    assert disease_state.stays_home[admitted].all(), "they do not go out"
    for _ in range(10):
        advance(model, disease_state, rng)
    assert not model.admitted(disease_state).any(), "home again after the illness"


def test_the_infectious_time_in_the_town_counts_only_those_not_in_hospital():
    model = ProdromeModel(ONE_DAY, **DAYS, admission_share=0.25, waning_infectiousness=0.5)
    p = model.parameters
    in_town = p["presymptomatic_days"] + p["prodrome_days"] + 0.75 * (
        p["infectious_days"] + 0.5 * p["waning_days"]
    )
    assert model.infectious_days() == pytest.approx(in_town)
