"""The record of who infected whom (epidemic_sandbox.engine.tracing)."""

import math

import numpy as np
import pytest
from config_factory import LIBRARY, disease_definition, experiment_config

from epidemic_sandbox.engine.epidemic_model import EpidemicModel
from epidemic_sandbox.engine.tracing import InfectionTrace

pytestmark = pytest.mark.usefixtures("synthetic_in_registry")


def traced_model(**parameters) -> EpidemicModel:
    diseases = {"a": disease_definition("plain", parameters.pop("initial", 0), **parameters)}
    return EpidemicModel(experiment_config(diseases), LIBRARY.town, trace=True)


def run_days(model, days: int):
    for _ in range(days * model.world["steps_per_day"]):
        model.step()
    return model.trace.diseases["a"].arrays()


def test_infections_without_anybody_infectious_around_are_from_outside():
    model = traced_model(transmission=0.0, outside_infections=50.0)
    record = run_days(model, 3)
    assert len(record["step"]) > 10
    assert (record["setting"] == model.trace.settings.index("outside")).all()
    assert (record["infector"] == -1).all() and (record["infector_record"] == -1).all()
    assert (record["infector_symptomatic"] == -1).all()


def test_without_outside_infections_every_infection_after_the_first_cases_has_an_infector():
    model = traced_model(initial=10, transmission=0.9, community_factor=5.0)
    record = run_days(model, 5)
    later = record["step"] > 0
    assert later.sum() > 10
    assert (record["infector"][later] >= 0).all()
    assert not (record["setting"] == model.trace.settings.index("outside")).any()


def test_the_chance_of_an_outside_infection_is_its_share_of_the_hazard():
    model = traced_model(transmission=1 - math.exp(-0.4), outside_infections=36.525)
    disease = model.diseases["a"]
    exposure = np.array([0.0, 0.25, 1.0], np.float32)
    share = InfectionTrace._outside_share(disease, np.arange(3), exposure, 1.0)
    assert share == pytest.approx([1.0, 0.1 / 0.2, 0.1 / 0.5], rel=1e-5)
    weighted = InfectionTrace._outside_share(disease, np.arange(3), exposure, np.full(3, 0.5))
    assert weighted == pytest.approx([1.0, 0.05 / 0.15, 0.05 / 0.45], rel=1e-5)


def test_an_infector_has_symptoms_from_the_onset_whatever_state_follows_it():
    # isolated cases (SEIQR) still infect their households here: they had their onset before
    diseases = {
        "a": disease_definition(
            "seiqr",
            20,
            transmission=0.6,
            community_factor=3.0,
            detection_delay_days=0.5,
            detection_coverage=1.0,
            quarantine_infectiousness=1.0,
        )
    }
    model = EpidemicModel(experiment_config(diseases), LIBRARY.town, trace=True)
    record = run_days(model, 8)
    rows = np.flatnonzero(record["infector_record"] >= 0)
    onset = record["onset_step"][record["infector_record"][rows]]
    began = (onset >= 0) & (onset <= record["step"][rows])
    assert (record["infector_symptomatic"][rows] == began).all()
    assert len(rows) > 20 and began.mean() > 0.5, "most infect while isolated, after the onset"
