"""The well-mixed driver agrees with the textbook results it is meant to be used against."""

import math

import numpy as np
import pytest
from mean_field import final_size, integrate, mean_course, run_well_mixed

from epidemic_sandbox.disease_models.sir import SirModel

IDENTICAL = {"susceptibility_cv": 0.0, "infectiousness_dispersion": 0.0}


def test_the_final_size_of_an_sir_epidemic_follows_the_final_size_relation():
    transmission, contacts, days_infectious = 0.4, 4.0, 5.0
    r0 = -math.log(1 - transmission) * contacts * days_infectious
    _, counts = mean_course(
        SirModel,
        runs=4,
        population=20_000,
        cases=100,
        days=120,
        contacts=contacts,
        transmission=transmission,
        infectious_days=days_infectious,
        immunity_days=36500,
        **IDENTICAL,
    )
    ever_infected = 1 - counts[-1, 0] / 20_000
    assert ever_infected == pytest.approx(final_size(r0, 100 / 20_000), abs=0.015)


def test_a_sir_epidemic_with_exponential_stays_follows_the_ode_in_time():
    transmission, contacts, days_infectious = 0.3, 3.0, 4.0
    beta = -math.log(1 - transmission) * contacts
    gamma = 1 / days_infectious

    def derivative(y):
        s, i, r = y
        return np.array([-beta * s * i, beta * s * i - gamma * i, gamma * i])

    population, cases = 40_000, 200
    ode = integrate(derivative, np.array([1 - cases / population, cases / population, 0.0]), 60)
    _, counts = mean_course(
        SirModel,
        runs=3,
        population=population,
        cases=cases,
        days=60,
        contacts=contacts,
        transmission=transmission,
        infectious_days=days_infectious,
        duration_shape=1.0,
        immunity_days=36500,
        **IDENTICAL,
    )
    infectious = counts[:, 1] / population
    # the half-step offset of the stays is within the tolerance of a daily comparison
    assert infectious.max() == pytest.approx(ode[:, 1].max(), rel=0.08)
    assert infectious.argmax() / 24 == pytest.approx(ode[:, 1].argmax() / 24, abs=2.0)


def test_without_infectious_people_nothing_happens():
    _, counts = run_well_mixed(
        SirModel, population=500, cases=0, days=5, contacts=10.0, transmission=0.5
    )
    assert counts[:, 0].min() == 500
