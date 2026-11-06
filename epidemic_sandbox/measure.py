"""Measuring how a disease behaves in the town, as studies of real epidemics measure it.

Every number that a disease file marks as Fitted is fitted with these measurements, and its
`[targets]` table names what was aimed at; `python -m epidemic_sandbox.calibrate <disease>`
measures a disease and compares it with its targets.

* The household attack rate: first cases are placed at random among people living in private
  households of two or more and nobody else is infected; nothing passes between households (the
  community factor is 0) and nobody is infected from outside, so the share of the other members of
  their households infected within the follow-up, in chains that go back to the first case, is the
  transmission within households that household studies report, without the infections that the
  town around would bring in. A first case infected again (a disease without immunity) is not a
  secondary case.
* The reproduction number, the generation interval and the dispersion of the number of people one
  person infects are taken from the early generations of an epidemic: the infections caused by the
  first cases (generation 1), by those (generation 2) and so on, up to the last generation whose
  own infections still fall, on average over the runs, within the first `EARLY_SHARE` of the town
  infected, so that the depletion of susceptible people does not lower what they do. The choice
  depends on generations, not on how many people a particular infection infected, so it does not
  select. A town of `EPIDEMIC_POPULATION` people, with the density of the default town, is large
  enough for generation 1 even at a reproduction number of 15. The reproduction number is also
  derived from the growth of the epidemic. The share of transmission before the onset of symptoms
  is that of the infections caused by people who had symptoms, as studies of pairs with known
  onsets measure it; the share of infections without symptoms and how many people they infect
  relative to the others are given alongside.
* Where infections happen, who is infected by age, and how long after the infection and after the
  onset of symptoms people die.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from epidemic_sandbox.epidemic_model import EpidemicModel
from epidemic_sandbox.experiment_config import ExperimentLibrary
from epidemic_sandbox.population import RESIDENT

# Model: an infection belongs to the early phase when, by the time it has done its infecting, at
# most this share of the town has been infected: the depletion of susceptible people then lowers
# its number of infections by at most this share.
EARLY_SHARE = 0.02
DEFAULT_WORLD = {"population": 3000, "arena_area": 4_000_000}
EPIDEMIC_POPULATION = 30_000
HOUSEHOLD_DAYS = 28  # the follow-up of a household study when its target gives none
AGE_GROUPS = ((0, 5), (5, 18), (18, 30), (30, 50), (50, 65), (65, 80), (80, 120))


@dataclass
class Experiment:
    """One disease alone in the default town: the disease name, changes of its parameters and
    of its age effects, and the world."""

    disease: str
    overrides: dict
    age_effects: dict | None = None
    population: int = DEFAULT_WORLD["population"]
    arena_area: float = DEFAULT_WORLD["arena_area"]

    def of_size(self, population: int) -> "Experiment":
        """The same disease in a town of `population` people, as dense as the default town."""
        area = round(DEFAULT_WORLD["arena_area"] * population / DEFAULT_WORLD["population"])
        return Experiment(self.disease, self.overrides, self.age_effects, population, area)

    def config(self, library: ExperimentLibrary, seed: int, initial_infected: int):
        disease = library.catalog[self.disease]
        effects = {key: [list(row) for row in rows] for key, rows in disease.age_effects.items()}
        effects.update(self.age_effects or {})
        world = {
            "population": self.population,
            "arena_area": self.arena_area,
            "seed": seed,
            "runs": 1,
        }
        return library.parse(
            {
                "world": world,
                "diseases": {
                    self.disease: {
                        "model": disease.model,
                        "initial_infected": initial_infected,
                        "parameters": {**disease.parameters, **self.overrides},
                        "age_effects": effects,
                    }
                },
                "interactions": [],
            }
        )


def household_attack_rate(
    library: ExperimentLibrary,
    experiment: Experiment,
    seeds: list[int],
    index_cases: int = 60,
    days: int = HOUSEHOLD_DAYS,
    workers: int | None = None,
) -> dict:
    """Share of the household members of first cases infected at home within `days`."""
    jobs = [(library, experiment, seed, index_cases, days) for seed in seeds]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_household_run, *zip(*jobs, strict=True)))
    infected = sum(result[0] for result in results)
    contacts = sum(result[1] for result in results)
    cases = sum(result[2] for result in results)
    per_run = np.array([result[0] / max(result[1], 1) for result in results])
    return {
        "household_attack_rate": infected / max(contacts, 1),
        "household_case_rate": cases / max(contacts, 1),
        "standard_error": float(per_run.std(ddof=1) / math.sqrt(len(per_run)))
        if len(per_run) > 1
        else math.nan,
        "contacts": contacts,
    }


def _household_run(library, experiment, seed, index_cases, days):
    # nothing passes between households and nobody is infected from outside: the members are
    # infected only at home, as household studies count it, and not first in an epidemic of the
    # town that the first cases would start
    alone = Experiment(
        experiment.disease,
        {**experiment.overrides, "community_factor": 0.0, "outside_infections": 0.0},
        experiment.age_effects,
        experiment.population,
        experiment.arena_area,
    )
    config = alone.config(library, seed, 0)
    model = EpidemicModel(config, library.town, trace=True)
    name = experiment.disease
    town = model.town
    size = np.bincount(town.household)
    private = (town.role != RESIDENT) & (size[town.household] >= 2)
    rng = np.random.default_rng(seed + 7919)
    households = rng.permutation(np.unique(town.household[private]))[:index_cases]
    first = np.array([rng.choice(np.flatnonzero(town.household == h)) for h in households])
    disease, state = model.diseases[name], model.disease_states[name]
    disease._enter(state, first, disease.INITIAL_CASE_STATE, rng)
    state.ever_infected[first] = True
    trace = model.trace
    trace._append(name, first, 0, np.full(len(first), -1), trace.settings.index("seed"))
    for _ in range(days * model.world["steps_per_day"]):
        model.step()
    record = trace.diseases[name].arrays()
    # with nothing passing between households every infection is by a member of the household (at
    # home or out together); those that go back to a first case are what a household study
    # attributes to it
    seed = record["infector"] < 0  # the first cases
    from_case = seed.copy()
    for row in range(len(from_case)):
        source = record["infector_record"][row]
        if source >= 0 and from_case[source]:
            from_case[row] = True
    secondary = from_case & ~seed
    # people, not episodes, and not a first case infected again by a member it infected
    traced = np.setdiff1d(record["person"][secondary], first)
    ill = np.setdiff1d(record["person"][secondary & (record["onset_step"] >= 0)], first)
    members = np.flatnonzero(np.isin(town.household, households))
    infected = np.count_nonzero(np.isin(traced, members))
    cases = np.count_nonzero(np.isin(ill, members))  # those who had symptoms
    return infected, len(members) - len(first), cases


def epidemic(
    library: ExperimentLibrary,
    experiment: Experiment,
    seeds: list[int],
    initial_infected: int = 10,
    days: int = 120,
    workers: int | None = None,
) -> dict:
    """The early reproduction number, generation interval, dispersion, settings, ages and deaths
    of epidemics started by `initial_infected` cases, one run per seed."""
    jobs = [(library, experiment, seed, initial_infected, days) for seed in seeds]
    with ProcessPoolExecutor(workers) as pool:
        runs = list(pool.map(_epidemic_run, *zip(*jobs, strict=True)))
    return summarise(runs, library.steps_per_day)


def _epidemic_run(library, experiment, seed, initial_infected, days):
    config = experiment.config(library, seed, initial_infected)
    model = EpidemicModel(config, library.town, trace=True)
    for _ in range(days * model.world["steps_per_day"]):
        model.step()
    record = model.trace.diseases[experiment.disease].arrays()
    return {
        "record": record,
        "settings": model.trace.settings,
        "age": model.town.age,
        "population": model.population,
        "states": model.disease_states[experiment.disease].state.copy(),
        "state_names": list(model.diseases[experiment.disease].STATE_COLORS),
    }


def endemic(
    library: ExperimentLibrary,
    experiment: Experiment,
    seeds: list[int],
    initial_infected: int,
    days: int,
    burn_in: int,
    workers: int | None = None,
) -> dict:
    """Infections per person and year, overall and by age, of a disease that stays in the town,
    counted after `burn_in` days, and the share of the runs in which it was still there at the
    end."""
    jobs = [(library, experiment, seed, initial_infected, days) for seed in seeds]
    with ProcessPoolExecutor(workers) as pool:
        runs = list(pool.map(_epidemic_run, *zip(*jobs, strict=True)))
    steps_per_day = library.steps_per_day
    years = (days - burn_in) / 365.25
    infections, by_age, people, persisted = 0, np.zeros(len(AGE_GROUPS)), 0, 0
    population = np.zeros(len(AGE_GROUPS))
    school_age, school_people, school_cases = 0, 0, 0  # 5-12 years, the strep throat cohort
    adult_people, adult_cases = 0, 0  # 30-49 years, the parents of that cohort
    names = runs[0]["state_names"]
    in_state = np.zeros(len(names))
    for run in runs:
        record, age = run["record"], run["age"]
        # every infection after the burn-in, in the town or from outside (the first cases are at
        # step 0)
        late = (record["step"] >= burn_in * steps_per_day) & (record["step"] > 0)
        infections += int(late.sum())
        people += run["population"]
        persisted += int(record["step"].max() >= (days - 1) * steps_per_day)
        infected_age = age[record["person"][late]]
        for index, (low, high) in enumerate(AGE_GROUPS):
            by_age[index] += np.count_nonzero((infected_age >= low) & (infected_age < high))
            population[index] += np.count_nonzero((age >= low) & (age < high))
        school = (age >= 5) & (age < 13)
        school_age += np.count_nonzero((infected_age >= 5) & (infected_age < 13))
        school_people += np.count_nonzero(school)
        ill = late & (record["onset_step"] >= 0)
        ill_age = age[record["person"][ill]]
        school_cases += np.count_nonzero((ill_age >= 5) & (ill_age < 13))
        adult_people += np.count_nonzero((age >= 30) & (age < 50))
        adult_cases += np.count_nonzero((ill_age >= 30) & (ill_age < 50))
        in_state += np.bincount(run["states"][school], minlength=len(names))
    return {
        "incidence_5_12": school_age / max(school_people, 1) / years,
        "case_incidence_5_12": school_cases / max(school_people, 1) / years,
        "case_incidence_30_49": adult_cases / max(adult_people, 1) / years,
        **{
            f"prevalence_5_12_{name}": float(value / max(school_people, 1))
            for name, value in zip(names, in_state, strict=True)
        },
        "incidence_per_person_year": infections / people / years,
        "incidence_by_age": {
            f"{low}-{high - 1}": float(by_age[index] / max(population[index], 1) / years)
            for index, (low, high) in enumerate(AGE_GROUPS)
        },
        **{
            f"incidence_{low}_{high - 1}": float(by_age[index] / max(population[index], 1) / years)
            for index, (low, high) in enumerate(AGE_GROUPS)
        },
        "persisting_share": persisted / len(runs),
    }


def summarise(runs: list[dict], steps_per_day: int) -> dict:
    """Pool the runs of `epidemic`."""
    offspring, intervals, settings, deaths = [], [], [], {"infection": [], "onset": []}
    before_symptoms, without_symptoms = [], []
    growth, attack, names = [], [], runs[0]["settings"]
    by_age = np.zeros(len(AGE_GROUPS))
    generations = [_generations(run["record"]) for run in runs]
    last = _last_early_generation(generations, runs)
    for run, generation in zip(runs, generations, strict=True):
        record, population = run["record"], run["population"]
        count = len(record["step"])
        attack.append(np.unique(record["person"]).size / population)
        seed = record["setting"] == names.index("seed")  # the first cases
        traced = record["infector_record"] >= 0  # not the first cases nor those from outside
        children = np.bincount(record["infector_record"][traced], minlength=count)
        early = (generation >= 1) & (generation <= last)
        offspring.append(children[early])
        without_symptoms.append(record["asymptomatic"][early] == 1)
        # the first cases start infectious, without a latent period, so the intervals of the people
        # they infect are too short: only infections by infected people count, those whose
        # infector is of an early generation
        timed = (generation >= 2) & (generation <= last + 1)
        intervals.append(
            (record["step"][timed] - record["step"][record["infector_record"][timed]])
            / steps_per_day
        )
        settings.append(np.bincount(record["setting"][early], minlength=len(names)))
        # as the serial interval studies count it: of the infections by people who had symptoms
        # (sooner or later), the share caused before their onset
        source = record["infector_record"][early]
        by_ill = (source >= 0) & (record["asymptomatic"][np.maximum(source, 0)] == 0)
        before_symptoms.append(record["infector_symptomatic"][early][by_ill] == 0)
        age = run["age"][record["person"][~seed]]
        for index, (low, high) in enumerate(AGE_GROUPS):
            by_age[index] += np.count_nonzero((age >= low) & (age < high))
        died = record["death_step"] >= 0
        deaths["infection"].append((record["death_step"] - record["step"])[died] / steps_per_day)
        with_onset = died & (record["onset_step"] >= 0)
        deaths["onset"].append(
            (record["death_step"] - record["onset_step"])[with_onset] / steps_per_day
        )
        growth.append(_growth_rate(record, population, steps_per_day))
    offspring = np.concatenate(offspring)
    without_symptoms = np.concatenate(without_symptoms)
    intervals = np.concatenate(intervals)
    settings = np.sum(settings, axis=0)
    rates = np.array([rate for rate in growth if np.isfinite(rate)])
    mean_rate = float(rates.mean()) if len(rates) else math.nan
    return {
        "runs": len(runs),
        "early_infections": int(len(offspring)),
        "early_generations": int(last),
        "reproduction_number": float(offspring.mean()) if len(offspring) else math.nan,
        "reproduction_number_se": float(offspring.std(ddof=1) / math.sqrt(len(offspring)))
        if len(offspring) > 1
        else math.nan,
        "growth_rate": mean_rate,
        "reproduction_number_from_growth": _euler_lotka(mean_rate, intervals),
        "generation_interval": float(intervals.mean()) if len(intervals) else math.nan,
        "share_before_symptoms": float(np.concatenate(before_symptoms).mean())
        if before_symptoms
        else math.nan,
        "dispersion_k": negative_binomial_k(offspring),
        "share_without_symptoms": float(without_symptoms.mean()) if len(offspring) else math.nan,
        "relative_offspring_without_symptoms": _ratio_of_means(
            offspring[without_symptoms], offspring[~without_symptoms]
        ),
        "setting_shares": {
            name: float(value / max(settings.sum(), 1))
            for name, value in zip(names, settings, strict=True)
            if value
        },
        "age_shares": {
            f"{low}-{high - 1}": float(value / max(by_age.sum(), 1))
            for (low, high), value in zip(AGE_GROUPS, by_age, strict=True)
        },
        "share_under_18": float(by_age[:2].sum() / max(by_age.sum(), 1)),
        "attack_rate": float(np.mean(attack)),
        "infection_to_death": _mean(deaths["infection"]),
        "onset_to_death": _mean(deaths["onset"]),
        "deaths": int(sum(len(values) for values in deaths["infection"])),
    }


def _generations(record) -> np.ndarray:
    """The generation of every infection: 0 for the first cases, 1 for those they infected..."""
    infector = record["infector_record"]
    generation = np.zeros(len(infector), np.int64)
    for row in np.flatnonzero(infector >= 0):  # an infector's row comes before the infected's
        generation[row] = generation[infector[row]] + 1
    return generation


def _last_early_generation(generations: list[np.ndarray], runs: list[dict]) -> int:
    """The last generation whose own infections still fall within the first `EARLY_SHARE` of the
    town infected, on average over the runs (at least generation 1)."""
    last = 1
    deepest = max((int(generation.max()) for generation in generations), default=0)
    for candidate in range(1, deepest):
        through = np.mean(
            [
                np.count_nonzero(generation <= candidate + 1) / run["population"]
                for generation, run in zip(generations, runs, strict=True)
            ]
        )
        if through > EARLY_SHARE:
            break
        last = candidate
    return last


def _ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    if not len(numerator) or not len(denominator) or denominator.mean() == 0:
        return math.nan
    return float(numerator.mean() / denominator.mean())


def _mean(parts: list[np.ndarray]) -> float:
    values = np.concatenate(parts) if parts else np.zeros(0)
    return float(values.mean()) if len(values) else math.nan


def _growth_rate(record, population, steps_per_day) -> float:
    """Exponential growth rate per day of the daily infections in the early phase."""
    early = record["step"][: max(int(EARLY_SHARE * population), 1)]
    days = early // steps_per_day
    days = days[days < days.max()]  # the last day is cut off by the share, not over
    if len(days) < 20 or days.max() < 7:
        return math.nan
    daily = np.bincount(days)
    time = np.arange(len(daily))
    start = int(np.argmax(np.cumsum(daily) >= daily.sum() * 0.2))
    fit = slice(start, len(daily))
    if len(daily[fit]) < 5:
        return math.nan
    return float(np.polyfit(time[fit], np.log1p(daily[fit]), 1)[0])


def _euler_lotka(rate: float, intervals: np.ndarray) -> float:
    """Reproduction number from the growth rate and the generation intervals (Wallinga and
    Lipsitch 2007): R = 1 / E[exp(-r T)]."""
    if not np.isfinite(rate) or not len(intervals):
        return math.nan
    return float(1.0 / np.mean(np.exp(-rate * intervals)))


_LGAMMA = np.frompyfunc(math.lgamma, 1, 1)


def negative_binomial_k(counts: np.ndarray, low: float = 1e-3, high: float = 1e3) -> float:
    """Maximum likelihood dispersion k of a negative binomial fitted to `counts` (the mean is the
    sample mean, as in Lloyd-Smith et al. 2005); `high` when the counts are not overdispersed."""
    counts = np.asarray(counts, float)
    mean = counts.mean() if len(counts) else 0.0
    if mean <= 0:
        return math.nan

    def log_likelihood(log_k: float) -> float:
        k = math.exp(log_k)
        terms = _LGAMMA(counts + k).astype(float) - math.lgamma(k)
        return float(
            terms.sum()
            + len(counts) * k * math.log(k / (k + mean))
            - counts.sum() * math.log(k + mean)
        )

    # golden-section search on log k; the terms that do not depend on k are left out
    left, right = math.log(low), math.log(high)
    ratio = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        first = right - ratio * (right - left)
        second = left + ratio * (right - left)
        if log_likelihood(first) < log_likelihood(second):
            left = first
        else:
            right = second
    return math.exp((left + right) / 2)
