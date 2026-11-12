"""Measuring how a disease behaves in the town, as studies of real epidemics measure it.

Every number that a disease file marks as Fitted is fitted with these measurements, and its
`[targets]` table names what was aimed at; `python -m epidemic_sandbox.analysis.calibrate <disease>`
measures a disease and compares it with its targets.

* The household attack rate: first cases are placed at random among people living in private
  households of two or more (of the ages of the first cases of the study, if it names them), and
  nobody outside their households can be infected, nor anybody from outside the town: the share of
  the other members (of the ages the study followed) infected within the follow-up, in chains of
  infections within the household that go back to its first case, is what household studies
  report, with every infected member counted wherever the members met (at home or out together),
  and without the infections the town around would bring in. A first case infected again (a
  disease without immunity) is not a secondary case.
* The reproduction number, the generation interval and the dispersion of the number of people one
  person infects are taken from the early generations of an epidemic: the infections caused by the
  first cases (generation 1), by those (generation 2) and so on, up to the last generation whose
  own infections still fall, on average over the runs, within the first `EARLY_SHARE` of the town
  infected, so that the depletion of susceptible people does not lower what they do. The choice
  depends on generations, not on how many people a particular infection infected, so it does not
  select. For a disease that infects many in one generation, a run starts from one case. The
  growth rate of the daily infections is fitted between `GROWTH_WINDOW` shares of the town
  infected, and the reproduction number is also derived from it. The share of transmission before
  the onset of symptoms is that of the infections caused by people who had symptoms, as studies
  of pairs with known onsets measure it; the share of infections without symptoms and how many
  people they infect relative to the others are given alongside.
* Where infections happen (and the share within households), the attack rate of the people met
  on visits, who is infected by age, the illness rates by age of the first `SEASON_INFECTED` of the
  town infected, the chance
  of infection per hour next to an infectious person, and how long after the infection and after
  the onset of symptoms people die.
"""

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from epidemic_sandbox.config.experiment import ExperimentLibrary
from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.engine.epidemic_model import EpidemicModel
from epidemic_sandbox.world.population import RESIDENT
from epidemic_sandbox.world.schedule import FIRST_ACTIVITY

# Model: an infection belongs to the early phase when, by the time it has done its infecting, at
# most this share of the town has been infected: the depletion of susceptible people then lowers
# its number of infections by at most this share.
EARLY_SHARE = 0.02
# Model: the generations counted for the reproduction number start at this one. The first cases
# are not typical people: they start infectious without a latent period and are chosen by chance,
# not by contact, so their own offspring are fewer (generation 0: COVID-19 1.4-1.6 against 2.0-2.4
# later; SARS-CoV-1 0.47 against 1.0-1.3 later). Beyond that the offspring of a generation depend
# on the window: they fall with the generation as households and workplaces saturate locally
# (COVID-19, 10 first cases, 30,000 people: 2.2, 2.2, 2.0, 2.0, 1.7, 1.5 for generations 1-6;
# influenza is flat from generation 0; SARS rises over generations 1-3 and then scatters around
# 1.0), so the number is a choice of window with an uncertainty of about 0.3, not a measurement
# of one quantity. `reproduction_number_from_growth` (the growth rate with the model's own
# generation intervals, Euler-Lotka) is reported beside it as the asymptotic estimate, and the
# fits and checks of the diseases are read with both. A target can set `first_generation`
# (measles, from one first case, depletes the town in two generations).
FIRST_GENERATION = 3
# Model: the growth rate is fitted to the daily infections from the day 0.3 % of the town has been
# infected to the day 3 % has been: after the first generations, whose numbers depend on how the
# first cases were placed, and before the depletion of susceptible people slows the growth.
GROWTH_WINDOW = (0.003, 0.03)
DEFAULT_WORLD = {"population": 3000, "area_km2": 4.0}
EPIDEMIC_POPULATION = 30_000
HOUSEHOLD_DAYS = 28  # the follow-up of a household study when its target gives none
AGE_GROUPS = ((0, 5), (5, 18), (18, 30), (30, 50), (50, 65), (65, 80), (80, 120))
# the age groups of the illness rates of CDC's influenza burden estimates, and the reference group
ILLNESS_GROUPS = ((0, 5), (5, 18), (18, 50), (50, 65), (65, 120))
ILLNESS_REFERENCE = (18, 50)
# Derived: the illness rates by age of a season are those of an epidemic that has infected this
# share of the town: CDC's 21-41 million symptomatic influenza illnesses in a US population of
# 310-330 million are 7-13 % of it a year (2010-11 to 2018-19), 9 % in the seasons of the age
# targets, which is 14 % infected at the 64 % of infections with symptoms.
SEASON_INFECTED = 0.14
HOURS_PER_DAY = 24


@dataclass
class Experiment:
    """One disease alone in the default town: the disease name, changes of its parameters and
    of its age effects, and the world."""

    disease: str
    overrides: dict
    age_effects: dict | None = None
    population: int = DEFAULT_WORLD["population"]
    area_km2: float = DEFAULT_WORLD["area_km2"]

    def of_size(self, population: int) -> "Experiment":
        """The same disease in a town of `population` people, as dense as the default town."""
        area = DEFAULT_WORLD["area_km2"] * population / DEFAULT_WORLD["population"]
        return Experiment(self.disease, self.overrides, self.age_effects, population, area)

    def config(self, library: ExperimentLibrary, seed: int, initial_infected: int):
        disease = library.catalog[self.disease]
        effects = {key: [list(row) for row in rows] for key, rows in disease.age_effects.items()}
        effects.update(self.age_effects or {})
        world = {
            "population": self.population,
            "area_km2": self.area_km2,
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

    def model(self, library: ExperimentLibrary):
        """The disease model with the parameters and age effects of the experiment."""
        config = self.config(library, seed=0, initial_infected=0)
        disease = config.diseases[self.disease]
        return DISEASE_MODELS[disease.model](
            1 / library.steps_per_day, age_effects=disease.age_effects, **disease.parameters
        )


def household_attack_rate(
    library: ExperimentLibrary,
    experiment: Experiment,
    seeds: list[int],
    index_cases: int = 40,
    days: int = HOUSEHOLD_DAYS,
    workers: int | None = None,
    index_ages: tuple[float, float] | None = None,
    contact_ages: tuple[float, float] | None = None,
) -> dict:
    """Share of the household members of first cases infected by their household within `days`;
    `index_ages` and `contact_ages` ([from, to) years) restrict the first cases and the members
    counted to the ages of a study."""
    jobs = [
        (library, experiment, seed, index_cases, days, index_ages, contact_ages) for seed in seeds
    ]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_household_run, *zip(*jobs, strict=True)))
    infected = sum(result[0] for result in results)
    contacts = sum(result[1] for result in results)
    cases = sum(result[2] for result in results)
    direct = sum(result[3] for result in results)
    chains = sum(result[4] for result in results)
    per_run = np.array([result[0] / max(result[1], 1) for result in results])
    return {
        "household_attack_rate": infected / max(contacts, 1),
        "household_case_rate": cases / max(contacts, 1),
        "household_direct_rate": direct / max(contacts, 1),
        # only the infections by a chain within the household back to the first case: lower by the
        # members infected by another index household (at school, in a shop)
        "household_chain_rate": chains / max(contacts, 1),
        "standard_error": float(per_run.std(ddof=1) / math.sqrt(len(per_run)))
        if len(per_run) > 1
        else math.nan,
        "contacts": contacts,
    }


def _household_run(library, experiment, seed, index_cases, days, index_ages, contact_ages):
    # nobody outside the households of the first cases can be infected (their susceptibility is
    # 0) and nobody is infected from outside: the members are infected only by members of their
    # household, wherever they meet, and not first in an epidemic of the town that the first cases
    # would start
    alone = Experiment(
        experiment.disease,
        {**experiment.overrides, "outside_infections": 0.0},
        experiment.age_effects,
        experiment.population,
        experiment.area_km2,
    )
    config = alone.config(library, seed, 0)
    model = EpidemicModel(config, library.town, trace=True)
    name = experiment.disease
    town = model.town
    size = np.bincount(town.household)
    private = (town.role != RESIDENT) & (size[town.household] >= 2)
    eligible = private if index_ages is None else private & _aged(town.age, index_ages)
    rng = np.random.default_rng(seed + 7919)
    households = rng.permutation(np.unique(town.household[eligible]))[:index_cases]
    first = np.array(
        [rng.choice(np.flatnonzero((town.household == h) & eligible)) for h in households],
        np.int64,
    )
    disease, state = model.diseases[name], model.disease_states[name]
    state.susceptibility[~np.isin(town.household, households)] = 0.0
    disease.seed_infections(state, first, rng)
    trace = model.trace
    trace._append(name, first, 0, np.full(len(first), -1), trace.settings.index("seed"))
    for _ in range(days * model.world["steps_per_day"]):
        model.step()
    record = trace.diseases[name].arrays()
    # chains of infections within one household that go back to its first case are what a
    # household study attributes to it; `direct` are those infected by the first case itself
    household = town.household[record["person"]]
    seeded = record["infector"] < 0
    from_case = seeded.copy()
    direct = np.zeros(len(from_case), np.bool_)
    for row in range(len(from_case)):
        source = record["infector_record"][row]
        if source >= 0 and from_case[source] and household[source] == household[row]:
            from_case[row] = True
            direct[row] = seeded[source]
    secondary = from_case & ~seeded
    members = np.flatnonzero(np.isin(town.household, households))
    contacts = np.setdiff1d(members, first)
    if contact_ages is not None:
        contacts = contacts[_aged(town.age[contacts], contact_ages)]
    # people, not episodes, and not a first case infected again by a member it infected; as the
    # studies count them, every contact infected during the follow-up (a few were infected by
    # members of another of the households followed together)
    infected = np.count_nonzero(np.isin(contacts, record["person"][record["step"] > 0]))
    chains = np.count_nonzero(np.isin(contacts, record["person"][secondary]))
    ill = secondary & (record["onset_step"] >= 0)
    cases = np.count_nonzero(np.isin(contacts, record["person"][ill]))
    first_hand = np.count_nonzero(np.isin(contacts, record["person"][direct]))
    return infected, len(contacts), cases, first_hand, chains


def _aged(age: np.ndarray, ages) -> np.ndarray:
    low, high = ages
    return (age >= low) & (age < high)


def epidemic(
    library: ExperimentLibrary,
    experiment: Experiment,
    seeds: list[int],
    initial_infected: int = 10,
    days: int = 120,
    workers: int | None = None,
    first_generation: int = FIRST_GENERATION,
) -> dict:
    """The early reproduction number, generation interval, growth, dispersion, settings, visits,
    ages, illness by age and deaths of epidemics started by `initial_infected` cases, one run per
    seed, and the chance of infection per hour next to an infectious person."""
    jobs = [(library, experiment, seed, initial_infected, days) for seed in seeds]
    with ProcessPoolExecutor(workers) as pool:
        runs = list(pool.map(_epidemic_run, *zip(*jobs, strict=True)))
    measured = summarise(runs, library.steps_per_day, first_generation)
    measured["hourly_risk"] = hourly_risk(experiment.model(library))
    return measured


def hourly_risk(disease) -> float:
    """The chance of infection in an hour at full weight (within near distance, in the same room)
    next to a person with an infection with symptoms, averaged over the days they spread it."""
    weighted = total = 0.0
    for course in disease.courses(asymptomatic=False):
        for _, mean, weight in course.stays:
            if weight > 0 and math.isfinite(mean):
                weighted += course.probability * mean * weight
                total += course.probability * mean
    transmission = disease.parameters["transmission"]
    if total <= 0 or transmission >= 1.0:
        return math.nan if total <= 0 else 1.0
    # next to the person (within near distance) in the same room: the close range and the air
    hazard = (
        -math.log(1.0 - transmission) / HOURS_PER_DAY * (1.0 + disease.parameters["room_exposure"])
    )
    return float(1.0 - math.exp(-hazard * weighted / total))


def _epidemic_run(library, experiment, seed, initial_infected, days, visits=True):
    config = experiment.config(library, seed, initial_infected)
    model = EpidemicModel(config, library.town, trace=True)
    name = experiment.disease
    pairs = set()
    for _ in range(days * model.world["steps_per_day"]):
        model.step()
        if visits:
            _note_visit_pairs(model, name, pairs)
    record = model.trace.diseases[name].arrays()
    return {
        "record": record,
        "settings": model.trace.settings,
        "age": model.town.age,
        "population": model.population,
        "states": model.disease_states[name].state.copy(),
        "state_names": list(model.diseases[name].STATE_COLORS),
        "visit_pairs": np.array(sorted(pairs), np.int64).reshape(-1, 2),
    }


def _note_visit_pairs(model, name: str, pairs: set) -> None:
    """Add to `pairs` (infection record of the infectious person, person met) everybody who is in
    a home with an infectious person when one of the two is a visitor there, of another
    household, and could catch the disease: the contacts of a visit, as contact tracing lists
    them."""
    town = model.town
    venue = model.venue
    homes = (venue >= 0) & (venue < model._households)
    on_a_visit = np.isin(model._code, FIRST_ACTIVITY + model._visits)
    visitors = np.flatnonzero(on_a_visit & homes & (venue != town.household))
    if not len(visitors):
        return
    disease, state = model.diseases[name], model.disease_states[name]
    present = np.flatnonzero(homes & np.isin(venue, venue[visitors]))
    spreading = (disease.infectious_weight(state) * state.infectiousness)[present] > 0
    if not spreading.any():
        return
    exposable = disease.exposable_mask(state)[present]
    current = model.trace.current[name]
    for infectious in present[spreading]:
        here = present[(venue[present] == venue[infectious]) & exposable]
        here = here[town.household[here] != town.household[infectious]]
        visiting = town.household[infectious] != venue[infectious]
        if not visiting:  # a host: only the visitors count
            here = here[town.household[here] != venue[infectious]]
        for other in here:
            pairs.add((int(current[infectious]), int(other)))


def endemic(
    library: ExperimentLibrary,
    experiment: Experiment,
    seeds: list[int],
    initial_infected: int,
    days: int,
    burn_in: int,
    workers: int | None = None,
    episode_days: float = 0.0,
) -> dict:
    """Infections per person and year, overall and by age, of a disease that stays in the town,
    counted after `burn_in` days, and the share of the runs in which it was still there at the
    end. Infections of a person that start within `episode_days` of the start of their previous
    one count as one episode, as studies that find infections by weekly swabs count them."""
    jobs = [(library, experiment, seed, initial_infected, days, False) for seed in seeds]
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
        new_episode = _new_episodes(record, episode_days * steps_per_day)
        # every episode after the burn-in, in the town or from outside (the first cases are at
        # step 0)
        late = (record["step"] >= burn_in * steps_per_day) & (record["step"] > 0) & new_episode
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
        ill = (
            (record["step"] >= burn_in * steps_per_day)
            & (record["step"] > 0)
            & (record["onset_step"] >= 0)
        )
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


def _new_episodes(record, window_steps: float) -> np.ndarray:
    """Which infections start a new episode: those that start at least `window_steps` after the
    start of the person's previous infection (every infection when the window is 0)."""
    new = np.ones(len(record["step"]), np.bool_)
    if window_steps <= 0 or not len(new):
        return new
    order = np.lexsort((record["step"], record["person"]))
    person, step = record["person"][order], record["step"][order]
    repeat = np.zeros(len(order), np.bool_)
    repeat[1:] = (person[1:] == person[:-1]) & (step[1:] - step[:-1] < window_steps)
    new[order] = ~repeat
    return new


def summarise(
    runs: list[dict], steps_per_day: int, first_generation: int = FIRST_GENERATION
) -> dict:
    """Pool the runs of `epidemic`."""
    offspring, intervals, settings, deaths = [], [], [], {"infection": [], "onset": []}
    before_symptoms, without_symptoms = [], []
    growth, attack, names = [], [], runs[0]["settings"]
    visits_infected = visits_met = 0
    by_age = np.zeros(len(AGE_GROUPS))
    ill_by_age, people_by_age = np.zeros(len(ILLNESS_GROUPS)), np.zeros(len(ILLNESS_GROUPS))
    generations = [_generations(run["record"]) for run in runs]
    last = max(_last_early_generation(generations, runs), first_generation)
    for run, generation in zip(runs, generations, strict=True):
        record, population = run["record"], run["population"]
        count = len(record["step"])
        attack.append(np.unique(record["person"]).size / population)
        seed = record["setting"] == names.index("seed")  # the first cases
        traced = record["infector_record"] >= 0  # not the first cases nor those from outside
        children = np.bincount(record["infector_record"][traced], minlength=count)
        early = (generation >= first_generation) & (generation <= last)
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
        # illness over the whole run: the infections with symptoms by age, per person
        in_season = np.arange(count) < SEASON_INFECTED * population  # the rows are in time order
        ill_age = run["age"][record["person"][(record["onset_step"] >= 0) & ~seed & in_season]]
        for index, (low, high) in enumerate(ILLNESS_GROUPS):
            ill_by_age[index] += np.count_nonzero((ill_age >= low) & (ill_age < high))
            people_by_age[index] += np.count_nonzero((run["age"] >= low) & (run["age"] < high))
        # visits: the people met on a visit by an infectious person of an early generation, and
        # those of them infected by that person there
        pairs = run.get("visit_pairs", np.zeros((0, 2), np.int64))
        if len(pairs):
            met = pairs[np.isin(pairs[:, 0], np.flatnonzero(early | seed))]
            visits_met += len(met)
            infected_on_visits = (record["setting"] == names.index("visit")) & traced
            sources = record["infector_record"][infected_on_visits]
            infected_people = record["person"][infected_on_visits]
            caught = {
                (int(s), int(p))
                for s, p in zip(sources, infected_people, strict=True)
                if early[s] or seed[s]
            }
            visits_infected += len(caught & {tuple(pair) for pair in met.tolist()})
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
    illness = ill_by_age / np.maximum(people_by_age, 1)
    reference = illness[ILLNESS_GROUPS.index(ILLNESS_REFERENCE)]
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
        "household_share": float(settings[names.index("household")] / max(settings.sum(), 1)),
        "visit_attack_rate": visits_infected / visits_met if visits_met else math.nan,
        "age_shares": {
            f"{low}-{high - 1}": float(value / max(by_age.sum(), 1))
            for (low, high), value in zip(AGE_GROUPS, by_age, strict=True)
        },
        "share_under_18": float(by_age[:2].sum() / max(by_age.sum(), 1)),
        **{
            f"illness_ratio_{low}_{high - 1}": float(value / reference) if reference else math.nan
            for (low, high), value in zip(ILLNESS_GROUPS, illness, strict=True)
            if (low, high) != ILLNESS_REFERENCE
        },
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
    """Exponential growth rate per day of the daily infections between the days on which the
    share `GROWTH_WINDOW` of the town had been infected (nan if the epidemic did not get there or
    the window is shorter than five days)."""
    steps = np.sort(record["step"][record["step"] > 0])
    low, high = (int(share * population) for share in GROWTH_WINDOW)
    if len(steps) <= high or high <= low:
        return math.nan
    first, last = steps[low] // steps_per_day, steps[high] // steps_per_day
    if last - first < 5:
        return math.nan
    daily = np.bincount(steps // steps_per_day, minlength=last + 1)[first : last + 1]
    days = np.arange(first, last + 1)
    return float(np.polyfit(days, np.log(np.maximum(daily, 1)), 1)[0])


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
