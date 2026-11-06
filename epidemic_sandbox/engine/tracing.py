"""Who infected whom, where and when: an optional record of a run.

It is what measuring a disease in the town needs: the reproduction number, the generation
interval, the household attack rate, where infections happen, how long after the onset people
die. The engine fills it when it is asked to (`EpidemicModel(..., trace=True)`); a run with a
trace is the same run as one without, because the choice of the infector uses its own random
numbers.

The infector of an infection is one of the infectious people in contact with the infected person
in that step, chosen with the chance of their share of the person's exposure, as the engine
weighs it (own household 1, everybody else the community factor), or nobody in the town: an
infection from outside, with the chance of the share of the outside infections in the person's
hazard.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from epidemic_sandbox.disease_models.disease_model_abstract import ASYMPTOMATIC, DAYS_PER_YEAR
from epidemic_sandbox.engine.spatial_kernels import pick_infectors
from epidemic_sandbox.world.population import PRESCHOOL, RESIDENT, SCHOOL, UNIVERSITY
from epidemic_sandbox.world.schedule import FIRST_ACTIVITY, HOME, LUNCH, WORK

TRACE_SEED = 20_261_105  # the attribution has its own random numbers, so the run is not changed
NOT_YET = -1
BASE_SETTINGS = (
    "seed",  # a first case of the run
    "outside",  # infected from outside the town (the disease's outside infections)
    "household",  # members of one private household, at home
    "care_home",  # residents and staff of a care home
    "visit",  # a visitor and the household they visit
    "preschool",
    "school",
    "university",
    "work",
    "lunch",
)


@dataclass
class DiseaseTrace:
    """The infections of one disease, one row each, in the order they happened.

    `infector_record` is the row of the infector's own infection (-1 for the first cases and
    the infections from outside), so that offspring and generation intervals can be counted;
    `infector_symptomatic` whether the infector had symptoms (was in a symptomatic state) when
    they infected (-1 without an infector). Steps are those of the engine; `onset_step` and
    `death_step` stay -1 if the person never had symptoms or did not die; `asymptomatic` is 1 for
    an infection that took the course without symptoms.
    """

    step: list = field(default_factory=list)
    person: list = field(default_factory=list)
    infector: list = field(default_factory=list)
    infector_record: list = field(default_factory=list)
    setting: list = field(default_factory=list)
    infector_symptomatic: list = field(default_factory=list)
    onset_step: list = field(default_factory=list)
    death_step: list = field(default_factory=list)
    asymptomatic: list = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(values, np.int64) for name, values in vars(self).items()}


class InfectionTrace:
    """The record of a run: a `DiseaseTrace` per disease and the names of the settings."""

    def __init__(self, model):
        activities = model.town_parameters.schedule.activities
        self.settings = (*BASE_SETTINGS, *(activity.name for activity in activities))
        self.diseases = {name: DiseaseTrace() for name in model.diseases}
        self.current = {name: np.full(model.population, -1, np.int64) for name in model.diseases}
        self._rng = np.random.default_rng(TRACE_SEED)
        # the step of the onset of each person's current infection (NOT_YET before it)
        self.onset = {name: np.full(model.population, NOT_YET, np.int64) for name in model.diseases}
        self._symptomatic = {}
        self._without_symptoms = {
            name: np.zeros(model.population, np.bool_) for name in model.diseases
        }
        for name in model.diseases:
            state = model.disease_states[name]
            seeds = np.flatnonzero(state.ever_infected)
            self._append(name, seeds, 0, np.full(len(seeds), -1), self.settings.index("seed"))
            self._symptomatic[name] = self._is_symptomatic(model, name)
            trace = self.diseases[name]
            for person in seeds[self._symptomatic[name][seeds]]:
                trace.onset_step[self.current[name][person]] = 0
                self.onset[name][person] = 0

    def record_infections(
        self, model, name, infected, infectiousness, grid, cells, exposure, relative=1.0
    ) -> None:
        """Note the people infected with `name` in this step, with their infectors; `exposure` and
        `relative` are what the engine drew them with."""
        if not len(infected):
            return
        disease = model.diseases[name]
        weight = relative * disease.outside_weight(model.disease_states[name])
        outside = self._outside_share(disease, infected, exposure, weight)
        from_outside = self._rng.random(len(infected)) < outside
        infectors = pick_infectors(
            model.x,
            model.y,
            infected.astype(np.int64),
            self._rng.random(len(infected)),
            infectiousness,
            model.town.household,
            model.venue,
            np.float32(disease.parameters["community_factor"]),
            *grid,
            cells,
            float(disease.contact_distance),
        )
        infectors[from_outside] = -1
        settings = self._settings_of(model, infected, infectors)
        settings[from_outside] = self.settings.index("outside")
        # an infector with symptoms is one whose illness has begun: in a symptomatic state or one
        # that follows it (isolation, a waning part), at the start of this step
        onset = self.onset[name][np.maximum(infectors, 0)]
        symptomatic = np.where(
            infectors >= 0, ((onset != NOT_YET) & (onset <= model.steps)).astype(np.int64), -1
        )
        self._append(name, infected, model.steps, infectors, settings, symptomatic)

    @staticmethod
    def _outside_share(disease, infected, exposure, relative) -> np.ndarray:
        """The share of the infections from outside in the hazard of each infected person;
        `relative` weights the outside part (interactions, the age effect), while the
        susceptibility and any protection weaken both parts alike."""
        rate = disease.parameters["outside_infections"] / DAYS_PER_YEAR
        if rate <= 0:
            return np.zeros(len(infected))
        transmission = disease.parameters["transmission"]
        contact = -math.log(1.0 - transmission) if transmission < 1.0 else math.inf
        exposed = np.asarray(exposure, np.float64)[infected]
        inside = np.where(exposed > 0, contact * exposed, 0.0)
        outside = rate * np.broadcast_to(np.asarray(relative, np.float64), exposure.shape)[infected]
        return outside / (outside + inside)

    def record_deaths(self, model, name, before: np.ndarray) -> None:
        """Note who died of `name` in this step; `before` are the states at its start."""
        dead = model.diseases[name].dead_index
        died = np.flatnonzero((model.disease_states[name].state == dead) & (before != dead))
        trace = self.diseases[name]
        for person in died:
            record = self.current[name][person]
            if record >= 0:
                trace.death_step[record] = model.steps

    def record_onsets(self, model) -> None:
        """Note who began to have symptoms in this step (at the end of the step), and whose
        infection took the course without symptoms."""
        for name in model.diseases:
            now = self._is_symptomatic(model, name)
            trace = self.diseases[name]
            disease = model.diseases[name]
            if ASYMPTOMATIC in disease.STATE_COLORS:
                state = model.disease_states[name].state
                without = state == disease.state_index(ASYMPTOMATIC)
                for person in np.flatnonzero(without & ~self._without_symptoms[name]):
                    record = self.current[name][person]
                    if record >= 0:
                        trace.asymptomatic[record] = 1
                self._without_symptoms[name] = without
            for person in np.flatnonzero(now & ~self._symptomatic[name]):
                record = self.current[name][person]
                if record >= 0 and trace.onset_step[record] == NOT_YET:
                    trace.onset_step[record] = model.steps + 1
                    self.onset[name][person] = model.steps + 1
            self._symptomatic[name] = now

    def _append(self, name, people, step, infectors, settings, symptomatic=-1) -> None:
        trace = self.diseases[name]
        current = self.current[name]
        settings = np.broadcast_to(settings, len(people))
        symptomatic = np.broadcast_to(symptomatic, len(people))
        for person, infector, setting, ill in zip(
            people, infectors, settings, symptomatic, strict=True
        ):
            trace.step.append(step)
            trace.person.append(int(person))
            trace.infector.append(int(infector))
            trace.infector_record.append(int(current[infector]) if infector >= 0 else -1)
            trace.setting.append(int(setting))
            trace.infector_symptomatic.append(int(ill))
            trace.onset_step.append(NOT_YET)
            trace.death_step.append(NOT_YET)
            trace.asymptomatic.append(0)
            current[person] = len(trace.step) - 1
            self.onset[name][person] = NOT_YET

    def _settings_of(self, model, people: np.ndarray, infectors: np.ndarray) -> np.ndarray:
        """The setting of each infection, from where the infected person is and who infected
        them: at home by a member of the household, at home by a visitor or on a visit, at a
        place of the day, at lunch or on an errand."""
        town, code = model.town, model._code[people].astype(np.int64)
        home, work = model._detour
        code = np.where(home[people], HOME, np.where(work[people], WORK, code))
        at_home = code == HOME
        known = infectors >= 0
        same_household = np.zeros(len(people), np.bool_)
        same_household[known] = town.household[infectors[known]] == town.household[people[known]]
        index = self.settings.index
        setting = np.full(len(people), index("work"))
        setting[at_home & same_household] = index("household")
        setting[(at_home & ~same_household) | self._visit(model, code)] = index("visit")
        setting[at_home & (town.role[people] == RESIDENT)] = index("care_home")
        place = town.place_kind[people]
        at_work = code == WORK
        education = ((PRESCHOOL, "preschool"), (SCHOOL, "school"), (UNIVERSITY, "university"))
        for kind, label in education:
            setting[at_work & (place == kind)] = index(label)
        setting[at_work & (place == RESIDENT)] = index("care_home")
        setting[code == LUNCH] = index("lunch")
        for offset, activity in enumerate(model.town_parameters.schedule.activities):
            if activity.kind != "visit":
                setting[code == FIRST_ACTIVITY + offset] = index(activity.name)
        return setting

    @staticmethod
    def _visit(model, code: np.ndarray) -> np.ndarray:
        visits = np.zeros(len(code), np.bool_)
        for offset, activity in enumerate(model.town_parameters.schedule.activities):
            if activity.kind == "visit":
                visits |= code == FIRST_ACTIVITY + offset
        return visits

    @staticmethod
    def _is_symptomatic(model, name) -> np.ndarray:
        disease = model.diseases[name]
        return np.isin(model.disease_states[name].state, disease._symptomatic_indices)
