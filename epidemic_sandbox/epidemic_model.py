import math

import numpy as np

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.experiment_config import ExperimentConfig, check_world
from epidemic_sandbox.schedule import DAYS_PER_WEEK, HOME, HOURS_PER_DAY, LUNCH, draw_day_schedule
from epidemic_sandbox.spatial_kernels import build_grid, count_infectious_contacts, move_toward
from epidemic_sandbox.town import (
    METERS_PER_KILOMETER,
    TownParameters,
    build_town,
    destination_coordinates,
)


class EpidemicModel:
    """People living in a town, going about their own days; infectious people infect those close by.

    Every person is a row in a set of arrays with their own age, role, habits, way of travelling
    and, for every disease, their own susceptibility, infectiousness and risk. Diseases can change
    each other's course through interactions. A run is reproducible from the seed whatever the
    number of CPU threads.
    """

    def __init__(self, config: ExperimentConfig, town_parameters: TownParameters):
        self.world = dict(config.world)
        check_world(self.world, town_parameters)
        self.town_parameters = town_parameters
        population = self.world["population"]
        self.rng = np.random.default_rng(self.world["seed"])
        self.town = build_town(
            population, float(self.world["arena_size"]), town_parameters, self.rng
        )
        self.x = self.town.home_x.copy()
        self.y = self.town.home_y.copy()
        steps_per_day = self.world["steps_per_day"]
        self.step_hours = HOURS_PER_DAY / steps_per_day
        self.diseases = {
            name: DISEASE_MODELS[disease.model](
                1 / steps_per_day, age_effects=disease.age_effects, **disease.parameters
            )
            for name, disease in config.diseases.items()
        }
        self.disease_states = {
            name: self.diseases[name].create_state(
                population,
                self.rng.choice(population, config.diseases[name].initial_infected, replace=False),
                self.rng,
                self.town.age,
            )
            for name in self.diseases
        }
        self.interactions = [
            (
                interaction,
                np.isin(list(self.diseases[interaction.source].STATE_COLORS), interaction.states),
            )
            for interaction in config.interactions
        ]
        self.steps = 0
        self.history = []
        self._people = np.arange(population)
        self._schedule = None
        self._pointer = np.zeros(population, np.int16)
        self._code = np.full(population, HOME, np.int8)
        self._target_x, self._target_y = self.x.copy(), self.y.copy()
        self._departure_time = np.zeros(population)
        self._planned_day = None
        self._follow_schedule()
        self._record_day()

    @property
    def population(self) -> int:
        return self.world["population"]

    @property
    def day(self) -> float:
        return self.steps / self.world["steps_per_day"]

    @property
    def weekday(self) -> int:
        """Day of the week, 0 is Monday; the run starts on Monday at midnight."""
        return (self.steps // self.world["steps_per_day"]) % DAYS_PER_WEEK

    @property
    def hour(self) -> float:
        """Hour of the day, 0 to 24."""
        steps_per_day = self.world["steps_per_day"]
        return (self.steps % steps_per_day) * HOURS_PER_DAY / steps_per_day

    @property
    def schedule(self):
        """Today's events of everybody (see schedule.DaySchedule); None before the first step."""
        return self._schedule

    def step(self) -> None:
        """Let people go about their day, then advance every disease by one step."""
        world = self.world
        self._follow_schedule()
        step_start = self.steps * self.step_hours
        walking_hours = np.clip(
            step_start + self.step_hours - np.maximum(step_start, self._departure_time),
            0.0,
            self.step_hours,
        )
        alive = ~self._dead()
        move_toward(
            self.x,
            self.y,
            self._target_x,
            self._target_y,
            self.town.home_x,
            self.town.home_y,
            self._goes_home(),
            walking_hours * self.town.speed_kmh * METERS_PER_KILOMETER * alive,
        )
        arena_size = float(world["arena_size"])
        longest = max(disease.contact_distance for disease in self.diseases.values())
        cells_per_side = max(1, min(int(arena_size // longest), 2 * math.isqrt(self.population)))
        grid = build_grid(self.x, self.y, arena_size, cells_per_side)
        exposures = {}
        for name, disease in self.diseases.items():
            disease_state = self.disease_states[name]
            contacts = count_infectious_contacts(
                self.x,
                self.y,
                disease.exposable_mask(disease_state),
                self._infectiousness(name),
                *grid,
                cells_per_side,
                float(disease.contact_distance),
            )
            exposures[name] = contacts * self._relative_susceptibility(name)
        infected = {
            name: disease.draw_infections(self.disease_states[name], exposures[name], self.rng)
            for name, disease in self.diseases.items()
        }
        # Deaths are settled before the infected enter, one disease after another in a random
        # order: whoever dies of one disease is dead for the rest and is not counted twice.
        names = list(self.diseases)
        for index in self.rng.permutation(len(names)):
            self.diseases[names[index]].finish_phases(self.disease_states[names[index]], self.rng)
            self._bury_the_dead()
        for name, disease in self.diseases.items():
            disease.apply_infections(self.disease_states[name], infected[name], self.rng)
        self.steps += 1
        if self.steps % world["steps_per_day"] == 0:
            self._record_day()

    def state_counts(self) -> dict[str, list[int]]:
        """Number of people in each state of each disease, in the order of the disease's states.

        The dead of a disease are its own victims: people killed by another disease are in no
        state of it, so every death is shown once, under the disease that caused it.
        """
        counts = {}
        for name, disease in self.diseases.items():
            state = self.disease_states[name]
            by_state = np.bincount(state.state, minlength=len(disease.STATE_COLORS))
            by_state[disease.dead_index] = state.deaths
            counts[name] = by_state.tolist()
        return counts

    def is_quiescent(self) -> bool:
        """True when no disease can change any more, so the run can be cut short."""
        return all(
            self.diseases[name].is_quiescent(self.disease_states[name]) for name in self.diseases
        )

    def current_code(self, person: int) -> int:
        """Where `person` is heading according to their schedule: HOME, WORK or an activity."""
        return int(self._code[person])

    def next_event_time(self, person: int) -> float | None:
        """Hour of the day of the next event in the schedule of `person`, or None."""
        when = self._schedule.times[person, self._pointer[person]]
        return float(when) if np.isfinite(when) else None

    def destination(self, person: int) -> tuple[float, float]:
        """Where `person` is heading: home if ill and staying in, otherwise the schedule's place."""
        if self._goes_home()[person]:
            return float(self.town.home_x[person]), float(self.town.home_y[person])
        return float(self._target_x[person]), float(self._target_y[person])

    def departure_time(self, person: int) -> float:
        """Simulated hour at which `person` sets off for their current destination."""
        return float(self._departure_time[person])

    def interaction_factors(self, person: int) -> dict[str, dict[str, float]]:
        """How the interactions currently change `person`'s susceptibility and infectiousness."""
        factors = {name: {"susceptibility": 1.0, "infectiousness": 1.0} for name in self.diseases}
        for interaction, source_states in self.interactions:
            if source_states[self.disease_states[interaction.source].state[person]]:
                factors[interaction.target][interaction.effect] *= interaction.factor
        return factors

    def _follow_schedule(self) -> None:
        """Draw a new day when one starts, then send off everybody whose next event is due.

        A person takes one event per step: when several are due, the later ones follow in the
        next steps, so that an outing shorter than a step is still visited for a step.
        """
        steps_per_day = self.world["steps_per_day"]
        day_index = self.steps // steps_per_day
        if day_index != self._planned_day:
            calendar = self.town_parameters.schedule.calendar
            self._schedule = draw_day_schedule(
                self.town.role,
                self.town.traits,
                self.town_parameters.schedule,
                calendar.is_day_off(day_index),
                self.rng,
                calendar.closed_places(day_index),
            )
            self._planned_day = day_index
            self._pointer[:] = 0
            away = np.flatnonzero(self._code != HOME)
            self._code[away] = HOME
            self._aim(away)
        until = (self.steps + 1) * self.step_hours - day_index * HOURS_PER_DAY
        times, codes = self._schedule.times, self._schedule.codes
        due = np.flatnonzero(times[self._people, self._pointer] <= until)
        pointer = self._pointer[due]
        self._code[due] = codes[due, pointer]
        self._departure_time[due] = day_index * HOURS_PER_DAY + times[due, pointer]
        self._pointer[due] += 1
        if len(due):
            self._aim(due)

    def _aim(self, people: np.ndarray) -> None:
        self._target_x[people], self._target_y[people] = destination_coordinates(
            self.town, self._code[people], people, self._schedule
        )

    def _bury_the_dead(self) -> None:
        """Whoever died of one disease is dead for all of them."""
        dead = self._dead()
        for name, disease in self.diseases.items():
            state = self.disease_states[name]
            disease.bury(state, np.flatnonzero(dead & (state.state != disease.dead_index)))

    def _dead(self) -> np.ndarray:
        dead = np.zeros(self.population, np.bool_)
        for name, disease in self.diseases.items():
            dead |= self.disease_states[name].state == disease.dead_index
        return dead

    def _goes_home(self) -> np.ndarray:
        """People who head home: ill and staying in, or ill and skipping errands and visits."""
        home = np.zeros(self.population, np.bool_)
        outing = self._code >= LUNCH
        for disease_state in self.disease_states.values():
            home |= disease_state.stays_home | (disease_state.limits_outings & outing)
        return home

    def _infectiousness(self, disease: str) -> np.ndarray:
        """Infectiousness of everybody for `disease`: 0 if not infectious, 1 if fully, or scaled."""
        disease_state = self.disease_states[disease]
        weights = (
            self.diseases[disease].infectious_mask(disease_state).astype(np.float32)
            * disease_state.infectiousness
        )
        for interaction, source_states in self.interactions:
            if interaction.target == disease and interaction.effect == "infectiousness":
                weights *= self._factor_where_in_states(interaction, source_states)
        return weights

    def _relative_susceptibility(self, disease: str) -> np.ndarray | np.float32:
        """How easily everybody catches `disease` compared with a person without interactions."""
        susceptibility = np.float32(1.0)
        for interaction, source_states in self.interactions:
            if interaction.target == disease and interaction.effect == "susceptibility":
                susceptibility = susceptibility * self._factor_where_in_states(
                    interaction, source_states
                )
        return susceptibility

    def _factor_where_in_states(self, interaction, source_states: np.ndarray) -> np.ndarray:
        in_states = source_states[self.disease_states[interaction.source].state]
        return np.where(in_states, np.float32(interaction.factor), np.float32(1.0))

    def _record_day(self) -> None:
        self.history.append(
            {
                "day": self.steps // self.world["steps_per_day"],
                "counts": self.state_counts(),
                "ever_infected": {
                    name: int(np.count_nonzero(state.ever_infected))
                    for name, state in self.disease_states.items()
                },
                "cumulative_infections": {
                    name: state.cumulative_infections for name, state in self.disease_states.items()
                },
                "deaths": {name: state.deaths for name, state in self.disease_states.items()},
            }
        )
