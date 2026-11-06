import math

import numpy as np

from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.experiment_config import ExperimentConfig, check_world
from epidemic_sandbox.population import RESIDENT
from epidemic_sandbox.schedule import (
    DAYS_PER_WEEK,
    FIRST_ACTIVITY,
    HOME,
    HOURS_PER_DAY,
    LUNCH,
    WORK,
    draw_day_schedule,
)
from epidemic_sandbox.spatial_kernels import (
    build_grid,
    count_infectious_contacts_by_household,
    move_toward,
)
from epidemic_sandbox.town import (
    METERS_PER_KILOMETER,
    PLACE_KINDS,
    TownParameters,
    build_town,
    destination_coordinates,
    destination_discs,
    destination_venues,
)
from epidemic_sandbox.tracing import InfectionTrace


class EpidemicModel:
    """People living in a town, going about their own days; infectious people infect those close by.

    Every person is a row in a set of arrays with their own age, role, habits, way of travelling
    and, for every disease, their own susceptibility, infectiousness and risk. Diseases can change
    each other's course through interactions. A run is reproducible from the seed whatever the
    number of CPU threads.
    """

    def __init__(
        self, config: ExperimentConfig, town_parameters: TownParameters, trace: bool = False
    ):
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
        # Whether everybody has reached the venue they are heading for; on the way there is nobody
        # to infect. Everybody starts at home.
        self._inside = np.ones(population, np.bool_)
        self._households = int(self.town.household.max()) + 1
        self._visit_activities = [
            index
            for index, activity in enumerate(town_parameters.schedule.activities)
            if activity.kind == "visit"
        ]
        self._detour = (np.zeros(population, np.bool_), np.zeros(population, np.bool_))
        self._residents = self.town.role == RESIDENT
        self._in_common = np.zeros(population, np.bool_)
        self._common_share = np.zeros(population)
        for code, kind in PLACE_KINDS.items():
            if kind in town_parameters.places:
                members = self.town.school_venue >= 0
                members &= self.town.place_kind == code
                self._common_share[members] = town_parameters.places[kind].common_share
        self.venue = self.town.household.astype(np.int64)
        self._follow_schedule()
        self.trace = InfectionTrace(self) if trace else None
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
        home, work = self._detours()
        changed = (home != self._detour[0]) | (work != self._detour[1])
        self._inside[changed] = False  # a new goal: home when ill, or back to the workplace
        self._detour = (home, work)
        self._move_about(home | work)
        step_start = self.steps * self.step_hours
        walking_hours = np.clip(
            step_start + self.step_hours - np.maximum(step_start, self._departure_time),
            0.0,
            self.step_hours,
        )
        alive = ~self._dead()
        other_goal = home | work
        goal_x = np.where(work, self.town.work_x, self.town.home_x)
        goal_y = np.where(work, self.town.work_y, self.town.home_y)
        move_toward(
            self.x,
            self.y,
            self._target_x,
            self._target_y,
            goal_x,
            goal_y,
            other_goal,
            walking_hours * self.town.speed_kmh * METERS_PER_KILOMETER * alive,
        )
        goal_x = np.where(other_goal, goal_x, self._target_x)
        goal_y = np.where(other_goal, goal_y, self._target_y)
        self._inside |= (self.x == goal_x) & (self.y == goal_y)
        self.venue = self._venues(home, work)
        arena_size = float(world["arena_size"])
        longest = max((disease.contact_distance for disease in self.diseases.values()), default=1.0)
        cells_per_side = max(1, min(int(arena_size // longest), 2 * math.isqrt(self.population)))
        grid = build_grid(self.x, self.y, arena_size, cells_per_side)
        exposures, infectiousness, relative = {}, {}, {}
        for name, disease in self.diseases.items():
            disease_state = self.disease_states[name]
            infectiousness[name] = self._infectiousness(name)
            contacts, own_household = count_infectious_contacts_by_household(
                self.x,
                self.y,
                disease.exposable_mask(disease_state),
                infectiousness[name],
                self.town.household,
                self.venue,
                *grid,
                cells_per_side,
                float(disease.contact_distance),
            )
            community = disease.parameters["community_factor"]
            relative[name] = self._relative_susceptibility(name)
            exposures[name] = (
                own_household + np.float32(community) * (contacts - own_household)
            ) * relative[name]
        infected = {
            name: disease.draw_infections(
                self.disease_states[name], exposures[name], self.rng, relative[name]
            )
            for name, disease in self.diseases.items()
        }
        # Deaths are settled before the infected enter, one disease after another in a random
        # order: whoever dies of one disease is dead for the rest and is not counted twice.
        names = list(self.diseases)
        for index in self.rng.permutation(len(names)):
            name = names[index]
            before = self.disease_states[name].state.copy() if self.trace else None
            self.diseases[name].finish_phases(self.disease_states[name], self.rng)
            if self.trace:
                self.trace.record_deaths(self, name, before)
            self._bury_the_dead()
        for name, disease in self.diseases.items():
            disease_state = self.disease_states[name]
            alive = infected[name][disease_state.state[infected[name]] != disease.dead_index]
            disease.apply_infections(disease_state, infected[name], self.rng)
            if self.trace:
                self.trace.record_infections(
                    self,
                    name,
                    alive,
                    infectiousness[name],
                    grid,
                    cells_per_side,
                    exposures[name],
                    relative[name],
                )
        if self.trace:
            self.trace.record_onsets(self)
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
        """Where `person` is heading: home if ill and staying in, back to the workplace if ill and
        skipping the lunch outing, otherwise the schedule's place."""
        home, work = self._detours()
        if home[person]:
            return float(self.town.home_x[person]), float(self.town.home_y[person])
        if work[person]:
            return float(self.town.work_x[person]), float(self.town.work_y[person])
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

    def _move_about(self, detour: np.ndarray) -> None:
        """People who have settled at a place move to another spot of it after their dwell time.

        The time at one spot is random with the mean `dwell_hours`; at school and kindergarten
        small children move about as everybody, on outings they stay with the adult who takes
        them out. Pupils and teachers spend the share `common_share` of their stays in the common
        rooms of the school (corridors, canteen) instead of their class. Residents of a care home
        move about the home while they are up (`day_hours`) and are in their rooms otherwise.
        `detour` are those who are not where their day takes them (see `_detours`).
        """
        town = self.town
        settled = (self.x == self._target_x) & (self.y == self._target_y)
        # moves happen at the steps, so this chance makes the mean stay `dwell_hours` long
        chance = min(1.0, self.step_hours / self.town_parameters.dwell_hours)
        away = (self._code != HOME) & ~detour
        on_their_own = (town.guardian < 0) | (self._code == WORK)
        residents = self._residents & (self._code == HOME) & ~detour
        start, end = self.town_parameters.care_home.day_hours
        up = start <= self.hour < end
        if not up:
            self._target_x[residents] = town.home_x[residents]
            self._target_y[residents] = town.home_y[residents]
        candidates = np.flatnonzero(settled & ((away & on_their_own) | (residents & up)))
        movers = candidates[self.rng.random(len(candidates)) < chance]
        if len(movers):
            center_x, center_y, radius = destination_discs(
                town, self._code[movers], movers, self._schedule
            )
            in_home = self._code[movers] == HOME
            center_x[in_home] = town.home_center_x[movers[in_home]]
            center_y[in_home] = town.home_center_y[movers[in_home]]
            radius[in_home] = town.home_radius[movers[in_home]]
            common = (self._code[movers] == WORK) & (
                self.rng.random(len(movers)) < self._common_share[movers]
            )
            # the common rooms are as crowded as the classes: they take the share of the school's
            # floor that the people in them at a time need
            center_x[common] = town.school_x[movers[common]]
            center_y[common] = town.school_y[movers[common]]
            radius[common] = town.school_radius[movers[common]] * np.sqrt(
                self._common_share[movers[common]]
            )
            self._in_common[movers] = common
            distance = radius * np.sqrt(self.rng.random(len(movers)))
            angle = self.rng.random(len(movers)) * 2 * math.pi
            arena_size = float(self.world["arena_size"])
            self._target_x[movers] = np.clip(center_x + distance * np.cos(angle), 0.0, arena_size)
            self._target_y[movers] = np.clip(center_y + distance * np.sin(angle), 0.0, arena_size)
        children = np.flatnonzero((town.guardian >= 0) & (self._code >= LUNCH))
        followers = children[np.isin(town.guardian[children], movers)]
        self._target_x[followers] = self._target_x[town.guardian[followers]]
        self._target_y[followers] = self._target_y[town.guardian[followers]]

    def _aim(self, people: np.ndarray) -> None:
        self._target_x[people], self._target_y[people] = destination_coordinates(
            self.town, self._code[people], people, self._schedule
        )
        self._inside[people] = False
        self._in_common[people] = False

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
        """People who head home: ill and staying in, or ill and skipping errands and visits, and
        the small children whose guardian takes them out and is heading home."""
        return self._detours()[0]

    def _detours(self) -> tuple[np.ndarray, np.ndarray]:
        """Who does not go where their day takes them, because of an illness: (home, work).

        Home: those who stay in, and on an errand, visit or walk those who skip outings, with the
        small children who were to go out with them. Work: those who skip outings and would go out
        for the lunch break; they stay at their workplace.
        """
        home = self._away()
        skips = np.zeros(self.population, np.bool_)
        for disease_state in self.disease_states.values():
            home |= disease_state.stays_home
            skips |= disease_state.limits_outings
        outing = self._code >= FIRST_ACTIVITY
        # Nobody visits a home where somebody is isolated. Ill people at home are visited as
        # usual: their contacts at home do not fall (Eames et al. 2010, Health Technol Assess
        # 14(34), Table 3: 2.38 a day when ill, 2.58 when healthy).
        isolated = self._confined()
        unwell = np.bincount(self.town.household, weights=isolated, minlength=self._households) > 0
        for index in self._visit_activities:
            visiting = np.flatnonzero(self._code == FIRST_ACTIVITY + index)
            host = self.town.activity_venue[index][visiting, self._schedule.choice[visiting, index]]
            home[visiting[unwell[host]]] = True
        home |= skips & outing
        children = np.flatnonzero((self.town.guardian >= 0) & outing)
        home[children] |= home[self.town.guardian[children]]
        work = skips & (self._code == LUNCH) & ~home
        return home, work

    def _venues(self, home: np.ndarray, work: np.ndarray) -> np.ndarray:
        """The venue everybody is in (see `town.Venues`), -1 for those on the way."""
        venue = destination_venues(self.town, self._code, self._people, self._schedule)
        common = self._in_common & (self._code == WORK) & ~home & ~work
        venue[common] = self.town.school_venue[common]
        venue[home] = self.town.household[home]
        venue[work] = self.town.work_venue[work]
        start, end = self.town_parameters.care_home.day_hours
        if not start <= self.hour < end:  # residents of care homes are in their rooms at night
            asleep = self._residents & (venue == self.town.household)
            venue[asleep] = self.town.room_venue[asleep]
        venue[~self._inside | self._away()] = -1
        return venue

    def _confined(self) -> np.ndarray:
        """Who is isolated at home by a disease (a confined state)."""
        confined = np.zeros(self.population, np.bool_)
        for name, disease in self.diseases.items():
            confined |= disease.confined_mask(self.disease_states[name])
        return confined

    def _away(self) -> np.ndarray:
        """Who is gravely ill (in hospital) with a disease and meets nobody in the town."""
        away = np.zeros(self.population, np.bool_)
        for name, disease in self.diseases.items():
            away |= disease.away_mask(self.disease_states[name])
        return away

    def _infectiousness(self, disease: str) -> np.ndarray:
        """Infectiousness of everybody for `disease`: 0 if not infectious, 1 if fully, or scaled."""
        disease_state = self.disease_states[disease]
        weights = (
            self.diseases[disease].infectious_weight(disease_state) * disease_state.infectiousness
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
