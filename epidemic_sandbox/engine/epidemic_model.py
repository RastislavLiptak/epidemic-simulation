import dataclasses
import math

import numpy as np

from epidemic_sandbox.config.experiment import ExperimentConfig, check_world
from epidemic_sandbox.disease_models import DISEASE_MODELS
from epidemic_sandbox.disease_models.disease_model_abstract import DEAD, DYING
from epidemic_sandbox.engine.person_kernels import (
    due_events,
    go_home_or_back_to_work,
    settled_movers,
    stay_flags,
    venues,
)
from epidemic_sandbox.engine.spatial_kernels import (
    build_grid,
    count_infectious_contacts_by_household,
    move_toward,
)
from epidemic_sandbox.engine.tracing import InfectionTrace
from epidemic_sandbox.world.interiors import LESSON_HOURS
from epidemic_sandbox.world.population import (
    ADULT_AGE,
    FEMALE,
    PRESCHOOL,
    RESIDENT,
    SCHOOL,
    UNIVERSITY,
)
from epidemic_sandbox.world.schedule import (
    DAYS_PER_WEEK,
    FIRST_ACTIVITY,
    HOME,
    HOURS_PER_DAY,
    LUNCH,
    WORK,
    draw_day_schedule,
    free_adults,
    grandparent_care,
)
from epidemic_sandbox.world.town import (
    METERS_PER_KILOMETER,
    TRANSIT,
    TownParameters,
    build_town,
    destination_coordinates,
    destination_discs,
)


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
        self._visits = np.array(self._visit_activities, np.int64)
        self._activity_venue = self._stacked_activity_venues()
        # the children who go out with a guardian, and who looks after them (see `_traits`)
        self._wards = np.flatnonzero(self.town.guardian >= 0)
        self._traits = self._with_grandparent_care()
        self._away_table, self._confined_table = self._state_tables()
        self._detour = (np.zeros(population, np.bool_), np.zeros(population, np.bool_))
        self._residents = self.town.role == RESIDENT
        self._in_common = np.zeros(population, np.bool_)
        self.venue = self.town.household.astype(np.int64)
        self._at_home = np.ones(population, np.bool_)
        # who died of which disease (its position in `diseases`, -1 for the living) and in which
        # step, for the hover card and for the deaths of the other diseases in the export
        self.cause_of_death = np.full(population, -1, np.int8)
        self.death_step = np.full(population, -1, np.int64)
        # how fast everybody moves now (m/h): to cover the trip in its time, see `_set_off`
        self._speed = self.town.speed_kmh * METERS_PER_KILOMETER
        self._trip_hours = np.full(population, np.nan)
        self._vehicles = Vehicles(self)
        # whose air is shared, by venue: the venues of the town that are not outdoors and every
        # vehicle (numbered after them, see `Vehicles`)
        indoor = self.town.indoor_venue
        if indoor is None:
            indoor = np.ones(self._vehicles.first_venue, np.bool_)
        self._indoor_venue = np.concatenate([indoor, np.ones(population, np.bool_)])
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
        """Today's events of everybody (see schedule.DaySchedule), drawn when the day began."""
        return self._schedule

    def step(self) -> None:
        """Let people go about their day, then advance every disease by one step."""
        world = self.world
        self._follow_schedule()
        home, work = self._detours()
        changed = (home != self._detour[0]) | (work != self._detour[1])
        self._inside[changed] = False  # a new goal: home when ill, or back to the workplace
        self._at_mode_speed(np.flatnonzero(changed))
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
            walking_hours * self._speed * alive,
        )
        goal_x = np.where(other_goal, goal_x, self._target_x)
        goal_y = np.where(other_goal, goal_y, self._target_y)
        self._inside |= (self.x == goal_x) & (self.y == goal_y)
        self.venue = self._venues(home, work)
        # who is in their own home: members of a household meet as such only there
        self._at_home = (self.venue >= 0) & ((self._code == HOME) | home) & ~work
        # passengers of public transport sit in their vehicle: there they are in contact
        contact_x, contact_y = self._vehicles.ride(self, step_start + self.step_hours)
        arena_size = float(world["arena_size"])
        longest = max((disease.contact_distance for disease in self.diseases.values()), default=1.0)
        cells_per_side = max(1, min(int(arena_size // longest), 2 * math.isqrt(self.population)))
        grid = build_grid(contact_x, contact_y, arena_size, cells_per_side)
        exposures, infectiousness, relative, room = {}, {}, {}, {}
        for name, disease in self.diseases.items():
            disease_state = self.disease_states[name]
            infectiousness[name] = self._infectiousness(name)
            contacts, own_household = count_infectious_contacts_by_household(
                contact_x,
                contact_y,
                disease.exposable_mask(disease_state),
                infectiousness[name],
                self.town.household,
                self._at_home,
                self.venue,
                *grid,
                cells_per_side,
                float(disease.contact_distance),
                float(disease.near_distance),
            )
            community = disease.parameters["community_factor"]
            relative[name] = self._relative_susceptibility(name)
            room[name] = self._room_exposure(disease, infectiousness[name])
            exposures[name] = (
                own_household + np.float32(community) * (contacts - own_household) + room[name]
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
            disease_state = self.disease_states[name]
            before = disease_state.state.copy() if self.trace else None
            deaths = disease_state.deaths
            self.diseases[name].finish_phases(disease_state, self.rng)
            if disease_state.deaths > deaths:
                self._note_deaths(index, disease_state.state == self.diseases[name].dead_index)
            if self.trace:
                self.trace.record_deaths(self, name, before)
            self._bury_the_dead()
        for name, disease in self.diseases.items():
            disease_state = self.disease_states[name]
            alive = infected[name][disease_state.state[infected[name]] != disease.dead_index]
            disease.apply_infections(disease_state, infected[name], self.rng)
            if self.trace:
                # the infector is looked for where people were in contact (in the vehicles)
                moved = self.x, self.y
                self.x, self.y = contact_x, contact_y
                self.trace.record_infections(
                    self,
                    name,
                    alive,
                    infectiousness[name],
                    grid,
                    cells_per_side,
                    exposures[name],
                    relative[name],
                    room[name],
                )
                self.x, self.y = moved
        if self.trace:
            self.trace.record_onsets(self)
        self.steps += 1
        if self.steps % world["steps_per_day"] == 0:
            self._record_day()

    def _room_exposure(self, disease, infectiousness: np.ndarray) -> np.ndarray | float:
        """The exposure of everybody to the infectious people who share an indoor venue with them,
        whatever the distance (the air of the room): `room_exposure` of the disease times the
        infectiousness of the others in the venue. 0 for a disease without the route."""
        per_hour = disease.parameters["room_exposure"]
        if per_hour <= 0:
            return 0.0
        venue = self.venue
        here = (venue >= 0) & (infectiousness > 0)
        total = np.bincount(
            venue[here], weights=infectiousness[here], minlength=len(self._indoor_venue)
        )
        others = total[np.maximum(venue, 0)] - np.where(here, infectiousness, 0.0)
        shared = (venue >= 0) & self._indoor_venue[np.maximum(venue, 0)]
        return np.where(shared, np.float32(per_hour) * others, 0.0).astype(np.float32)

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
        """True when no disease can change any more (nothing that follows can differ; the app does
        not stop a run by it, the measurements may)."""
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
            alive = ~self._dead()
            self._replace_dead_guardians(alive)
            self._schedule = draw_day_schedule(
                self.town.role,
                self._traits,
                self.town_parameters.schedule,
                calendar.is_day_off(day_index),
                self.rng,
                calendar.closed_places(day_index),
                alive,
            )
            self._free_adults = free_adults(self.town.traits, self._schedule.attends, alive)
            self._planned_day = day_index
            self._pointer[:] = 0
            away = np.flatnonzero(self._code != HOME)
            self._code[away] = HOME
            self._aim(away)
            self._at_mode_speed(away)
        until = (self.steps + 1) * self.step_hours - day_index * HOURS_PER_DAY
        times, codes = self._schedule.times, self._schedule.codes
        due = due_events(times, self._pointer, until)
        pointer = self._pointer[due]
        self._code[due] = codes[due, pointer]
        self._departure_time[due] = day_index * HOURS_PER_DAY + times[due, pointer]
        self._pointer[due] += 1
        if len(due):
            self._visit_hosts_at_home(due)
            self._aim(due)
            self._set_off(due, self._schedule.travel[due, pointer])

    def _set_off(self, people: np.ndarray, hours: np.ndarray | None) -> None:
        """`people` set off for their targets: each moves at the speed that covers the trip in its
        time `hours` (from the schedule; NaN, or no times, means at the speed of the mode)."""
        if hours is None:
            self._at_mode_speed(people)
            return
        distance = np.hypot(
            self._target_x[people] - self.x[people], self._target_y[people] - self.y[people]
        )
        timed = np.isfinite(hours) & (hours > 0)
        mode = self.town.speed_kmh[people] * METERS_PER_KILOMETER
        self._speed[people] = np.where(timed, distance / np.where(timed, hours, 1.0), mode)
        self._trip_hours[people] = np.where(timed, hours, np.nan)
        self._vehicles.board(self, people[timed & (distance > 0)])

    def _at_mode_speed(self, people: np.ndarray) -> None:
        """`people` move at the speed of their mode, on no timed trip (a detour, a new day)."""
        self._speed[people] = self.town.speed_kmh[people] * METERS_PER_KILOMETER
        self._trip_hours[people] = np.nan
        self._vehicles.leave(people)

    def _members_at_home(self) -> np.ndarray:
        """Who is at home now: living, arrived, heading for home by their day or by an illness and
        not kept at the workplace over lunch (the members of a household that a visitor finds)."""
        home, work = self._detour
        return self._inside & ((self._code == HOME) | home) & ~work & ~self._dead() & ~self._away()

    def _visit_households(self, activity: int, people: np.ndarray) -> np.ndarray:
        """The household of each favourite host of `people` for the visit `activity`, padded with
        -1 (the venue of the visit is the household's dwelling, or a care-home ward)."""
        hosts = self._activity_venue[activity, people]
        known = self.town.visit_host
        if known is not None and activity == self._visits[0]:
            households = np.full(hosts.shape, -1, np.int64)
            households[:, : known.shape[1]] = known[people]
            hosts = np.where(hosts >= 0, households, -1)
        return hosts

    def _visit_hosts_at_home(self, due: np.ndarray) -> None:
        """Visitors set off to a favourite household with somebody at home (a living member who
        was there in the last step): the one of today's plan, else another favourite with somebody
        at home; when nobody is at home in any of them, the visit is dropped and the visitor stays
        at home. A child on a visit with its guardian goes where the guardian goes. `due` are the
        people whose next event was taken in this step."""
        if not len(self._visits):
            return
        town = self.town
        visiting = due[np.isin(self._code[due], FIRST_ACTIVITY + self._visits)]
        if not len(visiting):
            return
        at_home = np.bincount(
            town.household[self._members_at_home()], minlength=self._households
        )
        cared = self._schedule.cared
        if cared is not None:  # a child taken to its grandparents goes there whatever happens
            visiting = visiting[~cared[visiting]]
        ward = town.guardian[visiting] >= 0
        choice = self._schedule.choice
        for activity in self._visits:
            who = visiting[(self._code[visiting] == FIRST_ACTIVITY + activity) & ~ward]
            hosts = self._visit_households(activity, who)  # padded with -1
            open_ = (hosts >= 0) & (at_home[np.maximum(hosts, 0)] > 0)
            keep = open_[np.arange(len(who)), choice[who, activity]]
            other = ~keep & open_.any(axis=1)
            choice[who[other], activity] = np.argmax(open_[other], axis=1)
            self._code[who[~keep & ~other]] = HOME
        children = visiting[ward]
        guardian = town.guardian[children]
        with_guardian = self._code[guardian] == self._code[children]
        for activity in self._visits:
            choice[children, activity] = choice[guardian, activity]
        self._code[children[~with_guardian]] = HOME

    def _move_about(self, detour: np.ndarray) -> None:
        """People who have settled at a place move to another spot of it after their dwell time.

        The time at one spot is random with the mean dwell time of the place. At work and at
        school people go between their own seat or desk (or a spot of their room where they have
        none) and the common area of the place (`common_share` of the time); teachers who go from
        class to class change class every lesson; care staff go between the day room of their
        ward and the rooms of its residents. At home people are in bed while they sleep, and
        awake they go between the common room of the dwelling and their own room. The mobile
        residents of a care home spend the day in the day room of their ward; the bedridden stay in
        bed. On outings small children stay with the adult who takes them out. `detour` are those
        who are not where their day takes them (see `_detours`).
        """
        town = self.town
        params = self.town_parameters
        start, end = params.care_home.day_hours
        up = start <= self.hour < end
        dwelling = params.dwelling
        awake = dwelling is None or not dwelling.asleep(self.hour)
        self._ensure_interiors()
        candidates = settled_movers(
            self.x,
            self.y,
            self._target_x,
            self._target_y,
            self._code,
            detour,
            town.guardian,
            self._residents,
            self._bedridden,
            up,
            awake,
            HOME,
            WORK,
        )
        at_home = (self._code == HOME) & ~detour
        if not awake:  # the night: everybody at home in bed
            asleep = at_home & ~self._residents
            self._target_x[asleep] = town.home_x[asleep]
            self._target_y[asleep] = town.home_y[asleep]
        if not up:  # at night the residents of a care home are in their rooms
            residents = self._residents & at_home
            self._target_x[residents] = town.home_x[residents]
            self._target_y[residents] = town.home_y[residents]
        # moves happen at the steps, so this chance makes the mean stay the dwell time long
        chance = np.minimum(1.0, self.step_hours / self._dwell[candidates])
        movers = candidates[self.rng.random(len(candidates)) < chance]
        if len(movers):
            self._new_spots(movers)
        self._breaks(detour)
        if self._rotation is not None and len(self._rotation.teacher):
            self._teach(detour)
        children = self._wards[self._code[self._wards] >= LUNCH]
        moved = np.zeros(self.population, np.bool_)
        moved[movers] = True
        guardian = town.guardian[children]
        # with the guardian on the same outing (not a child at its grandparents' meanwhile)
        followers = children[moved[guardian] & (self._code[guardian] == self._code[children])]
        self._target_x[followers] = self._target_x[town.guardian[followers]]
        self._target_y[followers] = self._target_y[town.guardian[followers]]

    def _ensure_interiors(self) -> None:
        """The arrays of where people are inside places, made at the first step."""
        if hasattr(self, "_dwell"):
            return
        town, params = self.town, self.town_parameters
        population = self.population
        dwell = np.full(population, params.dwell_hours)
        if town.dwell_hours is not None:
            dwell = town.dwell_hours.astype(float).copy()
        self._place_dwell = dwell
        self._home_dwell = np.full(
            population, params.dwelling.dwell_hours if params.dwelling else params.dwell_hours
        )
        self._home_dwell[self._residents] = params.care_home.dwell_hours
        self._dwell = dwell.copy()  # the dwell at a place stays; this one follows where people are
        self._bedridden = (
            town.bedridden if town.bedridden is not None else np.zeros(population, np.bool_)
        )
        self._lesson_venue = np.full(population, -1, np.int64)
        self._room_visit = np.full(population, -1, np.int64)
        self._care_staff = (town.place_kind == RESIDENT) & ~self._residents
        self._rotation = town.rotation
        self._last_lesson = -1
        # the pupils of a class go to the common rooms of their school together (the breaks)
        pupil = np.isin(town.role, (PRESCHOOL, SCHOOL, UNIVERSITY)) & (town.work_venue >= 0)
        pupil &= town.school_venue >= 0
        self._pupils = np.flatnonzero(pupil)
        self._is_pupil = pupil
        classes, self._pupil_class = np.unique(town.work_venue[self._pupils], return_inverse=True)
        share = np.zeros(len(classes))
        share[self._pupil_class] = (
            town.common_share[self._pupils] if town.common_share is not None else 0.0
        )
        self._class_share = share
        self._class_break = np.zeros(len(classes), np.bool_)

    def _new_spots(self, movers: np.ndarray) -> None:
        """Where the people who move go: a spot of their place, a room of their dwelling, ..."""
        town, params = self.town, self.town_parameters
        code = self._code[movers]
        center_x, center_y, radius = destination_discs(town, code, movers, self._schedule)
        x, y = np.empty(len(movers)), np.empty(len(movers))
        fixed = np.zeros(len(movers), np.bool_)
        # at home: the common room of the dwelling (or the ward's day room) or one's own room
        in_home = code == HOME
        center_x[in_home] = town.home_center_x[movers[in_home]]
        center_y[in_home] = town.home_center_y[movers[in_home]]
        radius[in_home] = town.home_radius[movers[in_home]]
        if params.dwelling is not None:
            own_room = in_home & ~self._residents[movers]
            own_room &= self.rng.random(len(movers)) >= params.dwelling.together_share
            fixed |= own_room
        # at work: the common area for `common_share` of the time, else one's seat or room (the
        # pupils go there with their class, see `_breaks`)
        at_work = code == WORK
        share = town.common_share[movers] if town.common_share is not None else 0.0
        common = at_work & (self.rng.random(len(movers)) < share) & (town.school_venue[movers] >= 0)
        pupil = self._is_pupil[movers]
        common[pupil] = self._in_common[movers[pupil]] & at_work[pupil]
        center_x[common] = town.school_x[movers[common]]
        center_y[common] = town.school_y[movers[common]]
        radius[common] = town.school_radius[movers[common]]
        seat = at_work & ~common
        if town.seat_fixed is not None:
            seat &= town.seat_fixed[movers]
        else:
            seat[:] = False
        fixed |= seat
        self._in_common[movers] = common
        distance = radius * np.sqrt(self.rng.random(len(movers)))
        angle = self.rng.random(len(movers)) * 2 * math.pi
        x[:] = center_x + distance * np.cos(angle)
        y[:] = center_y + distance * np.sin(angle)
        own = fixed & in_home
        x[own], y[own] = town.home_x[movers[own]], town.home_y[movers[own]]
        x[seat], y[seat] = town.work_x[movers[seat]], town.work_y[movers[seat]]
        self._care_rounds(movers, at_work & ~common, x, y)
        self._target_x[movers], self._target_y[movers] = x, y
        self._keep_in_arena(movers)
        self._dwell[movers] = np.where(in_home, self._home_dwell[movers], self._place_dwell[movers])

    def _breaks(self, detour: np.ndarray) -> None:
        """Every step each class is in the common rooms of its school with the chance of its
        `common_share` (a break, the canteen), all its pupils together; the pupils at school go
        there, or back to their seat (or a spot of the class) when the break ends."""
        if not len(self._pupils):
            return
        town = self.town
        on_break = self.rng.random(len(self._class_share)) < self._class_share
        pupils = self._pupils
        at_school = (self._code[pupils] == WORK) & ~detour[pupils]
        now = on_break[self._pupil_class]
        start = pupils[at_school & now & ~self._in_common[pupils]]
        end = pupils[at_school & ~now & self._in_common[pupils]]
        self._class_break = on_break
        if len(start):
            distance = town.school_radius[start] * np.sqrt(self.rng.random(len(start)))
            angle = self.rng.random(len(start)) * 2 * math.pi
            self._target_x[start] = town.school_x[start] + distance * np.cos(angle)
            self._target_y[start] = town.school_y[start] + distance * np.sin(angle)
            self._keep_in_arena(start)
            self._in_common[start] = True
        if len(end):
            fixed = (
                town.seat_fixed[end] if town.seat_fixed is not None else np.zeros(len(end), bool)
            )
            distance = town.work_radius[end] * np.sqrt(self.rng.random(len(end)))
            angle = self.rng.random(len(end)) * 2 * math.pi
            x = town.work_center_x[end] + distance * np.cos(angle)
            y = town.work_center_y[end] + distance * np.sin(angle)
            x[fixed], y[fixed] = town.work_x[end[fixed]], town.work_y[end[fixed]]
            self._target_x[end], self._target_y[end] = x, y
            self._keep_in_arena(end)
            self._in_common[end] = False

    def _keep_in_arena(self, people: np.ndarray) -> None:
        """A place at the edge of the arena has spots outside of it: nobody walks there."""
        arena_size = float(self.world["arena_size"])
        self._target_x[people] = np.clip(self._target_x[people], 0.0, arena_size)
        self._target_y[people] = np.clip(self._target_y[people], 0.0, arena_size)

    def _care_rounds(self, movers, at_work, x, y) -> None:
        """Care staff who move go to the room of a resident of their ward (`staff_room_share` of
        the time, standing by the bed) or else stay in the ward's day room (x, y are changed)."""
        town = self.town
        staff = at_work & self._care_staff[movers]
        self._room_visit[movers] = -1
        if not staff.any() or town.ward_start is None:
            return
        who = movers[staff]
        rounds = self.rng.random(len(who)) < self.town_parameters.care_home.staff_room_share
        ward = town.ward[who[rounds]]
        start, size = town.ward_start[ward], np.diff(town.ward_start)[ward]
        resident = town.ward_residents[start + (self.rng.random(len(ward)) * size).astype(np.int64)]
        index = np.flatnonzero(staff)[rounds]
        x[index] = town.home_x[resident] + 0.5
        y[index] = town.home_y[resident]
        self._room_visit[who[rounds]] = town.room_venue[resident]

    def _teach(self, detour: np.ndarray) -> None:
        """At the start of every lesson the teachers who go from class to class are given a class
        each (those left over stay in the staff room, the common area of the school)."""
        lesson = int(self.hour // LESSON_HOURS)
        if lesson == self._last_lesson:
            return
        self._last_lesson = lesson
        rotation, town = self._rotation, self.town
        teacher = rotation.teacher
        present = (self._code[teacher] == WORK) & ~detour[teacher]
        chosen = rotation.assign(present, self.rng)
        self._lesson_venue[teacher] = -1
        teaching = chosen >= 0
        who = teacher[teaching]
        self._lesson_venue[who] = rotation.class_venue[chosen[teaching]]
        distance = rotation.class_radius[chosen[teaching]] * np.sqrt(self.rng.random(len(who)))
        angle = self.rng.random(len(who)) * 2 * math.pi
        self._target_x[who] = rotation.class_x[chosen[teaching]] + distance * np.cos(angle)
        self._target_y[who] = rotation.class_y[chosen[teaching]] + distance * np.sin(angle)
        self._keep_in_arena(who)
        self._in_common[who] = False
        idle = teacher[present & ~teaching]
        self._target_x[idle], self._target_y[idle] = town.work_x[idle], town.work_y[idle]

    def _aim(self, people: np.ndarray) -> None:
        code = self._code[people]
        self._target_x[people], self._target_y[people] = destination_coordinates(
            self.town, code, people, self._schedule
        )
        dwelling = self.town_parameters.dwelling
        if dwelling is not None and not dwelling.asleep(self.hour):
            # whoever comes home during the day goes to the common room or to their own room, as
            # the people already at home do, not to the bed
            coming_home = people[code == HOME]
            if len(coming_home):
                self._ensure_interiors()
                self._new_spots(coming_home)
        self._inside[people] = False
        self._in_common[people] = False

    def _replace_dead_guardians(self, alive: np.ndarray) -> None:
        """Children whose guardian has died go out with another living adult of the household,
        chosen as guardians are (see `town._guardians`), to that adult's places and at their pace.
        A child without one keeps the dead guardian, who goes nowhere, so the child goes out only
        to its school or kindergarten (the relatives who would take such a child in are not in
        the town)."""
        town = self.town
        wards = self._wards
        orphaned = wards[~alive[town.guardian[wards]]]
        if not len(orphaned):
            return
        adults = np.flatnonzero((town.age >= ADULT_AGE) & ~self._residents & alive)
        order = adults[
            np.lexsort((town.age[adults], town.sex[adults] != FEMALE, town.household[adults]))
        ]
        households, first = np.unique(town.household[order], return_index=True)
        carer = np.full(self._households, -1, np.int64)
        carer[households] = order[first]
        new = carer[town.household[orphaned]]
        children, guardians = orphaned[new >= 0], new[new >= 0]
        if not len(children):
            return
        town.guardian[children] = guardians
        # their grandparents were the parents of the dead guardian: no care by them any more
        self._traits.care_hours[children] = 0.0
        self._traits.care_host[children] = -1
        for arrays in (
            town.activity_x,
            town.activity_y,
            town.activity_radius,
            town.activity_people,
            town.activity_venue,
        ):
            for values in arrays:
                values[children] = values[guardians]
        self._activity_venue[:, children] = self._activity_venue[:, guardians]
        town.speed_kmh[children] = town.speed_kmh[guardians]
        if town.travel_mode is not None:
            town.travel_mode[children] = town.travel_mode[guardians]
            self._vehicles.passengers[children] = self._vehicles.passengers[guardians]

    def _with_grandparent_care(self):
        """The traits of the town with the grandparent care of its children (see
        `schedule.grandparent_care`)."""
        town = self.town
        visit = self._visits[0] if len(self._visits) else None
        hosts = town.visit_host if visit is not None else None
        host, choice, hours = grandparent_care(
            town.guardian, town.age, town.visit_kin, hosts, self.town_parameters.schedule, self.rng
        )
        traits = town.traits
        return dataclasses.replace(traits, care_host=host, care_choice=choice, care_hours=hours)

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
        for the lunch break; they stay at their workplace. Nobody visits a home where somebody is
        isolated. Ill people at home are visited as usual: their contacts at home do not fall
        (Eames et al. 2010, Health Technol Assess 14(34), Table 3: 2.38 a day when ill, 2.58 when
        healthy).
        """
        # indexed by the household and by the venue of a visited host (a ward is never unwell)
        unwell = np.zeros(max(self._households, _venue_count(self.town)), np.bool_)
        if self.disease_states:
            stays, skips = stay_flags(
                self._diseases_by("state"),
                self._diseases_by("stays_home"),
                self._diseases_by("limits_outings"),
                self._away_table,
                self._confined_table,
                self.town.household,
                unwell,
            )
            self._stay_with_ill_children(stays)
        else:  # a town without diseases, as the validation runs it
            stays = skips = np.zeros(self.population, np.bool_)
        return go_home_or_back_to_work(
            self._code,
            self._schedule.choice,
            stays,
            skips,
            unwell,
            self._activity_venue,
            self._visits,
            self.town.guardian,
            FIRST_ACTIVITY,
            LUNCH,
        )

    def _stay_with_ill_children(self, stays: np.ndarray) -> None:
        """A child too young to spend a day alone who stays at home ill keeps the guardian at home
        with it when no other living adult of the household is free today (`stays` is changed)."""
        care = self.town_parameters.schedule.childcare
        wards = self._wards
        if care is None or not len(wards):
            return
        town = self.town
        ill = np.zeros(len(wards), np.bool_)
        for state in self.disease_states.values():
            ill |= state.stays_home[wards]
        ill &= town.age[wards] < care.alone_all_day_from
        ill &= ~self._away()[wards] & ~self._dead()[wards]
        needs = wards[ill & (self._free_adults[town.household[wards]] == 0)]
        stays[town.guardian[needs]] = True

    def _venues(self, home: np.ndarray, work: np.ndarray) -> np.ndarray:
        """The venue everybody is in (see `town.Venues`), -1 for those on the way: also the class
        a teacher who goes from class to class teaches and the room of a resident a carer is in."""
        start, end = self.town_parameters.care_home.day_hours
        town = self.town
        self._ensure_interiors()
        venue = venues(
            self._code,
            self._schedule.choice,
            town.household,
            town.work_venue,
            town.lunch_venue,
            self._activity_venue,
            self._in_common,
            town.school_venue,
            home,
            work,
            self._residents,
            town.room_venue,
            town.ward_venue if town.ward_venue is not None else town.room_venue,
            self._bedridden,
            not start <= self.hour < end,  # residents of care homes are in their rooms at night
            self._inside,
            self._away(),
            FIRST_ACTIVITY,
            WORK,
            LUNCH,
        )
        inside = (self._code == WORK) & (venue >= 0) & ~self._in_common & ~work & ~home
        for room in (self._lesson_venue, self._room_visit):
            there = inside & (room >= 0)
            venue[there] = room[there]
        return venue

    def _away(self) -> np.ndarray:
        """Who is gravely ill (in hospital) with a disease and meets nobody in the town."""
        away = np.zeros(self.population, np.bool_)
        for name, disease in self.diseases.items():
            away |= disease.away_mask(self.disease_states[name])
        return away

    def _diseases_by(self, attribute: str) -> tuple:
        """The array `attribute` of the state of each disease, in the order of the diseases."""
        return tuple(getattr(state, attribute) for state in self.disease_states.values())

    def _stacked_activity_venues(self) -> np.ndarray:
        """The venue of each favourite place of each activity as one array [activity, person,
        favourite], padded where an activity has fewer favourites."""
        lists = self.town.activity_venue
        stacked = np.full((len(lists), self.population, max(a.shape[1] for a in lists)), -1)
        for index, venues_of in enumerate(lists):
            stacked[index, :, : venues_of.shape[1]] = venues_of
        return stacked

    def _state_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """For each disease and state: whether the state keeps a person from their day (away in
        hospital, or dead: the children who were to go out with them stay at home) and whether it
        is being confined at home."""
        diseases = list(self.diseases.values())
        states = max((len(disease.STATE_COLORS) for disease in diseases), default=1)
        away, confined = (np.zeros((len(diseases), states), np.bool_) for _ in range(2))
        for row, disease in enumerate(diseases):
            away[row, : len(disease.STATE_COLORS)] = disease.state_table(
                [DYING, DEAD, *disease.away_states()]
            )
            confined[row, : len(disease.STATE_COLORS)] = disease.state_table(
                disease.CONFINED_STATES
            )
        return away, confined

    def _infectiousness(self, disease: str) -> np.ndarray:
        """Infectiousness of everybody for `disease`: 0 if not infectious, 1 if fully, or scaled."""
        disease_state = self.disease_states[disease]
        weights = (
            self.diseases[disease].infectious_weight(disease_state) * disease_state.infectiousness
        )
        for interaction, source_states in self.interactions:
            if interaction.target == disease and interaction.effect == "infectiousness":
                weights *= self._factor_where_in_states(interaction, source_states)
        asleep = self.diseases[disease].parameters["asleep_emission"]
        dwelling = self.town_parameters.dwelling
        if asleep < 1.0 and dwelling is not None and dwelling.asleep(self.hour):
            # in the hours in bed, whoever is at home is asleep: they only breathe
            weights = np.where(self._at_home, weights * np.float32(asleep), weights)
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

    def finish_day(self) -> None:
        """Record the state reached when a run ends before midnight, so that the course of the
        run includes its last hours (the days are recorded at midnight)."""
        if self.steps % self.world["steps_per_day"] and self.history[-1]["day"] != self.day:
            self._record_day(self.day)

    def _note_deaths(self, disease: int, dead: np.ndarray) -> None:
        """Note the cause and step of the deaths of this step among `dead` (the dead of the disease
        at position `disease`; those who died of another disease are noted already)."""
        died = np.flatnonzero(dead & (self.cause_of_death < 0))
        self.cause_of_death[died] = disease
        self.death_step[died] = self.steps

    def _record_day(self, day: float | None = None) -> None:
        self.history.append(
            {
                "day": self.steps // self.world["steps_per_day"] if day is None else day,
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


class Vehicles:
    """The public transport vehicles in which passengers meet (see [travel.vehicles]).

    The passengers by public transport who set off in the same step share vehicles of `load`
    people. A passenger rides for the middle `in_vehicle_share` of the trip, in the vehicle's own
    venue, seated somewhere on its floor; for the contacts the vehicle is placed where its first
    passenger set off (the venue keeps everybody else out).
    """

    def __init__(self, model: "EpidemicModel"):
        town, travel = model.town, model.town_parameters.travel
        self.load = travel.vehicle_load
        self.share = travel.in_vehicle_share
        self.radius = math.sqrt(travel.vehicle_floor_area / math.pi)
        mode = town.travel_mode if town.travel_mode is not None else np.zeros(len(town.age))
        self.passengers = mode == TRANSIT
        population = model.population
        self.number = np.full(population, -1, np.int64)
        self.x, self.y = np.zeros(population), np.zeros(population)
        self.first_venue = _venue_count(town)
        self.next = 0

    def board(self, model: "EpidemicModel", people: np.ndarray) -> None:
        """`people` set off on a timed trip; those who go by public transport take vehicles."""
        riders = people[self.passengers[people]]
        if not len(riders):
            return
        riders = riders[model.rng.permutation(len(riders))]
        group = np.arange(len(riders)) // self.load
        first = riders[np.searchsorted(group, group)]  # the first passenger of each vehicle
        self.number[riders] = (self.next + group) % model.population
        self.next = (self.next + int(group[-1]) + 1) % model.population
        distance = self.radius * np.sqrt(model.rng.random(len(riders)))
        angle = model.rng.random(len(riders)) * 2 * math.pi
        self.x[riders] = model.x[first] + distance * np.cos(angle)
        self.y[riders] = model.y[first] + distance * np.sin(angle)

    def leave(self, people: np.ndarray) -> None:
        self.number[people] = -1

    def ride(self, model: "EpidemicModel", now: float) -> tuple[np.ndarray, np.ndarray]:
        """Put the passengers who are in their vehicle at `now` (hours since the start) into its
        venue; returns where everybody is for the contacts (passengers at their seats)."""
        on_board = np.flatnonzero(self.number >= 0)
        if not len(on_board):
            return model.x, model.y
        trip = model._trip_hours[on_board]
        into = (now - model._departure_time[on_board]) / trip
        start = (1.0 - self.share) / 2
        riding = on_board[
            (into >= start) & (into <= start + self.share) & (model.venue[on_board] < 0)
            & ~model._inside[on_board]
        ]
        self.number[on_board[into > 1.0]] = -1  # the trip is over
        if not len(riding):
            return model.x, model.y
        model.venue[riding] = self.first_venue + self.number[riding]
        contact_x, contact_y = model.x.copy(), model.y.copy()
        arena = float(model.world["arena_size"])
        contact_x[riding] = np.clip(self.x[riding], 0.0, arena)
        contact_y[riding] = np.clip(self.y[riding], 0.0, arena)
        return contact_x, contact_y


def _venue_count(town) -> int:
    """The number of venues of the town (see `town.Venues`); vehicles are numbered after them."""
    if town.venue_count:
        return town.venue_count
    largest = [int(town.household.max())]
    for values in (town.work_venue, town.school_venue, town.lunch_venue, town.room_venue):
        if values is not None and len(values):
            largest.append(int(np.max(values)))
    for values in town.activity_venue or ():
        largest.append(int(np.max(values)))
    return max(largest) + 1
