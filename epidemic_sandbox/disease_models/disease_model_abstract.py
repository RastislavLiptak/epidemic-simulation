import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from epidemic_sandbox.parameter_spec import ParameterSpec, resolve_parameters

SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"
INFECTIOUS = "infectious"
RECOVERED = "recovered"
DEAD = "dead"
LIFELONG_DAYS = 36500
NEVER_DAYS = 1e9  # the timer of a stay that is never left
AGE_EFFECTS = ("susceptibility", "fatality", "branching")

TRANSMISSION = ParameterSpec(
    "Transmission (per day of contact)",
    "Chance that a susceptible person gets infected during one day spent within contact distance "
    "of one infectious person.",
    0.3,
    0.0,
    1.0,
    0.01,
)
CONTACT_DISTANCE = ParameterSpec(
    "Contact distance (m)",
    "How close an infectious person must be to infect someone: about 2 m for droplets, "
    "more for diseases that spread through the air of a room.",
    2.0,
    0.5,
    50.0,
    0.5,
)
DURATION_SHAPE = ParameterSpec(
    "Duration regularity (gamma shape)",
    "How regular the lengths of the disease phases are. 1 gives a very scattered "
    "(exponential) length, higher values make phases last close to their average.",
    4.0,
    1.0,
    50.0,
    0.5,
)
STAY_HOME = ParameterSpec(
    "Stay-at-home share (infectious)",
    "Share of people who stay at home for the whole time they are infectious.",
    0.0,
    0.0,
    1.0,
    0.05,
)
LIMIT_OUTINGS = ParameterSpec(
    "Fewer outings share (infectious)",
    "Share of infectious people who still go to work or school but skip errands, visits and "
    "leisure.",
    0.0,
    0.0,
    1.0,
    0.05,
)
FATALITY = ParameterSpec(
    "Fatality (per infection)",
    "Chance that an infection ends in death for a person whose age effect is 1; the age effects "
    "scale it up and down.",
    0.0,
    0.0,
    1.0,
    0.001,
)
SUSCEPTIBILITY_CV = ParameterSpec(
    "Susceptibility variation",
    "How much people differ in how easily they catch the disease (coefficient of variation).",
    0.3,
    0.0,
    3.0,
    0.05,
)
INFECTIOUSNESS_DISPERSION = ParameterSpec(
    "Infectiousness dispersion (gamma shape)",
    "How much infectious people differ in how much they spread it. Small values mean a few "
    "superspreaders cause most infections; 0 means everybody spreads equally.",
    0.0,
    0.0,
    100.0,
    0.05,
)
IMMUNITY_DAYS = ParameterSpec(
    "Immunity (days)",
    "Average time a recovered person stays immune before becoming susceptible again; "
    f"{LIFELONG_DAYS} days means for life.",
    float(LIFELONG_DAYS),
    1.0,
    float(LIFELONG_DAYS),
    1.0,
)
IMMUNITY_PROTECTION = ParameterSpec(
    "Protection while immune",
    "How well immunity protects: 1 is complete, lower values let immune people be reinfected "
    "at a reduced risk.",
    1.0,
    0.0,
    1.0,
    0.05,
)

COMMON_PARAMETERS = {
    "transmission": TRANSMISSION,
    "contact_distance": CONTACT_DISTANCE,
    "duration_shape": DURATION_SHAPE,
    "stay_home": STAY_HOME,
    "limit_outings": LIMIT_OUTINGS,
    "fatality": FATALITY,
    "susceptibility_cv": SUSCEPTIBILITY_CV,
    "infectiousness_dispersion": INFECTIOUSNESS_DISPERSION,
}
IMMUNITY_PARAMETERS = {"immunity_days": IMMUNITY_DAYS, "immunity_protection": IMMUNITY_PROTECTION}


@dataclass(frozen=True)
class Phase:
    """After a gamma-distributed time (mean in `duration_parameter`), go to `next_state`.

    A `fatal` phase ends in death for people who die of the disease instead. Of the survivors a
    share goes to `branch_state` instead of `next_state`: the value of `branch_parameter`, scaled
    by the person's "branching" age effect and capped at 1.
    """

    next_state: str
    duration_parameter: str
    fatal: bool = False
    branch_state: str | None = None
    branch_parameter: str | None = None


@dataclass(frozen=True)
class Rate:
    """People in the state move to `next_state` at a constant hazard, `rate_parameter` per day.

    The hazard applies from the day given by `start_parameter` (from the first day without it).
    """

    next_state: str
    rate_parameter: str
    start_parameter: str | None = None


@dataclass
class DiseaseState:
    """Per-person state of one disease: state, time left in the phase, traits and behaviour."""

    state: np.ndarray
    timer: np.ndarray
    ever_infected: np.ndarray
    stays_home: np.ndarray
    limits_outings: np.ndarray
    susceptibility: np.ndarray
    infectiousness: np.ndarray
    fatality_risk: np.ndarray
    branching: np.ndarray
    cumulative_infections: int = 0
    deaths: int = 0
    steps: int = 0


@dataclass(frozen=True)
class Course:
    """One way through an infection and the chance of taking it.

    `stays` lists the states passed through, each with its mean time in days (infinite when the
    state is never left) and its relative infectiousness; `end` is where the course stops:
    dead, susceptible again, or the state that is never left.
    """

    probability: float
    stays: tuple[tuple[str, float, float], ...]
    end: str


class DiseaseModelAbstract:
    """Course of one disease, advanced for the whole population at once in steps of `step_days`.

    A model is declared by its data. Required:

    - PARAMETERS (the common ones, plus those the model's own data refer to) and STATE_COLORS:
      the first state is the initial one, DEAD the absorbing one. The charts follow the states
      named "infectious" (all infectious states together) and "recovered" where they exist.
    - INFECTIOUS_STATES, INFECTION_ENTRY_STATE (where an infected person goes) and
      INITIAL_CASE_STATE (where the first cases start).
    - PHASES: timed transitions out of a state, to a next state and optionally, for a share of
      the people, to a branch state (see `Phase`). A state without a phase is never left.

    Optional, all empty by default:

    - STATE_INFECTIOUSNESS: infectious state -> parameter, the relative infectiousness of that
      state (an infectious state not listed spreads fully).
    - SYMPTOMATIC_STATES: states in which a person reacts to being ill, staying at home or
      skipping outings with the chances `stay_home` and `limit_outings` (default: all infectious
      states). CONFINED_STATES: states in which a person is always at home.
    - PROTECTION: state -> parameter, how well people in that state are protected against
      catching the disease (1 is complete; below 1 they can be infected at a reduced hazard).
    - RATES: spontaneous moves out of a state at a constant hazard per day (see `Rate`), such
      as vaccination; a state has either a phase or a rate.
    - STARTING_SHARES: state -> parameter, the share of the people who start in that state.
    """

    PARAMETERS: ClassVar[dict[str, ParameterSpec]]
    STATE_COLORS: ClassVar[dict[str, str]]
    INFECTIOUS_STATES: ClassVar[frozenset[str]]
    INFECTION_ENTRY_STATE: ClassVar[str]
    INITIAL_CASE_STATE: ClassVar[str]
    PHASES: ClassVar[dict[str, Phase]]
    STATE_INFECTIOUSNESS: ClassVar[dict[str, str]] = {}
    SYMPTOMATIC_STATES: ClassVar[frozenset[str] | None] = None
    CONFINED_STATES: ClassVar[frozenset[str]] = frozenset()
    PROTECTION: ClassVar[dict[str, str]] = {}
    RATES: ClassVar[dict[str, Rate]] = {}
    STARTING_SHARES: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        states = set(cls.STATE_COLORS)
        initial = next(iter(cls.STATE_COLORS))
        parameters = cls.PARAMETERS.keys()
        fatal = any(phase.fatal for phase in cls.PHASES.values())
        required = COMMON_PARAMETERS.keys() if fatal else COMMON_PARAMETERS.keys() - {"fatality"}
        if not required <= parameters:
            raise TypeError(f"{name}: PARAMETERS need {sorted(required)}")
        if DEAD not in states or not states >= cls.INFECTIOUS_STATES:
            raise TypeError(f"{name}: needs a dead state and infectious states that exist")
        if not {cls.INFECTION_ENTRY_STATE, cls.INITIAL_CASE_STATE} <= states:
            raise TypeError(f"{name}: entry and initial case states must be states")

        def known_state(state, what):
            if state not in states:
                raise TypeError(f"{name}: {what} uses an unknown state {state!r}")

        def known_parameter(parameter, what):
            if parameter not in parameters:
                raise TypeError(f"{name}: {what} uses an unknown parameter {parameter!r}")

        edges = {state: set() for state in states}
        for state, phase in cls.PHASES.items():
            what = f"phase {state}"
            known_state(state, what)
            known_state(phase.next_state, what)
            known_parameter(phase.duration_parameter, what)
            if state in (initial, DEAD):
                raise TypeError(f"{name}: the {state} state cannot have a phase")
            if (phase.branch_state is None) != (phase.branch_parameter is None):
                raise TypeError(f"{name}: {what} needs both branch_state and branch_parameter")
            edges[state].add(phase.next_state)
            if phase.branch_state is not None:
                known_state(phase.branch_state, what)
                known_parameter(phase.branch_parameter, what)
                edges[state].add(phase.branch_state)
            if DEAD in (phase.next_state, phase.branch_state):
                raise TypeError(f"{name}: {what} leads to dead; only a fatal phase kills")
        for state, rate in cls.RATES.items():
            what = f"rate of {state}"
            known_state(state, what)
            known_state(rate.next_state, what)
            known_parameter(rate.rate_parameter, what)
            if rate.start_parameter is not None:
                known_parameter(rate.start_parameter, what)
            if state != initial:
                raise TypeError(f"{name}: a rate leaves only the {initial} state, not {state}")
            if rate.next_state == DEAD:
                raise TypeError(f"{name}: {what} leads to dead; only a fatal phase kills")
            edges[state].add(rate.next_state)
        for state, parameter in cls.PROTECTION.items():
            known_state(state, f"protection of {state}")
            known_parameter(parameter, f"protection of {state}")
            if state == initial:
                raise TypeError(f"{name}: the {state} state cannot be protected")
        for state, parameter in cls.STATE_INFECTIOUSNESS.items():
            known_parameter(parameter, f"infectiousness of {state}")
            if state not in cls.INFECTIOUS_STATES:
                raise TypeError(f"{name}: {state} is not infectious, so it has no infectiousness")
        symptomatic = (
            cls.INFECTIOUS_STATES if cls.SYMPTOMATIC_STATES is None else cls.SYMPTOMATIC_STATES
        )
        for state in symptomatic:
            known_state(state, "the symptomatic states")
        for state in cls.CONFINED_STATES:
            known_state(state, "the confined states")
        if symptomatic & cls.CONFINED_STATES:
            raise TypeError(f"{name}: a state cannot be both confined and symptomatic")
        for state, parameter in cls.STARTING_SHARES.items():
            known_state(state, f"starting share of {state}")
            known_parameter(parameter, f"starting share of {state}")
            if state in (initial, DEAD):
                raise TypeError(f"{name}: nobody starts in the {state} state by a share")

        reachable = {cls.INFECTION_ENTRY_STATE, cls.INITIAL_CASE_STATE, initial}
        reachable |= {rate.next_state for rate in cls.RATES.values()}
        reachable |= set(cls.STARTING_SHARES)
        pending = list(reachable)
        while pending:
            for following in edges[pending.pop()] - reachable:
                reachable.add(following)
                pending.append(following)
        unreachable = states - reachable - {DEAD}
        if unreachable:
            raise TypeError(f"{name}: unreachable states {sorted(unreachable)}")
        _require_no_cycle(name, {s: targets - {initial} for s, targets in edges.items()})
        used = set(cls.STATE_INFECTIOUSNESS.values()) | set(cls.PROTECTION.values())
        used |= set(cls.STARTING_SHARES.values())
        used |= {phase.duration_parameter for phase in cls.PHASES.values()}
        used |= {phase.branch_parameter for phase in cls.PHASES.values()} - {None}
        for rate in cls.RATES.values():
            used |= {rate.rate_parameter, rate.start_parameter} - {None}
        unused = parameters - COMMON_PARAMETERS.keys() - used
        if unused:
            raise TypeError(f"{name}: parameters {sorted(unused)} are not used by the declaration")

    def __init__(self, step_days: float, age_effects: Mapping | None = None, **parameters: float):
        self.step_days = step_days
        self.parameters = resolve_parameters(self.PARAMETERS, parameters)
        self.age_effects = {name: _age_table(name, age_effects) for name in AGE_EFFECTS}
        self._require_effective_age_effects()
        if sum(self.parameters[parameter] for parameter in self.STARTING_SHARES.values()) > 1.0:
            raise ValueError("Starting shares add up to more than everybody")
        self._index = {state: index for index, state in enumerate(self.STATE_COLORS)}
        self._symptomatic = (
            self.INFECTIOUS_STATES if self.SYMPTOMATIC_STATES is None else self.SYMPTOMATIC_STATES
        )
        self._symptomatic_indices = [self._index[state] for state in self._symptomatic]
        self._state_weight = np.array(
            [self._relative_infectiousness(state) for state in self.STATE_COLORS], np.float32
        )
        self._moving = [
            self._index[state]
            for state in self.STATE_COLORS
            if self._is_left_over_time(state) or self._is_open_ended_infection(state)
        ]

    @classmethod
    def initial_state(cls) -> str:
        return next(iter(cls.STATE_COLORS))

    @classmethod
    def course_states(cls) -> frozenset[str]:
        """The states an infection runs through, from the one an infected person enters on."""
        found, pending = set(), [cls.INFECTION_ENTRY_STATE]
        while pending:
            state = pending.pop()
            if state in found or state in (cls.initial_state(), DEAD):
                continue
            found.add(state)
            phase = cls.PHASES.get(state)
            if phase is not None:
                pending += [s for s in (phase.next_state, phase.branch_state) if s is not None]
        return frozenset(found)

    @classmethod
    def exposable_states(cls) -> frozenset[str]:
        """The states infection can start from: the susceptible and those with some protection."""
        return frozenset({cls.initial_state(), *cls.PROTECTION})

    def protection_of(self, state: str) -> float:
        """How well people in `state` are protected against catching the disease, 0 to 1."""
        return self.parameters[self.PROTECTION[state]] if state in self.PROTECTION else 0.0

    @property
    def contact_distance(self) -> float:
        return self.parameters["contact_distance"]

    @property
    def dead_index(self) -> int:
        return self._index[DEAD]

    def create_state(
        self, population: int, initial_cases: np.ndarray, rng, age: np.ndarray
    ) -> DiseaseState:
        """Everybody starts susceptible except `initial_cases` and the starting shares."""
        susceptibility = self._age_multiplier("susceptibility", age)
        cv = self.parameters["susceptibility_cv"]
        if cv > 0:
            sigma = math.sqrt(math.log1p(cv**2))
            susceptibility = susceptibility * rng.lognormal(-(sigma**2) / 2, sigma, population)
        dispersion = self.parameters["infectiousness_dispersion"]
        infectiousness = (
            rng.gamma(dispersion, 1.0 / dispersion, population)
            if dispersion > 0
            else np.ones(population)
        )
        fatality = self.parameters.get("fatality", 0.0)
        disease_state = DiseaseState(
            state=np.zeros(population, np.int8),
            timer=np.zeros(population, np.float32),
            ever_infected=np.zeros(population, np.bool_),
            stays_home=np.zeros(population, np.bool_),
            limits_outings=np.zeros(population, np.bool_),
            susceptibility=susceptibility.astype(np.float32),
            infectiousness=infectiousness.astype(np.float32),
            fatality_risk=np.clip(
                fatality * self._age_multiplier("fatality", age), 0.0, 1.0
            ).astype(np.float32),
            branching=self._age_multiplier("branching", age).astype(np.float32),
        )
        if self.STARTING_SHARES:
            draw = rng.random(population)
            draw[initial_cases] = np.inf  # the first cases are in their own state
            upper = 0.0
            for state, parameter in self.STARTING_SHARES.items():
                lower, upper = upper, upper + self.parameters[parameter]
                self._enter(
                    disease_state, np.flatnonzero((draw >= lower) & (draw < upper)), state, rng
                )
        self._enter(disease_state, initial_cases, self.INITIAL_CASE_STATE, rng)
        disease_state.ever_infected[initial_cases] = True
        disease_state.cumulative_infections = len(initial_cases)
        return disease_state

    def infectious_weight(self, disease_state: DiseaseState) -> np.ndarray:
        """How much each person spreads because of their state: 0 if not infectious, 1 if fully."""
        return self._state_weight[disease_state.state]

    def state_infectiousness(self, state: str) -> float:
        """The relative infectiousness of people in `state`: 0 if not infectious, 1 if fully."""
        return float(self._state_weight[self._index[state]])

    def exposable_mask(self, disease_state: DiseaseState) -> np.ndarray:
        """Who can catch the disease: the susceptible and the partially protected."""
        mask = disease_state.state == 0
        for state, parameter in self.PROTECTION.items():
            if self.parameters[parameter] < 1.0:
                mask |= disease_state.state == self._index[state]
        return mask

    def advance(self, disease_state: DiseaseState, exposure: np.ndarray, rng) -> None:
        """One step of a single disease: new infections, expired phases, then the infected enter.

        `exposure` is each person's infectious contacts, already weighted by how infectious
        those contacts are and by interactions with other diseases. When several diseases run
        together the three parts are called separately (see `EpidemicModel.step`).
        """
        infected = self.draw_infections(disease_state, exposure, rng)
        self.finish_phases(disease_state, rng)
        self.apply_infections(disease_state, infected, rng)

    def draw_infections(self, disease_state: DiseaseState, exposure: np.ndarray, rng) -> np.ndarray:
        """Who catches the disease this step, from the states at its start."""
        exposure = exposure * disease_state.susceptibility
        infected = self._new_infections(disease_state, exposure, 0, 1.0, rng)
        for state, parameter in self.PROTECTION.items():
            protection = self.parameters[parameter]
            if protection < 1.0:
                reinfected = self._new_infections(
                    disease_state, exposure, self._index[state], 1.0 - protection, rng
                )
                infected = np.concatenate([infected, reinfected])
        return infected

    def finish_phases(self, disease_state: DiseaseState, rng) -> None:
        """Move on everybody whose phase has run out or who is moved by a rate.

        Some die at the end of a fatal phase; of the survivors some take the branch. Everybody
        takes at most one move per step, from the state they were in at its start.
        """
        step_days = self.step_days
        moved = self._moved_by_rates(disease_state, rng)
        finished = {}
        for state in self.PHASES:
            members = np.flatnonzero(disease_state.state == self._index[state])
            disease_state.timer[members] -= step_days
            finished[state] = members[disease_state.timer[members] <= 0]
        for state, people in finished.items():
            phase = self.PHASES[state]
            if phase.fatal:
                dies = rng.random(len(people)) < disease_state.fatality_risk[people]
                self._enter(disease_state, people[dies], DEAD, rng)
                disease_state.deaths += int(dies.sum())
                people = people[~dies]
            if phase.branch_state is not None:
                chance = np.minimum(
                    self.parameters[phase.branch_parameter] * disease_state.branching[people], 1.0
                )
                branched = rng.random(len(people)) < chance
                self._enter(disease_state, people[branched], phase.branch_state, rng)
                people = people[~branched]
            self._enter(disease_state, people, phase.next_state, rng)
        for state, people in moved.items():
            self._enter(disease_state, people, self.RATES[state].next_state, rng)
        disease_state.steps += 1

    def apply_infections(self, disease_state: DiseaseState, infected: np.ndarray, rng) -> None:
        """Let the infected enter the disease, except those who died in the meantime."""
        infected = infected[disease_state.state[infected] != self.dead_index]
        self._enter(disease_state, infected, self.INFECTION_ENTRY_STATE, rng)
        disease_state.ever_infected[infected] = True
        disease_state.cumulative_infections += len(infected)

    def bury(self, disease_state: DiseaseState, people: np.ndarray) -> None:
        """Mark `people` who died of another disease as dead for this one."""
        disease_state.state[people] = self._index[DEAD]
        disease_state.stays_home[people] = False
        disease_state.limits_outings[people] = False

    def courses(self) -> list[Course]:
        """Every way an infection can run, from the state an infected person enters.

        The chances are those of a person whose age effects are 1; a course that cannot happen
        is left out. Time spent in a state is its mean length, so the numbers derived from the
        courses are expectations over many people.
        """
        found = []

        def walk(state: str, probability: float, stays: tuple) -> None:
            if state in (self.initial_state(), DEAD):
                found.append(Course(probability, stays, state))
                return
            phase = self.PHASES.get(state)
            mean = math.inf if phase is None else self.parameters[phase.duration_parameter]
            if mean >= LIFELONG_DAYS:
                mean = math.inf
            stays = (*stays, (state, mean, float(self._state_weight[self._index[state]])))
            if math.isinf(mean):
                found.append(Course(probability, stays, state))
                return
            survive = 1.0
            if phase.fatal:
                died = min(self.parameters["fatality"], 1.0)
                if died > 0:
                    found.append(Course(probability * died, stays, DEAD))
                survive = 1.0 - died
            branch = 0.0
            if phase.branch_state is not None:
                branch = min(self.parameters[phase.branch_parameter], 1.0)
                if branch > 0:
                    walk(phase.branch_state, probability * survive * branch, stays)
            if survive * (1.0 - branch) > 0:
                walk(phase.next_state, probability * survive * (1.0 - branch), stays)

        walk(self.INFECTION_ENTRY_STATE, 1.0, ())
        return found

    def infectious_days(self) -> float:
        """Mean time an infected person spends spreading, each state weighed by its infectiousness.

        Infinite when an infectious state is never left.
        """
        return sum(
            course.probability
            * sum(mean * weight for _, mean, weight in course.stays if weight > 0)
            for course in self.courses()
        )

    def mean_field_r0(self, mean_contacts: float) -> float | None:
        """Reproduction number if every contact were new and independent.

        None when transmission is certain or the infection never ends. It overestimates the
        real value when contacts are shared, as in households and workplaces.
        """
        transmission = self.parameters["transmission"]
        days = self.infectious_days()
        if transmission >= 1.0 or math.isinf(days):
            return None
        return -math.log(1.0 - transmission) * mean_contacts * days

    def pair_transmission_probability(self) -> float:
        """Chance that an infectious person infects someone who stays next to them throughout.

        Exact for gamma distributed phases: 1 - E[exp(-hazard * infectiousness-weighted time)],
        the expectation over the ways the infection can run.
        """
        transmission = self.parameters["transmission"]
        if transmission >= 1.0:
            return 1.0
        hazard = -math.log(1.0 - transmission)
        if hazard == 0.0:
            return 0.0
        shape = self.parameters["duration_shape"]
        survival = 0.0
        for course in self.courses():
            kept = course.probability
            for _, mean, weight in course.stays:
                if weight > 0:
                    kept *= (
                        0.0
                        if math.isinf(mean)
                        else (1.0 + hazard * weight * mean / shape) ** -shape
                    )
            survival += kept
        return min(1.0, max(0.0, 1.0 - survival))

    def is_quiescent(self, disease_state: DiseaseState) -> bool:
        """True when nobody is in a state that can still change, so nothing changes any more."""
        return not np.isin(disease_state.state, self._moving).any()

    def lasts_for_life(self, state: str) -> bool:
        """True when people in `state` never leave it by themselves (a stay of `LIFELONG_DAYS`)."""
        phase = self.PHASES.get(state)
        return phase is not None and self.parameters[phase.duration_parameter] >= LIFELONG_DAYS

    def _is_left_over_time(self, state: str) -> bool:
        phase = self.PHASES.get(state)
        rate = self.RATES.get(state)
        return (
            phase is not None and self.parameters[phase.duration_parameter] < LIFELONG_DAYS
        ) or (rate is not None and self.parameters[rate.rate_parameter] > 0)

    def _is_open_ended_infection(self, state: str) -> bool:
        """An infectious state nobody leaves: as long as it lasts, others can still be infected."""
        phase = self.PHASES.get(state)
        never_left = phase is None or self.parameters[phase.duration_parameter] >= LIFELONG_DAYS
        return state in self.INFECTIOUS_STATES and never_left

    def _relative_infectiousness(self, state: str) -> float:
        if state not in self.INFECTIOUS_STATES:
            return 0.0
        return (
            self.parameters[self.STATE_INFECTIOUSNESS[state]]
            if state in self.STATE_INFECTIOUSNESS
            else 1.0
        )

    def _require_effective_age_effects(self) -> None:
        used = {
            "fatality": any(phase.fatal for phase in self.PHASES.values()),
            "branching": any(phase.branch_state is not None for phase in self.PHASES.values()),
        }
        for effect, applies in used.items():
            if self.age_effects[effect] and not applies:
                raise ValueError(
                    f"Age effect '{effect}' does nothing in {type(self).__name__}: "
                    f"it has no {'fatal phase' if effect == 'fatality' else 'branch'}"
                )

    def _moved_by_rates(self, disease_state: DiseaseState, rng) -> dict[str, np.ndarray]:
        """People moved by a rate in this step, by state; chosen from the states at its start."""
        clock = disease_state.steps * self.step_days
        moved = {}
        for state, rate in self.RATES.items():
            per_day = self.parameters[rate.rate_parameter]
            start = 0.0 if rate.start_parameter is None else self.parameters[rate.start_parameter]
            if per_day > 0 and clock >= start - 1e-9:
                members = np.flatnonzero(disease_state.state == self._index[state])
                chance = 1.0 - math.exp(-per_day * self.step_days)
                moved[state] = members[rng.random(len(members)) < chance]
        return moved

    def _new_infections(self, disease_state, exposure, state_index, scale, rng) -> np.ndarray:
        """People in `state_index` who catch the disease this step; `scale` weakens the exposure."""
        candidates = np.flatnonzero((disease_state.state == state_index) & (exposure > 0))
        chance = 1.0 - (1.0 - self.parameters["transmission"]) ** (
            exposure[candidates] * scale * self.step_days
        )
        return candidates[rng.random(len(candidates)) < chance]

    def _age_multiplier(self, effect: str, age: np.ndarray) -> np.ndarray:
        multiplier = np.ones(len(age))
        for start, end, value in self.age_effects[effect]:
            multiplier[(age >= start) & (age < end)] = value
        return multiplier

    def _enter(self, disease_state: DiseaseState, people: np.ndarray, state: str, rng) -> None:
        """Put `people` into `state`; a person moving between symptomatic states keeps their
        decision to stay at home or skip outings, anybody else draws it."""
        was_symptomatic = np.isin(disease_state.state[people], self._symptomatic_indices)
        disease_state.state[people] = self._index[state]
        symptomatic = state in self._symptomatic
        disease_state.stays_home[people] = state in self.CONFINED_STATES or (
            symptomatic
            and np.where(
                was_symptomatic,
                disease_state.stays_home[people],
                rng.random(len(people)) < self.parameters["stay_home"],
            )
        )
        disease_state.limits_outings[people] = symptomatic and np.where(
            was_symptomatic,
            disease_state.limits_outings[people],
            rng.random(len(people)) < self.parameters["limit_outings"],
        )
        phase = self.PHASES.get(state)
        if phase is not None:
            shape = self.parameters["duration_shape"]
            mean = self.parameters[phase.duration_parameter]
            stays = rng.gamma(shape, mean / shape, len(people))
            # a stay "for life" is never left, however irregular stays are; the gamma draw is
            # kept so that the random stream of the other stays does not depend on it
            disease_state.timer[people] = NEVER_DAYS if mean >= LIFELONG_DAYS else stays


def _require_no_cycle(name: str, edges: dict[str, set[str]]) -> None:
    """Raise TypeError when states lead back to themselves other than through the initial one."""
    done, open_ = set(), set()

    def visit(state: str) -> None:
        if state in open_:
            raise TypeError(f"{name}: states lead back to {state!r} in a cycle")
        if state in done:
            return
        open_.add(state)
        for following in edges[state]:
            visit(following)
        open_.discard(state)
        done.add(state)

    for state in edges:
        visit(state)


def _age_table(effect: str, age_effects: Mapping | None) -> tuple[tuple[float, float, float], ...]:
    """Validated [from age, to age, multiplier] ranges of an age effect; none means no effect."""
    unknown = set(age_effects or {}) - set(AGE_EFFECTS)
    if unknown:
        raise ValueError(f"Unknown age effects {sorted(unknown)}, available: {list(AGE_EFFECTS)}")
    table = tuple(
        tuple(float(value) for value in row) for row in (age_effects or {}).get(effect, ())
    )
    if any(len(row) != 3 or row[0] >= row[1] or row[2] < 0 for row in table):
        raise ValueError(f"Invalid age ranges in {effect}")
    return table
