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
AGE_EFFECTS = ("susceptibility", "fatality")

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
    "Chance that an infection ends in death, for a person of average age (see age effects).",
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

    A `fatal` phase ends in death for people who die of the disease instead.
    """

    next_state: str
    duration_parameter: str
    fatal: bool = False


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
    cumulative_infections: int = 0
    deaths: int = 0


class DiseaseModelAbstract:
    """Course of one disease, advanced for the whole population at once in steps of `step_days`.

    A model is declared by its data: PARAMETERS, STATE_COLORS (the first state is the initial
    one, DEAD the absorbing one), INFECTIOUS_STATES, INFECTION_ENTRY_STATE (where an infected
    person goes), INITIAL_CASE_STATE (where the first cases start), PHASES (timed transitions)
    and IMMUNE_STATE (whose people keep partial protection, or None).
    """

    PARAMETERS: ClassVar[dict[str, ParameterSpec]]
    STATE_COLORS: ClassVar[dict[str, str]]
    INFECTIOUS_STATES: ClassVar[frozenset[str]]
    INFECTION_ENTRY_STATE: ClassVar[str]
    INITIAL_CASE_STATE: ClassVar[str]
    PHASES: ClassVar[dict[str, Phase]]
    IMMUNE_STATE: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        states = set(cls.STATE_COLORS)
        if not COMMON_PARAMETERS.keys() <= cls.PARAMETERS.keys():
            raise TypeError(f"{cls.__name__}: PARAMETERS need {sorted(COMMON_PARAMETERS)}")
        if DEAD not in states or not states >= cls.INFECTIOUS_STATES:
            raise TypeError(f"{cls.__name__}: needs a dead state and infectious states that exist")
        if not {cls.INFECTION_ENTRY_STATE, cls.INITIAL_CASE_STATE} <= states:
            raise TypeError(f"{cls.__name__}: entry and initial case states must be states")
        if cls.IMMUNE_STATE is not None and (
            cls.IMMUNE_STATE not in states
            or not IMMUNITY_PARAMETERS.keys() <= cls.PARAMETERS.keys()
        ):
            raise TypeError(f"{cls.__name__}: invalid immune state or missing immunity parameters")
        for state, phase in cls.PHASES.items():
            if state not in states or phase.next_state not in states:
                raise TypeError(f"{cls.__name__}: phase {state} uses an unknown state")
            if phase.duration_parameter not in cls.PARAMETERS:
                raise TypeError(f"{cls.__name__}: phase {state} uses an unknown parameter")

    def __init__(self, step_days: float, age_effects: Mapping | None = None, **parameters: float):
        self.step_days = step_days
        self.parameters = resolve_parameters(self.PARAMETERS, parameters)
        self.age_effects = {name: _age_table(name, age_effects) for name in AGE_EFFECTS}
        self._index = {state: index for index, state in enumerate(self.STATE_COLORS)}
        self._infectious_by_index = np.array(
            [state in self.INFECTIOUS_STATES for state in self.STATE_COLORS]
        )
        self._timed = [
            self._index[state]
            for state, phase in self.PHASES.items()
            if self.parameters[phase.duration_parameter] < LIFELONG_DAYS
        ]

    @classmethod
    def initial_state(cls) -> str:
        return next(iter(cls.STATE_COLORS))

    @property
    def contact_distance(self) -> float:
        return self.parameters["contact_distance"]

    @property
    def dead_index(self) -> int:
        return self._index[DEAD]

    def create_state(
        self, population: int, initial_cases: np.ndarray, rng, age: np.ndarray
    ) -> DiseaseState:
        """Everybody starts susceptible except `initial_cases`; everybody has their own traits."""
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
        disease_state = DiseaseState(
            state=np.zeros(population, np.int8),
            timer=np.zeros(population, np.float32),
            ever_infected=np.zeros(population, np.bool_),
            stays_home=np.zeros(population, np.bool_),
            limits_outings=np.zeros(population, np.bool_),
            susceptibility=susceptibility.astype(np.float32),
            infectiousness=infectiousness.astype(np.float32),
            fatality_risk=np.clip(
                self.parameters["fatality"] * self._age_multiplier("fatality", age), 0.0, 1.0
            ).astype(np.float32),
        )
        self._enter(disease_state, initial_cases, self.INITIAL_CASE_STATE, rng)
        disease_state.ever_infected[initial_cases] = True
        disease_state.cumulative_infections = len(initial_cases)
        return disease_state

    def infectious_mask(self, disease_state: DiseaseState) -> np.ndarray:
        return self._infectious_by_index[disease_state.state]

    def exposable_mask(self, disease_state: DiseaseState) -> np.ndarray:
        """Who can catch the disease: the susceptible and the partially protected."""
        mask = disease_state.state == 0
        if self.IMMUNE_STATE is not None and self.parameters["immunity_protection"] < 1.0:
            mask |= disease_state.state == self._index[self.IMMUNE_STATE]
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
        if self.IMMUNE_STATE is not None and self.parameters["immunity_protection"] < 1.0:
            reinfected = self._new_infections(
                disease_state,
                exposure,
                self._index[self.IMMUNE_STATE],
                1.0 - self.parameters["immunity_protection"],
                rng,
            )
            infected = np.concatenate([infected, reinfected])
        return infected

    def finish_phases(self, disease_state: DiseaseState, rng) -> None:
        """Move on everybody whose phase has run out; some die at the end of a fatal phase."""
        step_days = self.step_days
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
            self._enter(disease_state, people, phase.next_state, rng)

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

    def infectious_days(self) -> float:
        """Mean total time a person spends in infectious states."""
        return sum(
            self.parameters[phase.duration_parameter]
            for state, phase in self.PHASES.items()
            if state in self.INFECTIOUS_STATES
        )

    def mean_field_r0(self, mean_contacts: float) -> float | None:
        """Reproduction number if every contact were new and independent.

        None when transmission is certain. It overestimates the real value when contacts are
        shared, as in households and workplaces.
        """
        transmission = self.parameters["transmission"]
        if transmission >= 1.0:
            return None
        return -math.log(1.0 - transmission) * mean_contacts * self.infectious_days()

    def pair_transmission_probability(self) -> float:
        """Chance that an infectious person infects someone who stays next to them throughout.

        Exact for gamma distributed phases: 1 - E[exp(-hazard * infectious time)].
        """
        transmission = self.parameters["transmission"]
        if transmission >= 1.0:
            return 1.0
        hazard = -math.log(1.0 - transmission)
        shape = self.parameters["duration_shape"]
        survival = 1.0
        for state, phase in self.PHASES.items():
            if state in self.INFECTIOUS_STATES:
                mean = self.parameters[phase.duration_parameter]
                survival *= (1.0 + hazard * mean / shape) ** (-shape)
        return 1.0 - survival

    def is_quiescent(self, disease_state: DiseaseState) -> bool:
        """True when nobody is in a timed phase that can still end, so nothing changes any more."""
        return not np.isin(disease_state.state, self._timed).any()

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
        disease_state.state[people] = self._index[state]
        infectious = state in self.INFECTIOUS_STATES
        disease_state.stays_home[people] = (
            infectious and rng.random(len(people)) < self.parameters["stay_home"]
        )
        disease_state.limits_outings[people] = (
            infectious and rng.random(len(people)) < self.parameters["limit_outings"]
        )
        phase = self.PHASES.get(state)
        if phase is not None:
            shape = self.parameters["duration_shape"]
            mean = self.parameters[phase.duration_parameter]
            disease_state.timer[people] = rng.gamma(shape, mean / shape, len(people))


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
