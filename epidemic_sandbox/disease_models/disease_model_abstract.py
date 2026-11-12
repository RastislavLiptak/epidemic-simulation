import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from epidemic_sandbox.config.parameter_spec import ParameterSpec, resolve_parameters

SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"
INFECTIOUS = "infectious"
RECOVERED = "recovered"
ASYMPTOMATIC = "asymptomatic"
DYING = "dying"
DEAD = "dead"
LIFELONG_DAYS = 36500
DAYS_PER_YEAR = 365.25
NEVER_DAYS = 1e9  # the timer of a stay that is never left
INFECTIOUSNESS_SAMPLE = 200_000  # draws of the infectiousness of an infection, for expectations
AGE_EFFECTS = (
    "susceptibility",
    "fatality",
    "branching",
    "starting_share",
    "outside_infections",
    "asymptomatic",
    "stay_home",
)
BRANCH_AGE_EFFECTS = ("branching", "asymptomatic")

TRANSMISSION = ParameterSpec(
    "Transmission (per day of contact)",
    "Chance that a susceptible person gets infected during one day spent within contact distance "
    "of one infectious person.",
    0.3,
    0.0,
    1.0,
    0.01,
)
COMMUNITY_FACTOR = ParameterSpec(
    "Spread outside the household (relative)",
    "How much more (or less) easily the disease passes between people who do not share a "
    "household than between members of one household, for the same time within contact "
    "distance. 1 treats both alike; higher values stand for what the town does not model: "
    "crowds, shared air, brief contacts.",
    1.0,
    0.0,
    100.0,
    0.1,
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
NEAR_DISTANCE = ParameterSpec(
    "Full-exposure distance (m)",
    "Up to this distance an infectious person exposes others fully; farther away, up to the "
    "contact distance, the exposure falls with the square of the distance. About 1 m for droplets "
    "and aerosols (the risk model of the NHS COVID-19 app, whose risk scores Ferretti et al. 2024, "
    "Nature 626:145, found proportional to the chance of transmission); the contact distance "
    "itself for a disease whose virus fills the air of a room evenly.",
    1.0,
    0.1,
    50.0,
    0.1,
)
ROOM_EXPOSURE = ParameterSpec(
    "Exposure in the air of a room (per hour, relative to close range)",
    "How much an infectious person exposes everybody who shares an indoor room with them, "
    "whatever the distance, relative to a person at the full-exposure distance: what virus in the "
    "air of a room does (airborne spread, as in Wells-Riley models). It adds to the exposure at "
    "close range; 0 means no spread beyond the contact distance. Outdoors there is none.",
    0.0,
    0.0,
    1000.0,
    0.001,
)
ASLEEP_EMISSION = ParameterSpec(
    "Infectiousness while asleep (relative to awake)",
    "How much an infectious person who is asleep (in bed at night) exposes others compared with "
    "one who is awake: what a person emits depends on what they do (breathing at rest emits far "
    "less than speaking; Buonanno et al. 2020, Environ Int 141:105794). 1 means no difference.",
    1.0,
    0.0,
    1.0,
    0.01,
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
    "Stay-at-home share (ill)",
    "Share of the ill who stay at home while they have symptoms, for a person whose stay-at-home "
    "age effect is 1 (see age effects). A state of the illness can have its own share.",
    0.0,
    0.0,
    1.0,
    0.01,
)
LIMIT_OUTINGS = ParameterSpec(
    "Fewer outings share (ill)",
    "Share of the ill who still go to work or school but skip errands, visits and leisure. A "
    "state of the illness can have its own share.",
    0.0,
    0.0,
    1.0,
    0.01,
)
FATALITY = ParameterSpec(
    "Fatality (per infection)",
    "Chance that an infection ends in death for a person whose age effect is 1; the age effects "
    "scale it up and down. Only infections with symptoms end in death: where a share of the "
    "infections has no symptoms, those with symptoms carry the deaths of all, so the chance per "
    "infection stays this one (capped at certain death).",
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
    "How much infections differ in how much they spread (a new draw for every infection). Small "
    "values mean a few superspreaders cause most infections; 0 means every infection spreads "
    "alike. It adds to the variation the town itself makes (households of different sizes, people "
    "who meet more), so it is set to give the published dispersion of the number of people one "
    "case infects, not taken from it.",
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

DEATH_DELAY = ParameterSpec(
    "Time to death (days)",
    "Average number of days that people who die of the disease live after their infectious "
    "course has ended. They are gravely ill (in hospital), meet nobody in the town and spread "
    "nothing; 0 means they die when the infectious course ends.",
    0.0,
    0.0,
    120.0,
    0.1,
)
SEVERE_PROTECTION = ParameterSpec(
    "Protection against death on reinfection",
    "How much an earlier infection lowers the chance of dying of a new one: 0 means a "
    "reinfection is as dangerous as a first infection, 1 that nobody dies of a reinfection. "
    "People who start the experiment immune (recovered) count as infected before.",
    0.0,
    0.0,
    1.0,
    0.01,
)
DEATH_PARAMETERS = {"death_delay_days": DEATH_DELAY, "severe_protection": SEVERE_PROTECTION}

OUTSIDE_INFECTIONS = ParameterSpec(
    "Infections from outside (per person and year)",
    "How often a susceptible person catches the disease outside the town, or from people the "
    "model does not have (healthy carriers, visitors), per year, for a person whose age effect "
    "is 1 (see age effects). 0 is a closed town, in which a disease dies out when its last case "
    "recovers.",
    0.0,
    0.0,
    50.0,
    0.001,
)

ASYMPTOMATIC_SHARE = ParameterSpec(
    "Infections without symptoms (share)",
    "Share of the infections that never cause symptoms, for a person whose asymptomatic age effect "
    "is 1 (see age effects). Such a person spreads the disease without knowing it, goes about the "
    "day as usual and does not die of it. 0 means every infection has symptoms.",
    0.0,
    0.0,
    1.0,
    0.01,
)
ASYMPTOMATIC_DAYS = ParameterSpec(
    "Infectious period without symptoms (days)",
    "Average number of days an infection without symptoms spreads the disease (used only when "
    "the share without symptoms is above 0).",
    5.0,
    0.1,
    60.0,
    0.1,
)
ASYMPTOMATIC_INFECTIOUSNESS = ParameterSpec(
    "Infectiousness without symptoms (relative, whole infection)",
    "How much an infection without symptoms spreads in all, compared with one with symptoms, for "
    "the same contacts: the ratio of the secondary attack rates of the contacts of people "
    "without and with symptoms that contact studies report. 1 is the same, 0 not at all. It is "
    "spread evenly over the days without symptoms; the behaviour of the ill (staying at home) "
    "comes on top of it.",
    1.0,
    0.0,
    2.0,
    0.01,
)
ASYMPTOMATIC_PARAMETERS = {
    "asymptomatic_share": ASYMPTOMATIC_SHARE,
    "asymptomatic_days": ASYMPTOMATIC_DAYS,
    "asymptomatic_infectiousness": ASYMPTOMATIC_INFECTIOUSNESS,
}

COMMON_PARAMETERS = {
    "transmission": TRANSMISSION,
    "contact_distance": CONTACT_DISTANCE,
    "near_distance": NEAR_DISTANCE,
    "room_exposure": ROOM_EXPOSURE,
    "asleep_emission": ASLEEP_EMISSION,
    "community_factor": COMMUNITY_FACTOR,
    "duration_shape": DURATION_SHAPE,
    "stay_home": STAY_HOME,
    "limit_outings": LIMIT_OUTINGS,
    "fatality": FATALITY,
    "susceptibility_cv": SUSCEPTIBILITY_CV,
    "infectiousness_dispersion": INFECTIOUSNESS_DISPERSION,
    "outside_infections": OUTSIDE_INFECTIONS,
}
IMMUNITY_PARAMETERS = {"immunity_days": IMMUNITY_DAYS, "immunity_protection": IMMUNITY_PROTECTION}


@dataclass(frozen=True)
class Phase:
    """After a gamma-distributed time (mean in `duration_parameter`), go to `next_state`.

    A `fatal` phase ends, for people who die of the disease, in the DYING state instead, which
    they leave dead after `death_delay_days`. Of the survivors a share goes to `branch_state`
    instead of `next_state`: the value of `branch_parameter`, scaled by the person's age effect
    `branch_age_effect` ("branching", or "asymptomatic" for the course without symptoms) and
    capped at 1.
    """

    next_state: str
    duration_parameter: str
    fatal: bool = False
    branch_state: str | None = None
    branch_parameter: str | None = None
    branch_age_effect: str = "branching"


@dataclass(frozen=True)
class Branch:
    """A share of the newly infected enters `state` instead of the entry state: the value of
    `parameter`, scaled by the person's age effect `age_effect` and capped at 1."""

    state: str
    parameter: str
    age_effect: str = "branching"


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
    reinfected: np.ndarray | None = None  # the person's current infection is not their first
    immune_before: np.ndarray | None = None  # infected before or started immune (recovered)
    outside: np.ndarray | None = None  # how often the person is infected from outside, by age
    asymptomatic: np.ndarray | None = None  # the "asymptomatic" age effect of the person
    home_scale: np.ndarray | None = None  # the "stay_home" age effect of the person
    # how the person reacts to the current infection: they stay at home in a state whose chance
    # of it (times the age effect) is above `home_draw`, and so on; drawn anew for every infection,
    # so that a person's decisions in the states of one illness are consistent
    home_draw: np.ndarray | None = None
    outings_draw: np.ndarray | None = None
    hospital_draw: np.ndarray | None = None
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
      A model with a fatal phase also has the DYING state and DEATH_PARAMETERS: those who die
      of the disease are gravely ill for a while, away from everybody, before they die.
    - INFECTIOUS_STATES, INFECTION_ENTRY_STATE (where an infected person goes) and
      INITIAL_CASE_STATE (where the first cases start).
    - PHASES: timed transitions out of a state, to a next state and optionally, for a share of
      the people, to a branch state (see `Phase`). A state without a phase is never left.

    Optional, all empty by default:

    - STATE_INFECTIOUSNESS: infectious state -> parameter, the relative infectiousness of that
      state (an infectious state not listed spreads fully).
    - SYMPTOMATIC_STATES: states in which a person has symptoms and reacts to being ill, staying
      at home or skipping outings with the chances `stay_home` (times the person's "stay_home"
      age effect, capped at 1) and `limit_outings` (default: all infectious states). BEHAVIOUR:
      symptomatic state -> (stay-at-home parameter, fewer-outings parameter) for a state with its
      own chances (an unspecific prodrome before the illness by which a disease is recognised).
      One draw per infection decides, so a person who stays at home in a state also does in any
      state with a higher chance. CONFINED_STATES: states in which a person is always at home.
    - HOSPITAL: symptomatic or confined state -> parameter, the share of the people in that
      state who are in hospital: away from the town, they meet nobody and infect nobody in it
      (one draw per infection, as for the behaviour). Those who die of a disease are in hospital
      while they are dying.
    - PROTECTION: state -> parameter, how well people in that state are protected against
      catching the disease (1 is complete; below 1 they can be infected at a reduced hazard).
    - RATES: spontaneous moves out of a state at a constant hazard per day (see `Rate`), such
      as vaccination; a state has either a phase or a rate.
    - STARTING_SHARES: state -> parameter, the share of the people who start in that state. The
      "starting_share" age effect scales every such share by age (capped at 1), so a disease can
      state, for instance, the immune share found by an age-resolved serological survey.
    - INFECTION_BRANCH: a share of the newly infected enters another state than
      INFECTION_ENTRY_STATE (see `Branch`), for a course that splits when it starts.

    Infections without symptoms are common to all models that have the ASYMPTOMATIC state: it
    is reached by a branch (of a phase, or INFECTION_BRANCH) with the share "asymptomatic_share"
    and the age effect "asymptomatic", left by a phase of "asymptomatic_days" that does not kill,
    and the model has ASYMPTOMATIC_PARAMETERS. The state is infectious and never symptomatic (a
    person in it behaves as usual); how much it spreads per day follows from
    "asymptomatic_infectiousness", what an infection without symptoms spreads in all relative to
    one with symptoms. The first cases always have symptoms (they are the cases noticed).
    """

    PARAMETERS: ClassVar[dict[str, ParameterSpec]]
    STATE_COLORS: ClassVar[dict[str, str]]
    INFECTIOUS_STATES: ClassVar[frozenset[str]]
    INFECTION_ENTRY_STATE: ClassVar[str]
    INITIAL_CASE_STATE: ClassVar[str]
    PHASES: ClassVar[dict[str, Phase]]
    INFECTION_BRANCH: ClassVar[Branch | None] = None
    STATE_INFECTIOUSNESS: ClassVar[dict[str, str]] = {}
    SYMPTOMATIC_STATES: ClassVar[frozenset[str] | None] = None
    CONFINED_STATES: ClassVar[frozenset[str]] = frozenset()
    PROTECTION: ClassVar[dict[str, str]] = {}
    RATES: ClassVar[dict[str, Rate]] = {}
    STARTING_SHARES: ClassVar[dict[str, str]] = {}
    BEHAVIOUR: ClassVar[dict[str, tuple[str, str]]] = {}
    HOSPITAL: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__
        states = set(cls.STATE_COLORS)
        initial = next(iter(cls.STATE_COLORS))
        parameters = cls.PARAMETERS.keys()
        fatal = any(phase.fatal for phase in cls.PHASES.values())
        required = COMMON_PARAMETERS.keys() if fatal else COMMON_PARAMETERS.keys() - {"fatality"}
        if fatal:
            required = required | DEATH_PARAMETERS.keys()
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
            if state in (initial, DEAD, DYING):
                raise TypeError(f"{name}: the {state} state cannot have a phase")
            if (phase.branch_state is None) != (phase.branch_parameter is None):
                raise TypeError(f"{name}: {what} needs both branch_state and branch_parameter")
            if phase.branch_age_effect not in BRANCH_AGE_EFFECTS:
                effect = phase.branch_age_effect
                raise TypeError(f"{name}: {what} has no branch age effect {effect}")
            edges[state].add(phase.next_state)
            if phase.branch_state is not None:
                known_state(phase.branch_state, what)
                known_parameter(phase.branch_parameter, what)
                edges[state].add(phase.branch_state)
            if {DEAD, DYING} & {phase.next_state, phase.branch_state}:
                raise TypeError(f"{name}: {what} leads to dead; only a fatal phase kills")
            if phase.fatal:
                edges[state].add(DYING)
        for state, rate in cls.RATES.items():
            what = f"rate of {state}"
            known_state(state, what)
            known_state(rate.next_state, what)
            known_parameter(rate.rate_parameter, what)
            if rate.start_parameter is not None:
                known_parameter(rate.start_parameter, what)
            if state != initial:
                raise TypeError(f"{name}: a rate leaves only the {initial} state, not {state}")
            if rate.next_state in (DEAD, DYING):
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
        for state, behaviour in cls.BEHAVIOUR.items():
            if state not in symptomatic:
                raise TypeError(f"{name}: only a symptomatic state has a behaviour, not {state}")
            if len(behaviour) != 2:
                raise TypeError(f"{name}: the behaviour of {state} needs two parameters")
            for parameter in behaviour:
                known_parameter(parameter, f"the behaviour of {state}")
        for state, parameter in cls.HOSPITAL.items():
            known_state(state, f"the hospital share of {state}")
            known_parameter(parameter, f"the hospital share of {state}")
            if state not in symptomatic | cls.CONFINED_STATES:
                raise TypeError(
                    f"{name}: a hospital state is one of the ill (symptomatic) or isolated "
                    f"(confined), not {state}"
                )
        for state, parameter in cls.STARTING_SHARES.items():
            known_state(state, f"starting share of {state}")
            known_parameter(parameter, f"starting share of {state}")
            if state in (initial, DEAD, DYING):
                raise TypeError(f"{name}: nobody starts in the {state} state by a share")

        if fatal != (DYING in states):
            raise TypeError(f"{name}: a model with a fatal phase, and only such, has a dying state")
        special = {DYING} & (
            set(cls.INFECTIOUS_STATES)
            | set(cls.PHASES)
            | set(cls.PROTECTION)
            | set(cls.RATES)
            | set(cls.STARTING_SHARES)
            | set(cls.CONFINED_STATES)
            | set(cls.SYMPTOMATIC_STATES or ())
            | set(cls.HOSPITAL)
        )
        if special:
            raise TypeError(f"{name}: the dying state is handled by the base class only")
        entry = cls.INFECTION_BRANCH
        if entry is not None:
            known_state(entry.state, "the infection branch")
            known_parameter(entry.parameter, "the infection branch")
            if entry.state in (initial, DEAD, DYING, cls.INFECTION_ENTRY_STATE):
                raise TypeError(f"{name}: the infection branch cannot lead to {entry.state}")
            if entry.age_effect not in BRANCH_AGE_EFFECTS:
                effect = entry.age_effect
                raise TypeError(f"{name}: the infection branch has no age effect {effect}")
        if ASYMPTOMATIC in states:
            cls._require_asymptomatic_course(name, symptomatic)

        reachable = {cls.INFECTION_ENTRY_STATE, cls.INITIAL_CASE_STATE, initial}
        if entry is not None:
            reachable.add(entry.state)
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
        used |= {parameter for pair in cls.BEHAVIOUR.values() for parameter in pair}
        used |= set(cls.HOSPITAL.values())
        used |= {phase.duration_parameter for phase in cls.PHASES.values()}
        used |= {phase.branch_parameter for phase in cls.PHASES.values()} - {None}
        if entry is not None:
            used.add(entry.parameter)
        if ASYMPTOMATIC in states:
            used |= ASYMPTOMATIC_PARAMETERS.keys()
        for rate in cls.RATES.values():
            used |= {rate.rate_parameter, rate.start_parameter} - {None}
        unused = parameters - COMMON_PARAMETERS.keys() - DEATH_PARAMETERS.keys() - used
        if unused:
            raise TypeError(f"{name}: parameters {sorted(unused)} are not used by the declaration")

    @classmethod
    def _require_asymptomatic_course(cls, name: str, symptomatic: frozenset[str]) -> None:
        """The ASYMPTOMATIC state is the common course without symptoms (see the class)."""
        if not ASYMPTOMATIC_PARAMETERS.keys() <= cls.PARAMETERS.keys():
            raise TypeError(f"{name}: the asymptomatic state needs ASYMPTOMATIC_PARAMETERS")
        if ASYMPTOMATIC not in cls.INFECTIOUS_STATES:
            raise TypeError(f"{name}: the asymptomatic state must be infectious")
        if ASYMPTOMATIC in symptomatic or ASYMPTOMATIC in cls.CONFINED_STATES:
            raise TypeError(f"{name}: the asymptomatic state cannot be symptomatic or confined")
        if ASYMPTOMATIC in cls.STATE_INFECTIOUSNESS:
            raise TypeError(
                f"{name}: the infectiousness of the asymptomatic state follows from "
                "asymptomatic_infectiousness"
            )
        if ASYMPTOMATIC in (cls.INFECTION_ENTRY_STATE, cls.INITIAL_CASE_STATE):
            raise TypeError(f"{name}: the asymptomatic state is reached by a branch only")
        phase = cls.PHASES.get(ASYMPTOMATIC)
        if (
            phase is None
            or phase.fatal
            or phase.duration_parameter != "asymptomatic_days"
            or phase.branch_state is not None
        ):
            raise TypeError(
                f"{name}: the asymptomatic state needs a phase of asymptomatic_days that does "
                "not kill and has no branch"
            )
        ways = [
            (other.branch_parameter, other.branch_age_effect)
            for other in cls.PHASES.values()
            if other.branch_state == ASYMPTOMATIC
        ]
        if cls.INFECTION_BRANCH is not None and cls.INFECTION_BRANCH.state == ASYMPTOMATIC:
            ways.append((cls.INFECTION_BRANCH.parameter, cls.INFECTION_BRANCH.age_effect))
        if any(other.next_state == ASYMPTOMATIC for other in cls.PHASES.values()) or ways != [
            ("asymptomatic_share", "asymptomatic")
        ]:
            raise TypeError(
                f"{name}: the asymptomatic state is reached by one branch only, with the share "
                "asymptomatic_share and the age effect asymptomatic"
            )

    def __init__(self, step_days: float, age_effects: Mapping | None = None, **parameters: float):
        self.step_days = step_days
        self.parameters = resolve_parameters(self.PARAMETERS, parameters)
        self.age_effects = {name: _age_table(name, age_effects) for name in AGE_EFFECTS}
        self._require_effective_age_effects()
        if sum(self.parameters[parameter] for parameter in self.STARTING_SHARES.values()) > 1.0:
            raise ValueError("Starting shares add up to more than everybody")
        if self.parameters["near_distance"] > self.parameters["contact_distance"]:
            raise ValueError("The full-exposure distance cannot exceed the contact distance")
        self._index = {state: index for index, state in enumerate(self.STATE_COLORS)}
        self._symptomatic = (
            self.INFECTIOUS_STATES if self.SYMPTOMATIC_STATES is None else self.SYMPTOMATIC_STATES
        )
        self._symptomatic_indices = [self._index[state] for state in self._symptomatic]
        # for each state: the chance of staying at home and of skipping outings (0 outside the
        # symptomatic states) and the share in hospital
        self._home_chance = np.zeros(len(self._index))
        self._outings_chance = np.zeros(len(self._index))
        for state in self._symptomatic:
            home, outings = self.BEHAVIOUR.get(state, ("stay_home", "limit_outings"))
            self._home_chance[self._index[state]] = self.parameters[home]
            self._outings_chance[self._index[state]] = self.parameters[outings]
        self._hospital_share = np.zeros(len(self._index))
        for state, parameter in self.HOSPITAL.items():
            self._hospital_share[self._index[state]] = self.parameters[parameter]
        self._state_weight = np.array(
            [self._relative_infectiousness(state) for state in self.STATE_COLORS], np.float32
        )
        if ASYMPTOMATIC in self._index:
            self._state_weight[self._index[ASYMPTOMATIC]] = self._asymptomatic_weight()
        self._moving = [
            self._index[state]
            for state in self.STATE_COLORS
            if self._is_left_over_time(state) or self._is_open_ended_infection(state)
        ]
        if DYING in self._index:
            self._moving.append(self._index[DYING])

    @classmethod
    def initial_state(cls) -> str:
        return next(iter(cls.STATE_COLORS))

    @classmethod
    def course_states(cls) -> frozenset[str]:
        """The states an infection runs through, from the one an infected person enters on."""
        found, pending = set(), [cls.INFECTION_ENTRY_STATE]
        if cls.INFECTION_BRANCH is not None:
            pending.append(cls.INFECTION_BRANCH.state)
        while pending:
            state = pending.pop()
            if state in found or state in (cls.initial_state(), DEAD, DYING):
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
    def near_distance(self) -> float:
        return self.parameters["near_distance"]

    @property
    def dead_index(self) -> int:
        return self._index[DEAD]

    def state_index(self, state: str) -> int:
        """The number of `state` in the per-person state array."""
        return self._index[state]

    def confined_mask(self, disease_state: DiseaseState) -> np.ndarray:
        """Who is in a confined state of the disease (isolated at home)."""
        indices = [self._index[state] for state in self.CONFINED_STATES]
        return np.isin(disease_state.state, indices)

    def state_table(self, states) -> np.ndarray:
        """A flag for each state of the disease: whether it is one of `states` (names)."""
        table = np.zeros(len(self._index), np.bool_)
        table[[self._index[state] for state in states if state in self._index]] = True
        return table

    def admitted(self, disease_state: DiseaseState) -> np.ndarray:
        """Who is admitted to hospital in a hospital state: their draw for the current infection
        is below the share of the state."""
        return disease_state.hospital_draw < self._hospital_share[disease_state.state]

    def away_mask(self, disease_state: DiseaseState) -> np.ndarray:
        """Who is in hospital with the disease (dying of it, or admitted in a hospital state) and
        meets nobody in the town."""
        away = self.admitted(disease_state)
        if DYING in self._index:
            away |= disease_state.state == self._index[DYING]
        return away

    def away_states(self) -> list[str]:
        """The states in which everybody is away from the town: dying, and the hospital states
        that admit everybody."""
        states = [DYING] if DYING in self._index else []
        return states + [
            state for state in self.HOSPITAL if self._hospital_share[self._index[state]] >= 1.0
        ]

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
        asymptomatic = self._age_multiplier("asymptomatic", age)
        # the deaths fall on the infections with symptoms: their risk is the risk per infection
        # divided by the chance of having symptoms, so that the risk per infection is kept
        with_symptoms = self._symptomatic_chance(asymptomatic)
        per_infection = self.parameters.get("fatality", 0.0) * self._age_multiplier("fatality", age)
        fatality_risk = np.divide(
            per_infection,
            with_symptoms,
            out=np.zeros(population),
            where=with_symptoms > 0,
        )
        disease_state = DiseaseState(
            state=np.zeros(population, np.int8),
            timer=np.zeros(population, np.float32),
            ever_infected=np.zeros(population, np.bool_),
            stays_home=np.zeros(population, np.bool_),
            limits_outings=np.zeros(population, np.bool_),
            susceptibility=susceptibility.astype(np.float32),
            infectiousness=infectiousness.astype(np.float32),
            fatality_risk=np.clip(fatality_risk, 0.0, 1.0).astype(np.float32),
            branching=self._age_multiplier("branching", age).astype(np.float32),
            reinfected=np.zeros(population, np.bool_),
            immune_before=np.zeros(population, np.bool_),
            outside=self._age_multiplier("outside_infections", age).astype(np.float32),
            asymptomatic=asymptomatic.astype(np.float32),
            home_scale=self._age_multiplier("stay_home", age).astype(np.float32),
            home_draw=rng.random(population).astype(np.float32),
            outings_draw=rng.random(population).astype(np.float32),
            hospital_draw=rng.random(population).astype(np.float32),
        )
        if self.STARTING_SHARES:
            starting = self._age_multiplier("starting_share", age)
            draw = rng.random(population)
            draw[initial_cases] = np.inf  # the first cases are in their own state
            upper = np.zeros(population)
            for state, parameter in self.STARTING_SHARES.items():
                lower, upper = upper, upper + np.minimum(self.parameters[parameter] * starting, 1.0)
                self._enter(
                    disease_state, np.flatnonzero((draw >= lower) & (draw < upper)), state, rng
                )
        if RECOVERED in self._index:
            disease_state.immune_before[disease_state.state == self._index[RECOVERED]] = True
        self.seed_infections(disease_state, initial_cases, rng)
        return disease_state

    def seed_infections(self, disease_state: DiseaseState, people, rng) -> None:
        """Infect `people` as first cases: they start in INITIAL_CASE_STATE (with symptoms, as
        the cases that are noticed) with the traits of a new infection."""
        people = np.asarray(people, np.int64)
        self._draw_infection_traits(disease_state, people, rng)
        self._enter(disease_state, people, self.INITIAL_CASE_STATE, rng)
        disease_state.ever_infected[people] = True
        disease_state.immune_before[people] = True
        disease_state.cumulative_infections += len(people)

    def _draw_infection_traits(self, disease_state: DiseaseState, people: np.ndarray, rng) -> None:
        """What belongs to an infection, not to the person: how much it spreads (the virus they
        shed, a new gamma draw) and how the person reacts to this illness (new behaviour draws)."""
        dispersion = self.parameters["infectiousness_dispersion"]
        if dispersion > 0:
            disease_state.infectiousness[people] = rng.gamma(
                dispersion, 1.0 / dispersion, len(people)
            )
        for draws in (
            disease_state.home_draw,
            disease_state.outings_draw,
            disease_state.hospital_draw,
        ):
            draws[people] = rng.random(len(people))

    def infectious_weight(self, disease_state: DiseaseState) -> np.ndarray:
        """How much each person spreads in the town because of their state: 0 if not infectious
        (or in hospital), 1 if fully."""
        return self._state_weight[disease_state.state] * ~self.admitted(disease_state)

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

    def draw_infections(
        self, disease_state: DiseaseState, exposure: np.ndarray, rng, relative=1.0
    ) -> np.ndarray:
        """Who catches the disease this step, from the states at its start. `relative` is how
        much more easily each person catches it than a person without interactions (`exposure`
        already holds it; the infections from outside are weighted by it here)."""
        weight = np.broadcast_to(
            np.asarray(relative, np.float64)
            * disease_state.susceptibility
            * self.outside_weight(disease_state),
            exposure.shape,
        )
        exposure = exposure * disease_state.susceptibility
        infected = self._new_infections(disease_state, exposure, weight, 0, 1.0, rng)
        for state, parameter in self.PROTECTION.items():
            protection = self.parameters[parameter]
            if protection < 1.0:
                reinfected = self._new_infections(
                    disease_state, exposure, weight, self._index[state], 1.0 - protection, rng
                )
                infected = np.concatenate([infected, reinfected])
        return infected

    @staticmethod
    def outside_weight(disease_state: DiseaseState) -> np.ndarray | float:
        """How often each person is infected from outside relative to the parameter (the
        "outside_infections" age effect)."""
        return 1.0 if disease_state.outside is None else disease_state.outside

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
        dying = np.zeros(0, np.int64)
        if DYING in self._index:
            members = np.flatnonzero(disease_state.state == self._index[DYING])
            disease_state.timer[members] -= step_days
            dying = members[disease_state.timer[members] <= 0]
        self._die(disease_state, dying, rng)
        for state, people in finished.items():
            phase = self.PHASES[state]
            if phase.fatal:
                protection = self.parameters["severe_protection"]
                risk = disease_state.fatality_risk[people] * np.where(
                    disease_state.reinfected[people], 1.0 - protection, 1.0
                )
                dies = rng.random(len(people)) < risk
                if self.parameters["death_delay_days"] > 0:
                    self._enter(disease_state, people[dies], DYING, rng)
                else:
                    self._die(disease_state, people[dies], rng)
                people = people[~dies]
            if phase.branch_state is not None:
                branched = self._branched(
                    disease_state, people, phase.branch_parameter, phase.branch_age_effect, rng
                )
                self._enter(disease_state, people[branched], phase.branch_state, rng)
                people = people[~branched]
            self._enter(disease_state, people, phase.next_state, rng)
        for state, people in moved.items():
            self._enter(disease_state, people, self.RATES[state].next_state, rng)
        disease_state.steps += 1

    def _branched(
        self, disease_state: DiseaseState, people: np.ndarray, parameter: str, effect: str, rng
    ) -> np.ndarray:
        """Which of `people` take a branch whose share is `parameter`, scaled by the age effect
        `effect` and capped at 1 (no random numbers are drawn for a share of 0)."""
        share = self.parameters[parameter]
        if share <= 0:
            return np.zeros(len(people), np.bool_)
        scale = disease_state.asymptomatic if effect == "asymptomatic" else disease_state.branching
        chance = np.minimum(share * scale[people], 1.0)
        return rng.random(len(people)) < chance

    def _symptomatic_chance(self, asymptomatic_scale) -> np.ndarray | float:
        """The chance that an infection has symptoms, for the "asymptomatic" age effect given."""
        if ASYMPTOMATIC not in self.STATE_COLORS:
            return np.ones_like(asymptomatic_scale, dtype=np.float64)
        share = self.parameters["asymptomatic_share"]
        return 1.0 - np.minimum(share * np.asarray(asymptomatic_scale, np.float64), 1.0)

    def _asymptomatic_weight(self) -> float:
        """The infectiousness per day of the state without symptoms: what an infection with
        symptoms spreads in all (its states' days weighted by their infectiousness), times the
        relative infectiousness of an infection without symptoms, spread over its days."""
        with_symptoms = sum(
            course.probability * sum(days * weight for _, days, weight in course.stays if weight)
            for course in self.courses(asymptomatic=False)
        )
        if math.isinf(with_symptoms):
            return self.parameters["asymptomatic_infectiousness"]
        return (
            self.parameters["asymptomatic_infectiousness"]
            * with_symptoms
            / self.parameters["asymptomatic_days"]
        )

    def _die(self, disease_state: DiseaseState, people: np.ndarray, rng) -> None:
        self._enter(disease_state, people, DEAD, rng)
        disease_state.deaths += len(people)

    def apply_infections(self, disease_state: DiseaseState, infected: np.ndarray, rng) -> None:
        """Let the infected enter the disease, except those who died in the meantime."""
        infected = infected[disease_state.state[infected] != self.dead_index]
        # how much a person spreads and how they behave are properties of an infection, drawn
        # anew for every infection (before the entry, which applies the behaviour of the state)
        self._draw_infection_traits(disease_state, infected, rng)
        entry = self.INFECTION_BRANCH
        if entry is None:
            self._enter(disease_state, infected, self.INFECTION_ENTRY_STATE, rng)
        else:
            branched = self._branched(
                disease_state, infected, entry.parameter, entry.age_effect, rng
            )
            self._enter(disease_state, infected[branched], entry.state, rng)
            self._enter(disease_state, infected[~branched], self.INFECTION_ENTRY_STATE, rng)
        disease_state.reinfected[infected] = disease_state.immune_before[infected]
        disease_state.ever_infected[infected] = True
        disease_state.immune_before[infected] = True
        disease_state.cumulative_infections += len(infected)

    def bury(self, disease_state: DiseaseState, people: np.ndarray) -> None:
        """Mark `people` who died of another disease as dead for this one."""
        disease_state.state[people] = self._index[DEAD]
        disease_state.stays_home[people] = False
        disease_state.limits_outings[people] = False

    def courses(self, asymptomatic: bool = True) -> list[Course]:
        """Every way an infection can run, from the state an infected person enters.

        The chances are those of a person whose age effects are 1; a course that cannot happen
        is left out. Time spent in a state is its mean length, so the numbers derived from the
        courses are expectations over many people. With `asymptomatic` False only the courses
        with symptoms, as if every infection had them.
        """
        found = []
        with_symptoms = float(self._symptomatic_chance(1.0))
        died_with_symptoms = 0.0
        if "fatality" in self.parameters and with_symptoms > 0:
            died_with_symptoms = min(self.parameters["fatality"] / with_symptoms, 1.0)

        def branch_share(state: str, parameter: str) -> float:
            if state == ASYMPTOMATIC and not asymptomatic:
                return 0.0
            return min(self.parameters[parameter], 1.0)

        def walk(state: str, probability: float, stays: tuple) -> None:
            if state in (self.initial_state(), DEAD):
                found.append(Course(probability, stays, state))
                return
            phase = self.PHASES.get(state)
            mean = math.inf if phase is None else self.parameters[phase.duration_parameter]
            if mean >= LIFELONG_DAYS:
                mean = math.inf
            weight = self._state_weight[self._index[state]] * (
                1.0 - self._hospital_share[self._index[state]]
            )
            stays = (*stays, (state, mean, float(weight)))
            if math.isinf(mean):
                found.append(Course(probability, stays, state))
                return
            survive = 1.0
            if phase.fatal:
                died = died_with_symptoms
                if died > 0:
                    delay = self.parameters["death_delay_days"]
                    found.append(Course(probability * died, (*stays, (DYING, delay, 0.0)), DEAD))
                survive = 1.0 - died
            branch = 0.0
            if phase.branch_state is not None:
                branch = branch_share(phase.branch_state, phase.branch_parameter)
                if branch > 0:
                    walk(phase.branch_state, probability * survive * branch, stays)
            if survive * (1.0 - branch) > 0:
                walk(phase.next_state, probability * survive * (1.0 - branch), stays)

        entry = self.INFECTION_BRANCH
        branch = 0.0 if entry is None else branch_share(entry.state, entry.parameter)
        if branch > 0:
            walk(entry.state, branch, ())
        if branch < 1.0:
            walk(self.INFECTION_ENTRY_STATE, 1.0 - branch, ())
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

        Exact for gamma distributed phases: 1 - E[exp(-hazard * X * infectiousness-weighted
        time)], the expectation over the ways the infection can run and over the infectiousness X
        of the infection (gamma with the shape `infectiousness_dispersion`, mean 1; 1 if that is
        0). The expectation over X is taken over a fixed large sample of it.
        """
        transmission = self.parameters["transmission"]
        if transmission >= 1.0:
            return 1.0
        hazard = -math.log(1.0 - transmission)
        if hazard == 0.0:
            return 0.0
        shape = self.parameters["duration_shape"]
        dispersion = self.parameters["infectiousness_dispersion"]
        scale = (
            np.random.default_rng(0).gamma(dispersion, 1.0 / dispersion, INFECTIOUSNESS_SAMPLE)
            if dispersion > 0
            else np.ones(1)
        )
        survival = 0.0
        for course in self.courses():
            kept = np.full(len(scale), course.probability)
            for _, mean, weight in course.stays:
                if weight > 0:
                    kept *= (
                        0.0
                        if math.isinf(mean)
                        else (1.0 + hazard * scale * weight * mean / shape) ** -shape
                    )
            survival += float(kept.mean())
        return min(1.0, max(0.0, 1.0 - survival))

    def is_quiescent(self, disease_state: DiseaseState) -> bool:
        """True when nobody is in a state that can still change, so nothing changes any more (never
        while infections come from outside)."""
        if self.parameters["outside_infections"] > 0:
            return False
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
        branch_effects = {
            phase.branch_age_effect
            for phase in self.PHASES.values()
            if phase.branch_state is not None
        }
        if self.INFECTION_BRANCH is not None:
            branch_effects.add(self.INFECTION_BRANCH.age_effect)
        used = {
            "fatality": any(phase.fatal for phase in self.PHASES.values()),
            "branching": "branching" in branch_effects,
            "starting_share": bool(self.STARTING_SHARES),
            "asymptomatic": ASYMPTOMATIC in self.STATE_COLORS,
        }
        for effect, applies in used.items():
            if self.age_effects[effect] and not applies:
                missing = {
                    "fatality": "fatal phase",
                    "branching": "branch",
                    "starting_share": "starting share",
                    "asymptomatic": "course without symptoms",
                }[effect]
                raise ValueError(
                    f"Age effect '{effect}' does nothing in {type(self).__name__}: "
                    f"it has no {missing}"
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

    def _new_infections(
        self, disease_state, exposure, weight, state_index, scale, rng
    ) -> np.ndarray:
        """People in `state_index` who catch the disease this step; `scale` weakens the exposure.

        The hazard is that of the infectious contacts, -ln(1 - transmission) per day of exposure,
        plus that of infections from outside the town, weighted by `weight`, the person's
        susceptibility (which `exposure` already holds).
        """
        outside = self.parameters["outside_infections"] / DAYS_PER_YEAR
        if outside > 0:
            candidates = np.flatnonzero(disease_state.state == state_index)
        else:
            candidates = np.flatnonzero((disease_state.state == state_index) & (exposure > 0))
        transmission = self.parameters["transmission"]
        contact = -math.log(1.0 - transmission) if transmission < 1.0 else math.inf
        exposed = exposure[candidates]
        hazard = np.where(exposed > 0, contact * exposed, 0.0) + outside * weight[candidates]
        chance = 1.0 - np.exp(-hazard * scale * self.step_days)
        return candidates[rng.random(len(candidates)) < chance]

    def _age_multiplier(self, effect: str, age: np.ndarray) -> np.ndarray:
        multiplier = np.ones(len(age))
        for start, end, value in self.age_effects[effect]:
            multiplier[(age >= start) & (age < end)] = value
        return multiplier

    def _enter(self, disease_state: DiseaseState, people: np.ndarray, state: str, rng) -> None:
        """Put `people` into `state`. How they behave there and whether they are in hospital
        follows from the chances of the state and the person's draws for the current infection:
        a person stays at home in every state whose chance (times their age effect) exceeds
        their draw, so the decisions in the states of one illness are consistent."""
        index = self._index[state]
        disease_state.state[people] = index
        home = np.minimum(self._home_chance[index] * disease_state.home_scale[people], 1.0)
        disease_state.stays_home[people] = (state in self.CONFINED_STATES) | (
            disease_state.home_draw[people] < home
        )
        disease_state.limits_outings[people] = (
            disease_state.outings_draw[people] < self._outings_chance[index]
        )
        # an admitted person does not go out either: the children who were to go with them stay
        disease_state.stays_home[people] |= (
            disease_state.hospital_draw[people] < self._hospital_share[index]
        )
        phase = self.PHASES.get(state)
        if phase is not None or state == DYING:
            shape = self.parameters["duration_shape"]
            name = "death_delay_days" if state == DYING else phase.duration_parameter
            mean = self.parameters[name]
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
