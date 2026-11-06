"""What one person is, what they are doing right now and why: the data behind the hover card."""

import numpy as np

from epidemic_sandbox.engine.epidemic_model import EpidemicModel
from epidemic_sandbox.world.population import RESIDENT, ROLES, SEXES
from epidemic_sandbox.world.schedule import FIRST_ACTIVITY, HOME, LUNCH, WORK

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
ARRIVED_WITHIN_METERS = 1.0
MAX_NEIGHBOURS = 8
PLACE_OF_ROLE = {
    "preschool": "kindergarten",
    "school": "school",
    "university": "university",
    "worker": "workplace",
}
ROLE_LABELS = {
    "homebound": "stays at home (infant, unemployed or on leave)",
    "preschool": "goes to kindergarten",
    "school": "goes to school",
    "university": "studies at a university",
    "worker": "works at a workplace",
    "remote": "works from home",
    "retired": "retired",
    "resident": "lives in a care home",
}
ACTIVITY_LABELS = {
    "shopping": "shopping",
    "food": "eating out",
    "walk": "out for a walk",
    "visit": "visiting family or friends",
}


def describe_person(model: EpidemicModel, person: int) -> dict:
    """JSON-ready description of one person; raises ValueError for an unknown person."""
    if (
        isinstance(person, bool)
        or not isinstance(person, int)
        or not 0 <= person < model.population
    ):
        raise ValueError(f"Unknown person {person!r}")
    town = model.town
    role = ROLES[town.role[person]]
    return {
        "id": person,
        "age": int(town.age[person]),
        "sex": SEXES[town.sex[person]],
        "role": role,
        "role_label": ROLE_LABELS[role],
        "clock": _clock(model),
        "household": _household(model, person),
        "place": _place(model, person, role),
        "traits": _traits(model, person),
        "now": _now(model, person, role),
        "timetable": _timetable(model, person, role),
        "health": _health(model, person),
        "nearby": _nearby(model, person),
    }


def _clock(model: EpidemicModel) -> str:
    return f"{WEEKDAYS[model.weekday]} {_hour_label(model.hour)}"


def _hour_label(hours: float) -> str:
    return f"{int(hours) % 24:02d}:{int(hours % 1 * 60):02d}"


def _health_summary(model: EpidemicModel, person: int) -> dict[str, str]:
    return {
        name: list(disease.STATE_COLORS)[model.disease_states[name].state[person]]
        for name, disease in model.diseases.items()
    }


def _household(model: EpidemicModel, person: int) -> dict:
    town = model.town
    first = int(np.searchsorted(town.household, town.household[person], side="left"))
    last = int(np.searchsorted(town.household, town.household[person], side="right"))
    return {
        "id": int(town.household[person]),
        "members": [
            {
                "id": member,
                "age": int(town.age[member]),
                "sex": SEXES[town.sex[member]],
                "role": ROLES[town.role[member]],
                "health": _health_summary(model, member),
            }
            for member in range(first, last)
        ],
    }


def _place(model: EpidemicModel, person: int, role: str) -> dict | None:
    town = model.town
    if role not in PLACE_OF_ROLE:
        return None
    kind = PLACE_OF_ROLE[role]
    if role == "worker" and town.at_school[person]:
        kind = "school (as staff)"
    elif role == "worker" and town.place_kind[person] == RESIDENT:
        kind = "care home (as staff)"
    distance = np.hypot(
        town.work_x[person] - town.home_x[person], town.work_y[person] - town.home_y[person]
    )
    return {
        "kind": kind,
        "people": int(town.work_people[person]),
        "distance_m": float(distance),
    }


def _traits(model: EpidemicModel, person: int) -> dict:
    town = model.town
    return {
        "sociability": float(town.sociability[person]),
        "speed_kmh": float(town.speed_kmh[person]),
        "usual_start_shift_h": float(town.start_habit[person]),
        "usual_end_shift_h": float(town.end_habit[person]),
    }


def _label(model: EpidemicModel, role: str, code: int) -> str:
    if code == HOME:
        return "home"
    if code == WORK:
        return PLACE_OF_ROLE.get(role, "work")
    if code == LUNCH:
        return "lunch break"
    return ACTIVITY_LABELS[model.town_parameters.schedule.activities[code - FIRST_ACTIVITY].name]


def _now(model: EpidemicModel, person: int, role: str) -> dict:
    town = model.town
    x, y = float(model.x[person]), float(model.y[person])
    destination_x, destination_y = model.destination(person)
    ill_at_home = any(state.stays_home[person] for state in model.disease_states.values())
    heading_home = (destination_x, destination_y) == (
        float(town.home_x[person]),
        float(town.home_y[person]),
    )
    code = HOME if heading_home else model.current_code(person)
    label = _label(model, role, code)
    remaining = float(np.hypot(destination_x - x, destination_y - y))
    now_hours = model.steps * model.step_hours
    if remaining > ARRIVED_WITHIN_METERS:
        speed = float(town.speed_kmh[person]) * 1000
        departure = model.departure_time(person)
        if departure > now_hours:
            activity, arrives = f"about to leave for {label}", departure + remaining / speed
        else:
            activity, arrives = f"on the way to {label}", now_hours + remaining / speed
        arrives_at = _hour_label(arrives)
    else:
        activity = "at home" if code == HOME else f"at {label}" if code == WORK else label
        arrives_at = None
    return {
        "activity": activity,
        "arrives": arrives_at,
        "reason": _reason(model, person, role, code, ill_at_home),
    }


def _reason(model: EpidemicModel, person: int, role: str, code: int, ill_at_home: bool) -> str:
    if ill_at_home:
        ill = [name for name, state in model.disease_states.items() if state.stays_home[person]]
        return f"is infectious with {', '.join(ill)} and stays at home"
    schedule = model.schedule
    next_time = model.next_event_time(person)
    upcoming = (
        ""
        if next_time is None
        else f"; next: {_label(model, role, int(schedule.codes[person, model._pointer[person]]))} "
        f"at {_hour_label(next_time)}"
    )
    if code == HOME:
        if schedule.absent[person]:
            place = PLACE_OF_ROLE.get(role, "workplace")
            return f"does not go to the {place} today (leave, illness or home office)"
        return f"nothing planned right now{upcoming}"
    if code == WORK:
        return f"it is their time at the {PLACE_OF_ROLE.get(role, 'workplace')}{upcoming}"
    return f"{_label(model, role, code)} is on today's plan{upcoming}"


def _timetable(model: EpidemicModel, person: int, role: str) -> list[dict]:
    """Today's events of the person with their own times."""
    schedule = model.schedule
    entries = []
    for when, code in zip(schedule.times[person], schedule.codes[person], strict=True):
        if np.isfinite(when):
            entries.append(
                {"at": _hour_label(when), "what": f"to {_label(model, role, int(code))}"}
            )
    if schedule.absent[person]:
        entries.append(
            {"at": None, "what": f"does not go to the {PLACE_OF_ROLE.get(role, 'workplace')} today"}
        )
    return entries


def _health(model: EpidemicModel, person: int) -> dict:
    factors = model.interaction_factors(person)
    health = {}
    for name, disease in model.diseases.items():
        disease_state = model.disease_states[name]
        state = list(disease.STATE_COLORS)[disease_state.state[person]]
        timed = state in disease.PHASES and not disease.lasts_for_life(state)
        health[name] = {
            "model": type(disease).__name__.removesuffix("Model").upper(),
            "state": state,
            "days_left": float(disease_state.timer[person]) if timed else None,
            "infectious": state in disease.INFECTIOUS_STATES,
            "ever_infected": bool(disease_state.ever_infected[person]),
            "susceptibility": float(
                disease_state.susceptibility[person]
                * factors[name]["susceptibility"]
                * (1.0 - disease.protection_of(state))
            ),
            "infectiousness": float(
                disease_state.infectiousness[person]
                * disease.state_infectiousness(state)
                * factors[name]["infectiousness"]
            ),
            "fatality_risk": float(disease_state.fatality_risk[person]),
        }
    return health


def _nearby(model: EpidemicModel, person: int) -> dict:
    """People within the contact distance of the longest-reaching disease, the nearest first."""
    radius = max(disease.contact_distance for disease in model.diseases.values())
    distance = np.hypot(model.x - model.x[person], model.y - model.y[person])
    close = np.flatnonzero(distance <= radius)
    close = close[close != person]
    nearest = close[np.argsort(distance[close])][:MAX_NEIGHBOURS]
    return {
        "count": int(len(close)),
        "people": [
            {
                "id": int(other),
                "age": int(model.town.age[other]),
                "distance_m": float(distance[other]),
                "health": _health_summary(model, int(other)),
            }
            for other in nearest
        ],
    }
