"""Kernels that look at every person once where NumPy needed a dozen passes over the arrays.

A step asks of every person where they are (which venue), whether an illness keeps them from their
day, and so on. Written with array expressions each answer is a chain of temporaries that travel
through the memory again and again; here one pass over the people reads what it needs and writes
the answer. The results are the same as the expressions give, number for number (the tests keep the
expressions as the reference).

Starting the threads takes 50 to 90 microseconds a kernel, measured, which is more than a pass
over 200,000 people takes when it only compares (4 microseconds). The passes here read a dozen
arrays for each person, a millisecond for 200,000 people on one thread, and are worth the threads
from 20,000 people; the few passes that do almost nothing stay on one thread.
"""

import numpy as np
from numba import njit, prange

from epidemic_sandbox.engine.spatial_kernels import sized_kernel

PARALLEL_FROM_PEOPLE = 20_000  # a pass over the people with this much work for each of them


@sized_kernel(from_people=PARALLEL_FROM_PEOPLE)
def venues(
    code,
    choice,
    household,
    work_venue,
    lunch_venue,
    activity_venue,
    in_common,
    school_venue,
    home,
    work,
    resident,
    room_venue,
    ward_venue,
    bedridden,
    night,
    inside,
    away,
    first_activity,
    work_code,
    lunch_code,
):
    """The venue everybody is in (see `town.Venues`), -1 for those on the way or away.

    A person is at the place of their `code` (the workplace, the lunch place, the place of an
    activity, else at home); the people in the common area of their school or workplace are
    there; whoever goes home or back to work is there. The residents of a care home who are at
    home are in the day room of their ward, at `night` and when they are bedridden in their room.
    """
    count = code.shape[0]
    venue = np.empty(count, np.int64)
    for person in prange(count):
        place = code[person]
        here = work_venue[person] if place == work_code else np.int64(household[person])
        if place == lunch_code:
            here = lunch_venue[person]
        for index in range(len(activity_venue)):
            if place == first_activity + index:
                here = activity_venue[index, person, choice[person, index]]
        if in_common[person] and place == work_code and not home[person] and not work[person]:
            here = school_venue[person]
        if home[person]:
            here = np.int64(household[person])
        if work[person]:
            here = work_venue[person]
        if resident[person] and here == household[person]:
            here = room_venue[person] if night or bedridden[person] else ward_venue[person]
        if not inside[person] or away[person]:
            here = -1
        venue[person] = here
    return venue


@sized_kernel(from_people=PARALLEL_FROM_PEOPLE)
def stay_flags(states, stays_home, limits_outings, away_table, confined_table, household, unwell):
    """For every person, from the diseases (tuples with one array each): whether they are away in
    hospital or stay at home because of an illness (`stays`), and whether an illness makes them
    skip outings (`skips`). The tables say, for each disease and state, whether the state is being
    away or being confined; `unwell` is marked for the households with somebody confined (many
    people may write the same value to a household, and nothing else is written there)."""
    count = states[0].shape[0]
    stays = np.zeros(count, np.bool_)
    skips = np.zeros(count, np.bool_)
    for person in prange(count):
        stay, skip = False, False
        for disease in range(len(states)):
            state = states[disease][person]
            stay = stay or away_table[disease, state] or stays_home[disease][person]
            skip = skip or limits_outings[disease][person]
            if confined_table[disease, state]:
                unwell[household[person]] = True
        stays[person], skips[person] = stay, skip
    return stays, skips


@sized_kernel(from_people=PARALLEL_FROM_PEOPLE)
def go_home_or_back_to_work(
    code,
    choice,
    stays,
    skips,
    unwell,
    activity_venue,
    visits,
    guardian,
    first_activity,
    lunch_code,
):
    """Who does not go where their day takes them, because of an illness: (home, work).

    Home: those who stay in and those who are visiting a household with somebody isolated, those
    who skip outings and are on one, and the small children who were to go out with them. Work:
    those who skip outings and would go out for the lunch break; they stay at their workplace.
    """
    count = code.shape[0]
    home = np.empty(count, np.bool_)
    work = np.empty(count, np.bool_)
    for person in prange(count):
        goes = False
        # the person, then (for a child who is out) the guardian, whose day the child follows
        whose = 2 if guardian[person] >= 0 and code[person] >= first_activity else 1
        for turn in range(whose):
            who = np.int64(person) if turn == 0 else guardian[person]
            here = stays[who]
            if code[who] >= first_activity:
                for position in range(len(visits)):
                    activity = visits[position]
                    if code[who] == first_activity + activity:
                        host = activity_venue[activity, who, choice[who, activity]]
                        here = here or unwell[host]
                here = here or skips[who]
            goes = goes or here
        home[person] = goes
        work[person] = skips[person] and code[person] == lunch_code and not goes
    return home, work


@njit(cache=True, nogil=True)
def due_events(times, pointer, until):
    """Who has an event in their schedule due at the time `until` (hours of the day): the people
    whose next event, the one `pointer` points to, is not later."""
    due = np.empty(times.shape[0], np.int64)
    found = 0
    for person in range(times.shape[0]):
        if times[person, pointer[person]] <= until:
            due[found] = person
            found += 1
    return due[:found].copy()


@njit(cache=True, nogil=True)
def settled_movers(
    x,
    y,
    target_x,
    target_y,
    code,
    detour,
    guardian,
    resident,
    bedridden,
    up,
    awake,
    home_code,
    work_code,
):
    """Who has settled at their spot and may move to another one: those who are away from home
    where their day takes them and are on their own (not small children with a guardian, except
    at school); at home everybody while `awake`, the residents of a care home while they are
    `up` and not bedridden."""
    chosen = np.empty(x.shape[0], np.int64)
    found = 0
    for person in range(x.shape[0]):
        if x[person] != target_x[person] or y[person] != target_y[person]:
            continue
        if detour[person]:
            continue
        if code[person] != home_code:
            moves = guardian[person] < 0 or code[person] == work_code
        elif resident[person]:
            moves = up and not bedridden[person]
        else:
            moves = awake
        if moves:
            chosen[found] = person
            found += 1
    return chosen[:found].copy()


@sized_kernel(from_people=PARALLEL_FROM_PEOPLE)
def sort_events(times, codes, travel):
    """The events of every person (a row of `times`, of `codes` and of `travel`) in the order of
    their times; events at the same time stay in the order they had. What
    `np.argsort(kind="stable")` and `np.take_along_axis` give, without the sort of the whole table
    and the copies of it."""
    ordered_times, ordered_codes, ordered_travel = times.copy(), codes.copy(), travel.copy()
    for person in prange(times.shape[0]):
        for position in range(1, times.shape[1]):
            time, code = ordered_times[person, position], ordered_codes[person, position]
            trip = ordered_travel[person, position]
            before = position - 1
            while before >= 0 and ordered_times[person, before] > time:
                ordered_times[person, before + 1] = ordered_times[person, before]
                ordered_codes[person, before + 1] = ordered_codes[person, before]
                ordered_travel[person, before + 1] = ordered_travel[person, before]
                before -= 1
            ordered_times[person, before + 1] = time
            ordered_codes[person, before + 1] = code
            ordered_travel[person, before + 1] = trip
    return ordered_times, ordered_codes, ordered_travel
