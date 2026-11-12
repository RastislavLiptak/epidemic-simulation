"""Derive polymod_durations_de_pl.csv from the POLYMOD contact survey: contacts by place and by
how long they lasted, and how many of them are met every day.

Mossong J, Hens N, Jit M, et al. (2008), Social contacts and mixing patterns relevant to the spread
of infectious diseases, PLoS Med 5(3): e74. Data: Zenodo, doi:10.5281/zenodo.3874557 (CC BY 4.0),
files hh_common.csv, participant_common.csv, sday.csv, contact_common.csv.

Germany and Poland, as in polymod_contacts.py (the place of a contact is the first of home, work,
school, transport, leisure, otherplace that is marked; the last three are "other"). For every age
band and place: the mean number of contacts per participant and day that lasted 1 hour or more
("1-4 hours" and "more than 4 hours" of the survey) and more than 4 hours, and the share of the
contacts of 1 hour or more that are met daily or almost daily (weekdays and weekend days weighted
5:2 for the means), with the standard errors of the means (the two kinds of days as independent
samples) and of the shares.

    python polymod_durations.py <directory with the POLYMOD files> > ../polymod_durations_de_pl.csv
"""

import csv
import sys
from collections import defaultdict

import numpy as np

COUNTRIES = {"DE", "PL"}
PLACES = ("home", "work", "school", "transport", "leisure", "otherplace")
OTHER = ("transport", "leisure", "otherplace")
KEYS = ("home", "school", "work", "other")
BANDS = (
    (0, 5),
    (5, 10),
    (10, 15),
    (15, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 120),
)
ONE_HOUR, FOUR_HOURS = ("4", "5"), ("5",)  # duration_multi: 4 is 1-4 hours, 5 more than 4 hours
DAILY = "1"  # frequency_multi: daily or almost daily


def rows(directory: str, name: str) -> list[dict]:
    with open(f"{directory}/{name}", newline="") as table:
        return list(csv.DictReader(table))


def weighted(values: list, weekend: list) -> tuple[float, float]:
    """The 5:2 weighted mean of weekday and weekend participants and its standard error."""
    data, days = np.array(values, float), np.array(weekend, bool)
    weekdays, weekends = data[~days], data[days]
    mean = (5 * weekdays.mean() + 2 * weekends.mean()) / 7
    error = np.sqrt(
        (5 / 7) ** 2 * weekdays.var(ddof=1) / len(weekdays)
        + (2 / 7) ** 2 * weekends.var(ddof=1) / len(weekends)
    )
    return float(mean), float(error)


def main(directory: str) -> None:
    households = {row["hh_id"]: row for row in rows(directory, "hh_common.csv")}
    participants = {}
    for row in rows(directory, "participant_common.csv"):
        country = households[row["hh_id"]]["country"]
        if row["part_age"] not in ("NA", "") and country in COUNTRIES:
            participants[row["part_id"]] = int(float(row["part_age"]))
    weekend = {
        row["part_id"]: row["dayofweek"] in ("0", "6") for row in rows(directory, "sday.csv")
    }
    counts = defaultdict(lambda: defaultdict(float))
    for row in rows(directory, "contact_common.csv"):
        person = row["part_id"]
        if person not in participants:
            continue
        place = next((place for place in PLACES if row[f"cnt_{place}"] == "TRUE"), None)
        if place is None:
            continue
        key = "other" if place in OTHER else place
        counts[person][f"{key}_all"] += 1
        counts[person][f"{key}_1h"] += row["duration_multi"] in ONE_HOUR
        counts[person][f"{key}_4h"] += row["duration_multi"] in FOUR_HOURS
        counts[person][f"{key}_daily"] += (
            row["frequency_multi"] == DAILY and row["duration_multi"] in ONE_HOUR
        )
    columns = [f"{key}_{kind}" for key in KEYS for kind in ("1h", "4h", "daily_share")]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ["age_from", "age_to", "participants", *columns, *(f"{column}_se" for column in columns)]
    )
    for low, high in BANDS:
        people = [person for person, age in participants.items() if low <= age < high]
        days = [weekend.get(person, False) for person in people]
        means, errors = [], []
        for key in KEYS:
            for kind in ("1h", "4h"):
                mean, error = weighted([counts[p][f"{key}_{kind}"] for p in people], days)
                means.append(mean)
                errors.append(error)
            total = sum(counts[p][f"{key}_1h"] for p in people)
            daily = sum(counts[p][f"{key}_daily"] for p in people)
            share = daily / total if total else float("nan")
            means.append(share)
            errors.append(np.sqrt(share * (1 - share) / total) if total else float("nan"))
        writer.writerow(
            [low, high, len(people), *(f"{v:.3f}" for v in means), *(f"{v:.3f}" for v in errors)]
        )


if __name__ == "__main__":
    main(sys.argv[1])
