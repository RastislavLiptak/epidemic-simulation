"""Derive polymod_contacts_de_pl.csv from the POLYMOD contact survey.

Mossong J, Hens N, Jit M, et al. (2008), Social contacts and mixing patterns relevant to the spread
of infectious diseases, PLoS Med 5(3): e74. Data: Zenodo, doi:10.5281/zenodo.3874557 (CC BY 4.0),
files hh_common.csv, participant_common.csv, sday.csv, contact_common.csv.

Germany and Poland, the two neighbours of Czechia in the survey. Mean number of contacts per
participant and day by the participant's age band and the place of the contact (the first of
home, work, school, transport, leisure, otherplace that is marked; transport, leisure and
otherplace together are "other"), weekdays and weekend days weighted 5:2 (dayofweek 0 is Sunday,
6 Saturday), and the standard errors of these means (the two kinds of days as independent
samples).

    python polymod_contacts.py <directory with the POLYMOD files> > ../polymod_contacts_de_pl.csv
"""

import csv
import sys
from collections import defaultdict

import numpy as np

COUNTRIES = {"DE", "PL"}
PLACES = ("home", "work", "school", "transport", "leisure", "otherplace")
OTHER = ("transport", "leisure", "otherplace")
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


def rows(directory: str, name: str) -> list[dict]:
    with open(f"{directory}/{name}", newline="") as table:
        return list(csv.DictReader(table))


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
        counts[person]["all"] += 1
        if place is not None:
            counts[person]["other" if place in OTHER else place] += 1
    keys = ("all", "home", "school", "work", "other")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["age_from", "age_to", "participants", *keys, *(f"{key}_se" for key in keys)])
    for low, high in BANDS:
        days = ([], [])  # the contacts of every participant of the band, weekdays and weekends
        for person, age in participants.items():
            if low <= age < high:
                days[int(weekend.get(person, False))].append([counts[person][key] for key in keys])
        weekdays, weekends = (np.array(values, float).reshape(-1, len(keys)) for values in days)
        means = (5 * weekdays.mean(axis=0) + 2 * weekends.mean(axis=0)) / 7
        # standard error of the weighted mean of the two independent samples
        errors = np.sqrt(
            (5 / 7) ** 2 * weekdays.var(axis=0, ddof=1) / len(weekdays)
            + (2 / 7) ** 2 * weekends.var(axis=0, ddof=1) / len(weekends)
        )
        writer.writerow(
            [
                low,
                high,
                len(weekdays) + len(weekends),
                *(f"{mean:.2f}" for mean in means),
                *(f"{error:.2f}" for error in errors),
            ]
        )


if __name__ == "__main__":
    main(sys.argv[1])
