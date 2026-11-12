# Reference data

Hard data the simulated population is validated against (`python -m epidemic_sandbox.analysis.validate`).

| File | Content | Source |
|------|---------|--------|
| `czechia_population_2019.csv` | Population of Czechia on 1 January 2019 by single year of age and sex, everybody from 99 counted as 99 | Eurostat, demo_pjan |
| `czechia_contacts_prem2021.csv` | Mean number of contacts per day by age group of the person and of the contact, in households, schools, workplaces and other places | Prem K, van Zandvoort K, Klepac P, et al. (2021), *Projecting contact matrices in 177 geographical regions: an update and comparison with empirical data for the COVID-19 era*, PLoS Comput Biol 17(7): e1009098. Data: https://github.com/kieshaprem/synthetic-contact-matrices (`synthetic_contacts_2021.csv`, setting "overall", country CZE) |
| `polymod_contacts_de_pl.csv` | Mean number of contacts per day, and its standard error, by age band and place, in Germany and Poland (the two neighbours of Czechia in the survey), weekdays and weekend days weighted 5:2 | Mossong J, Hens N, Jit M, et al. (2008), *Social contacts and mixing patterns relevant to the spread of infectious diseases*, PLoS Med 5(3): e74. Data: Zenodo, doi:10.5281/zenodo.3874557 (CC BY 4.0); derived by `derivations/polymod_contacts.py` |
| `polymod_durations_de_pl.csv` | Mean number of contacts per day that lasted 1 hour or more and more than 4 hours, and the share of the contacts of 1 hour or more that are met daily, by age band and place, in Germany and Poland | Mossong et al. (2008), the same data; made by `derivations/polymod_durations.py` |
| `czechia_reference.toml` | Published aggregates: age, households, living alone, families, births, employment, young people, care homes, local units (workplaces), the shares of long contacts at home (`household_contacts`), the hours a day spent with the household (`time_together`) and the share of people at home at 10 a.m. of a workday (`whereabouts`) | Czech Statistical Office, census 2021 (households) and Eurostat, mostly 2019 (a link and the year in each section) |

The scripts in `derivations/` make the POLYMOD files from the survey data (Zenodo); the shares in `[household_contacts]` (62.9 %, 21.1 %, 2,894 contacts) were read from the same data by hand and have no script yet.

The contact matrices are synthetic: they were projected from contact surveys of other countries onto
Czech demography, so they are the best available estimate for Czechia, not a measurement.

The reference year is 2019, the last before the pandemic, so that the town describes ordinary life. The household structure is from the 2021 census, which has the details; the household series of the labour force survey show no break around 2020-21.

The contact matrices of Prem et al. are used for who meets whom by age (their structure); the levels, how many people a person meets in a day, are checked against the measured POLYMOD survey, whose standard errors are known.
