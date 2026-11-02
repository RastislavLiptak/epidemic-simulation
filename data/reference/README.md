# Reference data

Hard data the simulated population is validated against (`python -m epidemic_sandbox.validate`).

| File | Content | Source |
|------|---------|--------|
| `czechia_population_2019.csv` | Population of Czechia on 1 January 2019 by single year of age and sex, everybody from 99 counted as 99 | Eurostat, demo_pjan |
| `czechia_contacts_prem2021.csv` | Mean number of contacts per day by age group of the person and of the contact, in households, schools, workplaces and other places | Prem K, van Zandvoort K, Klepac P, et al. (2021), *Projecting contact matrices in 177 geographical regions: an update and comparison with empirical data for the COVID-19 era*, PLoS Comput Biol 17(7): e1009098. Data: https://github.com/kieshaprem/synthetic-contact-matrices (`synthetic_contacts_2021.csv`, setting "overall", country CZE) |
| `czechia_reference.toml` | Published aggregates: age, households, living alone, families, births, employment, young people, care homes, enterprises | Czech Statistical Office, census 2021 (households) and Eurostat, mostly 2019 (a link and the year in each section) |

The contact matrices are synthetic: they were projected from contact surveys of other countries onto
Czech demography, so they are the best available estimate for Czechia, not a measurement.

The reference year is 2019, the last before the pandemic, so that the town describes ordinary life. The household structure is from the 2021 census, which has the details; the household series of the labour force survey show no break around 2020-21.
