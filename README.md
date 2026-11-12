# Epidemic sandbox

Agent-based model of several interacting diseases spreading in a simulated town, with an animated web UI. Semester project for the course Modelling in Biocybernetics.

Every person is simulated individually: they live in a household, go to school or work, spend their free time and meet others, and infectious people infect those close to them. Several diseases can run at once, each with its own model, parameters and effects on how people behave, and they can influence each other.

## Requirements and installation

Python 3.12 or newer and a modern browser.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -c constraints.txt -e ".[dev]"
```

## Usage

```bash
python -m epidemic_sandbox         # then open http://localhost:8001
python -m epidemic_sandbox --help  # port, host, config files and worker processes
```

In the UI you set up an experiment (town, diseases, number of runs), watch the runs and hover over a person to see who they are, and look at the results and export them once the experiment is finished.

An experiment belongs to the server, not to the open page: it goes on when the screen is locked or the page is closed, and the page (opened again in the same browser) shows it where it has got to. A run that nobody watches is computed as fast as the machine can, and one without a maximum duration is paused after a while (`[session]` in `configs/app.toml`). The runs of an experiment are computed by one worker process per CPU core, so that several runs take little longer than one; `--workers 0` computes them in the server itself. The machine must stay awake: a computer that goes to sleep stops everything.

## Project layout

```
configs/              all settings that are not code (see below)
data/reference/       published data the population is validated against
epidemic_sandbox/
  config/             loading and checking of the settings
  world/              the people and the town: population, schedules, places
  disease_models/     one module per disease model
  engine/             the model that steps the people and diseases, its kernels, the run workers
  app/                the web server, sessions, what the UI is told
  analysis/           validation of the population, calibration of the diseases
  web/                the UI: index.html, css/, js/, vendor/
tests/                mirrors the package; shared helpers are in tests/support/
```

## Configuration

Everything that is not code is in `configs/`; each file explains its values in comments.

| File | Content |
|------|---------|
| `app.toml` | Server, limits and UI timing |
| `town.toml` | The population and the town: ages, households, work and school, places, daily schedules, travel |
| `experiment.toml` | World settings of the default experiment |
| `diseases/<name>.toml` | One disease; every file in the directory is offered in the UI, those with `enabled = true` start in the default experiment |
| `interactions/<name>.toml` | One interaction between two diseases |

To add a disease, copy a file in `configs/diseases/` and change it. To add an interaction between diseases, copy a file in `configs/interactions/`. A wrong value stops the start with an error that names it.

## Adding a disease model

Add a module `<name>.py` to `epidemic_sandbox/disease_models/` with one class `<Name>Model` derived from `DiseaseModelAbstract`; it is found by its file name and needs no registration. A model is declared by data (parameters, states, phases), so an existing model is the best template and the docstring of `DiseaseModelAbstract` lists what can be declared. The UI picks up its states and parameters automatically. Add a disease that uses it in `configs/diseases/` (copy a real disease file, change the model name and the numbers, and give each number a source or mark it as an estimate); the tests that every model has to pass run for it without further work.

## Validation

```bash
python -m epidemic_sandbox.analysis.validate
```

compares the generated population with the data in `data/reference/` (sources are listed there). Shares and means are tested for equivalence with the published values; who meets whom is compared with published contact matrices, and how many people each age meets with contact surveys. The same checks run in the tests. They validate the population and the contacts, not the diseases.

```bash
python -m epidemic_sandbox.analysis.calibrate covid19
```

measures a disease in the town the way studies of real outbreaks measure it (household attack rate, reproduction number, generation interval, where infections happen, incidence of a disease that stays) and compares the results with the published targets in the `[targets]` table of its file. The comments of a disease file say which values were fitted to which target; `--set name=value` tries another value. The reproduction number is counted from the third generation of an epidemic that starts from a few first cases (`first_generation` and `initial` of a target change that), because the first generations are not typical; a disease file lists the checks it does not meet under "Known limitations".

## Scope

The town is self-contained and its people do not age, move or change jobs during a run; a disease can be brought in from outside at a constant rate (`outside_infections`). Every model can let a share of the infections run without symptoms (`asymptomatic_share`): those people behave as usual, spread as much as the disease file says and do not die of it. There are no seasons and no hospital beds: the gravely ill, and the cases a disease admits (measles by law, SARS), are away from the town until they recover or die. Infection spreads at close range (weighted by distance) and, for a disease with `room_exposure`, through the air of an indoor room at any distance. Distances, schedules and contacts are simplified; `configs/town.toml` and `epidemic_sandbox/world/town.py` are where to look when you need to know how.

## Development

```bash
pytest          # tests, including the validation
ruff check .    # style
```

The web UI is plain HTML, CSS and JavaScript in `epidemic_sandbox/web/` (no build step) with a bundled copy of [Chart.js](https://www.chartjs.org).
