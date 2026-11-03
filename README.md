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
python -m epidemic_sandbox --help  # port, host and which config files to use
```

In the UI you set up an experiment (town, diseases, number of runs), watch the runs and hover over a person to see who they are, and look at the results and export them once the experiment is finished.

## Configuration

Everything that is not code is in `configs/`; each file explains its values in comments.

| File | Content |
|------|---------|
| `app.toml` | Server, limits and UI timing |
| `town.toml` | The population and the town: ages, households, work and school, places, daily schedules, travel |
| `experiment.toml` | Default experiment and the interactions between diseases |
| `diseases/<name>.toml` | One disease; every file in the directory is offered in the UI |

To add a disease, copy a file in `configs/diseases/` and change it. To add an interaction between diseases, copy an entry in `experiment.toml`. A wrong value stops the start with an error that names it.

## Adding a disease model

Add a module `<name>.py` to `epidemic_sandbox/disease_models/` with one class `<Name>Model` derived from `DiseaseModelAbstract`; it is found by its file name and needs no registration. A model is declared by data (parameters, states, phases), so an existing model is the best template and the docstring of `DiseaseModelAbstract` lists what can be declared. The UI picks up its states and parameters automatically. Add a demo disease for it in `configs/diseases/`; the tests that every model has to pass run for it without further work.

## Validation

```bash
python -m epidemic_sandbox.validate
```

compares the generated population with the data in `data/reference/` (sources are listed there). Shares and means are tested for equivalence with the published values, and who meets whom is compared with published contact matrices. The same checks run in the tests. They validate the population and the structure of contacts, not the epidemics: a disease has to be fitted to real data by whoever defines it.

## Scope

The town is self-contained and its people do not age, move or change jobs during a run. Distances, schedules and contacts are simplified; `configs/town.toml` and `epidemic_sandbox/town.py` are where to look when you need to know how.

## Development

```bash
pytest          # tests, including the validation
ruff check .    # style
```

The web UI is plain HTML, CSS and JavaScript in `epidemic_sandbox/web/` (no build step) with a bundled copy of [Chart.js](https://www.chartjs.org).
