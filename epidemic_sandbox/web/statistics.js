const sum = (values) => values.reduce((total, value) => total + value, 0);
const mean = (values) => sum(values) / values.length;

/** Everything the runs have reported so far: one record per simulated day and run. */
export class History {
  constructor(run) {
    this.diseases = run.diseases;
    this.population = run.population;
    this.records = run.seeds.map(() => []);
  }

  push(frame) {
    frame.runs.forEach((data, run) => this.records[run].push(...data.history));
  }

  get runCount() {
    return this.records.length;
  }

  get dayCount() {
    return this.records[0].length;
  }

  disease(name) {
    return this.diseases.find((disease) => disease.name === name);
  }

  /** Counts per state over the days; `run` null averages all runs. */
  series(name, run) {
    const states = this.disease(name).states;
    const runs = run === null ? this.records : [this.records[run]];
    return states.map((_, state) =>
      Array.from({ length: this.dayCount }, (__, day) => mean(runs.map((records) => records[day].counts[name][state]))),
    );
  }

  /** Infectious people over the days in one run (or averaged over all runs when `run` is null). */
  infectious(name, run) {
    const states = this.disease(name).states;
    const columns = states.flatMap((state, index) => (state.infectious ? [index] : []));
    const counts = this.series(name, run);
    return Array.from({ length: this.dayCount }, (_, day) => sum(columns.map((column) => counts[column][day])));
  }

  /** Summary numbers of one disease in one run. */
  metrics(name, run) {
    const records = this.records[run];
    const infectious = this.infectious(name, run);
    const peak = Math.max(...infectious);
    const last = records.at(-1);
    const lastInfectiousDay = infectious.reduce((found, count, day) => (count > 0 ? day : found), 0);
    return {
      everInfected: last.ever_infected[name],
      attackRate: last.ever_infected[name] / this.population,
      peakInfectious: peak,
      peakDay: infectious.indexOf(peak),
      deaths: last.deaths[name],
      totalInfections: last.cumulative_infections[name],
      lastInfectiousDay,
    };
  }

  /** `metrics` of every run, plus the mean over runs. */
  allMetrics(name) {
    const perRun = this.records.map((_, run) => this.metrics(name, run));
    const average = Object.fromEntries(
      Object.keys(perRun[0]).map((key) => [key, mean(perRun.map((metrics) => metrics[key]))]),
    );
    return { perRun, average };
  }

  /** Long-format CSV: one row per run, day, disease and state. */
  csv() {
    const lines = ["run,day,disease,state,people"];
    this.records.forEach((records, run) => {
      for (const record of records) {
        for (const disease of this.diseases) {
          disease.states.forEach((state, index) => {
            lines.push(`${run + 1},${record.day},${disease.name},${state.name},${record.counts[disease.name][index]}`);
          });
        }
      }
    });
    return `${lines.join("\n")}\n`;
  }
}
