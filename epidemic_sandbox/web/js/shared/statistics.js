const sum = (values) => values.reduce((total, value) => total + value, 0);
const OTHER_DEATHS = "dead of another disease";
const mean = (values) => sum(values) / values.length;

/**
 * Everything the runs have reported so far: one record per simulated day and run, taken at
 * midnight, and a last one at the hour the experiment was finished if that was not midnight.
 */
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

  /** The day of every record (the last one may be a fraction of a day). */
  get days() {
    return this.records[0].map((record) => record.day);
  }

  /** The time of the last record, in days. */
  get lastDay() {
    return this.dayCount ? this.records[0].at(-1).day : 0;
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
      peakDay: records[infectious.indexOf(peak)].day,
      deaths: last.deaths[name],
      totalInfections: last.cumulative_infections[name],
      lastInfectiousDay: records[lastInfectiousDay].day,
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

  /**
   * Long-format CSV: one row per run, day, disease and state. A disease's dead are its own
   * victims; the people another disease killed are in no state of it, so each disease has a row
   * "dead of another disease" too, and the rows of a disease add up to the population.
   */
  csv() {
    const lines = ["run,day,disease,state,people"];
    this.records.forEach((records, run) => {
      for (const record of records) {
        const allDeaths = sum(Object.values(record.deaths));
        for (const disease of this.diseases) {
          const row = (state, people) => lines.push(`${run + 1},${record.day},${disease.name},${state},${people}`);
          disease.states.forEach((state, index) => row(state.name, record.counts[disease.name][index]));
          if (this.diseases.length > 1) row(OTHER_DEATHS, allDeaths - record.deaths[disease.name]);
        }
      }
    });
    return `${lines.join("\n")}\n`;
  }
}
