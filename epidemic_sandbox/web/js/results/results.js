import {
  applyTheme,
  buildRunsChart,
  buildStackedChart,
  chartedStates,
  dailyPoints,
  diseaseColor,
  fillRunsChart,
} from "../shared/charts.js";
import { element } from "../shared/fields.js";
import { formatMetric } from "../shared/sidebar.js";

const COLUMNS = [
  ["Ever infected", "everInfected", (value) => Math.round(value).toLocaleString()],
  ["Share of people", "attackRate", (value) => `${(100 * value).toFixed(1)} %`],
  ["Peak infectious", "peakInfectious", (value) => Math.round(value).toLocaleString()],
  ["Peak on day", "peakDay", (value) => value.toFixed(0)],
  ["Deaths caused", "deaths", (value) => Math.round(value).toLocaleString()],
  ["Infections in total", "totalInfections", (value) => Math.round(value).toLocaleString()],
  ["Last infectious day", "lastInfectiousDay", (value) => value.toFixed(0)],
];

function download(name, text) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

/** The course of the finished experiment: summary numbers and charts for every disease. */
export class ResultsView {
  constructor(container) {
    this.container = container;
    this.charts = [];
    this.themeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    this.themeQuery.addEventListener("change", () => this.charts.forEach(applyTheme));
  }

  /** Build the screen from the course of the finished experiment. */
  show(history) {
    this.charts.forEach((chart) => chart.destroy());
    this.charts = [];
    if (!history.dayCount) {
      this.container.replaceChildren(element("p", "card__lead", "Nothing to show yet: no day has been simulated."));
      return;
    }
    const head = element("section", "card results__head");
    const text = element("div");
    text.append(
      element("h2", "card__title", "Experiment finished"),
      element(
        "p",
        "card__lead",
        `${history.runCount} ${history.runCount === 1 ? "run" : "runs"}, ${history.dayCount - 1} simulated days, ${history.population.toLocaleString()} people each.`,
      ),
    );
    const save = element("button", "button", "Download CSV");
    save.type = "button";
    save.addEventListener("click", () => download("experiment.csv", history.csv()));
    head.append(text, save);
    this.container.replaceChildren(head, ...history.diseases.map((disease, index) => this.buildDisease(history, disease, index)));
  }

  buildDisease(history, disease, index) {
    const color = diseaseColor(index);
    const card = element("section", "card");
    const title = element("header", "stat__head");
    const swatch = element("span", "disease__swatch");
    swatch.style.background = color;
    title.append(swatch, element("h2", "card__title", disease.name), element("span", "badge", disease.model));
    const tableBox = element("div", "table-wrap");
    tableBox.append(this.buildTable(history, disease.name));
    card.append(title, tableBox);

    const charts = element("div", "results__charts");
    const stackedBox = element("div", "results__chart");
    const stackedCanvas = element("canvas");
    stackedBox.append(element("h3", "controls__title", "People by state"), this.runSelector(history, disease, stackedCanvas), stackedCanvas);
    const infectiousBox = element("div", "results__chart");
    const infectiousCanvas = element("canvas");
    infectiousBox.append(element("h3", "controls__title", "Infectious, recovered and dead people, every run and the mean"), infectiousCanvas);
    charts.append(stackedBox, infectiousBox);
    card.append(charts);

    const runs = buildRunsChart(infectiousCanvas, history.runCount, chartedStates(disease));
    fillRunsChart(runs, history, disease.name);
    this.charts.push(runs);
    const stacked = buildStackedChart(stackedCanvas, disease.states);
    this.charts.push(stacked);
    this.fillStacked(stacked, history, disease.name, null);
    stackedCanvas.stackedChart = stacked;
    return card;
  }

  fillStacked(chart, history, name, run) {
    const series = history.series(name, run);
    chart.data.datasets.forEach((dataset, state) => {
      dataset.data = dailyPoints(series[state]);
    });
    chart.options.scales.y.max = history.population;
    chart.options.scales.x.max = Math.max(history.dayCount - 1, 1);
    chart.update("none");
  }

  runSelector(history, disease, canvas) {
    const select = element("select", "select");
    const options = [[history.runCount > 1 ? "Mean of all runs" : "Run 1", "all"]];
    if (history.runCount > 1) history.records.forEach((_, run) => options.push([`Run ${run + 1}`, String(run)]));
    for (const [label, value] of options) {
      const option = element("option", "", label);
      option.value = value;
      select.append(option);
    }
    select.addEventListener("change", () => {
      this.fillStacked(canvas.stackedChart, history, disease.name, select.value === "all" ? null : Number(select.value));
    });
    select.hidden = history.runCount === 1;
    return select;
  }

  buildTable(history, name) {
    const { perRun } = history.allMetrics(name);
    const table = element("table", "table");
    const header = element("tr");
    header.append(element("th", "", ""), ...COLUMNS.map(([label]) => element("th", "", label)));
    const head = element("thead");
    head.append(header);
    table.append(head);
    const body = element("tbody");
    perRun.forEach((metrics, run) => {
      const row = element("tr");
      row.append(element("th", "", `Run ${run + 1}`), ...COLUMNS.map(([, key, format]) => element("td", "", format(metrics[key]))));
      body.append(row);
    });
    if (perRun.length > 1) {
      const row = element("tr", "table__total");
      row.append(
        element("th", "", "Mean (range)"),
        ...COLUMNS.map(([, key, format]) => element("td", "", formatMetric(perRun, key, format))),
      );
      body.append(row);
    }
    table.append(body);
    return table;
  }
}
