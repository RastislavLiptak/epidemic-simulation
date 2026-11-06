import {
  applyTheme,
  buildRunsChart,
  buildStackedChart,
  chartedStates,
  dailyPoints,
  diseaseColor,
  fillRunsChart,
  visibleColor,
} from "./charts.js";
import { element } from "./fields.js";

const percent = (share) => `${(100 * share).toFixed(1)} %`;

/** `average` of a metric with the range over runs when there are several. */
export function formatMetric(perRun, key, format) {
  const values = perRun.map((metrics) => metrics[key]);
  const average = values.reduce((total, value) => total + value, 0) / values.length;
  if (values.length === 1) return format(average);
  return `${format(average)} (${format(Math.min(...values))} – ${format(Math.max(...values))})`;
}

const people = (count) => Math.round(count).toLocaleString();

/**
 * One card per disease with live statistics. For all runs (`selected` null) the numbers are
 * averages and the chart has the course of every run and the mean of infectious, recovered and
 * dead people; for one run they are that run's own.
 */
export class LiveStatistics {
  constructor(container, run, selected) {
    this.run = run;
    this.selected = selected;
    this.cards = run.diseases.map((disease, index) => this.buildCard(disease, index));
    container.replaceChildren(...this.cards.map((card) => card.element));
    this.themeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    this.themeListener = () =>
      this.cards.forEach((card) => {
        applyTheme(card.chart);
        this.paintDots(card);
      });
    this.themeQuery.addEventListener("change", this.themeListener);
  }

  destroy() {
    this.themeQuery.removeEventListener("change", this.themeListener);
    this.cards.forEach(({ chart }) => chart.destroy());
  }

  buildCard(disease, index) {
    const card = element("article", "card");
    const head = element("header", "stat__head");
    const swatch = element("span", "disease__swatch");
    swatch.style.background = diseaseColor(index);
    head.append(swatch, element("h2", "card__title", disease.name), element("span", "badge", disease.model));

    const counts = element("ul", "counts");
    const dots = [];
    const values = disease.states.map((state) => {
      const item = element("li", "counts__item");
      const dot = element("span", "counts__dot");
      dots.push(dot);
      const value = element("span", "counts__value", "0");
      item.append(dot, `${state.name} `, value);
      counts.append(item);
      return value;
    });
    const metrics = element("dl", "stat__metrics");
    const chartBox = element("div", "stat__chart");
    const canvas = element("canvas");
    chartBox.append(canvas);
    card.append(head, counts, metrics, chartBox);
    const chart =
      this.selected === null
        ? buildRunsChart(canvas, this.run.seeds.length, chartedStates(disease))
        : buildStackedChart(canvas, disease.states);
    const built = { element: card, disease, values, dots, metrics, chart, days: -1 };
    this.paintDots(built);
    return built;
  }

  /** The dots have the colors of the lines in the chart of all runs, which that chart adjusts to its background. */
  paintDots({ disease, dots }) {
    dots.forEach((dot, state) => {
      const color = disease.states[state].color;
      dot.style.background = this.selected === null ? visibleColor(color) : color;
    });
  }

  update(history, frame, isNewRun) {
    const selected = this.selected;
    const runs = selected === null ? frame.runs : [frame.runs[selected]];
    for (const card of this.cards) {
      const name = card.disease.name;
      card.disease.states.forEach((_, state) => {
        const average = runs.reduce((total, data) => total + data.counts[name][state], 0) / runs.length;
        card.values[state].textContent = people(average);
      });
      if (!history.dayCount) continue;
      const { perRun } = history.allMetrics(name);
      const shown = selected === null ? perRun : [perRun[selected]];
      card.metrics.replaceChildren(
        ...[
          ["Ever infected", formatMetric(shown, "attackRate", percent)],
          ["Peak infectious", formatMetric(shown, "peakInfectious", people)],
          ["Peak on day", formatMetric(shown, "peakDay", (day) => day.toFixed(0))],
          ["Deaths caused", formatMetric(shown, "deaths", people)],
        ].flatMap(([label, value]) => [element("dt", "", label), element("dd", "", value)]),
      );
      if (isNewRun || card.days !== history.dayCount) {
        this.fillChart(card, history);
        card.days = history.dayCount;
      }
    }
  }

  fillChart(card, history) {
    const name = card.disease.name;
    if (this.selected === null) {
      fillRunsChart(card.chart, history, name);
      return;
    }
    const series = history.series(name, this.selected);
    card.chart.data.datasets.forEach((dataset, state) => (dataset.data = dailyPoints(series[state])));
    card.chart.options.scales.y.max = history.population;
    card.chart.update("none");
  }
}
