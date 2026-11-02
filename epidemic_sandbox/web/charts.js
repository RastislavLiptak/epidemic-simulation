const SUGGESTED_DAYS = 30;
const MAX_CHART_POINTS = 1000;
const MIN_BRIGHTNESS_GAP = 40; // of 255: a line closer to the background than this cannot be seen

/** Chart points of a series of daily values; a very long series is thinned to keep charts fast. */
export function dailyPoints(values) {
  const stride = Math.ceil(values.length / MAX_CHART_POINTS);
  const points = [];
  for (let day = 0; day < values.length; day += stride) points.push({ x: day, y: values[day] });
  const last = values.length - 1;
  if (last % stride !== 0) points.push({ x: last, y: values[last] });
  return points;
}

export function themeColors() {
  const style = getComputedStyle(document.body);
  return { text: style.getPropertyValue("--muted"), grid: style.getPropertyValue("--border") };
}

function axisOptions(title, colors, extra = {}) {
  return { title: { display: true, text: title, color: colors.text }, ticks: { color: colors.text }, grid: { color: colors.grid }, ...extra };
}

function brightness(color) {
  const channel = (start) => parseInt(color.slice(start, start + 2), 16);
  return 0.299 * channel(1) + 0.587 * channel(3) + 0.114 * channel(5);
}

/** A color for a line on the current background: one lost in it gives way to the text color. */
export function visibleColor(color) {
  const style = getComputedStyle(document.body);
  const background = style.getPropertyValue("--surface").trim();
  const lost = Math.abs(brightness(color) - brightness(background)) < MIN_BRIGHTNESS_GAP;
  return lost ? style.getPropertyValue("--text").trim() : color;
}

/** Colors that tell diseases apart in charts; the arena tells them apart by wedge position. */
export const DISEASE_COLORS = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6", "#78716c", "#84cc16", "#06b6d4"];

/** Lines of the number of infectious people over the days, one per disease. */
export function buildInfectiousChart(canvas, diseases) {
  const colors = themeColors();
  return new Chart(canvas, {
    type: "line",
    data: {
      datasets: diseases.map((disease, index) => ({
        label: disease.name,
        data: [],
        borderColor: DISEASE_COLORS[index % DISEASE_COLORS.length],
        backgroundColor: DISEASE_COLORS[index % DISEASE_COLORS.length],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.25,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { type: "linear", min: 0, suggestedMax: SUGGESTED_DAYS, ...axisOptions("day", colors, { ticks: { precision: 0, color: colors.text } }) },
        y: { min: 0, ...axisOptions("infectious", colors) },
      },
    },
  });
}

/** Stacked area chart of the number of people in every state of one disease. */
export function buildStackedChart(canvas, states) {
  const colors = themeColors();
  return new Chart(canvas, {
    type: "line",
    data: {
      datasets: states.map((state, index) => ({
        label: state.name,
        data: [],
        backgroundColor: state.color,
        borderColor: state.color,
        borderWidth: 1,
        pointRadius: 0,
        fill: index === 0 ? "origin" : "-1",
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { type: "linear", min: 0, suggestedMax: SUGGESTED_DAYS, ...axisOptions("day", colors, { ticks: { precision: 0, color: colors.text } }) },
        y: { stacked: true, min: 0, ...axisOptions("people", colors) },
      },
    },
  });
}

const CHARTED_STATES = ["infectious", "recovered", "dead"];

/** The states of a disease that the charts of all runs show, in the order of their lines. */
export function chartedStates(disease) {
  return CHARTED_STATES.flatMap((name) => disease.states.filter((state) => state.name === name));
}

function paintLine(dataset) {
  const color = visibleColor(dataset.stateColor);
  dataset.borderColor = dataset.run === null ? color : `${color}55`;
  dataset.backgroundColor = color;
}

/**
 * People over the days in the given states (name and color): a thin line per run and a bold one
 * for their mean. The lines have the colors of the counts in the card of the disease, which serve
 * as the legend. Hovering shows only the means: the values of every run at one point would be
 * too many to read.
 */
export function buildRunsChart(canvas, runCount, states) {
  const colors = themeColors();
  const datasets = states.flatMap((state) =>
    [...Array.from({ length: runCount }, (_, run) => run), null].map((run) => ({
      label: run === null ? state.name : `${state.name}, run ${run + 1}`,
      state: state.name,
      run,
      stateColor: state.color,
      data: [],
      borderWidth: run === null ? 3 : 1,
      pointRadius: 0,
      hoverRadius: run === null ? 4 : 0,
      order: run === null ? 0 : 1,
    })),
  );
  datasets.forEach(paintLine);
  return new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => item.dataset.run === null,
          callbacks: { label: (item) => `${item.dataset.label}: ${Math.round(item.parsed.y).toLocaleString()}` },
        },
      },
      scales: {
        x: { type: "linear", min: 0, ...axisOptions("day", colors, { ticks: { precision: 0, color: colors.text } }) },
        y: { min: 0, ...axisOptions("people", colors) },
      },
    },
  });
}

/** Put the course of every run, and the mean of all runs, into a chart of `buildRunsChart`. */
export function fillRunsChart(chart, history, name) {
  const states = history.disease(name).states;
  const mean = history.series(name, null);
  const perRun = history.records.map((_, run) => history.series(name, run));
  const course = (series, state) => {
    const columns = states.flatMap((entry, index) => (state === "infectious" ? entry.infectious : entry.name === state) ? [index] : []);
    return Array.from({ length: history.dayCount }, (_, day) => columns.reduce((total, column) => total + series[column][day], 0));
  };
  for (const dataset of chart.data.datasets) {
    dataset.data = dailyPoints(course(dataset.run === null ? mean : perRun[dataset.run], dataset.state));
  }
  chart.update("none");
}

export function applyTheme(chart) {
  const colors = themeColors();
  chart.data.datasets.filter((line) => line.stateColor).forEach(paintLine);
  for (const axis of Object.values(chart.options.scales)) {
    axis.ticks.color = colors.text;
    axis.title.color = colors.text;
    axis.grid.color = colors.grid;
  }
  chart.update("none");
}

/** Bars of the age structure of the town with the census shares as a line. */
export function buildAgeChart(canvas, ageGroups) {
  const colors = themeColors();
  return new Chart(canvas, {
    data: {
      labels: ageGroups.labels,
      datasets: [
        { type: "bar", label: "town (sample)", data: [], backgroundColor: "#6366f1aa", order: 2 },
        {
          type: "line",
          label: "census",
          data: ageGroups.census_shares.map((share) => 100 * share),
          borderColor: "#f59e0b",
          backgroundColor: "#f59e0b",
          borderWidth: 2,
          pointRadius: 0,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: true, labels: { color: colors.text, boxWidth: 12 } } },
      scales: {
        x: { ...axisOptions("age", colors), ticks: { color: colors.text, maxRotation: 90, minRotation: 90, font: { size: 10 } } },
        y: { min: 0, ...axisOptions("% of people", colors) },
      },
    },
  });
}
