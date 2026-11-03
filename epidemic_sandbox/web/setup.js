import { buildAgeChart, diseaseColor } from "./charts.js";
import { setBusy } from "./busy.js";
import { buildField, buildOptionalNumber, element, transmissionChips } from "./fields.js";

const DURATION = "max_days";
const RUNS = "runs";
const SVG = "http://www.w3.org/2000/svg";
const GRAPH_WIDTH = 440;
const NODE_RADIUS = 16;
const EDGE_BEND = 26;
const EDGE_BEND_STEP = 24;
const KINDS = ["protects", "promotes", "neutral"];

function svg(tag, attributes = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  return node;
}

function workingShare(roleShares) {
  return ["preschool", "school", "university", "worker"].reduce((sum, role) => sum + roleShares[role], 0);
}

function formatValue(value) {
  return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(4)));
}

/** Whether an interaction reduces the target disease, increases it, or (factor 1) changes nothing. */
function interactionKind(interaction) {
  if (interaction.factor < 1) return "protects";
  return interaction.factor > 1 ? "promotes" : "neutral";
}

/** What happens to a person who is in the source disease's states, as the end of a sentence. */
function describeEffect(interaction) {
  const { factor, target, effect } = interaction;
  const less = `${formatValue((1 - factor) * 100)} % less`;
  const times = `${formatValue(factor)}× as`;
  if (effect === "susceptibility") {
    if (factor === 0) return `cannot catch ${target}`;
    if (factor === 1) return `catch ${target} as easily as usual`;
    return factor < 1 ? `are ${less} likely to catch ${target}` : `are ${times} likely to catch ${target}`;
  }
  if (factor === 0) return `do not infect anyone with ${target}`;
  if (factor === 1) return `infect others with ${target} as much as usual`;
  return factor < 1 ? `infect others with ${target} ${less}` : `infect others with ${target} ${times} much`;
}

/** The screen where an experiment is composed: the world, which diseases take part, how many runs. */
export class SetupScreen {
  constructor(container, setup, handlers) {
    this.container = container;
    this.setup = setup;
    this.handlers = handlers;
    this.world = Object.fromEntries(setup.world.map((control) => [control.name, control.value]));
    this.diseases = setup.catalog.map((disease) => ({
      ...disease,
      enabled: false,
      initialInfected: disease.initial_infected,
      color: diseaseColor(setup.catalog.indexOf(disease)),
    }));
    this.estimates = {};
    this.previewError = null;
    this.starting = false;
    this.render();
  }

  render() {
    const left = element("div", "setup__column");
    left.append(this.buildWorldCard(), this.buildDiseasesCard(), this.buildInteractionsCard());
    const right = element("div", "setup__column setup__column--summary");
    right.append(this.buildSummaryCard());
    this.container.replaceChildren(left, right);
    this.refreshState();
  }

  enabledDiseases() {
    return this.diseases.filter((disease) => disease.enabled);
  }

  /** The experiment as the server expects it: world settings and the chosen diseases by name. */
  definition() {
    return {
      world: { ...this.world },
      diseases: Object.fromEntries(
        this.enabledDiseases().map((disease) => [disease.name, { initial_infected: disease.initialInfected }]),
      ),
    };
  }

  showPreview(preview) {
    this.previewError = preview.error ?? null;
    this.estimates = preview.error ? {} : preview.diseases;
    this.summary.textContent = "";
    if (!preview.error) {
      this.summary.append(
        this.summaryRow("Mean age", `${preview.population.mean_age.toFixed(1)} years`, "Mean age of a sample of the town."),
        this.summaryRow("People per household", preview.population.mean_household_size.toFixed(2), "Census 2021: 2.15."),
        this.summaryRow(
          "Working or studying",
          `${(100 * workingShare(preview.population.role_shares)).toFixed(0)} %`,
          "Share of people who go to a workplace, school or university on weekdays.",
        ),
      );
      this.ageChart.data.datasets[0].data = preview.population.age_shares.map((share) => 100 * share);
      this.ageChart.update("none");
    }
    this.refreshState();
  }

  /** The experiment is being built by the server: the Start button waits and cannot be pressed again. */
  setStarting(starting) {
    this.starting = starting;
    setBusy(this.startButton, starting);
    this.refreshState();
  }

  /** The summary of the town is being worked out: it is dimmed and a spinner shows in its title. */
  setLoadingPreview(loading) {
    this.summaryCard.classList.toggle("is-loading", loading);
    this.summaryCard.setAttribute("aria-busy", String(loading));
  }

  summaryRow(label, value, hint) {
    const row = element("div", "summary__row");
    row.title = hint;
    row.append(element("span", "summary__label", label), element("strong", "", value));
    return row;
  }

  changed() {
    this.refreshState();
    this.handlers.onChange();
  }

  refreshState() {
    const ready = this.enabledDiseases().length > 0 && !this.previewError;
    this.startButton.disabled = !ready || this.starting;
    for (const [name, slot] of this.chipSlots) slot.replaceChildren(...transmissionChips(this.estimates[name]));
    this.notice.textContent = this.previewError ?? (this.enabledDiseases().length ? "" : "Enable at least one disease.");
  }

  buildWorldCard() {
    const card = element("section", "card");
    card.append(element("h2", "card__title", "Town"));
    const list = element("div", "controls");
    for (const control of this.setup.world.filter((entry) => ![DURATION, RUNS].includes(entry.name))) {
      list.append(
        buildField(control, (value) => {
          this.world[control.name] = value;
          this.changed();
        }).element,
      );
    }
    card.append(list);
    return card;
  }

  buildDiseasesCard() {
    const card = element("section", "card");
    card.append(
      element("h2", "card__title", "Diseases"),
      element(
        "p",
        "card__lead",
        "Choose which diseases spread together and how many people start infected. What a disease is like is defined in its file in configs/diseases/, not here. One person can have several diseases at once.",
      ),
    );
    this.chipSlots = new Map();
    this.diseaseList = element("div", "diseases");
    this.diseaseList.append(...this.diseases.map((disease) => this.buildDiseaseCard(disease)));
    card.append(this.diseaseList);
    return card;
  }

  buildDiseaseCard(disease) {
    const card = element("details", "disease");
    card.open = disease.enabled;
    card.classList.toggle("disease--disabled", !disease.enabled);

    const head = element("summary", "disease__head");
    const toggle = element("input", "disease__toggle");
    toggle.type = "checkbox";
    toggle.checked = disease.enabled;
    toggle.title = "Include this disease in the experiment";
    toggle.addEventListener("click", (event) => event.stopPropagation());
    toggle.addEventListener("change", () => {
      disease.enabled = toggle.checked;
      card.classList.toggle("disease--disabled", !disease.enabled);
      card.open = disease.enabled;
      this.renderInteractions();
      this.changed();
    });
    const swatch = element("span", "disease__swatch");
    swatch.style.background = disease.color;
    const chipSlot = element("span", "chips");
    this.chipSlots.set(disease.name, chipSlot);
    head.append(toggle, swatch, element("span", "disease__name", disease.name), element("span", "badge", disease.model), chipSlot);

    const body = element("div", "disease__body");
    body.append(
      buildField({ ...this.setup.initial_infected, value: disease.initialInfected }, (value) => {
        disease.initialInfected = value;
        this.changed();
      }).element,
      this.buildParameters(disease),
    );
    card.append(head, body);
    return card;
  }

  buildParameters(disease) {
    const details = element("details", "parameters");
    details.append(element("summary", "parameters__summary", "Disease parameters (read-only, from its file)"));
    const list = element("dl", "parameters__list");
    for (const parameter of disease.parameters) {
      const term = element("dt", "", parameter.label);
      term.title = parameter.description;
      list.append(term, element("dd", "", formatValue(parameter.value)));
    }
    details.append(list);
    for (const [effect, rows] of Object.entries(disease.age_effects)) {
      if (!rows.length) continue;
      details.append(
        element(
          "p",
          "field__hint",
          `${effect} by age: ${rows.map(([from, to, factor]) => `${from}–${to} ×${formatValue(factor)}`).join(", ")}`,
        ),
      );
    }
    return details;
  }

  buildInteractionsCard() {
    const card = element("section", "card");
    this.interactionsCard = card;
    card.append(
      element("h2", "card__title", "Interactions between diseases"),
      element(
        "p",
        "card__lead",
        "While a person is in some state of one disease, another disease changes for them: below ×1 it is reduced (cross-immunity, viral interference), above ×1 it is increased (weakened defences, coinfection).",
      ),
    );
    this.graphBox = element("div", "network");
    this.interactionList = element("ul", "interactions");
    card.append(this.graphBox, this.interactionList);
    this.renderInteractions();
    return card;
  }

  isActive(interaction) {
    const enabled = this.enabledDiseases().map((disease) => disease.name);
    return enabled.includes(interaction.source) && enabled.includes(interaction.target);
  }

  /** Shown only when at least two diseases are chosen, and then only with the interactions between the chosen ones. */
  renderInteractions() {
    const enabled = this.enabledDiseases();
    this.interactionsCard.hidden = enabled.length < 2;
    if (enabled.length < 2) return;
    const interactions = this.setup.interactions.filter((interaction) => this.isActive(interaction));
    this.graphBox.replaceChildren(this.buildGraph(enabled, interactions));
    this.interactionList.replaceChildren(...interactions.map((interaction) => this.buildInteraction(interaction)));
  }

  diseaseTag(name) {
    const tag = element("span", "interaction__disease");
    const swatch = element("span", "disease__swatch");
    swatch.style.background = this.diseases.find((disease) => disease.name === name).color;
    tag.append(swatch, name);
    return tag;
  }

  /** One interaction as a card: who changes whom, by how much, and in which state of the source. */
  buildInteraction(interaction) {
    const kind = interactionKind(interaction);
    const item = element("li", `interaction interaction--${kind}`);
    const head = element("div", "interaction__head");
    const factor = element("strong", "interaction__factor", `×${formatValue(interaction.factor)}`);
    factor.title = this.setup.effects[interaction.effect];
    head.append(
      this.diseaseTag(interaction.source),
      element("span", "interaction__arrow", "→"),
      this.diseaseTag(interaction.target),
      factor,
    );
    item.append(
      head,
      element(
        "p",
        "interaction__text",
        `People who are ${interaction.states.join(" or ")} in ${interaction.source} ${describeEffect(interaction)}.`,
      ),
    );
    return item;
  }

  /** The chosen diseases as nodes on a circle, interactions as arrows from the disease that matters to the one it changes. */
  buildGraph(diseases, interactions) {
    const height = diseases.length === 2 ? 150 : 250;
    const graph = svg("svg", { viewBox: `0 0 ${GRAPH_WIDTH} ${height}`, class: "network__svg", role: "img" });
    graph.setAttribute("aria-label", "Chosen diseases and their interactions");
    const defs = svg("defs");
    for (const kind of KINDS) {
      const marker = svg("marker", { id: `arrow-${kind}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
      marker.append(svg("path", { d: "M0 0L10 5L0 10z", class: `network__arrow network__arrow--${kind}` }));
      defs.append(marker);
    }
    graph.append(defs);

    const count = diseases.length;
    const centerX = GRAPH_WIDTH / 2;
    const centerY = height / 2 - 6;
    const radiusX = GRAPH_WIDTH / 2 - 70;
    const radiusY = height / 2 - 38;
    const firstAngle = count === 2 ? Math.PI : -Math.PI / 2;
    const position = new Map(
      diseases.map((disease, index) => {
        const angle = firstAngle + (2 * Math.PI * index) / count;
        return [disease.name, { x: centerX + radiusX * Math.cos(angle), y: centerY + radiusY * Math.sin(angle) }];
      }),
    );

    const drawn = new Map();
    for (const interaction of interactions) {
      const from = position.get(interaction.source);
      const to = position.get(interaction.target);
      const pair = `${interaction.source}>${interaction.target}`;
      const parallel = drawn.get(pair) ?? 0;
      drawn.set(pair, parallel + 1);
      const kind = interactionKind(interaction);
      const middle = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 };
      const bend = EDGE_BEND + EDGE_BEND_STEP * parallel;
      const length = Math.hypot(to.x - from.x, to.y - from.y) || 1;
      const control = { x: middle.x - ((to.y - from.y) / length) * bend, y: middle.y + ((to.x - from.x) / length) * bend };
      const shorten = (point, other) => {
        const distance = Math.hypot(point.x - other.x, point.y - other.y) || 1;
        return { x: point.x + ((other.x - point.x) / distance) * (NODE_RADIUS + 3), y: point.y + ((other.y - point.y) / distance) * (NODE_RADIUS + 3) };
      };
      const start = shorten(from, control);
      const end = shorten(to, control);
      const edge = svg("g", { class: `network__edge network__edge--${kind}` });
      edge.append(
        svg("path", { d: `M${start.x} ${start.y} Q${control.x} ${control.y} ${end.x} ${end.y}`, "marker-end": `url(#arrow-${kind})` }),
      );
      const label = svg("text", { x: (start.x + 2 * control.x + end.x) / 4, y: (start.y + 2 * control.y + end.y) / 4 - 4, "text-anchor": "middle" });
      label.textContent = `×${formatValue(interaction.factor)}`;
      edge.append(label);
      graph.append(edge);
    }

    for (const disease of diseases) {
      const { x, y } = position.get(disease.name);
      const node = svg("g", { class: "network__node" });
      node.append(svg("circle", { cx: x, cy: y, r: NODE_RADIUS, fill: disease.color }));
      const label = svg("text", { x, y: y + NODE_RADIUS + 13, "text-anchor": "middle" });
      label.textContent = disease.name;
      node.append(label);
      graph.append(node);
    }
    return graph;
  }

  buildSummaryCard() {
    const card = element("section", "card card--summary");
    this.summaryCard = card;
    card.append(element("h2", "card__title", "Experiment"));
    this.summary = element("div", "summary");
    this.notice = element("p", "summary__error");
    this.startButton = element("button", "button button--primary button--wide", "Start");
    this.startButton.type = "button";
    this.startButton.addEventListener("click", () => this.handlers.start());
    const ageBox = element("div", "card__chart card__chart--short");
    const ageCanvas = element("canvas");
    ageBox.append(ageCanvas);
    this.ageChart = buildAgeChart(ageCanvas, this.setup.age_groups);
    const runs = buildField(
      this.setup.world.find((control) => control.name === RUNS),
      (value) => {
        this.world[RUNS] = value;
        this.changed();
      },
    );
    const duration = buildOptionalNumber(
      this.setup.world.find((control) => control.name === DURATION),
      (value) => {
        this.world[DURATION] = value;
        this.changed();
      },
    );
    card.append(this.summary, ageBox, runs.element, duration.element, this.notice, this.startButton);
    return card;
  }
}
