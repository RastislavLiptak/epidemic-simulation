import { Arena } from "./arena.js";
import { diseaseColor } from "../shared/charts.js";
import { element } from "../shared/fields.js";
import { basicPerson, renderPerson } from "./person.js";
import { LiveStatistics } from "../shared/sidebar.js";

const TOOLTIP_OFFSET_PX = 16;

/**
 * The runs of an experiment. The overview shows the course of every run in the charts of each
 * disease and a tile per run; opening a run shows its town with every person, whom you can
 * hover to see who they are, and the statistics of that run alone. Only the run that is open
 * is drawn, so the number of runs hardly matters.
 */
export class RunsView {
  constructor(container, tooltip, run, handlers) {
    this.container = container;
    this.tooltip = tooltip;
    this.run = run;
    this.handlers = handlers;
    this.history = null;
    this.latest = null;
    this.selected = null;
    this.statistics = null;
    this.arena = null;
    this.people = null;
    this.hovered = null;
    this.household = [];
    this.info = new Map();
    this.tiles = [];
    this.show(null);
  }

  destroy() {
    this.clear();
  }

  clear() {
    this.statistics?.destroy();
    this.arena?.destroy();
    this.statistics = this.arena = this.people = this.hovered = null;
    this.household = [];
    this.tooltip.hidden = true;
  }

  /** Show the overview (`null`) or one run. */
  show(selected) {
    this.clear();
    this.selected = selected;
    this.handlers.watch(selected);
    if (selected === null) this.buildOverview();
    else this.buildDetail(selected);
    if (this.latest) this.update(this.latest, this.history, true);
  }

  buildOverview() {
    const statistics = element("div", "runs__statistics");
    this.statistics = new LiveStatistics(statistics, this.run, null);
    const heading = element("h2", "runs__heading", "Runs");
    const tiles = element("div", "runs__tiles");
    this.tiles = this.run.seeds.map((seed, index) => this.buildTile(index, seed));
    tiles.append(...this.tiles.map((tile) => tile.element));
    this.container.replaceChildren(
      element("h2", "runs__heading", "All runs"),
      statistics,
      heading,
      element("p", "field__hint", "Open a run to see its town and the people in it."),
      tiles,
    );
    this.container.className = "runs runs--overview";
  }

  buildTile(index, seed) {
    const tile = element("button", "card tile");
    tile.type = "button";
    tile.addEventListener("click", () => this.show(index));
    const head = element("span", "tile__head");
    head.append(element("strong", "", `Run ${index + 1}`), element("span", "badge badge--muted", `seed ${seed}`));
    const rows = this.run.diseases.map((disease, position) => {
      const row = element("span", "tile__row");
      const swatch = element("span", "disease__swatch");
      swatch.style.background = diseaseColor(position);
      const text = element("span", "tile__text");
      row.append(swatch, element("span", "tile__name", disease.name), text);
      return { row, text, disease };
    });
    tile.append(head, ...rows.map(({ row }) => row));
    return { element: tile, rows };
  }

  buildDetail(index) {
    const back = element("button", "button", "← All runs");
    back.type = "button";
    back.addEventListener("click", () => this.show(null));
    const head = element("div", "runs__bar");
    head.append(back, element("h2", "runs__heading", `Run ${index + 1}`), element("span", "badge badge--muted", `seed ${this.run.seeds[index]}`));

    const canvas = element("canvas");
    const stage = element("div", "run__arena");
    stage.append(canvas);
    this.arena = new Arena(canvas, { hover: (person, event) => this.hover(person, event) });
    this.arena.configure(this.run.diseases);

    const statistics = element("div", "runs__statistics runs__statistics--column");
    this.statistics = new LiveStatistics(statistics, this.run, index);
    const body = element("div", "runs__detail");
    body.append(stage, statistics);
    this.container.replaceChildren(head, body);
    this.container.className = "runs runs--detail";
  }

  /** Show a frame of all runs; `isNewRun` clears the charts, `history` has everything so far. */
  update(frame, history, isNewRun) {
    this.latest = frame;
    this.history = history;
    if (this.selected === null) this.updateTiles(frame, history);
    else {
      const data = frame.runs[this.selected];
      if (data.x && this.people) {
        this.arena.pushFrame({ run_id: frame.run_id, hour: frame.hour, x: data.x, y: data.y, states: data.states });
      }
    }
    this.statistics.update(history, frame, isNewRun);
  }

  updateTiles(frame, history) {
    this.tiles.forEach((tile, run) => {
      tile.rows.forEach(({ text, disease }) => {
        const counts = frame.runs[run].counts[disease.name];
        const infectious = disease.states.reduce((sum, state, at) => sum + (state.infectious ? counts[at] : 0), 0);
        const ever = history.dayCount ? history.metrics(disease.name, run).attackRate : 0;
        text.textContent = `${infectious.toLocaleString()} infectious · ${(100 * ever).toFixed(1)} % ever infected`;
      });
    });
  }

  /** The people of the open run: their ages and roles. */
  showWatching(message) {
    if (message.run === this.selected && message.run !== null) this.people = message;
  }

  hover(person, event) {
    if (person === null || !this.people) {
      this.hovered = null;
      this.household = [];
      this.tooltip.hidden = true;
      this.highlight();
      return;
    }
    const changed = !this.hovered || this.hovered.person !== person;
    this.hovered = { person };
    this.pointer = event;
    if (changed) {
      this.household = this.info.get(`${this.selected}:${person}`)?.household.members.map((member) => member.id) ?? [];
      this.renderTooltip();
      this.handlers.inspect(this.selected, person);
    }
    this.placeTooltip(event);
    this.highlight();
  }

  renderTooltip() {
    const person = this.hovered.person;
    const known = this.info.get(`${this.selected}:${person}`);
    const states = this.latest.runs[this.selected].states;
    const info = known ?? basicPerson(person, this.run, this.people, states);
    this.tooltip.replaceChildren(renderPerson(info, this.run.diseases));
    this.tooltip.hidden = false;
  }

  placeTooltip(event) {
    const box = this.tooltip.getBoundingClientRect();
    const left = Math.min(event.clientX + TOOLTIP_OFFSET_PX, window.innerWidth - box.width - 8);
    const top = Math.min(event.clientY + TOOLTIP_OFFSET_PX, window.innerHeight - box.height - 8);
    this.tooltip.style.left = `${Math.max(8, left)}px`;
    this.tooltip.style.top = `${Math.max(8, top)}px`;
  }

  /** The server's description of a person arrived. */
  showPerson(info) {
    this.info.set(`${info.run}:${info.id}`, info);
    if (this.info.size > 200) this.info.delete(this.info.keys().next().value);
    if (this.hovered && this.selected === info.run && this.hovered.person === info.id) {
      this.household = info.household.members.map((member) => member.id);
      this.renderTooltip();
      this.placeTooltip(this.pointer);
      this.highlight();
    }
  }

  highlight() {
    this.arena?.setHighlight({ hovered: this.hovered?.person ?? null, household: this.household });
  }
}
