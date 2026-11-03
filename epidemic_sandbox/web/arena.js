const FLASH_MS = 700;
const HALO_ALPHA_HEX = "22";
const HALO_RADIUS_FACTOR = 2.6;
const NIGHT_COLOR = "15, 23, 60";
const NIGHT_DARKNESS = 0.35;
const NOON_HOUR = 13;
const HIT_SLACK_PX = 5;
const MIN_GAP_MS = 16;
const MAX_GAP_MS = 1500;
const GAP_SMOOTHING = 0.3;
const DELAY_IN_GAPS = 1.5;
const KEPT_SNAPSHOTS = 4;

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const dotRadius = (size) => clamp(size * 0.008, 3, 8);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const arenas = new Set();
let loopRunning = false;

function loop(now) {
  for (const arena of arenas) if (arena.visible) arena.draw(now);
  requestAnimationFrame(loop);
}

/**
 * Animated view of the people of one run. Frames arrive unevenly, so the view is drawn a little
 * in the past, between the two frames that surround that moment: every dot moves at a constant
 * pace from frame to frame and people walk instead of jumping.
 */
export class Arena {
  constructor(canvas, handlers) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.handlers = handlers;
    this.diseases = [];
    this.current = null;
    this.snapshots = [];
    this.displayed = new Float32Array(0);
    this.gapMs = 100;
    this.lastArrival = null;
    this.flashes = new Map();
    this.hovered = null;
    this.household = [];
    this.visible = true;
    this.size = 0;
    this.resizer = new ResizeObserver(() => this.resize());
    this.resizer.observe(canvas);
    this.watcher = new IntersectionObserver((entries) => {
      this.visible = entries.at(-1).isIntersecting;
    });
    this.watcher.observe(canvas);
    canvas.addEventListener("mousemove", (event) => handlers.hover(this.personAt(event), event));
    canvas.addEventListener("mouseleave", () => handlers.hover(null));
    arenas.add(this);
    if (!loopRunning) {
      loopRunning = true;
      requestAnimationFrame(loop);
    }
  }

  destroy() {
    arenas.delete(this);
    this.resizer.disconnect();
    this.watcher.disconnect();
  }

  configure(diseases) {
    this.diseases = diseases;
    this.current = null;
    this.hovered = null;
    this.household = [];
    this.lastArrival = null;
    this.flashes.clear();
  }

  /** Mark the hovered person and the members of their household. */
  setHighlight({ hovered, household }) {
    this.hovered = hovered;
    this.household = household;
  }

  /** `frame` is {run_id, hour, x, y, states}: positions in 0..1 and a state index per disease. */
  pushFrame(frame) {
    const now = performance.now();
    const isNewRun = !this.current || this.current.run_id !== frame.run_id;
    if (isNewRun) {
      this.flashes.clear();
      this.lastArrival = null;
    } else {
      this.recordNewInfections(frame, now);
    }
    if (this.lastArrival !== null) {
      const gap = clamp(now - this.lastArrival, MIN_GAP_MS, MAX_GAP_MS);
      this.gapMs += (gap - this.gapMs) * GAP_SMOOTHING;
    }
    this.lastArrival = now;
    this.current = frame;
    const positions = new Float32Array(2 * frame.x.length);
    for (let person = 0; person < frame.x.length; person++) {
      positions[2 * person] = frame.x[person];
      positions[2 * person + 1] = frame.y[person];
    }
    if (isNewRun) this.snapshots = [];
    this.snapshots.push({ at: now, positions });
    if (this.snapshots.length > KEPT_SNAPSHOTS) this.snapshots.shift();
    if (isNewRun || this.displayed.length !== positions.length) this.displayed = positions.slice();
  }

  /** Where everybody is at `now`: between the two snapshots around `now` minus the delay. */
  interpolate(now) {
    const snapshots = this.snapshots;
    const at = now - DELAY_IN_GAPS * this.gapMs;
    let next = snapshots.findIndex((snapshot) => snapshot.at > at);
    if (reducedMotion.matches || next === -1) next = snapshots.length;
    const after = snapshots[Math.min(next, snapshots.length - 1)];
    const before = snapshots[Math.max(next - 1, 0)];
    const span = after.at - before.at;
    const share = span > 0 ? clamp((at - before.at) / span, 0, 1) : 1;
    for (let index = 0; index < this.displayed.length; index++) {
      this.displayed[index] = before.positions[index] + (after.positions[index] - before.positions[index]) * share;
    }
  }

  resize() {
    const ratio = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    this.size = width;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  /** Index of the person under the mouse, or null. */
  personAt(event) {
    if (!this.size || !this.current) return null;
    const box = this.canvas.getBoundingClientRect();
    const padding = dotRadius(this.size);
    const span = this.size - 2 * padding;
    const reach = dotRadius(this.size) + HIT_SLACK_PX;
    const mouseX = event.clientX - box.left;
    const mouseY = event.clientY - box.top;
    let best = null;
    let bestDistance = reach * reach;
    for (let person = 0; person < this.displayed.length / 2; person++) {
      const dx = padding + this.displayed[2 * person] * span - mouseX;
      const dy = padding + this.displayed[2 * person + 1] * span - mouseY;
      const distance = dx * dx + dy * dy;
      if (distance <= bestDistance) {
        best = person;
        bestDistance = distance;
      }
    }
    return best;
  }

  recordNewInfections(frame, now) {
    for (const disease of this.diseases) {
      const before = this.current.states[disease.name];
      const after = frame.states[disease.name];
      for (let person = 0; person < after.length; person++) {
        if (
          before[person] !== after[person] &&
          disease.states[before[person]].exposable &&
          disease.states[after[person]].in_course
        ) {
          this.flashes.set(`${disease.name}:${person}`, {
            person,
            since: now,
            color: disease.states[after[person]].color,
          });
        }
      }
    }
  }

  draw(now) {
    if (!this.current || !this.size) return;
    this.interpolate(now);
    const size = this.size;
    const positions = this.canvasPositions(size);
    const context = this.context;
    context.clearRect(0, 0, size, size);
    this.drawNight(context, size);
    this.drawHalos(context, positions, now, size);
    this.drawPeople(context, positions, size);
    this.drawFlashes(context, positions, now, size);
    this.drawHighlights(context, positions, size);
  }

  /** Canvas coordinates of everyone; the arena is inset so that dots at the walls are not clipped. */
  canvasPositions(size) {
    const padding = dotRadius(size);
    const span = size - 2 * padding;
    const positions = new Float32Array(this.displayed.length);
    for (let index = 0; index < positions.length; index++) {
      positions[index] = padding + this.displayed[index] * span;
    }
    return positions;
  }

  drawNight(context, size) {
    const daylight = 0.5 * (1 + Math.cos((2 * Math.PI * (this.current.hour - NOON_HOUR)) / 24));
    context.fillStyle = `rgba(${NIGHT_COLOR}, ${NIGHT_DARKNESS * (1 - daylight)})`;
    context.fillRect(0, 0, size, size);
  }

  drawHalos(context, positions, now, size) {
    const pulse = reducedMotion.matches ? 0 : 0.06;
    for (const disease of this.diseases) {
      const states = this.current.states[disease.name];
      for (let person = 0; person < states.length; person++) {
        const state = disease.states[states[person]];
        if (!state.infectious) continue;
        const wobble = 1 + pulse * Math.sin(now / 350 + person);
        context.beginPath();
        context.arc(
          positions[2 * person],
          positions[2 * person + 1],
          dotRadius(size) * HALO_RADIUS_FACTOR * wobble,
          0,
          2 * Math.PI,
        );
        context.fillStyle = state.color + HALO_ALPHA_HEX;
        context.fill();
      }
    }
  }

  /** Every person is a disc split into one wedge per disease, clockwise from the top. */
  drawPeople(context, positions, size) {
    const radius = dotRadius(size);
    const wedge = (2 * Math.PI) / this.diseases.length;
    context.lineWidth = 1;
    context.strokeStyle = getComputedStyle(document.body).getPropertyValue("--arena");
    for (let person = 0; person < positions.length / 2; person++) {
      const x = positions[2 * person];
      const y = positions[2 * person + 1];
      this.diseases.forEach((disease, slot) => {
        const state = disease.states[this.current.states[disease.name][person]];
        const start = -Math.PI / 2 + slot * wedge;
        context.beginPath();
        if (this.diseases.length > 1) context.moveTo(x, y);
        context.arc(x, y, radius, start, start + wedge);
        context.closePath();
        context.fillStyle = state.color;
        context.fill();
      });
      context.beginPath();
      context.arc(x, y, radius, 0, 2 * Math.PI);
      context.stroke();
    }
  }

  drawFlashes(context, positions, now, size) {
    const baseRadius = dotRadius(size);
    for (const [key, flash] of this.flashes) {
      const age = (now - flash.since) / FLASH_MS;
      if (age >= 1) {
        this.flashes.delete(key);
        continue;
      }
      context.beginPath();
      context.arc(
        positions[2 * flash.person],
        positions[2 * flash.person + 1],
        baseRadius * (1.5 + 3 * age),
        0,
        2 * Math.PI,
      );
      context.strokeStyle = flash.color;
      context.globalAlpha = 1 - age;
      context.lineWidth = 2;
      context.stroke();
      context.globalAlpha = 1;
    }
  }

  drawHighlights(context, positions, size) {
    const radius = dotRadius(size);
    const accent = getComputedStyle(document.body).getPropertyValue("--accent");
    const ring = (person, extra, width) => {
      if (person === null || 2 * person >= positions.length) return;
      context.beginPath();
      context.arc(positions[2 * person], positions[2 * person + 1], radius + extra, 0, 2 * Math.PI);
      context.lineWidth = width;
      context.strokeStyle = accent;
      context.stroke();
    };
    this.household.forEach((member) => ring(member, 3, 1.5));
    ring(this.hovered, 5, 2);
  }
}
