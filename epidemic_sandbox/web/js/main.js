import { setBusy } from "./shared/busy.js";
import { ResultsView } from "./results/results.js";
import { RunsView } from "./runs/runs.js";
import { SetupScreen } from "./setup/setup.js";
import { connect } from "./socket.js";
import { History } from "./shared/statistics.js";

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const PAUSE_ICON = "M6 5h4v14H6zM14 5h4v14h-4z";
const PLAY_ICON = "M8 5v14l11-7z";
const SESSION_KEY = "epidemic-sandbox-session";
const ui = await (await fetch("/api/ui")).json();

const elements = {
  screens: {
    setup: document.getElementById("screen-setup"),
    runs: document.getElementById("screen-runs"),
    results: document.getElementById("screen-results"),
  },
  tabs: document.getElementById("tabs"),
  transport: document.getElementById("transport"),
  playIcon: document.getElementById("button-play-icon"),
  playLabel: document.getElementById("button-play-label"),
  play: document.getElementById("button-play"),
  stepDay: document.getElementById("button-step-day"),
  finish: document.getElementById("button-finish"),
  restart: document.getElementById("button-restart"),
  speed: document.getElementById("select-speed"),
  day: document.getElementById("clock-day"),
  step: document.getElementById("clock-step"),
  tooltip: document.getElementById("tooltip"),
  toast: document.getElementById("toast"),
};

let setupScreen = null;
let runsView = null;
let history = null;
const results = new ResultsView(elements.screens.results);
let startedDefinition = null;
let lastFrame = null;
let previewTimer = null;
let toastTimer = null;
let previewsInFlight = 0;
let currentScreen = "setup";
const pending = { start: false, step: false };

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (elements.toast.hidden = true), ui.toast_ms);
}

/** The name of this browser's experiment on the server; it outlives the page and the connection. */
function sessionToken() {
  try {
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) return stored;
  } catch {
    // storage may be blocked; the experiment is then lost with the page
  }
  const token = (crypto.randomUUID?.() ?? `${Math.random()}${Math.random()}`).replace(/[^A-Za-z0-9]/g, "");
  try {
    localStorage.setItem(SESSION_KEY, token);
  } catch {
    // see above
  }
  return token;
}

const send = connect({
  session: sessionToken(),
  reconnectDelayMs: ui.reconnect_delay_ms,
  onMessage(message) {
    switch (message.type) {
      case "hello": return setupScreen ? requestPreview() : buildSetup(message.setup);
      case "preview": return showPreview(message);
      case "started": return startRun(message);
      case "frame": return showFrame(message);
      case "watching": return runsView?.showWatching(message);
      case "person": return runsView?.showPerson(message);
      case "error": return showError(message.message);
    }
  },
  onConnectionChange(connected) {
    if (connected) return;
    forgetPending();
    showToast("Connection lost, reconnecting…");
  },
  onReplaced() {
    showToast("This experiment is open in another window; reload the page to bring it back here.");
  },
});

/** Send a command the user asked for; say so when it cannot be sent, instead of doing nothing. */
function command(message) {
  if (send(message)) return true;
  showToast("Not connected to the simulation, reconnecting…");
  return false;
}

/** Nothing is waited for any more (an answer came, failed or the connection was lost). */
function forgetPending() {
  pending.start = pending.step = false;
  previewsInFlight = 0;
  setupScreen?.setStarting(false);
  setupScreen?.setLoadingPreview(false);
  updateControls();
}

function showError(message) {
  pending.start = pending.step = false;
  setupScreen?.setStarting(false);
  updateControls();
  showToast(message);
}

function showPreview(message) {
  previewsInFlight = Math.max(0, previewsInFlight - 1);
  setupScreen.setLoadingPreview(previewsInFlight > 0);
  setupScreen.showPreview(message);
}

/** Each screen is reachable only when it makes sense, and moving between them never changes the run. */
function updateTabs() {
  const hasExperiment = runsView !== null;
  const rules = {
    setup: [!(hasExperiment && lastFrame?.playing), "Pause or finish the run first"],
    runs: [hasExperiment, "Start an experiment first"],
    results: [lastFrame?.finished ?? false, "Available once the experiment is finished"],
  };
  for (const button of elements.tabs.children) {
    const [enabled, reason] = rules[button.dataset.screen];
    button.disabled = !enabled;
    button.title = enabled ? "" : reason;
  }
}

/** Play, +1 day and Finish only make sense for a run that goes on; nothing is pressed twice. */
function updateControls() {
  const finished = lastFrame?.finished ?? false;
  for (const control of [elements.play, elements.stepDay, elements.finish]) {
    control.disabled = finished || pending.start;
  }
  elements.stepDay.disabled ||= pending.step;
  elements.restart.disabled = pending.start;
  setBusy(elements.stepDay, pending.step);
  setBusy(elements.restart, pending.start);
}

function showScreen(name) {
  currentScreen = name;
  for (const [key, screen] of Object.entries(elements.screens)) screen.hidden = key !== name;
  for (const button of elements.tabs.children) button.setAttribute("aria-pressed", String(button.dataset.screen === name));
  elements.transport.hidden = name === "setup";
  if (name !== "runs") elements.tooltip.hidden = true;
  if (name === "results") results.show(history);
}

function buildSetup(setup) {
  runsView?.destroy();
  runsView = lastFrame = history = null;
  setupScreen = null;
  forgetPending();
  setupScreen = new SetupScreen(elements.screens.setup, setup, {
    onChange() {
      clearTimeout(previewTimer);
      previewTimer = setTimeout(requestPreview, ui.preview_delay_ms);
    },
    start: () => startExperiment(setupScreen.definition()),
  });
  updateTabs();
  showScreen("setup");
  requestPreview();
}

function requestPreview() {
  if (!send({ type: "preview", experiment: setupScreen.definition() })) return;
  previewsInFlight += 1;
  setupScreen.setLoadingPreview(true);
}

function startExperiment(definition) {
  if (!command({ type: "start", experiment: definition })) return;
  startedDefinition = definition;
  pending.start = true;
  setupScreen.setStarting(true);
  updateControls();
}

/**
 * A run started, or (`resumed`) the server is catching this page up with an experiment that went
 * on without it: the page was closed, or the connection broke. The frame that follows carries the
 * whole history so far.
 */
function startRun({ run, resumed, experiment }) {
  const reconnected = resumed && runsView !== null;
  pending.start = false;
  setupScreen.setStarting(false);
  runsView?.destroy();
  lastFrame = null;
  if (resumed) startedDefinition = experiment;
  history = new History(run);
  runsView = new RunsView(elements.screens.runs, elements.tooltip, run, {
    inspect: (runIndex, person) => send({ type: "inspect", run: runIndex, person }),
    watch: (runIndex) => send({ type: "watch", run: runIndex }),
  });
  updateTabs();
  updateControls();
  showScreen(reconnected ? currentScreen : "runs");
}

function showFrame(frame) {
  try {
    drawFrame(frame);
  } finally {
    send({ type: "ack" });
  }
}

function drawFrame(frame) {
  if (!runsView) return;
  const isNewRun = !lastFrame || lastFrame.run_id !== frame.run_id;
  const justFinished = frame.finished && !lastFrame?.finished;
  history.push(frame);
  runsView.update(frame, history, isNewRun);
  elements.day.textContent = Math.floor(frame.day);
  const clock = `${WEEKDAYS[frame.weekday]} ${String(Math.floor(frame.hour)).padStart(2, "0")}:${String(Math.floor((frame.hour % 1) * 60)).padStart(2, "0")}`;
  elements.step.textContent = `${clock} · ${frame.step_ms.toFixed(1)} ms/step`;
  elements.playLabel.textContent = frame.playing ? "Pause" : "Play";
  elements.playIcon.setAttribute("d", frame.playing ? PAUSE_ICON : PLAY_ICON);
  if (isNewRun) elements.speed.value = String(frame.speed);
  lastFrame = frame;
  if (!frame.playing) pending.step = false;
  updateControls();
  updateTabs();
  if (justFinished) {
    showToast("The experiment is finished.");
    showScreen("results");
  }
}

function togglePlay() {
  command({ type: lastFrame?.playing ? "pause" : "play" });
}

elements.tabs.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (button && !button.disabled) showScreen(button.dataset.screen);
});
elements.play.addEventListener("click", togglePlay);
elements.stepDay.addEventListener("click", () => {
  if (!command({ type: "step_day" })) return;
  pending.step = true;
  updateControls();
});
elements.finish.addEventListener("click", () => command({ type: "finish" }));
elements.restart.addEventListener("click", () => startExperiment(startedDefinition));
elements.speed.addEventListener("change", () => command({ type: "speed", value: Number(elements.speed.value) }));
document.addEventListener("keydown", (event) => {
  const isTyping = event.target.matches("input, select, textarea, button, summary");
  const running = !elements.screens.runs.hidden && !lastFrame?.finished && !pending.start;
  if (event.code === "Space" && running && !isTyping) {
    event.preventDefault();
    togglePlay();
  }
});
