import { element } from "./fields.js";

function stateColor(diseases, disease, state) {
  const states = diseases.find((entry) => entry.name === disease)?.states;
  return states?.find((entry) => entry.name === state)?.color ?? "#999";
}

function healthDot(diseases, disease, state) {
  const dot = element("span", "counts__dot");
  dot.style.background = stateColor(diseases, disease, state);
  dot.title = `${disease}: ${state}`;
  return dot;
}

function healthDots(diseases, health) {
  const dots = element("span", "person__dots");
  for (const [disease, state] of Object.entries(health)) dots.append(healthDot(diseases, disease, state));
  return dots;
}

function line(label, ...content) {
  const row = element("p", "person__line");
  const value = element("span", "person__value");
  value.append(...content);
  row.append(element("span", "person__label", label), value);
  return row;
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

function peopleList(items, diseases, describe) {
  const list = element("p", "person__members");
  for (const item of items) {
    const entry = element("span", "person__member", `${describe(item)} `);
    entry.append(healthDots(diseases, item.health));
    list.append(entry);
  }
  return list;
}

function healthLine(diseases, disease, health) {
  const row = line(disease, healthDot(diseases, disease, health.state), ` ${health.state}`);
  if (health.days_left !== null) row.append(` · ${health.days_left.toFixed(1)} days left`);
  row.append(` · susceptibility ×${health.susceptibility.toFixed(2)}`);
  if (health.infectious) row.append(` · infectiousness ×${health.infectiousness.toFixed(2)}`);
  if (health.fatality_risk > 0) row.append(` · fatality ${(100 * health.fatality_risk).toFixed(1)} %`);
  return row;
}

/**
 * Card with what is known about one person. `info` is the server's description, or the basic
 * local one (no `now`) while the answer is on its way.
 */
export function renderPerson(info, diseases) {
  const card = element("div", "person");
  const title = element("p", "person__title");
  const sex = info.sex ? ` · ${info.sex}` : "";
  title.append(element("strong", "", `Person #${info.id}`), ` · ${info.age} years${sex} · ${info.role}`);
  card.append(title);
  if (!info.now) {
    for (const [disease, health] of Object.entries(info.health)) {
      card.append(line(disease, healthDot(diseases, disease, health.state), ` ${health.state}`));
    }
    return card;
  }

  card.append(
    element("p", "person__activity", `${info.now.activity}${info.now.arrives ? ` (arrives ${info.now.arrives})` : ""}`),
    element("p", "person__reason", info.now.reason),
  );
  for (const [disease, health] of Object.entries(info.health)) card.append(healthLine(diseases, disease, health));

  const members = info.household.members.filter((member) => member.id !== info.id);
  card.append(line("household", plural(info.household.members.length, "person")));
  if (members.length) {
    card.append(peopleList(members, diseases, (member) => `${member.age} · ${member.role}`));
  }
  if (info.place) {
    card.append(
      line(
        info.place.kind,
        `${plural(info.place.people, "person")}, ${(info.place.distance_m / 1000).toFixed(2)} km from home`,
      ),
    );
  }
  const traits = info.traits;
  card.append(
    line(
      "traits",
      `${traits.speed_kmh.toFixed(1)} km/h · sociability ×${traits.sociability.toFixed(2)} · ` +
        `usually ${traits.usual_start_shift_h >= 0 ? "+" : ""}${traits.usual_start_shift_h.toFixed(1)} h at the start of the day`,
    ),
    line("today", info.timetable.map((entry) => `${entry.at ?? "—"} ${entry.what}`).join(" · ") || "nothing planned"),
    line("nearby", `${plural(info.nearby.count, "person")} within contact distance`),
  );
  if (info.nearby.people.length) {
    card.append(peopleList(info.nearby.people, diseases, (other) => `${other.age} · ${other.distance_m.toFixed(1)} m`));
  }
  card.append(element("p", "person__hint", info.clock));
  return card;
}

/** What is known about a person without asking the server: age, role and current health. */
export function basicPerson(person, run, people, states) {
  const health = {};
  for (const disease of run.diseases) {
    health[disease.name] = { state: disease.states[states[disease.name][person]].name };
  }
  return { id: person, age: people.ages[person], role: run.role_names[people.roles[person]], health, now: null };
}
