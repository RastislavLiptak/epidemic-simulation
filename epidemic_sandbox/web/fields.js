export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function decimalsOf(step) {
  return step < 1 ? String(step).split(".")[1].length : 0;
}

/** A labelled slider (or number box) with a hint; `onChange(value)` fires on every user edit. */
export function buildField(control, onChange) {
  const field = element("div", "field");
  const id = `field-${Math.random().toString(36).slice(2)}`;
  const head = element("div", "field__head");
  const label = element("label", "field__label", control.label);
  label.htmlFor = id;
  const output = element("output", "field__value");
  const input = element("input", "field__input");
  input.id = id;
  input.type = control.control === "number" ? "number" : "range";
  input.min = control.min;
  input.max = control.max;
  input.step = control.step;
  input.value = control.value;
  field.append(head, input, element("p", "field__hint", control.description));
  head.append(label, output);

  const read = () => Number(input.value);
  const showValue = () => {
    output.textContent = input.type === "number" ? "" : read().toFixed(decimalsOf(control.step));
  };
  input.addEventListener("input", () => {
    showValue();
    onChange(read());
  });
  showValue();
  return {
    element: field,
    read,
    write(value) {
      input.value = value;
      showValue();
    },
  };
}

/** A number box that may stay empty, which reads as 0 (the setting is off); `onChange(value)` on edits. */
export function buildOptionalNumber(control, onChange) {
  const field = element("div", "field");
  const id = `field-${Math.random().toString(36).slice(2)}`;
  const label = element("label", "field__label", control.label);
  label.htmlFor = id;
  const input = element("input", "field__input");
  input.id = id;
  input.type = "number";
  input.min = 1;
  input.max = control.max;
  input.step = control.step;
  input.placeholder = "no limit";
  input.value = control.value || "";
  field.append(label, input, element("p", "field__hint", control.description));
  const read = () => (input.value === "" ? 0 : Number(input.value));
  input.addEventListener("input", () => onChange(read()));
  return { element: field, read };
}

/** Badges with the estimates of one disease, from a preview entry. */
export function transmissionChips(estimate) {
  if (!estimate) return [];
  const contacts = element("span", "badge badge--muted", `${estimate.mean_contacts.toFixed(1)} contacts`);
  contacts.title = "Average number of people within this disease's contact distance over a week.";
  const household = element("span", "badge badge--muted", `household risk ${(100 * estimate.household_risk).toFixed(0)} %`);
  household.title = "Chance that an infectious person infects someone who stays next to them at home for the whole infectious period.";
  return [contacts, household];
}
