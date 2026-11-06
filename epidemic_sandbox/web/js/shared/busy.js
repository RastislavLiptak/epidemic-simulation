/** Show that `button` waits for the server: a spinner replaces its icon. Disabling is up to the caller. */
export function setBusy(button, busy) {
  button.classList.toggle("button--busy", busy);
  button.setAttribute("aria-busy", String(busy));
}
