/** The server closes a connection with this code when another window took over the experiment. */
const OPENED_ELSEWHERE = 4001;

/**
 * Keep a WebSocket to the server open; returns `send(command)`, which says whether it was sent.
 * The `session` names the experiment: a connection that reuses it continues that experiment,
 * whether the connection broke or the page was closed and opened again.
 */
export function connect({ session, reconnectDelayMs, onMessage, onConnectionChange, onReplaced }) {
  let socket;

  function open() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${location.host}/ws?session=${session}`);
    socket.onopen = () => onConnectionChange(true);
    socket.onmessage = (event) => onMessage(JSON.parse(event.data));
    socket.onclose = (event) => {
      onConnectionChange(false);
      if (event.code === OPENED_ELSEWHERE) onReplaced();
      else setTimeout(open, reconnectDelayMs);
    };
  }

  open();
  return (command) => {
    if (socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(command));
    return true;
  };
}
