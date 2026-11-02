/** Keep a WebSocket to the simulation open; returns `send(command)`, which says whether it was sent. */
export function connect({ reconnectDelayMs, onMessage, onConnectionChange }) {
  let socket;

  function open() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${location.host}/ws`);
    socket.onopen = () => onConnectionChange(true);
    socket.onmessage = (event) => onMessage(JSON.parse(event.data));
    socket.onclose = () => {
      onConnectionChange(false);
      setTimeout(open, reconnectDelayMs);
    };
  }

  open();
  return (command) => {
    if (socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(command));
    return true;
  };
}
