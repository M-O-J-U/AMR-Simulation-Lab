// AMR Simulation Lab — live stream client (WS /ws; protocol in api/stream.py).
// Snapshot on connect, then one diff per step. While disconnected: poll GET
// /state every 2.5 s and reconnect with backoff (0.5 s -> 5 s).
(function (AMR) {
  const S = AMR.store;
  let ws = null, pollTimer = null, reconnectDelay = 500;

  function setConn(text, color) {
    const el = document.getElementById('conn-display');
    if (el) { el.textContent = text; el.style.color = color; }
  }

  function unpackGrid(p) {
    const raw = atob(p.b64), out = new Array(p.w);
    for (let x = 0; x < p.w; x++) {
      const col = new Array(p.h);
      for (let y = 0; y < p.h; y++) col[y] = raw.charCodeAt(x * p.h + y) / 255 * p.scale;
      out[x] = col;
    }
    return out;
  }

  function send(msg) { if (ws && S.wsOpen) ws.send(JSON.stringify(msg)); }
  function resync() { S.seq = null; send({ type: 'resync' }); }

  function handle(f) {
    if (f.type === 'snapshot') {
      if (f.reason === 'reset') { S.resetLocal(); AMR.ui && AMR.ui.onReset(); }
      S.applySnapshot(f);
      if (f.status) AMR.ui.applyStatus(f.status);
    } else if (f.type === 'diff') {
      if (!S.applyDiff(f)) resync();
    } else if (f.type === 'status') {
      AMR.ui.applyStatus(f);
    } else if (f.type === 'detail') {
      const b = f.bacterium && S.bmap.get(f.id);
      if (b) AMR.DETAIL_FIELDS.forEach(k => { b[k] = f.bacterium[k]; });
    }
  }

  async function poll() {
    try {
      S.applyPolled(await AMR.api.state());
    } catch (e) {
      const el = document.getElementById('lbl-pop');
      if (el) el.innerHTML = '<span>STATUS</span>Cannot reach server';
    }
  }
  function startPolling() { if (!pollTimer) { poll(); pollTimer = setInterval(poll, 2500); } }
  function stopPolling() { clearInterval(pollTimer); pollTimer = null; }

  function down() {
    S.wsOpen = false; ws = null; S.seq = null;
    setConn('POLLING', 'var(--amber)');
    startPolling();
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(5000, reconnectDelay * 2);
  }

  function connect() {
    try { ws = new WebSocket(AMR.WS_URL); } catch (e) { down(); return; }
    ws.onopen = () => {
      S.wsOpen = true; reconnectDelay = 500; stopPolling(); setConn('LIVE', 'var(--green)');
      if (S.selectedId !== null) send({ type: 'inspect', id: S.selectedId });
    };
    ws.onmessage = e => handle(JSON.parse(e.data));
    ws.onclose = down;
  }

  AMR.stream = {
    connect, send, unpackGrid,
    inspect: id => send({ type: 'inspect', id }),
    refreshIfPolling: () => { if (!S.wsOpen) poll(); },
  };
})(window.AMR);
