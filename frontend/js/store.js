// AMR Simulation Lab — client state store.
// Applies snapshot/diff frames (same rules as RefClient in tests/test_stream.py
// and the protocol in api/stream.py) and records what the renderer needs to
// animate: per-cell motion (from -> to), births, deaths, and discrete events.
(function (AMR) {
  const S = {
    state: null,          // same shape as GET /state
    bmap: new Map(),      // id -> bacterium dict (the objects inside state.bacteria)
    seq: null,
    selectedId: null,
    isPlaying: false,
    speed: 5,
    wsOpen: false,
    motion: new Map(),    // id -> {fx, fy, tx, ty, t0, dur, s0 (scale start)}
    ghosts: [],           // dying cells fading out: {b, x, y, t0}
    effects: [],          // {kind, t0, ...} discrete event animations
    history: { step: [], total: [], resistant: [], biofilm: [], persist: [], genePct: {} },
    cellHist: null,       // {id, step:[], fitness:[], energy:[], stress:[], damage:[]}
    listeners: [],
  };
  AMR.store = S;

  const MAX_HIST = 600, MAX_GHOSTS = 1200, MAX_EFFECTS = 400;

  S.on = fn => S.listeners.push(fn);
  const emit = (kind, info) => S.listeners.forEach(fn => { try { fn(kind, info); } catch (e) { console.error(e); } });

  // Motion duration: close to the time until the next frame, so cells glide
  // continuously while playing; a short tween for single manual steps.
  S.motionMs = () => S.isPlaying ? AMR.util.clamp(0.9 * AMR.playIntervalMs(S.speed), 80, 450) : 260;

  // Rendered (interpolated) grid position of a cell at time `now`.
  S.renderPos = (id, now) => {
    const m = S.motion.get(id);
    const b = S.bmap.get(id);
    if (!m) return b ? { x: b.pos[0], y: b.pos[1], s: 1 } : null;
    const t = m.dur > 0 ? AMR.util.clamp((now - m.t0) / m.dur, 0, 1) : 1;
    if (t >= 1) { S.motion.delete(id); return { x: m.tx, y: m.ty, s: 1 }; }   // finished: idle again
    const e = AMR.util.easeInOut(t);
    return { x: m.fx + (m.tx - m.fx) * e, y: m.fy + (m.ty - m.fy) * e, s: m.s0 + (1 - m.s0) * e };
  };

  function retarget(id, fromPos, toPos, now, dur, s0 = 1) {
    S.motion.set(id, { fx: fromPos.x, fy: fromPos.y, tx: toPos[0], ty: toPos[1], t0: now, dur, s0 });
  }

  function pushHistory(st) {
    const h = S.history, s = st.stats;
    if (h.step.length && h.step[h.step.length - 1] === s.step) {
      // same step (a command frame): overwrite the last point
      h.total[h.total.length - 1] = s.total_bacteria; h.resistant[h.resistant.length - 1] = s.resistant_bacteria;
      h.biofilm[h.biofilm.length - 1] = s.biofilm_bacteria; h.persist[h.persist.length - 1] = s.persister_cells;
    } else {
      h.step.push(s.step); h.total.push(s.total_bacteria); h.resistant.push(s.resistant_bacteria);
      h.biofilm.push(s.biofilm_bacteria); h.persist.push(s.persister_cells);
    }
    const n = h.step.length, tot = Math.max(1, s.total_bacteria);
    const dist = s.gene_distribution || {};
    for (const g of Object.keys(dist)) if (!h.genePct[g]) h.genePct[g] = new Array(n - 1).fill(0);
    for (const g of Object.keys(h.genePct)) {
      const arr = h.genePct[g], v = 100 * (dist[g] || 0) / tot;
      if (arr.length === n) arr[n - 1] = v; else arr.push(v);
    }
    if (n > MAX_HIST) {
      for (const k of ['step', 'total', 'resistant', 'biofilm', 'persist']) h[k].shift();
      for (const g of Object.keys(h.genePct)) h.genePct[g].shift();
    }
  }

  function recordCell() {
    if (S.selectedId === null) return;
    const b = S.bmap.get(S.selectedId);
    if (!b) return;
    if (!S.cellHist || S.cellHist.id !== b.id) S.cellHist = { id: b.id, step: [], fitness: [], energy: [], stress: [], damage: [] };
    const c = S.cellHist, step = S.state.stats.step;
    if (c.step.length && c.step[c.step.length - 1] === step) return;
    c.step.push(step); c.fitness.push(b.fitness); c.energy.push(b.energy);
    c.stress.push(b.stress_level); c.damage.push(b.antibiotic_damage);
    if (c.step.length > 200) for (const k of ['step', 'fitness', 'energy', 'stress', 'damage']) c[k].shift();
  }

  S.resetLocal = () => {
    S.motion.clear(); S.ghosts = []; S.effects = []; S.cellHist = null; S.selectedId = null;
    S.history = { step: [], total: [], resistant: [], biofilm: [], persist: [], genePct: {} };
  };

  S.applySnapshot = f => {
    S.seq = f.seq;
    S.state = f.state;
    S.bmap = new Map(f.state.bacteria.map(b => [b.id, b]));
    S.motion.clear(); S.ghosts = [];
    pushHistory(S.state); recordCell();
    emit('snapshot', f);
  };

  // Returns false if the frame reveals a gap/mismatch (caller must resync).
  S.applyDiff = f => {
    if (S.seq === null || f.seq <= S.seq) return true;          // stale: ignore
    if (f.base_seq !== S.seq) return false;                      // gap
    const now = performance.now(), dur = S.motionMs();
    for (const id of f.removed) {
      const b = S.bmap.get(id);
      if (b) {
        const p = S.renderPos(id, now) || { x: b.pos[0], y: b.pos[1] };
        if (S.ghosts.length < MAX_GHOSTS) S.ghosts.push({ b, x: p.x, y: p.y, t0: now });
      }
      S.bmap.delete(id); S.motion.delete(id);
    }
    if (f.stepped) for (const b of S.bmap.values()) b.age += 1;   // age only sent when it didn't advance by 1
    for (const [id, d] of f.changed) {
      const b = S.bmap.get(id);
      if (!b) continue;
      if (d.pos) retarget(id, S.renderPos(id, now) || { x: b.pos[0], y: b.pos[1] }, d.pos, now, dur);
      Object.assign(b, d);
    }
    for (const b of f.added) {
      S.bmap.set(b.id, b);
      const parent = b.parent_id !== null && b.parent_id !== undefined ? S.renderPos(b.parent_id, now) : null;
      if (parent) retarget(b.id, parent, b.pos, now, dur, 0.35);          // division: glide out of the parent
      else retarget(b.id, { x: b.pos[0], y: b.pos[1] }, b.pos, now, 320, 0); // spawn: grow in place
    }
    for (const [id, d] of Object.entries(f.detail)) { const b = S.bmap.get(+id); if (b) Object.assign(b, d); }

    const st = S.state;
    st.stats = f.stats;
    st.event_log = st.event_log.concat(f.log).slice(-20);
    const hgt = f.events.filter(e => e.kind === 'hgt');
    st.hgt_events = st.hgt_events.concat(hgt.map(e => ({ step: e.step, donor: e.donor, recipient: e.recipient, gene: e.gene, pos: e.pos }))).slice(-50);
    for (const [k, p] of Object.entries(f.heatmaps)) st.antibiotic_heatmaps[k] = { data: AMR.stream.unpackGrid(p), max: p.max, color: p.color, name: p.name };
    if (f.nutrient) st.nutrient_heatmap = AMR.stream.unpackGrid(f.nutrient);
    st.bacteria = [...S.bmap.values()];
    if (st.bacteria.length !== f.n_bacteria) return false;
    S.seq = f.seq;

    // Discrete event animations
    for (const e of f.events) {
      if (S.effects.length >= MAX_EFFECTS) break;
      if (e.kind === 'hgt') S.effects.push({ kind: 'hgt', t0: now, donor: e.donor, recipient: e.recipient, pos: e.pos, gene: e.gene });
      else if (e.kind === 'sos_on' || e.kind === 'biofilm_on' || e.kind === 'persister_on') S.effects.push({ kind: e.kind, t0: now, id: e.id, pos: e.pos });
    }
    pushHistory(st); recordCell();
    emit('diff', f);
    return true;
  };

  // Polling fallback: whole state; animate cells that moved.
  S.applyPolled = state => {
    const now = performance.now(), dur = 600;
    const old = S.bmap;
    const next = new Map(state.bacteria.map(b => [b.id, b]));
    for (const [id, b] of next) {
      const o = old.get(id);
      if (o && (o.pos[0] !== b.pos[0] || o.pos[1] !== b.pos[1]))
        retarget(id, S.renderPos(id, now) || { x: o.pos[0], y: o.pos[1] }, b.pos, now, dur);
    }
    for (const [id, o] of old) if (!next.has(id) && S.ghosts.length < MAX_GHOSTS) S.ghosts.push({ b: o, x: o.pos[0], y: o.pos[1], t0: now });
    const newHgt = (state.hgt_events || []).filter(e => !S.state || e.step > (S.state.stats.step || -1));
    for (const e of newHgt.slice(-20)) S.effects.push({ kind: 'hgt', t0: now, donor: e.donor, recipient: e.recipient, pos: e.pos, gene: e.gene });
    S.state = state; S.bmap = next; S.seq = null;
    pushHistory(state); recordCell();
    emit('poll', state);
  };
})(window.AMR);
