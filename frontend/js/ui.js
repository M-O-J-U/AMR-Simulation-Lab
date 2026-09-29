// AMR Simulation Lab — controls, panels and canvas interaction.
(function (AMR) {
  const S = AMR.store, U = AMR.util, $ = id => document.getElementById(id);
  const UI = AMR.ui = {};
  let maxHGT = 1, lastLogStep = -1, lastUI = 0, uiPending = false;
  const faint = t => `<div style="font-size:9px;color:var(--text-faint);font-style:italic">${t}</div>`;

  // ── status / play ─────────────────────────────────────────────────────────
  UI.applyStatus = st => {
    S.isPlaying = !!st.playing;
    if (st.speed !== undefined) {
      S.speed = st.speed;
      const r = $('rng-speed');
      if (document.activeElement !== r) { r.value = st.speed; $('val-speed').textContent = st.speed; }
    }
    const btn = $('btn-play');
    if (S.isPlaying) { btn.textContent = '⏸ Pause'; btn.className = 'topbar-btn tbtn-reset'; }
    else { btn.textContent = '▶ Play'; btn.className = 'topbar-btn tbtn-play'; }
    UI.schedule();
  };

  UI.togglePlay = async () => {
    const want = !S.isPlaying;
    try { await (want ? AMR.api.resume() : AMR.api.pause()); if (!S.wsOpen) UI.applyStatus({ playing: want }); } catch (e) {}
  };
  let speedTimer = null;
  UI.setSpeed = v => { clearTimeout(speedTimer); speedTimer = setTimeout(() => AMR.api.speed(parseInt(v)).catch(() => {}), 120); };
  UI.stepOnce = async () => { try { await AMR.api.step(1); AMR.stream.refreshIfPolling(); } catch (e) {} };

  // ── commands ──────────────────────────────────────────────────────────────
  UI.applyAB = async () => {
    const key = $('sel-ab').value, conc = parseFloat($('rng-dose').value) / 10, mode = $('sel-mode').value;
    // "Spot - concentrated in center": the model needs a centre for spot mode.
    const center = mode === 'spot' && S.state ? [Math.floor(S.state.grid_width / 2), Math.floor(S.state.grid_height / 2)] : null;
    try { await AMR.api.applyAntibiotic(key, conc, mode, center, center ? 10 : null); AMR.stream.refreshIfPolling(); } catch (e) {}
  };
  UI.clearAllAB = async () => {
    for (const k of AMR.ANTIBIOTICS) { try { await AMR.api.removeAntibiotic(k); } catch (e) {} }
    AMR.stream.refreshIfPolling();
  };
  UI.spawn = async () => {
    try { await AMR.api.spawn($('sel-spawn').value, parseInt($('rng-spawn').value)); AMR.stream.refreshIfPolling(); } catch (e) {}
  };
  UI.reset = async () => {
    try {
      await AMR.api.reset($('sel-scenario').value, parseInt($('rng-pop').value));
      // With the stream up, the 'reset' snapshot calls onReset() in every open client.
      if (!S.wsOpen) { S.resetLocal(); UI.onReset(); AMR.stream.refreshIfPolling(); }
    } catch (e) {}
  };
  UI.onReset = () => {
    maxHGT = 1; lastLogStep = -1;
    $('event-scroll').innerHTML = '';
    AMR.inspector.clear();
    $('gnn-predict-content').innerHTML = faint('Click "Run Prediction" to see transfer predictions');
    $('advisory-content').innerHTML = faint('Click "Treatment Advisory" for recommendations');
    $('analytics-content').innerHTML = faint('Run an analysis above');
    AMR.charts.reset();
  };

  UI.setMode = (mode, el) => {
    AMR.render.setMode(mode);
    document.querySelectorAll('#chip-row .chip').forEach(c => c.classList.toggle('active', c === el));
    updateLegend();
  };

  // ── panels ────────────────────────────────────────────────────────────────
  function updateLegend() {
    const legend = $('canvas-legend'), data = AMR.LEGENDS[AMR.render.mode];
    if (!data) { legend.classList.remove('visible'); return; }
    legend.classList.add('visible');
    $('legend-title').textContent = data.title;
    $('legend-items').innerHTML = data.items.map(i => `<div class="legend-item"><div class="legend-dot" style="background:${i.color}"></div><div class="legend-lbl">${i.label}</div></div>`).join('');
  }

  function updatePanels() {
    uiPending = false; lastUI = performance.now();
    const state = S.state; if (!state) return;
    const s = state.stats;
    $('step-display').textContent = `Step ${s.step}`;
    const bio = state.biology, sd = $('scenario-display');
    sd.textContent = (s.scenario || '-') + (bio ? ` · ${bio.name}${bio.warnings && bio.warnings.length ? ' ⚠' : ''}` : '');
    sd.title = bio ? `Biology: ${bio.name}` + (bio.warnings && bio.warnings.length ? ' | ' + bio.warnings.join(' | ') : '') : '';
    sd.style.color = bio && bio.warnings && bio.warnings.length ? 'var(--amber)' : 'var(--text-dim)';
    const rPct = s.total_bacteria > 0 ? Math.round(s.resistant_bacteria / s.total_bacteria * 100) : 0;
    $('lbl-pop').innerHTML = `<span>ALIVE</span>${s.total_bacteria}`;
    $('lbl-res').innerHTML = `<span>RESISTANT</span>${s.resistant_bacteria} (${rPct}%)`;
    $('lbl-hgt').innerHTML = `<span>HGT EVENTS</span>${s.hgt_total}`;
    const pill = $('status-pill');
    if (s.total_bacteria === 0) { pill.className = 'pill pill-dead'; pill.textContent = 'EXTINCT'; }
    else if (S.isPlaying) { pill.className = 'pill pill-run'; pill.innerHTML = '&nbsp;RUNNING'; }
    else { pill.className = 'pill pill-pause'; pill.textContent = 'PAUSED'; }
    $('st-total').textContent = s.total_bacteria;
    $('st-resist').textContent = s.resistant_bacteria;
    $('st-resist-pct').textContent = s.total_bacteria > 0 ? `${rPct}% of population` : '-';
    $('st-biofilm').textContent = s.biofilm_bacteria;
    $('st-persist').textContent = s.persister_cells;
    $('st-fit').textContent = s.avg_fitness.toFixed(3);
    $('st-genes').textContent = s.avg_resistance_genes.toFixed(2);
    $('st-hgt').textContent = s.hgt_total;
    $('bar-fit').style.width = (s.avg_fitness * 100) + '%';
    $('bar-genes').style.width = (Math.min(s.avg_resistance_genes / 5, 1) * 100) + '%';
    maxHGT = Math.max(maxHGT, s.hgt_total, 1);
    $('bar-hgt').style.width = (s.hgt_total / maxHGT * 100) + '%';

    const sp = s.species_counts || {};
    $('species-list').innerHTML = Object.keys(sp).length === 0 ? '<div class="ab-none">No bacteria present</div>'
      : Object.entries(sp).map(([name, cnt]) => {
        const meta = AMR.SPECIES[name] || { color: '#aaa', who: 'HIGH' };
        return `<div class="species-row"><div class="sp-dot" style="background:${meta.color}"></div><div><div class="sp-name" style="font-style:italic">${U.esc(U.shortSpecies(name))}</div><div class="sp-who ${meta.who === 'CRITICAL' ? 'who-crit' : 'who-high'}">WHO ${meta.who}</div></div><div class="sp-cnt" style="color:${meta.color}">${cnt}</div></div>`;
      }).join('');

    const ab = state.antibiotic_heatmaps || {};
    $('ab-list').innerHTML = Object.keys(ab).length === 0 ? '<div class="ab-none">None - bacteria are untreated</div>'
      : Object.values(ab).map(hm => `<div class="ab-row"><div class="ab-swatch" style="background:${hm.color}"></div><div><div class="ab-name">${U.esc(hm.name)}</div><div class="ab-class">Max: ${(hm.max || 0).toFixed(2)} µg/mL</div></div></div>`).join('');

    const ev = $('event-scroll');
    const fresh = (state.event_log || []).filter(e => e.step > lastLogStep);
    for (const e of fresh) {
      const cls = e.type === 'hgt_burst' ? 'ev-hgt' : e.type.includes('antibiotic') ? 'ev-ab' : e.type === 'extinction' ? 'ev-extinct' : e.type === 'simulation_start' ? 'ev-milestone' : 'ev-science';
      const d = document.createElement('div'); d.className = `ev-line ${cls}`;
      d.innerHTML = `<span class="ev-step">${String(e.step).padStart(4, '0')}</span><div class="ev-dot"></div><span class="ev-text">${U.esc(e.message)}</span>`;
      ev.appendChild(d);
    }
    if (fresh.length) { lastLogStep = fresh[fresh.length - 1].step; ev.scrollTop = ev.scrollHeight; while (ev.children.length > 300) ev.removeChild(ev.firstChild); }
    AMR.inspector.refresh();
  }
  // DOM panels at most ~6x/s; the canvas animates independently.
  UI.schedule = () => {
    if (uiPending) return; uiPending = true;
    setTimeout(() => requestAnimationFrame(updatePanels), Math.max(0, 160 - (performance.now() - lastUI)));
  };

  // ── GNN + analytics panels (unchanged behaviour; MIC now covers every drug) ─
  UI.checkGNN = async () => {
    const badge = $('gnn-status-badge');
    try {
      const d = await AMR.api.gnnStatus();
      if (d.model_ready && d.trained) { badge.className = 'gnn-badge gnn-ready'; badge.textContent = `Model ready - ${d.n_predictions_run} predictions run`; }
      else if (d.model_ready) { badge.className = 'gnn-badge gnn-nomodel'; badge.textContent = 'Untrained - run train-gnn first'; }
      else { badge.className = 'gnn-badge gnn-nomodel'; badge.textContent = 'Not loaded'; }
    } catch (e) { badge.className = 'gnn-badge gnn-nomodel'; badge.textContent = 'Server offline'; }
  };

  UI.gnnPredict = async () => {
    const btn = $('btn-gnn-predict'), el = $('gnn-predict-content');
    btn.textContent = 'Running...'; btn.disabled = true;
    el.innerHTML = '<div style="font-size:9px;color:var(--cyan);font-family:var(--mono)">Running GNN inference...</div>';
    try {
      const d = await AMR.api.gnnPredict();
      if (d.error) { el.innerHTML = `<div style="color:var(--red);font-size:9px">${U.esc(d.error)}</div>`; return; }
      const sorted = Object.entries(d.gene_transfer_probs || {}).sort((a, b) => b[1] - a[1]), maxP = (sorted[0] && sorted[0][1]) || 1;
      el.innerHTML = `<div style="font-size:9px;color:var(--text-dim);margin-bottom:6px">${d.n_nodes} cells analyzed | ${d.n_edges} edges | ${d.n_predicted_transfers} predicted transfers | ${d.inference_time_ms.toFixed(0)}ms</div>` +
        sorted.map(([g, p]) => `<div class="gnn-gene-row"><div class="gnn-gene-lbl" title="${U.esc(g)}">${U.esc(g)}</div><div class="gnn-bar-bg"><div class="gnn-bar-fill" style="width:${Math.round(p / maxP * 100)}%;background:${p > 0.05 ? 'var(--red)' : p > 0.02 ? 'var(--amber)' : 'var(--teal)'}"></div></div><div class="gnn-prob">${(p * 100).toFixed(2)}%</div></div>`).join('');
      UI.checkGNN();
    } catch (e) { el.innerHTML = `<div style="color:var(--red);font-size:9px">Failed: ${U.esc(e.message)}</div>`; }
    finally { btn.textContent = 'Run Prediction'; btn.disabled = false; }
  };

  UI.gnnAdvisory = async () => {
    const btn = $('btn-gnn-advisory'), el = $('advisory-content');
    btn.textContent = 'Analysing...'; btn.disabled = true;
    el.innerHTML = '<div style="font-size:9px;color:var(--cyan);font-family:var(--mono)">Running advisory...</div>';
    try {
      const d = await AMR.api.gnnAdvisory();
      if (d.error) { el.innerHTML = `<div style="color:var(--red);font-size:9px">${U.esc(d.error)}</div>`; return; }
      const risk = d.risk_level || 'UNKNOWN';
      const list = (a, none) => (a && a.length ? a.join(', ') : none);
      const recs = (d.recommendations || []).slice(0, 2);
      el.innerHTML = `<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px"><span class="risk-badge risk-${risk}">${risk} RISK</span><span style="font-size:9px;color:var(--text-dim)">${d.n_high_risk_cells || 0} high-risk cells</span></div>
        <div style="font-size:9px;margin-bottom:4px"><span style="color:var(--text-dim)">Imminent genes: </span><span style="color:var(--purple);font-family:var(--mono)">${U.esc(list(d.imminent_resistance, 'none'))}</span></div>
        <div style="font-size:9px;margin-bottom:4px"><span style="color:var(--text-dim)">Threatened drugs: </span><span style="color:var(--red);font-family:var(--mono)">${U.esc(list(d.threatened_antibiotics, 'none'))}</span></div>
        <div style="font-size:9px;margin-bottom:6px"><span style="color:var(--text-dim)">Effective now: </span><span style="color:var(--green);font-family:var(--mono)">${U.esc(list(d.safe_antibiotics, 'none of the available antibiotics'))}</span></div>
        ${recs.length ? '<div style="font-size:8px;color:var(--text-faint);margin-bottom:4px;letter-spacing:.08em;text-transform:uppercase">Top recommendations</div>' : ''}
        ${recs.map(r => `<div style="font-size:9px;padding:3px 0;border-bottom:1px solid var(--border)"><span style="color:var(--cyan);font-family:var(--mono)">${U.esc(r.antibiotic)}</span> <span style="color:var(--text-faint)">${(r.pct_susceptible || 0).toFixed(0)}% susceptible</span></div>`).join('')}
        <div class="advisory-text">${U.esc(d.advisory_text || '')}</div>`;
    } catch (e) { el.innerHTML = `<div style="color:var(--red);font-size:9px">Failed: ${U.esc(e.message)}</div>`; }
    finally { btn.textContent = 'Treatment Advisory'; btn.disabled = false; }
  };

  UI.micAnalysis = async () => {
    const el = $('analytics-content');
    el.innerHTML = '<div style="font-size:9px;color:var(--cyan);font-family:var(--mono)">Running MIC analysis...</div>';
    try {
      const d = await AMR.api.micAll();
      const rows = Object.entries(d.antibiotics || {});
      if (!rows.length) { el.innerHTML = '<div style="color:var(--amber);font-size:9px">No data - start simulation first</div>'; return; }
      el.innerHTML = `<div style="font-size:9px;color:var(--text-dim);margin-bottom:6px">Model-estimated MIC category per drug (EUCAST S/R breakpoints) - population ${d.population}</div>` +
        rows.map(([k, m]) => `<div class="mic-row"><span style="font-family:var(--mono);color:var(--text-dim)">${U.esc(m.antibiotic)}</span>
          <div class="mic-bar" title="S ${m.susceptible_pct}% / I ${m.intermediate_pct}% / R ${m.resistant_pct}%  (S≤${m.breakpoint_S}, R>${m.breakpoint_R} µg/mL)">
            <span style="width:${m.susceptible_pct}%;background:var(--green)"></span><span style="width:${m.intermediate_pct}%;background:var(--amber)"></span><span style="width:${m.resistant_pct}%;background:var(--red)"></span></div>
          <span style="font-family:var(--mono);color:var(--red);text-align:right">${m.resistant_pct.toFixed(0)}%R</span></div>`).join('') +
        '<div style="font-size:8px;color:var(--text-faint);margin-top:4px">Green S · amber I · red R. Estimated from carried genes and EUCAST expected phenotypes; not measured MICs.</div>';
    } catch (e) { el.innerHTML = `<div style="color:var(--red);font-size:9px">Failed: ${U.esc(e.message)}</div>`; }
  };

  UI.diversity = async () => {
    const el = $('analytics-content');
    el.innerHTML = '<div style="font-size:9px;color:var(--cyan);font-family:var(--mono)">Computing diversity...</div>';
    try {
      const d = await AMR.api.diversity(), h = d.shannon_diversity || 0;
      const col = h > 2 ? 'var(--red)' : h > 1 ? 'var(--amber)' : 'var(--green)';
      el.innerHTML = `<div style="font-size:9px;color:var(--text-dim);margin-bottom:8px">Shannon Diversity Index</div>
        <div style="text-align:center;margin-bottom:8px"><div style="font-family:var(--mono);font-size:28px;font-weight:600;color:${col}">${h.toFixed(4)}</div><div style="font-size:8px;color:var(--text-dim)">H (population: ${d.population})</div></div>
        <div style="height:6px;background:var(--bg0);border-radius:3px;overflow:hidden;margin-bottom:6px"><div style="height:100%;width:${Math.min(h / 3 * 100, 100)}%;background:${col};border-radius:3px;transition:width .5s"></div></div>
        <div style="font-size:9px;color:var(--text-dim)">${U.esc(d.interpretation)}</div>`;
    } catch (e) { el.innerHTML = `<div style="color:var(--red);font-size:9px">Failed: ${U.esc(e.message)}</div>`; }
  };

  UI.recommend = async () => {
    const el = $('analytics-content');
    el.innerHTML = '<div style="font-size:9px;color:var(--cyan);font-family:var(--mono)">Computing recommendations...</div>';
    try {
      const recs = ((await AMR.api.recommend()).recommendations || []).slice(0, 5);
      if (!recs.length) { el.innerHTML = '<div style="color:var(--amber);font-size:9px">No bacteria to analyse</div>'; return; }
      el.innerHTML = '<div style="font-size:9px;color:var(--text-dim);margin-bottom:6px">Treatment Recommendations (by efficacy score)</div>' +
        recs.map((r, i) => { const col = i === 0 ? 'var(--green)' : i === 1 ? 'var(--cyan)' : 'var(--teal)';
          return `<div style="margin-bottom:6px"><div style="display:flex;justify-content:space-between;font-size:9px;margin-bottom:2px"><span style="color:${col};font-family:var(--mono)">${U.esc(r.antibiotic)}</span><span style="color:var(--text-dim)">${(r.pct_susceptible || 0).toFixed(0)}% susceptible</span></div>
            <div style="height:5px;background:var(--bg0);border-radius:3px;overflow:hidden"><div style="height:100%;width:${Math.min(Math.round(r.score || 0), 100)}%;background:${col};border-radius:3px"></div></div>
            <div style="font-size:8px;color:var(--text-faint);margin-top:2px">${U.esc(r.rationale || '')}</div></div>`; }).join('');
    } catch (e) { el.innerHTML = `<div style="color:var(--red);font-size:9px">Failed: ${U.esc(e.message)}</div>`; }
  };

  // ── canvas interaction ────────────────────────────────────────────────────
  UI.initCanvas = canvas => {
    let hoverRaf = null;
    canvas.addEventListener('mousemove', e => {
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(() => {
        hoverRaf = null;
        const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
        AMR.inspector.tooltip(AMR.render.hitTest(x, y), x, y, canvas);
      });
    });
    canvas.addEventListener('mouseleave', () => { $('tip').style.display = 'none'; });
    canvas.addEventListener('click', e => {
      const r = canvas.getBoundingClientRect(), b = AMR.render.hitTest(e.clientX - r.left, e.clientY - r.top);
      if (b) AMR.inspector.select(b.id);
    });
  };

  UI.init = () => {
    S.on(() => UI.schedule());
    document.querySelectorAll('[data-toggle]').forEach(t => t.addEventListener('change', () => {
      AMR.render.toggles[t.dataset.toggle] = t.checked; AMR.render.markDirty();
    }));
    updateLegend();
  };
})(window.AMR);
