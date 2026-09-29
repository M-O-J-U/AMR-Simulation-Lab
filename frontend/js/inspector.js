// AMR Simulation Lab — selected-cell inspector (live) and hover tooltip.
(function (AMR) {
  const S = AMR.store, U = AMR.util;
  const I = AMR.inspector = { ref: null };
  let lastRender = 0, lastGone = null;

  const stateColors = { growing: 'var(--green)', stressed: 'var(--red)', biofilm: 'var(--amber)', dormant: 'var(--text-dim)', dying: 'var(--red)' };

  function geneTag(g) {
    const r = I.ref && I.ref.genes[g];
    const tip = r ? `${g} — ${r.mechanism}; resists: ${r.drug_classes.join(', ')}; CARD ${r.card_id}${r.transferable ? '' : '; not transferable in this model'}` : g;
    return `<span class="insp-gene-tag" title="${U.esc(tip)}">${U.esc(g)}</span>`;
  }

  function bar(label, v, color) {
    const pct = Math.round(U.clamp(v || 0, 0, 1) * 100);
    return `<span class="k">${label}</span><div class="insp-bar"><div style="width:${pct}%;background:${color}"></div></div><span class="v">${(v ?? 0).toFixed(3)}</span>`;
  }

  function sparkline(canvas) {
    const h = S.cellHist; if (!canvas || !h || h.step.length < 2) return;
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, ht = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = ht * dpr;
    const c = canvas.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, w, ht);
    const n = h.step.length;
    const line = (arr, color) => {
      c.beginPath();
      arr.forEach((v, i) => { const x = i / (n - 1) * w, y = ht - 2 - U.clamp(v || 0, 0, 1) * (ht - 4); i ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.strokeStyle = color; c.lineWidth = 1.2; c.stroke();
    };
    line(h.fitness, '#00e87a'); line(h.energy, '#20b4ff'); line(h.stress, '#ff3a5c'); line(h.damage, '#ffb020');
  }

  I.render = b => {
    const el = document.getElementById('inspector');
    const sc = stateColors[b.state] || 'var(--text-dim)';
    const parentAlive = b.parent_id !== null && b.parent_id !== undefined && S.bmap.has(b.parent_id);
    const parent = b.parent_id === null || b.parent_id === undefined ? 'founder / spawned'
      : parentAlive ? `<span class="insp-link" data-select="${b.parent_id}">#${b.parent_id}</span>` : `#${b.parent_id} (gone)`;
    const genes = b.resistance_genes.length ? b.resistance_genes.map(geneTag).join('')
      : '<span style="color:var(--text-faint);font-size:9px">No resistance genes - fully susceptible</span>';
    const detail = k => (b[k] === undefined ? '–' : b[k]);
    el.innerHTML = `
      ${lastGone ? `<div class="insp-gone">No longer alive (gone by step ${lastGone}) - showing its last state</div>` : ''}
      <div class="insp-head">
        <div class="insp-name" style="color:${b.color_hex};font-style:italic">${U.esc(U.shortSpecies(b.species))}</div>
        <span class="insp-id">#${b.id}</span>
      </div>
      <div class="insp-kv"><span class="insp-key">State</span><strong class="insp-val" style="color:${sc}">${b.state}</strong></div>
      <div class="insp-kv"><span class="insp-key">Generation · age</span><span class="insp-val">${b.generation} · ${b.age} steps</span></div>
      <div class="insp-kv"><span class="insp-key">Parent cell</span><span class="insp-val">${parent}</span></div>
      <div class="insp-bars">
        ${bar('Fitness', b.fitness, 'var(--green)')}
        ${bar('Energy', b.energy, 'var(--cyan)')}
        ${bar('Stress', b.stress_level, 'var(--red)')}
        ${bar('AB damage', b.antibiotic_damage, 'var(--amber)')}
      </div>
      <canvas class="insp-spark" id="insp-spark" title="Recent history: fitness (green), energy (blue), stress (red), antibiotic damage (amber)"></canvas>
      <div class="insp-flags">
        <span class="flag sos ${b.sos_active ? 'on' : ''}">SOS${b.sos_active ? ' · mutation ×8' : ''}</span>
        <span class="flag bio ${b.in_biofilm ? 'on' : ''}">Biofilm</span>
        <span class="flag per ${b.is_persister ? 'on' : ''}">Persister</span>
      </div>
      <div class="insp-kv"><span class="insp-key">Mutations · HGT events</span><span class="insp-val">${b.total_mutations} · ${b.hgt_events}</span></div>
      <div class="insp-kv"><span class="insp-key">Offspring · local density</span><span class="insp-val">${detail('offspring_count')} · ${detail('local_density')}</span></div>
      <div class="insp-section">Resistance genes (${b.resistance_genes.length})</div>
      <div class="insp-genes">${genes}</div>`;
    sparkline(document.getElementById('insp-spark'));
  };

  I.select = id => {
    S.selectedId = id; S.cellHist = null; lastGone = null;
    AMR.stream.inspect(id);
    const b = S.bmap.get(id);
    if (b) I.render(b);
    AMR.render.markDirty();
  };

  I.refresh = () => {
    if (S.selectedId === null || !S.state) return;
    const now = performance.now();
    if (now - lastRender < 250) return;
    lastRender = now;
    const b = S.bmap.get(S.selectedId);
    if (b) { lastGone = null; I.render(b); }
    else if (!lastGone) {
      lastGone = S.state.stats.step;
      const el = document.getElementById('inspector');
      el.insertAdjacentHTML('afterbegin', `<div class="insp-gone">No longer alive (gone by step ${lastGone}) - showing its last state</div>`);
    }
  };

  I.clear = () => {
    S.selectedId = null; lastGone = null;
    document.getElementById('inspector').innerHTML = '<div class="insp-empty">Click a bacterium on the canvas to inspect it</div>';
  };

  I.tooltip = (b, px, py, canvas) => {
    const tip = document.getElementById('tip');
    if (!b) { tip.style.display = 'none'; return; }
    const sc = { growing: '#00e87a', stressed: '#ff3a5c', biofilm: '#ffb020', dormant: '#8899aa', dying: '#ff4444' }[b.state] || '#888';
    const genes = b.resistance_genes.length ? b.resistance_genes.map(g => `<span style="display:inline-block;background:rgba(192,96,255,.15);color:var(--purple);font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:3px;margin:1px">${U.esc(g)}</span>`).join(' ')
      : '<span style="color:var(--text-faint);font-size:9px;">None - susceptible</span>';
    tip.style.display = 'block';
    tip.style.left = Math.min(px + 14, canvas.clientWidth - 220) + 'px';
    tip.style.top = Math.min(py + 14, canvas.clientHeight - 180) + 'px';
    tip.innerHTML = `<div class="tip-name" style="color:${b.color_hex}"><i>${U.esc(U.shortSpecies(b.species))}</i></div>
      <span class="tip-state" style="background:${sc}22;color:${sc};border:1px solid ${sc}44">${b.state.toUpperCase()}</span>
      <div class="tip-row"><span>Fitness</span><strong>${b.fitness.toFixed(3)}</strong></div>
      <div class="tip-row"><span>Energy</span><strong>${b.energy.toFixed(2)}</strong></div>
      <div class="tip-row"><span>Stress</span><strong>${(b.stress_level ?? 0).toFixed(2)}</strong></div>
      <div class="tip-row"><span>Generation</span><strong>${b.generation}</strong></div>
      <div class="tip-row"><span>SOS active</span><strong>${b.sos_active ? 'YES - mutating fast' : 'No'}</strong></div>
      <div class="tip-genes"><div style="font-size:8px;color:var(--text-faint);margin-bottom:3px">RESISTANCE GENES:</div>${genes}</div>`;
  };

  I.init = async () => {
    try { I.ref = await AMR.api.reference(); } catch (e) { I.ref = null; }
    document.getElementById('inspector').addEventListener('click', e => {
      const t = e.target.closest('[data-select]'); if (t) I.select(+t.dataset.select);
    });
    S.on(kind => { if (kind === 'snapshot' && I.ref) AMR.api.reference().then(r => { I.ref = r; }).catch(() => {}); I.refresh(); });
  };
})(window.AMR);
