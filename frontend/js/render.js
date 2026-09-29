// AMR Simulation Lab — petri-dish renderer (canvas 2D).
//  - requestAnimationFrame loop; redraws only while something changes/animates
//  - cells interpolate between simulation steps (store.renderPos)
//  - sprites cached per colour/shape/radius, so thousands of cells are drawImage calls
//  - heatmaps rendered once per data change into small bitmaps, upscaled smoothly
//  - discrete events animated: HGT arc donor->recipient, SOS/biofilm/persister rings, deaths
(function (AMR) {
  const S = AMR.store, U = AMR.util;
  const R = AMR.render = { mode: 'species', prevMode: null, modeT0: 0, toggles: { ab: true, hgt: true, nut: false } };
  let canvas, ctx, W = 0, H = 0, dpr = 1, sx = 1, sy = 1, rad = 4;
  let gridLayer = null, heatLayer = null, heatKey = null;
  let dirty = true, lastFrame = 0, fpsEma = 16;
  const sprites = new Map();

  // ── geometry ──────────────────────────────────────────────────────────────
  function gw() { return (S.state && S.state.grid_width) || 80; }
  function gh() { return (S.state && S.state.grid_height) || 60; }
  R.toPx = (x, y) => [(x + 0.5) * sx, (y + 0.5) * sy];
  R.markDirty = () => { dirty = true; };

  function resize() {
    const wrap = canvas.parentElement;
    dpr = window.devicePixelRatio || 1;
    W = wrap.clientWidth; H = wrap.clientHeight;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sx = W / gw(); sy = H / gh();
    rad = Math.max(2.5, Math.min(5.5, sx * 0.46));
    sprites.clear(); gridLayer = null; heatKey = null;
    dirty = true;
  }

  // ── sprites ───────────────────────────────────────────────────────────────
  function sprite(key, w, h, draw) {
    let s = sprites.get(key);
    if (!s) {
      s = document.createElement('canvas');
      s.width = Math.ceil(w * dpr); s.height = Math.ceil(h * dpr);
      const c = s.getContext('2d'); c.scale(dpr, dpr); draw(c, w, h);
      s._w = w; s._h = h;
      sprites.set(key, s);
    }
    return s;
  }
  function bodySprite(color, shape) {
    const w = shape === 'bacillus' ? rad * 3.6 : rad * 2.2, h = rad * 2.2;
    return sprite(`b|${color}|${shape}|${rad}`, w, h, (c, w, h) => {
      c.beginPath();
      if (shape === 'bacillus') c.ellipse(w / 2, h / 2, rad * 1.7, rad * 0.65, 0, 0, Math.PI * 2);
      else c.arc(w / 2, h / 2, rad, 0, Math.PI * 2);
      c.fillStyle = color; c.fill();
    });
  }
  function ringSprite(shape) {
    const w = shape === 'bacillus' ? rad * 3.6 : rad * 2.2, h = rad * 2.2;
    return sprite(`r|${shape}|${rad}`, w, h, (c, w, h) => {
      c.beginPath();
      if (shape === 'bacillus') c.ellipse(w / 2, h / 2, rad * 1.7, rad * 0.65, 0, 0, Math.PI * 2);
      else c.arc(w / 2, h / 2, rad, 0, Math.PI * 2);
      c.strokeStyle = '#ff3a5c'; c.lineWidth = 1.1; c.stroke();
    });
  }
  function glowSprite(rgb) {
    const R0 = rad + 7;
    return sprite(`g|${rgb}|${rad}`, R0 * 2, R0 * 2, (c, w) => {
      const g = c.createRadialGradient(w / 2, w / 2, rad * 0.6, w / 2, w / 2, R0);
      g.addColorStop(0, `rgba(${rgb},0.9)`); g.addColorStop(1, `rgba(${rgb},0)`);
      c.fillStyle = g; c.fillRect(0, 0, w, w);
    });
  }

  // ── colour by visual mode ────────────────────────────────────────────────
  const q = (v, n) => Math.round(U.clamp(v, 0, 1) * n) / n;   // quantise continuous values (sprite cache)
  function colorFor(b, mode) {
    switch (mode) {
      case 'resistance': return AMR.RES_COLORS[Math.min(b.gene_count, 4)];
      case 'fitness': return U.lerpHex('#ff0000', '#00e87a', q(b.fitness, 24));
      case 'state': return AMR.STATE_COLORS[b.state] || '#888888';
      case 'energy': return U.lerpHex('#ff0000', '#20b4ff', q(b.energy, 24));
      default: return b.color_hex;
    }
  }
  R.colorFor = colorFor;
  R.setMode = mode => { R.prevMode = R.mode; R.mode = mode; R.modeT0 = performance.now(); dirty = true; };

  // ── static layers ─────────────────────────────────────────────────────────
  function drawGrid() {
    if (!gridLayer) {
      gridLayer = document.createElement('canvas');
      gridLayer.width = canvas.width; gridLayer.height = canvas.height;
      const c = gridLayer.getContext('2d'); c.scale(dpr, dpr);
      c.fillStyle = '#020609'; c.fillRect(0, 0, W, H);
      c.strokeStyle = 'rgba(20,60,90,0.3)'; c.lineWidth = 0.5;
      for (let x = 0; x < gw(); x += 10) { c.beginPath(); c.moveTo(x * sx, 0); c.lineTo(x * sx, H); c.stroke(); }
      for (let y = 0; y < gh(); y += 10) { c.beginPath(); c.moveTo(0, y * sy); c.lineTo(W, y * sy); c.stroke(); }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(gridLayer, 0, 0); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // Heatmaps are data[x][y] (numpy grids shaped (width, height)); same alpha
  // encoding as before, but rendered to a small bitmap and smoothly upscaled.
  function drawHeat() {
    const st = S.state;
    const maps = R.toggles.ab ? Object.values(st.antibiotic_heatmaps || {}) : [];
    const nut = R.toggles.nut ? st.nutrient_heatmap : null;
    const key = [nut, ...maps.map(m => m.data)];
    if (!heatKey || key.length !== heatKey.length || key.some((v, i) => v !== heatKey[i])) {
      heatKey = key;
      const ref = (maps[0] && maps[0].data) || nut;
      if (!ref || !ref.length) { heatLayer = null; return; }
      const nw = ref.length, nh = ref[0].length;
      heatLayer = document.createElement('canvas'); heatLayer.width = nw; heatLayer.height = nh;
      const c = heatLayer.getContext('2d');
      const layer = (data, rgb, alphaFn) => {
        const img = c.createImageData(nw, nh);
        for (let x = 0; x < nw; x++) for (let y = 0; y < nh; y++) {
          const i = (y * nw + x) * 4, a = alphaFn(data[x][y]);
          img.data[i] = rgb.r; img.data[i + 1] = rgb.g; img.data[i + 2] = rgb.b; img.data[i + 3] = Math.round(a * 255);
        }
        const tmp = document.createElement('canvas'); tmp.width = nw; tmp.height = nh;
        tmp.getContext('2d').putImageData(img, 0, 0); c.drawImage(tmp, 0, 0);
      };
      if (nut) layer(nut, { r: 0, g: 200, b: 80 }, v => (v > 0.05 ? v * 0.07 : 0));
      for (const m of maps) if (m.data && m.data.length) layer(m.data, U.hexRgb(m.color), v => (v > 0.01 ? Math.min(0.6, v * 0.28) : 0));
    }
    if (heatLayer) { ctx.imageSmoothingEnabled = true; ctx.drawImage(heatLayer, 0, 0, W, H); }
  }

  // ── cells ─────────────────────────────────────────────────────────────────
  function drawCell(b, x, y, scale, alpha, mode) {
    const s = bodySprite(colorFor(b, mode), b.shape);
    const w = s._w * scale, h = s._h * scale;
    ctx.globalAlpha = alpha;
    ctx.drawImage(s, x - w / 2, y - h / 2, w, h);
  }

  function drawCells(now) {
    const mode = R.mode, fade = R.prevMode && now - R.modeT0 < 300 ? (now - R.modeT0) / 300 : 1;
    if (fade >= 1) R.prevMode = null;
    const sosG = glowSprite('255,58,92'), bioG = glowSprite('255,176,32');
    for (const b of S.bmap.values()) {
      const p = S.renderPos(b.id, now); if (!p) continue;
      const [x, y] = R.toPx(p.x, p.y);
      // Glows first. SOS glow intensity tracks the cell's real stress_level.
      if (b.sos_active) { ctx.globalAlpha = 0.18 + 0.5 * U.clamp(b.stress_level || 0, 0, 1); ctx.drawImage(sosG, x - sosG._w / 2, y - sosG._h / 2, sosG._w, sosG._h); }
      if (b.in_biofilm) { ctx.globalAlpha = 0.28; ctx.drawImage(bioG, x - bioG._w / 2, y - bioG._h / 2, bioG._w, bioG._h); }
      // Body: dormant/dying dimmed as before; antibiotic damage fades the cell further.
      let a = b.state === 'dormant' ? 0.38 : b.state === 'dying' ? 0.55 : 1.0;
      a *= 1 - 0.45 * U.clamp(b.antibiotic_damage || 0, 0, 1);
      if (fade < 1) { drawCell(b, x, y, p.s, a * (1 - fade), R.prevMode); drawCell(b, x, y, p.s, a * fade, mode); }
      else drawCell(b, x, y, p.s, a, mode);
      // Resistance ring: opacity by acquired-gene count (same encoding as before)
      if (b.gene_count > 0) {
        const r = ringSprite(b.shape), w = r._w * p.s, h = r._h * p.s;
        ctx.globalAlpha = Math.min(0.9, b.gene_count * 0.28); ctx.drawImage(r, x - w / 2, y - h / 2, w, h);
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawGhosts(now) {
    const keep = [];
    for (const g of S.ghosts) {
      const t = (now - g.t0) / 450;
      if (t >= 1) continue;
      keep.push(g);
      const [x, y] = R.toPx(g.x, g.y);
      drawCell(g.b, x, y, 1 - 0.6 * t, 0.7 * (1 - t), R.mode);
    }
    S.ghosts = keep; ctx.globalAlpha = 1;
  }

  function drawEffects(now) {
    const keep = [];
    for (const e of S.effects) {
      const age = now - e.t0;
      if (e.kind === 'hgt') {
        if (age > 1100) continue; keep.push(e);
        if (!R.toggles.hgt) continue;
        const dp = S.renderPos(e.donor, now), rp = S.renderPos(e.recipient, now);
        const to = rp ? R.toPx(rp.x, rp.y) : R.toPx(e.pos[0], e.pos[1]);
        const from = dp ? R.toPx(dp.x, dp.y) : to;
        const mx = (from[0] + to[0]) / 2, my = (from[1] + to[1]) / 2 - 12;   // arc control point
        if (age < 700) {
          const t = age / 700;
          ctx.strokeStyle = `rgba(192,96,255,${0.7 * (1 - t * 0.5)})`; ctx.lineWidth = 1.4;
          ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.quadraticCurveTo(mx, my, to[0], to[1]); ctx.stroke();
          const it = 1 - t, px = it * it * from[0] + 2 * it * t * mx + t * t * to[0], py = it * it * from[1] + 2 * it * t * my + t * t * to[1];
          ctx.fillStyle = '#e0b0ff'; ctx.beginPath(); ctx.arc(px, py, 2.4, 0, Math.PI * 2); ctx.fill();
        } else {
          const t = (age - 700) / 400;
          ctx.strokeStyle = `rgba(192,96,255,${0.95 * (1 - t)})`; ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(to[0], to[1], rad + 2 + 10 * t, 0, Math.PI * 2); ctx.stroke();
        }
      } else {
        const dur = e.kind === 'biofilm_on' ? 700 : 600;
        if (age > dur) continue; keep.push(e);
        const p = S.renderPos(e.id, now); const [x, y] = p ? R.toPx(p.x, p.y) : R.toPx(e.pos[0], e.pos[1]);
        const t = age / dur;
        const col = e.kind === 'sos_on' ? '255,58,92' : e.kind === 'biofilm_on' ? '255,176,32' : '136,153,170';
        const r = e.kind === 'persister_on' ? rad + 10 * (1 - t) : rad + 2 + 9 * t;
        ctx.strokeStyle = `rgba(${col},${0.9 * (1 - t)})`; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
      }
    }
    S.effects = keep;
  }

  function drawSelection(now) {
    if (S.selectedId === null) return;
    const p = S.renderPos(S.selectedId, now); if (!p) return;
    const [x, y] = R.toPx(p.x, p.y);
    ctx.beginPath(); ctx.arc(x, y, rad + 5, 0, Math.PI * 2);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.setLineDash([3, 2]); ctx.stroke(); ctx.setLineDash([]);
  }

  // ── frame loop ────────────────────────────────────────────────────────────
  function frame(now) {
    requestAnimationFrame(frame);
    const animating = S.motion.size || S.ghosts.length || S.effects.length || R.prevMode;
    if (!S.state || !(dirty || animating)) return;
    dirty = false;
    const t0 = performance.now();
    drawGrid(); drawHeat(); drawCells(now); drawGhosts(now); drawEffects(now); drawSelection(now);
    const ms = performance.now() - t0;
    fpsEma = 0.9 * fpsEma + 0.1 * ms;
    R.lastDrawMs = ms;
    if (now - lastFrame > 500) {
      lastFrame = now;
      const el = document.getElementById('fps');
      if (el) el.textContent = `draw ${fpsEma.toFixed(1)} ms · ${S.bmap.size} cells`;
    }
  }
  R.drawOnce = () => { const now = performance.now(); drawGrid(); drawHeat(); drawCells(now); drawGhosts(now); drawEffects(now); drawSelection(now); };

  // Nearest cell to a canvas point (uses rendered, interpolated positions).
  R.hitTest = (mx, my) => {
    const now = performance.now(); let best = null, bd = (rad * 2.2) ** 2;
    for (const b of S.bmap.values()) {
      const p = S.renderPos(b.id, now); if (!p) continue;
      const [x, y] = R.toPx(p.x, p.y), d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd) { bd = d; best = b; }
    }
    return best;
  };

  R.init = el => {
    canvas = el; ctx = canvas.getContext('2d');
    new ResizeObserver(resize).observe(canvas.parentElement);
    resize();
    S.on(kind => { if (kind === 'snapshot') resize(); dirty = true; });
    requestAnimationFrame(frame);
  };
})(window.AMR);
