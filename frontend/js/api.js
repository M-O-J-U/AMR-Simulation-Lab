// AMR Simulation Lab — REST commands (the stream carries state; REST carries commands).
(function (AMR) {
  const json = { 'Content-Type': 'application/json' };
  async function req(path, body, method) {
    const opts = body === undefined ? { method: method || 'GET' } : { method: method || 'POST', headers: json, body: JSON.stringify(body) };
    const r = await fetch(`${AMR.API}${path}`, opts);
    if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
    return r.json();
  }
  AMR.api = {
    state: () => req('/state'),
    step: n => req('/step', { n_steps: n }),
    applyAntibiotic: (key, conc, mode, center, radius) =>
      req('/apply_antibiotic', { antibiotic_key: key, concentration: conc, mode, ...(center ? { center, radius } : {}) }),
    removeAntibiotic: key => req('/remove_antibiotic', { antibiotic_key: key }),
    spawn: (key, count) => req('/spawn_bacteria', { germ_key: key, count }),
    reset: (scenario, n) => req('/reset', { scenario, initial_bacteria: n }),
    resume: () => req('/resume', {}),
    pause: () => req('/pause', {}),
    speed: s => req('/speed', { speed: s }),
    reference: () => req('/reference'),
    gnnStatus: () => req('/gnn/status'),
    gnnPredict: () => req('/gnn/predict', { threshold: 0.31, max_nodes: 300, max_edge_distance: 3 }),
    gnnAdvisory: () => req('/gnn/advisory', { available_antibiotics: AMR.ANTIBIOTICS }),
    mic: key => req(`/analytics/mic?antibiotic_key=${encodeURIComponent(key)}`),
    micAll: () => req('/analytics/mic_all'),
    diversity: () => req('/analytics/diversity'),
    recommend: () => req('/analytics/recommend'),
  };
})(window.AMR);
