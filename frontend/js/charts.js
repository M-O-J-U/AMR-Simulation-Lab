// AMR Simulation Lab — live charts (uPlot, vendored in frontend/vendor/uplot).
//  - Population: total / resistant / biofilm / persisters per step
//  - Genes: % of living cells carrying each resistance gene
//  - Drugs: % of living cells classed resistant per antibiotic, from the
//    server's model-based MIC estimate (/analytics/mic_all; sampled ~every 2 s)
(function (AMR) {
  const S = AMR.store;
  const C = AMR.charts = { tab: 'genes' };
  let pop = null, gene = null, drug = null, geneList = [], pending = false;
  const drugHist = { step: [], pct: {} };
  let lastDrugStep = -1, drugBusy = false;

  const PALETTE = ['#c060ff', '#20b4ff', '#ff9800', '#00e87a', '#ff3a5c', '#ffd700', '#00d4c0',
                   '#ff66cc', '#9fa8ff', '#b0ff5a', '#ffb3a0'];
  const axis = { stroke: '#5a8099', grid: { stroke: 'rgba(32,120,180,0.10)', width: 1 },
                 ticks: { stroke: 'rgba(32,120,180,0.2)' }, font: '9px IBM Plex Mono' };

  // Compact one-line legend above the plot (uPlot's own legend wraps badly in
  // the 180 px panels); the plot takes whatever height is left.
  function size(el) {
    const leg = el.querySelector('.chart-legend');
    return { width: Math.max(120, el.clientWidth - 8), height: Math.max(50, el.clientHeight - (leg ? leg.offsetHeight : 0) - 12) };
  }

  function make(el, series, yUnit) {
    el.innerHTML = `<div class="chart-legend" title="y: ${yUnit}; x: simulation step">` +
      series.map(s => `<span><i style="background:${s.stroke}"></i>${s.label}</span>`).join('') +
      `<span class="unit">${yUnit}</span></div>`;
    const { width, height } = size(el);
    return new uPlot({
      width, height, legend: { show: false },
      cursor: { drag: { x: false, y: false } },
      scales: { x: { time: false } },
      axes: [{ ...axis, size: 22 }, { ...axis, size: 40 }],
      series: [{ label: 'step' }, ...series.map(s => ({ width: 1.5, points: { show: false }, ...s }))],
    }, [[], ...series.map(() => [])], el);
  }

  function buildGeneChart() {
    const el = document.getElementById('chart-genes');
    if (gene) gene.destroy();
    gene = make(el, geneList.map((g, i) => ({ label: g, stroke: PALETTE[i % PALETTE.length] })), '% cells');
  }

  function buildDrugChart() {
    const el = document.getElementById('chart-drugs');
    if (drug) drug.destroy();
    drug = make(el, AMR.ANTIBIOTICS.map((k, i) => ({ label: k, stroke: PALETTE[(i + 3) % PALETTE.length] })), '% resistant');
  }

  function update() {
    pending = false;
    const h = S.history;
    if (pop) pop.setData([h.step, h.total, h.resistant, h.biofilm, h.persist]);
    if (gene && C.tab === 'genes') gene.setData([h.step, ...geneList.map(g => h.genePct[g] || h.step.map(() => 0))]);
    if (drug && C.tab === 'drugs') drug.setData([drugHist.step, ...AMR.ANTIBIOTICS.map(k => drugHist.pct[k] || [])]);
  }
  C.schedule = () => { if (!pending) { pending = true; setTimeout(() => requestAnimationFrame(update), 200); } };

  async function sampleDrugs() {
    if (drugBusy || !S.state || S.state.stats.step === lastDrugStep) return;
    drugBusy = true;
    try {
      const d = await AMR.api.micAll();
      lastDrugStep = d.step;
      if (drugHist.step.length && drugHist.step[drugHist.step.length - 1] >= d.step) { drugHist.step = []; drugHist.pct = {}; }
      drugHist.step.push(d.step);
      for (const k of AMR.ANTIBIOTICS) {
        (drugHist.pct[k] = drugHist.pct[k] || []).push(d.antibiotics[k] ? d.antibiotics[k].resistant_pct : 0);
      }
      if (drugHist.step.length > 300) { drugHist.step.shift(); for (const k in drugHist.pct) drugHist.pct[k].shift(); }
      C.schedule();
    } catch (e) { /* server busy/offline: next sample */ } finally { drugBusy = false; }
  }

  C.setTab = (tab, btn) => {
    C.tab = tab;
    document.querySelectorAll('#chart-tabs .tab').forEach(t => t.classList.toggle('active', t === btn));
    document.getElementById('chart-genes').style.display = tab === 'genes' ? '' : 'none';
    document.getElementById('chart-drugs').style.display = tab === 'drugs' ? '' : 'none';
    document.getElementById('chart-note').textContent = tab === 'genes'
      ? '% of living cells carrying each gene'
      : '% classed resistant: the simulation\'s model-based MIC estimate vs EUCAST breakpoints (not measured MICs)';
    C.resize(); C.schedule();
  };

  C.reset = () => { drugHist.step = []; drugHist.pct = {}; lastDrugStep = -1; C.schedule(); };

  C.resize = () => {
    for (const [c, id] of [[pop, 'chart-pop'], [gene, 'chart-genes'], [drug, 'chart-drugs']]) {
      const el = document.getElementById(id);
      if (c && el && el.offsetParent !== null) c.setSize(size(el));
    }
  };

  C.init = async () => {
    pop = make(document.getElementById('chart-pop'), [
      { label: 'Total', stroke: '#20b4ff', fill: 'rgba(32,180,255,0.06)' },
      { label: 'Resistant', stroke: '#ff3a5c', dash: [4, 3] },
      { label: 'Biofilm', stroke: '#ffb020' },
      { label: 'Persisters', stroke: '#c060ff' }], 'cells');
    try { geneList = Object.keys((await AMR.api.reference()).genes); }
    catch (e) { geneList = ['blaTEM-1', 'blaCTX-M-15', 'blaKPC-2', 'blaNDM-1', 'mexAB-oprM', 'acrAB-tolC', 'gyrA_S83L', 'mcr-1', 'tetM', 'vanA']; }
    buildGeneChart(); buildDrugChart();
    C.resize();
    C.setTab('genes', document.querySelector('#chart-tabs .tab'));
    S.on(() => C.schedule());
    setInterval(() => { if (C.tab === 'drugs') sampleDrugs(); }, 2000);
    new ResizeObserver(() => C.resize()).observe(document.getElementById('bottom'));
  };
})(window.AMR);
