// AMR Simulation Lab — configuration and constants.
// Plain script (no build step); everything hangs off window.AMR.
window.AMR = window.AMR || {};

(function (AMR) {
  // Served by the API server at /ui/ -> same origin. Opened from disk -> local server.
  AMR.API = location.protocol.startsWith('http') ? location.origin : 'http://127.0.0.1:8000';
  AMR.WS_URL = AMR.API.replace(/^http/, 'ws') + '/ws';

  // Fields the server only sends for the inspected cell (api/stream.py DETAIL_FIELDS).
  AMR.DETAIL_FIELDS = ['offspring_count', 'local_density'];

  AMR.SPECIES = {
    'Escherichia coli': { color: '#4CAF50', who: 'HIGH' },
    'Klebsiella pneumoniae': { color: '#FF9800', who: 'CRITICAL' },
    'Acinetobacter baumannii': { color: '#F44336', who: 'CRITICAL' },
    'Pseudomonas aeruginosa': { color: '#9C27B0', who: 'CRITICAL' },
    'Staphylococcus aureus (MRSA)': { color: '#FFD700', who: 'HIGH' },
  };

  AMR.ANTIBIOTICS = ['ciprofloxacin', 'meropenem', 'colistin', 'vancomycin', 'ampicillin', 'tetracycline'];

  AMR.STATE_COLORS = { growing: '#00e87a', stressed: '#ff3a5c', biofilm: '#ffb020', dormant: '#334455', dying: '#ff0000' };
  AMR.RES_COLORS = ['#2d6a2d', '#7aff4a', '#ffaa00', '#ff5500', '#ff0033'];

  // Legends shown on the canvas for each visual mode (unchanged wording).
  AMR.LEGENDS = {
    resistance: { title: 'Resistance Gene Count', items: [
      { color: '#2d6a2d', label: '0 genes - fully susceptible' },
      { color: '#7aff4a', label: '1 gene' },
      { color: '#ffaa00', label: '2 genes' },
      { color: '#ff5500', label: '3 genes' },
      { color: '#ff0033', label: '4+ genes - highly resistant' }] },
    fitness: { title: 'Bacterial Fitness (0-1)', items: [
      { color: '#ff0000', label: 'Low fitness (< 0.3)' },
      { color: '#ff8800', label: 'Medium (0.3-0.6)' },
      { color: '#00ff88', label: 'High fitness (> 0.7)' }] },
    state: { title: 'Cell Physiological State', items: [
      { color: '#00ff88', label: 'Growing - dividing normally' },
      { color: '#ff3366', label: 'Stressed - SOS response active' },
      { color: '#ffaa00', label: 'Biofilm - inside biofilm colony' },
      { color: '#334455', label: 'Dormant - persister (tolerant)' },
      { color: '#ff0000', label: 'Dying - antibiotic damage critical' }] },
    energy: { title: 'Cell Energy Level (0-1)', items: [
      { color: '#ff0000', label: 'Starving (< 0.2)' },
      { color: '#ffaa00', label: 'Low energy (0.2-0.5)' },
      { color: '#00d4ff', label: 'Healthy energy (> 0.7)' }] },
  };

  // Play timing — mirrors api/server.py play_timing(), used to pace motion.
  AMR.playIntervalMs = speed => Math.max(80, 650 - speed * 28);

  AMR.util = {
    hexRgb(hex) {
      if (!hex || hex.length < 6) return { r: 100, g: 100, b: 100 };
      const h = hex.replace('#', '');
      return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16) };
    },
    lerpHex(a, b, t) {
      const ca = AMR.util.hexRgb(a), cb = AMR.util.hexRgb(b);
      const c = k => Math.round(ca[k] + (cb[k] - ca[k]) * t).toString(16).padStart(2, '0');
      return `#${c('r')}${c('g')}${c('b')}`;
    },
    clamp: (v, lo, hi) => Math.max(lo, Math.min(hi, v)),
    easeInOut: t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    esc: s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])),
    shortSpecies: sp => sp.split(' ').slice(0, 2).join(' '),
  };
})(window.AMR);
