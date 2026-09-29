// AMR Simulation Lab — entry point.
(function (AMR) {
  document.addEventListener('DOMContentLoaded', async () => {
    const canvas = document.getElementById('petri-canvas');
    AMR.render.init(canvas);
    AMR.ui.init();
    AMR.ui.initCanvas(canvas);
    await Promise.all([AMR.inspector.init(), AMR.charts.init()]);
    AMR.stream.connect();
    AMR.ui.checkGNN();
    setInterval(AMR.ui.checkGNN, 15000);
  });
})(window.AMR);
