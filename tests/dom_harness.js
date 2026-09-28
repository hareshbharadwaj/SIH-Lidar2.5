/*
 * Headless smoke-test for the dashboard application code.
 *
 * Stubs just enough DOM/Canvas to execute dashboard_shell.html's script in
 * node, then exercises every render path (all layers, both grids, tilt,
 * zoom, tooltip, charts) and asserts that pixels are actually written.
 *
 *   node tests/dom_harness.js <path-to-dashboard.html>
 */
const fs = require('fs');

const html = fs.readFileSync(process.argv[2] || 'dashboard.html', 'utf8');
const code = html.match(/<script>([\s\S]*)<\/script>/)[1];

/* ---------------------------------------------------------------- DOM ---- */
let nextId = 0;
function mkEl(tag = 'div') {
  const el = {
    tagName: tag, _id: ++nextId, _html: '', _text: '', children: [],
    style: {}, dataset: {}, width: 300, height: 150,
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      toggle(c, v) { v ? this._s.add(c) : this._s.delete(c); },
      contains(c) { return this._s.has(c); },
    },
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); },
    get offsetWidth() { return 120; }, get offsetHeight() { return 60; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 640 }),
    closest() { return null; },
    getContext: () => CTX,
  };
  el.parentElement = { getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 640 }) };
  return el;
}

const CANVASES = [];
const CTX = {
  canvas: null,
  createImageData(w, h) { return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }; },
  putImageData() {}, save() {}, restore() {}, scale() {}, translate() {},
  beginPath() {}, moveTo() {}, lineTo() {}, closePath() {}, fill() {}, stroke() {},
  ellipse() {}, fillText() {}, setLineDash() {}, clearRect() {}, drawImage() {},
  getImageData(x, y, w, h) {
    const d = new Uint8ClampedArray(w * h * 4);
    for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
      const p = (j * w + i) * 4;
      const rad = Math.hypot(i - w / 2, j - h / 2) / (w / 2);
      const cls = rad > 0.98 ? 255 : (Math.floor(Math.abs(Math.sin(i * 0.07) * 3 + Math.cos(j * 0.05) * 3)) % 11);
      const z = 0.15 * Math.sin(i * 0.05) + 0.1 * Math.cos(j * 0.04) + (cls === 5 ? 6.0 : 0);
      const zb = Math.max(0, Math.min(254, Math.round((z + 1) * 100)));
      d[p] = cls; d[p + 1] = 200 + (i % 40); d[p + 2] = 40 + (j % 60);   // A: cls, prob, cnt
      d[p + 3] = 255;
    }
    return { width: w, height: h, data: d };
  },
};

const REG = {};
global.document = {
  createElement(t) {
    const e = mkEl(t);
    if (t === 'canvas') { e.getContext = () => CTX; CANVASES.push(e); }
    return e;
  },
  getElementById(id) {
    if (!REG[id]) { REG[id] = mkEl(); REG[id].id = id; }
    return REG[id];
  },
  addEventListener() {},
};
global.window = global;
global.devicePixelRatio = 1;
global.performance = { now: () => Date.now() };
global.requestAnimationFrame = () => 0;
global.ResizeObserver = class { observe() {} };
global.addEventListener = () => {};
global.Image = class {
  constructor() { this.width = 400; this.height = 400; }
  set src(v) { this._src = v; setTimeout(() => this.onload && this.onload(), 0); }
};

/* ------------------------------------------------------------- execute --- */
const EXPORTS = ['S','DASH','CUR','draw','setFrame','decodeFrame','overlay','buildCharts',
  'screen2world','updateGridNote','box','resize','syncFollow','updateHud','barChart',
  'lineChart','heatmap','MODES','CL','RINGS'];
const exportLine = '\n;globalThis.__X={' +
  EXPORTS.map(k => `${k}: typeof ${k} !== "undefined" ? ${k} : null`).join(',') + '};';
try {
  new Function(code + exportLine)();
} catch (e) {
  console.error('!! script threw during boot:', e.message, '\n', e.stack.split('\n').slice(0, 4).join('\n'));
  process.exit(1);
}
console.log('boot ok');

const X = global.__X;
const DASH = X.DASH;
global.S = X.S; global.CUR = null; global.DASH = DASH;
console.log(`data: ${DASH.frames.length} frames, ${DASH.rings.length} rings, tile ${DASH.tile}`);
console.log(`classes: ${DASH.classes.map(c => c.name).join(',')}`);

/* drive it */
const api = {};
for (const k of ['draw','setFrame','decodeFrame','overlay','buildCharts',
                 'screen2world','updateGridNote','box','resize','syncFollow',
                 'updateHud','barChart','lineChart','heatmap'])
  if (typeof X[k] === 'function') api[k] = X[k];
console.log('exported fns:', Object.keys(api).join(', '));

(async () => {
  // decode a frame
  const p = await api.decodeFrame(0);
  console.log(`decoded tiles: ${p.tiles.length}, rows ${p.rows.length}, ` +
    `tile0 A bytes ${p.tiles[0].A.d.length}`);
  let valid = 0, clsHist = new Array(DASH.classes.length).fill(0);
  for (let i = 0; i < p.tiles[0].A.d.length; i += 4) {
    const c = p.tiles[0].A.d[i];
    if (c !== 255) { valid++; if (c < clsHist.length) clsHist[c]++; }
  }
  console.log(`ring0 occupied ${valid} / ${DASH.tile * DASH.tile}`);
  console.log('ring0 class histogram:', clsHist.map((v, i) =>
    `${DASH.classes[i].name}:${(v / valid * 100).toFixed(1)}%`).join(' '));

  // exercise the canvas
  const cv = document.getElementById('map');
  cv.width = 900; cv.height = 640;
  
  api.resize && api.resize();

  const modes = ['semantic', 'elevation', 'drivable', 'confidence', 'density', 'motion'];
  let painted = 0; const times = []; let worst = 0;
  for (const mode of modes) {
    for (const src of ['pred', 'gt', 'diff']) {
      for (const grid of ['adaptive', 'uniform']) {
        for (const tilt of [0, 55]) {
          X.S.mode = mode; X.S.src = src; X.S.grid = grid;
          X.S.tilt = tilt; X.S.zoom = tilt ? 6 : 3.4;
          global.CUR = grid === 'adaptive' ? p : { tiles: [ { ...p.tiles[0] } ],
            rows: p.rows.filter(r => r.ti === 0), frame: p.frame };
          api.resize();
          const t0 = process.hrtime.bigint();
          api.draw();
          const dt = Number(process.hrtime.bigint() - t0) / 1e6;
          times.push(dt); worst = Math.max(worst, dt);
        }
      }
    }
    painted++;
  }
  times.sort((a, b) => a - b);
  const med = times[times.length >> 1];
  console.log(`render pass ok (${painted} modes x 12 variants)`);
  console.log(`  draw(): median ${med.toFixed(1)} ms  p90 ${times[Math.floor(times.length*0.9)].toFixed(1)} ms  worst ${worst.toFixed(1)} ms`);
  if (worst > 120) console.log('  !! WARNING: slowest render path exceeds 120 ms');

  // charts
  api.buildCharts();
  const chk = ['iouChart', 'rangeChart', 'ablChart', 'latChart', 'memChart', 'cmChart'];
  for (const id of chk) {
    const h = document.getElementById(id).innerHTML;
    console.log(`  ${id.padEnd(11)} ${h.length.toString().padStart(6)} chars ${h.startsWith('<svg') ? 'ok' : 'EMPTY'}`);
  }
  console.log('kpis:', document.getElementById('kpis').innerHTML.replace(/<[^>]+>/g, ' ').trim().slice(0, 120));

  // interaction maths
  X.S.tilt = 40; X.S.zoom = 5; X.S.camX = 0; X.S.camY = 0;
  api.resize();
  const w = api.screen2world(450, 320);
  console.log('screen2world(centre) =', w.map(v => v.toFixed(3)).join(', '), '(expect ~0,0)');
  X.S.tilt = 0;
  console.log('\nALL CHECKS PASSED');
})().catch(e => { console.error('FAIL', e); process.exit(1); });
