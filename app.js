// Application: camera in arbitrary precision, input, reference-orbit management, UI.
'use strict';
(() => {
  const BASE_H = 2.6;        // world-space height of the view at zoom 0
  const MIN_ZOOM = -3;
  const MAX_ZOOM = 4000;     // log2 magnification cap (~10^1204); raise if you have patience
  const WHEEL_TAU = 0.18;    // seconds; wheel impulses decay with this time constant
  const ZOOM_PER_NOTCH = 0.6;
  const MAX_PIXELS = 2.6e6;

  const PALETTES = [
    { name: 'Classic', stops: [[0, [0, 7, 100]], [0.16, [32, 107, 203]], [0.42, [237, 255, 255]], [0.6425, [255, 170, 0]], [0.8575, [0, 2, 0]], [1, [0, 7, 100]]] },
    { name: 'Ember', stops: [[0, [10, 2, 0]], [0.2, [120, 20, 10]], [0.4, [240, 90, 20]], [0.6, [255, 200, 80]], [0.75, [255, 250, 220]], [0.9, [80, 30, 40]], [1, [10, 2, 0]]] },
    { name: 'Ocean', stops: [[0, [3, 10, 40]], [0.25, [10, 80, 140]], [0.5, [60, 200, 210]], [0.7, [235, 250, 250]], [0.85, [30, 50, 110]], [1, [3, 10, 40]]] },
    { name: 'Aurora', cos: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], [0.0, 0.33, 0.67]] },
    { name: 'Candy', cos: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], [0.3, 0.2, 0.2]] },
    { name: 'Mono', stops: [[0, [0, 0, 0]], [0.5, [255, 255, 255]], [1, [0, 0, 0]]] },
  ];

  const $ = (id) => document.getElementById(id);
  const canvas = $('view');

  // ---------------------------------------------------------------- state
  const S = {
    cx: 0n, cy: 0n, cbits: 128,
    zoom: 0,
    wheelVel: 0, wheelAnchor: null,
    keyZoom: 0, keyZoomSmooth: 0,
    keyPan: [0, 0], keyPanSmooth: [0, 0],
    pendingPan: [0, 0], panVel: [0, 0],
    pinchZoom: 0, pinchAnchor: null,
    autoZoom: false, speed: 0.8,
    iterExp: 0, density: 6, phase: 0, palette: 0,
    cursor: null,
    fraction: 0.5,
    idleFrames: 0, havePrev: false, forceCompute: true,
    width: 0, height: 0,
    lastMaxIter: 0,
  };

  let renderer;
  try {
    renderer = new Renderer(canvas);
  } catch (err) {
    $('fatal').textContent = err.message;
    $('fatal').classList.remove('hidden');
    return;
  }

  // ---------------------------------------------------------------- helpers
  function pixelSize(zoom) {
    const psLog = Math.log2(BASE_H / Math.max(1, S.height)) - zoom;
    const ec = Math.floor(psLog);
    return { pm: Math.pow(2, psLog - ec), ec };
  }

  function setBits(bits) {
    S.cx = BigFixed.shift(S.cx, bits - S.cbits);
    S.cy = BigFixed.shift(S.cy, bits - S.cbits);
    S.cbits = bits;
  }

  function ensurePrecision(ec) {
    const need = -ec + 64;
    if (S.cbits < need || S.cbits > need + 256) setBits(need + 64);
  }

  // (a - b) / pixelSize in pixels, where a and b are fixed point at the given bit counts.
  function fixedDiffPx(a, abits, b, bbits, pm, ec) {
    const d = a - BigFixed.shift(b, abits - bbits);
    const fe = BigFixed.toFloatExp(d, abits);
    if (fe.m === 0) return 0;
    const de = fe.e - ec;
    if (de > 60) return fe.m > 0 ? 1e30 : -1e30;
    if (de < -120) return 0;
    return (fe.m * Math.pow(2, de)) / pm;
  }

  function zoomBy(dz, anchor) {
    if (dz === 0) return;
    const nz = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, S.zoom + dz));
    dz = nz - S.zoom;
    if (dz === 0) return;
    const { pm, ec } = pixelSize(S.zoom);
    const ax = (anchor ? anchor[0] : S.width / 2) - S.width / 2;
    const ay = (anchor ? anchor[1] : S.height / 2) - S.height / 2;
    const k = 1 - Math.pow(2, -dz);
    S.cx += BigFixed.fromFloatExp(ax * k * pm, ec, S.cbits);
    S.cy += BigFixed.fromFloatExp(ay * k * pm, ec, S.cbits);
    S.zoom = nz;
  }

  function panBy(dx, dy) {
    if (dx === 0 && dy === 0) return;
    const { pm, ec } = pixelSize(S.zoom);
    S.cx -= BigFixed.fromFloatExp(dx * pm, ec, S.cbits);
    S.cy -= BigFixed.fromFloatExp(dy * pm, ec, S.cbits);
  }

  function maxIterations() {
    const base = 256 + 40 * Math.max(0, S.zoom);
    return Math.max(64, Math.min(400000, Math.round(base * Math.pow(2, S.iterExp))));
  }

  function toGl(e) {
    const r = canvas.getBoundingClientRect();
    const sx = S.width / r.width, sy = S.height / r.height;
    return [(e.clientX - r.left) * sx, S.height - (e.clientY - r.top) * sy];
  }

  // ---------------------------------------------------------------- palettes
  function buildPalette(p, n = 512) {
    const out = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      const t = i / n;
      let r, g, b;
      if (p.cos) {
        const [a, bb, c, d] = p.cos;
        r = a[0] + bb[0] * Math.cos(6.28318 * (c[0] * t + d[0]));
        g = a[1] + bb[1] * Math.cos(6.28318 * (c[1] * t + d[1]));
        b = a[2] + bb[2] * Math.cos(6.28318 * (c[2] * t + d[2]));
        r *= 255; g *= 255; b *= 255;
      } else {
        const st = p.stops;
        let k = 0;
        while (k < st.length - 2 && t >= st[k + 1][0]) k++;
        const [t0, c0] = st[k], [t1, c1] = st[k + 1];
        let u = (t - t0) / (t1 - t0);
        u = 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)));
        r = c0[0] + (c1[0] - c0[0]) * u;
        g = c0[1] + (c1[1] - c0[1]) * u;
        b = c0[2] + (c1[2] - c0[2]) * u;
      }
      out[i * 4] = Math.max(0, Math.min(255, r));
      out[i * 4 + 1] = Math.max(0, Math.min(255, g));
      out[i * 4 + 2] = Math.max(0, Math.min(255, b));
      out[i * 4 + 3] = 255;
    }
    renderer.setPalette(out, n);
  }

  // ---------------------------------------------------------------- reference orbit
  const REF = { cur: null, pending: null, nextId: 1, workerId: 0 };
  const worker = createRefWorker();

  function requestRef(iters) {
    const id = REF.nextId++;
    REF.pending = {
      id, cx: S.cx, cy: S.cy, bits: S.cbits,
      orbit: new Float32Array(Math.max(4096, iters) * 2), len: 0,
      escaped: false, done: false, target: iters,
    };
    REF.workerId = id;
    worker.postMessage({ type: 'start', id, cx: S.cx.toString(), cy: S.cy.toString(), bits: S.cbits, iters });
  }

  function extendRef(r, iters) {
    r.target = iters;
    r.done = false;
    worker.postMessage({ type: 'extend', id: r.id, iters });
  }

  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type !== 'chunk') return;
    const r = REF.pending && REF.pending.id === msg.id ? REF.pending
      : REF.cur && REF.cur.id === msg.id ? REF.cur : null;
    if (!r) return;
    const need = (msg.start + msg.data.length / 2) * 2;
    if (need > r.orbit.length) {
      const grown = new Float32Array(Math.max(need, r.orbit.length * 2));
      grown.set(r.orbit);
      r.orbit = grown;
    }
    r.orbit.set(msg.data, msg.start * 2);
    r.len = msg.len;
    r.escaped = msg.escaped;
    r.done = msg.done;
    if (r === REF.pending) {
      if (r.done) {
        REF.cur = r;
        REF.pending = null;
        renderer.setReference(r.orbit, r.len);
        S.forceCompute = true;
      }
    } else {
      renderer.setReference(r.orbit, r.len);
      S.forceCompute = true;
    }
  };

  function refIsStale(r, ec, pm) {
    if (-ec + 40 > r.bits) return true;
    const ox = fixedDiffPx(S.cx, S.cbits, r.cx, r.bits, pm, ec);
    const oy = fixedDiffPx(S.cy, S.cbits, r.cy, r.bits, pm, ec);
    return Math.abs(ox) > S.width * 0.6 || Math.abs(oy) > S.height * 0.6;
  }

  function manageRef(maxIter, ec, pm) {
    const cur = REF.cur, pend = REF.pending;
    if (pend) {
      if (refIsStale(pend, ec, pm)) requestRef(maxIter);
      return;
    }
    if (!cur || refIsStale(cur, ec, pm)) {
      requestRef(maxIter);
      return;
    }
    if (!cur.escaped && cur.done && cur.target < maxIter && REF.workerId === cur.id) {
      extendRef(cur, maxIter);
    }
  }

  // ---------------------------------------------------------------- resize
  function handleResize() {
    const r = canvas.getBoundingClientRect();
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (r.width * r.height * dpr * dpr > MAX_PIXELS) dpr = Math.sqrt(MAX_PIXELS / (r.width * r.height));
    const w = Math.max(1, Math.floor(r.width * dpr)), h = Math.max(1, Math.floor(r.height * dpr));
    if (w !== S.width || h !== S.height) {
      S.width = w; S.height = h;
      renderer.resize(w, h);
      S.havePrev = false;
      S.forceCompute = true;
    }
  }

  // ---------------------------------------------------------------- main loop
  let lastT = 0;
  let statT = 0;
  let fpsAcc = 0, fpsN = 0;

  function frame(t) {
    requestAnimationFrame(frame);
    const dt = lastT ? Math.min(0.1, Math.max(0.001, (t - lastT) / 1000)) : 1 / 60;
    lastT = t;
    handleResize();

    const prevCx = S.cx, prevCy = S.cy, prevBits = S.cbits, prevZoom = S.zoom;

    // --- wheel / double-click impulses (exponential decay, anchored at the cursor)
    if (S.wheelVel !== 0) {
      const decay = Math.exp(-dt / WHEEL_TAU);
      const dz = S.wheelVel * WHEEL_TAU * (1 - decay);
      S.wheelVel *= decay;
      if (Math.abs(S.wheelVel) < 1e-3) S.wheelVel = 0;
      zoomBy(dz, S.wheelAnchor || S.cursor);
    }
    // --- pinch
    if (S.pinchZoom !== 0) {
      zoomBy(S.pinchZoom, S.pinchAnchor);
      S.pinchZoom = 0;
    }
    // --- keyboard / auto zoom (smoothly ramped, anchored at the screen centre)
    const zoomTarget = S.autoZoom ? S.speed : S.keyZoom * S.speed * 1.5;
    S.keyZoomSmooth += (zoomTarget - S.keyZoomSmooth) * (1 - Math.exp(-dt / 0.15));
    if (Math.abs(S.keyZoomSmooth) < 1e-4 && zoomTarget === 0) S.keyZoomSmooth = 0;
    if (S.keyZoomSmooth !== 0) zoomBy(S.keyZoomSmooth * dt, null);
    if (S.autoZoom && S.zoom >= MAX_ZOOM) setAuto(false);

    // --- panning: drag, inertia, keys
    let px = S.pendingPan[0], py = S.pendingPan[1];
    S.pendingPan[0] = S.pendingPan[1] = 0;
    if (S.panVel[0] !== 0 || S.panVel[1] !== 0) {
      const decay = Math.exp(-dt / 0.25);
      px += S.panVel[0] * 0.25 * (1 - decay);
      py += S.panVel[1] * 0.25 * (1 - decay);
      S.panVel[0] *= decay; S.panVel[1] *= decay;
      if (Math.hypot(S.panVel[0], S.panVel[1]) < 1) S.panVel[0] = S.panVel[1] = 0;
    }
    const kp = Math.min(S.width, S.height) * 0.6;
    for (let i = 0; i < 2; i++) {
      S.keyPanSmooth[i] += (S.keyPan[i] * kp - S.keyPanSmooth[i]) * (1 - Math.exp(-dt / 0.12));
      if (Math.abs(S.keyPanSmooth[i]) < 0.5 && S.keyPan[i] === 0) S.keyPanSmooth[i] = 0;
    }
    px += S.keyPanSmooth[0] * dt;
    py += S.keyPanSmooth[1] * dt;
    panBy(px, py);

    // --- precision, iterations, reference
    const { pm, ec } = pixelSize(S.zoom);
    ensurePrecision(ec);
    const maxIter = maxIterations();
    manageRef(maxIter, ec, pm);

    const moved = S.cx !== BigFixed.shift(prevCx, S.cbits - prevBits)
      || S.cy !== BigFixed.shift(prevCy, S.cbits - prevBits)
      || S.zoom !== prevZoom;
    const changed = moved || S.forceCompute || maxIter !== S.lastMaxIter;
    S.lastMaxIter = maxIter;

    const cur = REF.cur;
    if (changed) S.idleFrames = 0; else S.idleFrames++;
    const idleBudget = S.fraction >= 1 ? 1 : 64;
    const active = cur && (changed || S.idleFrames <= idleBudget || !cur.done);

    if (active) {
      const offsetPx = [
        fixedDiffPx(S.cx, S.cbits, cur.cx, cur.bits, pm, ec),
        fixedDiffPx(S.cy, S.cbits, cur.cy, cur.bits, pm, ec),
      ];
      const prevPs = pixelSize(prevZoom);
      const prevShift = [
        fixedDiffPx(S.cx, S.cbits, prevCx, prevBits, prevPs.pm, prevPs.ec),
        fixedDiffPx(S.cy, S.cbits, prevCy, prevBits, prevPs.pm, prevPs.ec),
      ];
      const prevScale = Math.pow(2, prevZoom - S.zoom);
      const havePrev = S.havePrev && prevScale > 0.25 && prevScale < 4
        && Math.abs(prevShift[0]) < 1e5 && Math.abs(prevShift[1]) < 1e5;

      // adapt the recompute fraction to the measured frame time while moving
      if (moved && S.havePrev) {
        if (dt > 0.024) S.fraction = Math.max(1 / 64, S.fraction * 0.7);
        else if (dt < 0.0175) S.fraction = Math.min(1, S.fraction * 1.1);
      }
      const cost = S.width * S.height * (havePrev ? S.fraction : 1) * maxIter;
      const bands = Math.max(1, Math.min(32, Math.ceil(cost / 4e8)));

      renderer.render({
        refLen: cur.len, refEscaped: cur.escaped, havePrev,
        offsetPx, pm, ec, maxIter, fraction: S.fraction, prevScale, prevShift, bands,
        density: Math.pow(2, S.density), phase: S.phase, interior: [0, 0, 0],
      });
      S.havePrev = true;
      S.forceCompute = false;
      fpsAcc += dt; fpsN++;
    } else if (S.presentDirty && S.havePrev) {
      renderer.present({ density: Math.pow(2, S.density), phase: S.phase, interior: [0, 0, 0] });
    }
    S.presentDirty = false;

    if (t - statT > 250) {
      statT = t;
      updateStats(maxIter, active, fpsN ? fpsN / fpsAcc : 0);
      fpsAcc = 0; fpsN = 0;
    }
  }

  function updateStats(maxIter, active, fps) {
    const dec = S.zoom * Math.LOG10E * Math.LN2;
    $('stZoom').textContent = '10^' + dec.toFixed(2);
    $('stIter').textContent = maxIter.toLocaleString();
    $('stBits').textContent = S.cbits + ' bits';
    const pct = Math.round(S.fraction * 100);
    $('stRender').textContent = active ? `${pct}% / frame · ${fps.toFixed(0)} fps` : 'idle';
    const r = REF.cur, refEl = $('stRef');
    if (REF.pending) {
      refEl.textContent = `computing ${REF.pending.len.toLocaleString()} / ${REF.pending.target.toLocaleString()}`;
      refEl.className = 'v warn';
    } else if (r) {
      refEl.textContent = (r.done ? '' : 'extending · ') + r.len.toLocaleString() + (r.escaped ? ' (escaped)' : '');
      refEl.className = r.done ? 'v' : 'v warn';
    }
  }

  // ---------------------------------------------------------------- input
  const pointers = new Map();
  let drag = null;   // {x, y, t, vx, vy}
  let pinch = null;  // {dist, mid}

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    const p = toGl(e);
    pointers.set(e.pointerId, p);
    if (pointers.size === 1) {
      drag = { x: p[0], y: p[1], t: performance.now(), vx: 0, vy: 0 };
      S.panVel[0] = S.panVel[1] = 0;
      pinch = null;
    } else if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = { dist: Math.hypot(a[0] - b[0], a[1] - b[1]), mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] };
      drag = null;
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    const p = toGl(e);
    if (e.pointerType === 'mouse') S.cursor = p;
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, p);
    if (pointers.size === 1 && drag) {
      const now = performance.now();
      const dx = p[0] - drag.x, dy = p[1] - drag.y;
      const ddt = Math.max(1, now - drag.t) / 1000;
      S.pendingPan[0] += dx; S.pendingPan[1] += dy;
      drag.vx = drag.vx * 0.5 + (dx / ddt) * 0.5;
      drag.vy = drag.vy * 0.5 + (dy / ddt) * 0.5;
      drag.x = p[0]; drag.y = p[1]; drag.t = now;
    } else if (pointers.size === 2 && pinch) {
      const [a, b] = [...pointers.values()];
      const dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      if (dist > 0 && pinch.dist > 0) {
        S.pinchZoom += Math.log2(dist / pinch.dist);
        S.pinchAnchor = mid;
      }
      S.pendingPan[0] += mid[0] - pinch.mid[0];
      S.pendingPan[1] += mid[1] - pinch.mid[1];
      pinch = { dist, mid };
    }
  });

  const endPointer = (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);
    if (drag && pointers.size === 0) {
      if (performance.now() - drag.t < 80 && Math.hypot(drag.vx, drag.vy) > 50) {
        S.panVel[0] = drag.vx; S.panVel[1] = drag.vy;
      }
      drag = null;
    }
    if (pointers.size < 2) pinch = null;
    if (pointers.size === 1) {
      const p = [...pointers.values()][0];
      drag = { x: p[0], y: p[1], t: performance.now(), vx: 0, vy: 0 };
    }
  };
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);
  canvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') S.cursor = null; });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const notches = e.deltaMode === 1 ? e.deltaY / 3 : e.deltaMode === 2 ? e.deltaY : e.deltaY / 100;
    S.wheelVel += (-notches * ZOOM_PER_NOTCH) / WHEEL_TAU;
    S.wheelAnchor = toGl(e);
  }, { passive: false });

  canvas.addEventListener('dblclick', (e) => {
    S.wheelAnchor = toGl(e);
    S.wheelVel += 3 / WHEEL_TAU;
  });

  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) e.target.blur();
    switch (e.key) {
      case 'w': case 'W': case '+': case '=': S.keyZoom = 1; break;
      case 's': case 'S': case '-': case '_': S.keyZoom = -1; break;
      case 'ArrowLeft': S.keyPan[0] = 1; break;
      case 'ArrowRight': S.keyPan[0] = -1; break;
      case 'ArrowUp': S.keyPan[1] = -1; break;
      case 'ArrowDown': S.keyPan[1] = 1; break;
      case ' ': setAuto(!S.autoZoom); e.preventDefault(); break;
      case '[': setDensity(S.density - 0.25); break;
      case ']': setDensity(S.density + 0.25); break;
      case ',': setIterExp(S.iterExp - 0.25); break;
      case '.': setIterExp(S.iterExp + 0.25); break;
      case 'c': case 'C': setPalette((S.palette + 1) % PALETTES.length); break;
      case 'r': case 'R': resetView(); break;
      case 'h': case 'H': $('help').classList.toggle('hidden'); break;
      default: return;
    }
    if (e.key.startsWith('Arrow')) e.preventDefault();
  });
  window.addEventListener('keyup', (e) => {
    switch (e.key) {
      case 'w': case 'W': case '+': case '=': if (S.keyZoom > 0) S.keyZoom = 0; break;
      case 's': case 'S': case '-': case '_': if (S.keyZoom < 0) S.keyZoom = 0; break;
      case 'ArrowLeft': case 'ArrowRight': S.keyPan[0] = 0; break;
      case 'ArrowUp': case 'ArrowDown': S.keyPan[1] = 0; break;
    }
  });
  window.addEventListener('blur', () => { S.keyZoom = 0; S.keyPan[0] = S.keyPan[1] = 0; });

  // ---------------------------------------------------------------- UI
  function setAuto(on) {
    S.autoZoom = on;
    $('btnAuto').classList.toggle('on', on);
    $('btnAuto').textContent = on ? '❚❚ Auto zoom' : '▶ Auto zoom';
  }
  function setDensity(v) {
    S.density = Math.min(12, Math.max(2, v));
    $('inDensity').value = S.density;
    S.presentDirty = true;
  }
  function setIterExp(v) {
    S.iterExp = Math.min(4, Math.max(-2, v));
    $('inIter').value = S.iterExp;
  }
  function setPalette(i) {
    S.palette = i;
    $('selPalette').value = i;
    buildPalette(PALETTES[i]);
    S.presentDirty = true;
  }
  function resetView() {
    S.cbits = 128;
    S.cx = BigFixed.fromNumber(-0.6, S.cbits);
    S.cy = 0n;
    S.zoom = 0;
    S.wheelVel = 0; S.panVel[0] = S.panVel[1] = 0;
    setAuto(false);
    S.havePrev = false;
    S.forceCompute = true;
  }
  function toast(msg) {
    const el = $('toast');
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.add('hidden'), 1800);
  }

  function viewToHash() {
    return '#c=' + BigFixed.toHex(S.cx) + ',' + BigFixed.toHex(S.cy) + ',' + S.cbits
      + '&z=' + S.zoom.toFixed(4) + '&i=' + S.iterExp.toFixed(2) + '&d=' + S.density.toFixed(2) + '&p=' + S.palette;
  }
  function hashToView(hash) {
    try {
      const q = new URLSearchParams(hash.replace(/^#/, ''));
      const c = (q.get('c') || '').split(',');
      if (c.length !== 3) return false;
      const bits = parseInt(c[2], 10);
      if (!(bits > 0)) return false;
      S.cx = BigFixed.fromHex(c[0]);
      S.cy = BigFixed.fromHex(c[1]);
      S.cbits = bits;
      S.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, parseFloat(q.get('z')) || 0));
      if (q.has('i')) setIterExp(parseFloat(q.get('i')) || 0);
      if (q.has('d')) setDensity(parseFloat(q.get('d')) || 6);
      if (q.has('p')) setPalette(Math.min(PALETTES.length - 1, Math.max(0, parseInt(q.get('p'), 10) || 0)));
      return true;
    } catch (_) { return false; }
  }

  for (let i = 0; i < PALETTES.length; i++) {
    const o = document.createElement('option');
    o.value = i; o.textContent = PALETTES[i].name;
    $('selPalette').appendChild(o);
  }
  $('selPalette').addEventListener('change', (e) => setPalette(parseInt(e.target.value, 10)));
  $('inDensity').addEventListener('input', (e) => setDensity(parseFloat(e.target.value)));
  $('inIter').addEventListener('input', (e) => setIterExp(parseFloat(e.target.value)));
  $('inSpeed').addEventListener('input', (e) => { S.speed = parseFloat(e.target.value); });
  $('btnAuto').addEventListener('click', () => setAuto(!S.autoZoom));
  $('btnReset').addEventListener('click', resetView);
  $('btnHelp').addEventListener('click', () => $('help').classList.toggle('hidden'));
  const out = $('btnOut');
  out.addEventListener('pointerdown', () => { S.keyZoom = -1; setAuto(false); });
  const outUp = () => { if (S.keyZoom < 0) S.keyZoom = 0; };
  out.addEventListener('pointerup', outUp);
  out.addEventListener('pointerleave', outUp);
  out.addEventListener('pointercancel', outUp);
  $('btnLink').addEventListener('click', async () => {
    const url = location.href.split('#')[0] + viewToHash();
    history.replaceState(null, '', viewToHash());
    try { await navigator.clipboard.writeText(url); toast('Link copied'); }
    catch (_) { toast('Link is in the address bar'); }
  });
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    $('fatal').textContent = 'The GPU context was lost. Reload the page to continue.';
    $('fatal').classList.remove('hidden');
  });

  S.speed = parseFloat($('inSpeed').value);
  resetView();
  setPalette(0);
  if (!hashToView(location.hash)) setDensity(6);
  requestAnimationFrame(frame);
})();
