// Fixed-point BigInt helpers and the reference-orbit worker.
// A coordinate is stored as a BigInt `v` with `bits` fractional bits: value = v * 2^-bits.
// The worker is created from a Blob so the page also works when opened from file://.
'use strict';

const BigFixed = (() => {
  const HEXBITS = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];

  function bitLength(v) {
    if (v < 0n) v = -v;
    if (v === 0n) return 0;
    const h = v.toString(16);
    return (h.length - 1) * 4 + HEXBITS[parseInt(h[0], 16)];
  }

  // Multiply by 2^delta (delta may be negative; right shift floors).
  function shift(v, delta) {
    return delta >= 0 ? v << BigInt(delta) : v >> BigInt(-delta);
  }

  // m * 2^e (m a normal double) -> fixed point with `bits` fractional bits.
  function fromFloatExp(m, e, bits) {
    if (m === 0 || !isFinite(m)) return 0n;
    const ee = Math.floor(Math.log2(Math.abs(m)));
    const mi = BigInt(Math.round(m * Math.pow(2, 52 - ee)));
    return shift(mi, bits + e + ee - 52);
  }

  function fromNumber(f, bits) {
    return fromFloatExp(f, 0, bits);
  }

  // fixed point -> {m, e} with value = m * 2^e and |m| in [2^52, 2^53) (or m = 0).
  function toFloatExp(v, bits) {
    if (v === 0n) return { m: 0, e: 0 };
    const neg = v < 0n;
    const a = neg ? -v : v;
    const bl = bitLength(a);
    const sh = bl - 53;
    const mi = sh > 0 ? a >> BigInt(sh) : a << BigInt(-sh);
    const m = Number(mi);
    return { m: neg ? -m : m, e: sh - bits };
  }

  function toNumber(v, bits) {
    const fe = toFloatExp(v, bits);
    return fe.m * Math.pow(2, fe.e);
  }

  function toHex(v) {
    return (v < 0n ? '-' : '') + (v < 0n ? -v : v).toString(16);
  }

  function fromHex(s) {
    return s[0] === '-' ? -BigInt('0x' + s.slice(1)) : BigInt('0x' + s);
  }

  return { bitLength, shift, fromFloatExp, fromNumber, toFloatExp, toNumber, toHex, fromHex };
})();

// ---------------------------------------------------------------------------
// Reference orbit worker. Iterates z -> z^2 + c for the view centre in fixed-point
// BigInt arithmetic and streams the orbit back as float32 pairs in chunks.
// ---------------------------------------------------------------------------
function refWorkerMain() {
  const HEXBITS = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
  const CHUNK = 2048;
  const BIG_MIN = 1n << 30n;
  let s = null;
  let scheduled = false;

  function toFloat(v, s) {
    if (v === 0n) return 0;
    const neg = v < 0n;
    const a = neg ? -v : v;
    // Fast path: the value is at least 2^-30 in magnitude, so the top 60 bits are enough.
    const big = a >> s.fastShift;
    let f;
    if (big >= BIG_MIN) {
      f = Number(big) * s.fastScale;
    } else {
      const h = a.toString(16);
      const bl = (h.length - 1) * 4 + HEXBITS[parseInt(h[0], 16)];
      const sh = bl - 53;
      const mi = sh > 0 ? a >> BigInt(sh) : a;
      f = Number(mi) * Math.pow(2, (sh > 0 ? sh : 0) - s.bits);
    }
    return neg ? -f : f;
  }

  function run() {
    scheduled = false;
    if (!s || s.escaped || s.n >= s.target) return;
    const count = Math.min(CHUNK, s.target - s.n);
    const buf = new Float32Array(count * 2);
    let zx = s.zx, zy = s.zy, i = 0;
    const bb = s.bbits, bb1 = bb - 1n;
    for (; i < count; i++) {
      const fx = toFloat(zx, s), fy = toFloat(zy, s);
      buf[2 * i] = fx;
      buf[2 * i + 1] = fy;
      if (fx * fx + fy * fy > 1e10) { s.escaped = true; i++; break; }
      const xx = (zx * zx) >> bb;
      const yy = (zy * zy) >> bb;
      const xy = (zx * zy) >> bb1;
      zx = xx - yy + s.cx;
      zy = xy + s.cy;
    }
    s.zx = zx; s.zy = zy;
    const start = s.n;
    s.n += i;
    const out = i === count ? buf : buf.slice(0, i * 2);
    const done = s.escaped || s.n >= s.target;
    self.postMessage({ type: 'chunk', id: s.id, start, data: out, escaped: s.escaped, done, len: s.n }, [out.buffer]);
    if (!done && !scheduled) { scheduled = true; setTimeout(run, 0); }
  }

  self.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'start') {
      const bits = msg.bits;
      s = {
        id: msg.id, bits, bbits: BigInt(bits),
        cx: BigInt(msg.cx), cy: BigInt(msg.cy),
        zx: 0n, zy: 0n, n: 0, escaped: false, target: msg.iters,
        fastShift: BigInt(Math.max(0, bits - 60)),
        fastScale: Math.pow(2, Math.max(0, bits - 60) - bits),
      };
    } else if (msg.type === 'extend') {
      if (!s || s.id !== msg.id) return;
      s.target = Math.max(s.target, msg.iters);
    } else {
      return;
    }
    if (!scheduled) { scheduled = true; setTimeout(run, 0); }
  };
}

function createRefWorker() {
  const src = '(' + refWorkerMain.toString() + ')();';
  const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  const w = new Worker(url);
  URL.revokeObjectURL(url);
  return w;
}
