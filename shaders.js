// GLSL sources. The compute pass evaluates the Mandelbrot set with perturbation theory:
// each pixel iterates a small delta against a high-precision reference orbit (uploaded as
// a float texture), carrying its own power-of-two exponent so deltas far below float32
// range stay exact. Deltas rebase onto the start of the reference whenever the orbit
// passes closer to zero than the delta itself, which removes perturbation glitches.
'use strict';

const SHADER_VS = `#version 300 es
precision highp float;
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() { gl_Position = vec4(P[gl_VertexID], 0.0, 1.0); }
`;

const SHADER_COMPUTE_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform sampler2D uRef;      // reference orbit, RG32F, REFW texels wide
uniform int uRefLen;
uniform int uRefEscaped;
uniform sampler2D uPrev;     // previous frame's smooth-iteration buffer
uniform int uHavePrev;
uniform vec2 uRes;
uniform vec2 uOffsetPx;      // (view centre - reference point) / pixel size, in pixels
uniform float uPm;           // pixel size = uPm * 2^uEc, uPm in [1, 2)
uniform int uEc;
uniform int uMaxIter;
uniform int uFrame;
uniform float uFraction;     // fraction of pixels recomputed this frame
uniform float uPrevScale;    // reprojection: prevPx = c + (px - c) * scale + shift
uniform vec2 uPrevShift;
uniform int uRefGrew;        // 1 when the reference orbit gained entries since last frame

out vec4 outColor;

#define REFW 2048
#define REFSHIFT 11
#define BAIL 65536.0
#define MU_INTERIOR -1.0
#define MU_NOTREADY -2.0
#define MU_NONE     -3.0

const int BAYER[64] = int[64](
   0, 32,  8, 40,  2, 34, 10, 42,
  48, 16, 56, 24, 50, 18, 58, 26,
  12, 44,  4, 36, 14, 46,  6, 38,
  60, 28, 52, 20, 62, 30, 54, 22,
   3, 35, 11, 43,  1, 33,  9, 41,
  51, 19, 59, 27, 49, 17, 57, 25,
  15, 47,  7, 39, 13, 45,  5, 37,
  63, 31, 55, 23, 61, 29, 53, 21);

vec2 refAt(int i) {
  return texelFetch(uRef, ivec2(i & (REFW - 1), i >> REFSHIFT), 0).rg;
}

// 2^e split into three factors that are each representable; applied one at a time so
// intermediate products saturate correctly (to 0 or inf) instead of producing NaN.
vec3 expParts(int e) {
  int e1 = clamp(e, -126, 127); e -= e1;
  int e2 = clamp(e, -126, 127); e -= e2;
  int e3 = clamp(e, -126, 127);
  return vec3(exp2(float(e1)), exp2(float(e2)), exp2(float(e3)));
}
vec2 scale2(vec2 v, vec3 p) {
  v = v * p.x;
  v = v * p.y;
  v = v * p.z;
  return v;
}

float fetchPrev(ivec2 p) {
  p = clamp(p, ivec2(0), ivec2(uRes) - 1);
  return texelFetch(uPrev, p, 0).r;
}

float samplePrev(vec2 pp) {
  vec2 q = pp - 0.5;
  vec2 f = fract(q);
  ivec2 i = ivec2(floor(q));
  float a = fetchPrev(i);
  float b = fetchPrev(i + ivec2(1, 0));
  float c = fetchPrev(i + ivec2(0, 1));
  float d = fetchPrev(i + ivec2(1, 1));
  if (min(min(a, b), min(c, d)) >= 0.0) {
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }
  return fetchPrev(ivec2(floor(pp)));
}

float iterate(vec2 p) {
  if (uRefLen < 2) return MU_NOTREADY;
  int ec = uEc;
  vec2 dcM = (p - 0.5 * uRes + uOffsetPx) * uPm;   // delta c = dcM * 2^ec
  vec2 d = dcM;                                    // delta   = d   * 2^e
  int e = ec;
  vec3 P = expParts(e);        // 2^e
  vec3 N = expParts(-e);       // 2^-e
  vec3 C = expParts(0);        // 2^(ec - e)
  int m = 1;                   // reference index: z_n = Z[m] + delta
  vec2 Z;

  for (int n = 1; n < uMaxIter; n++) {
    if (m >= uRefLen) return MU_NOTREADY;
    Z = refAt(m);
    vec2 z = Z + scale2(d, P);
    float zz = dot(z, z);
    if (zz > BAIL) {
      return float(n) + 4.0 - log2(log2(zz));
    }
    if (m == uRefLen - 1) {
      // Reference escaped here but this pixel did not: continue from an absolute restart.
      d = z; e = 0; m = 0; Z = vec2(0.0);
      P = expParts(0); N = P; C = expParts(ec);
    } else {
      vec2 zs = scale2(Z, N) + d;
      if (dot(zs, zs) < dot(d, d)) {
        d = zs; m = 0; Z = vec2(0.0);
      }
    }

    if (ec - e > 100) {
      // delta is negligible next to delta-c: next delta is delta-c to float precision.
      d = dcM; e = ec;
      P = expParts(e); N = expParts(-e); C = expParts(0);
    } else {
      vec2 dsq = vec2(d.x * d.x - d.y * d.y, 2.0 * d.x * d.y);
      d = 2.0 * vec2(Z.x * d.x - Z.y * d.y, Z.x * d.y + Z.y * d.x) + scale2(dsq, P) + scale2(dcM, C);
    }
    m++;

    float mag = max(abs(d.x), abs(d.y));
    if (mag > 65536.0 || mag < 1.52587890625e-5) {
      if (mag > 0.0) {
        int sh = int(floor(log2(mag)));
        d *= exp2(float(-sh));
        e += sh;
        P = expParts(e); N = expParts(-e); C = expParts(ec - e);
      }
    }
  }
  return MU_INTERIOR;
}

void main() {
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec2 p = gl_FragCoord.xy;

  float prev = MU_NONE;
  if (uHavePrev == 1) {
    vec2 pp = 0.5 * uRes + (p - 0.5 * uRes) * uPrevScale + uPrevShift;
    if (all(greaterThanEqual(pp, vec2(0.5))) && all(lessThanEqual(pp, uRes - 0.5))) {
      prev = samplePrev(pp);
    }
  }

  int slot = (BAYER[(ip.y & 7) * 8 + (ip.x & 7)] + uFrame * 39) & 63;
  bool none = prev < MU_NOTREADY - 0.5;
  bool must = none || (prev < MU_INTERIOR - 0.5 && uRefGrew == 1);
  bool doCompute = must || (float(slot) + 0.5 < uFraction * 64.0);

  float mu = prev;
  if (doCompute) {
    float c = iterate(p);
    if (c > MU_NOTREADY + 0.5) mu = c;
    else if (must) mu = MU_NOTREADY;
  }
  outColor = vec4(mu, 0.0, 0.0, 1.0);
}
`;

const SHADER_DISPLAY_FS = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uMu;
uniform sampler2D uPal;
uniform float uDensityInv;
uniform float uPhase;
uniform vec3 uInterior;
out vec4 outColor;
void main() {
  float mu = texelFetch(uMu, ivec2(gl_FragCoord.xy), 0).r;
  vec3 c;
  if (mu < 0.0) {
    c = uInterior;
  } else {
    float t = fract(mu * uDensityInv + uPhase);
    c = texture(uPal, vec2(t, 0.5)).rgb;
  }
  outColor = vec4(c, 1.0);
}
`;

// Summarises blocks of the iteration buffer for the autopilot:
// r = interior fraction, g = max smooth iteration, b = mean of escaped pixels, a = unresolved fraction.
const SHADER_ANALYZE_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uMu;
uniform vec2 uRes;
uniform ivec2 uCells;
out vec4 outColor;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 cs = uRes / vec2(uCells);
  ivec2 p0 = ivec2(vec2(c) * cs);
  ivec2 p1 = min(ivec2(vec2(c + 1) * cs), ivec2(uRes));
  ivec2 step = max(ivec2(1), (p1 - p0) / 12);
  float n = 0.0, inside = 0.0, mx = 0.0, sum = 0.0, unknown = 0.0;
  for (int y = p0.y; y < p1.y; y += step.y) {
    for (int x = p0.x; x < p1.x; x += step.x) {
      float mu = texelFetch(uMu, ivec2(x, y), 0).r;
      n += 1.0;
      if (mu >= 0.0) { mx = max(mx, mu); sum += mu; }
      else if (mu > -1.5) inside += 1.0;
      else unknown += 1.0;
    }
  }
  float esc = max(1.0, n - inside - unknown);
  outColor = vec4(inside / max(1.0, n), mx, sum / esc, unknown / max(1.0, n));
}
`;
