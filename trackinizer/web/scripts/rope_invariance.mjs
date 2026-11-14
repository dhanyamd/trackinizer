// DOES THE RoPE DOT PRODUCT REALLY ONLY SEE THE GAP? Measure it, do not assert it.
//
// One seeded head (d = 128, base = 10000, mulberry32 seed 0 through Box-Muller), four
// measurements, one JSON object on stdout. Every number on the canvas comes from here.
//
//   1. SLIDE  — hold the gap at m - n = 4 and slide the pair along the sequence,
//               s = 0 .. 4095. Record the drawn plane's dot product and the whole
//               128-dim head's, against the closed form that names no position at all.
//   2. GAP    — sweep the gap from -32 to +32 at two different anchors (37 and 1000)
//               and compare the two curves: if the score is a function of the gap, the
//               anchor cannot matter.
//   3. FP32   — repeat the slide with the rotation ANGLE rounded to single precision
//               (Math.fround(m * theta)) and everything else in double. This isolates
//               one mechanism: how much of the absolute position survives rounding.
//               It is a model of an fp32 kernel's angle, not a full fp32 kernel.
//   4. BREAK  — rotate only q, which is absolute position rather than relative, and
//               slide the same pair. The quantity that was pinned now moves.
//
// Run: node scripts/rope_invariance.mjs

const D = 128;
const BASE = 10000;
// The plane the canvas draws. Picked for legibility, not for the result: theta_7 turns
// 20.9 degrees per step, so one step of the slide is visible and 18 steps close a circle,
// and this head's q and k have comparable length there (0.81 and 0.97). The invariance is
// per-plane, so every other plane behaves the same and plane 7 is not special.
const PLANE = 7;
const GAP = 4; // m - n, held fixed while the pair slides
const SHIFTS = 4096;

/** theta_i = base^(-2i/d): the per-plane angular frequency RoPE defines. */
const THETA = Array.from({ length: D / 2 }, (_, i) => BASE ** ((-2 * i) / D));

/** mulberry32: a small deterministic generator, so the head is the same head every run. */
function mulberry32(a) {
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** n standard normals, Box-Muller over the generator above. */
function normals(n, seed) {
  const rand = mulberry32(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

/** Rotate every 2-D plane of `vec` by its own frequency times the position. */
function rope(vec, pos, theta, roundAngle = false) {
  const out = new Array(vec.length);
  for (let i = 0; i < theta.length; i++) {
    const ang = roundAngle ? Math.fround(pos * theta[i]) : pos * theta[i];
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const a = vec[2 * i];
    const b = vec[2 * i + 1];
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  return out;
}

/** The score RoPE actually computes: rotate each end by its own position, then dot. */
function headDot(q, k, m, n, theta, roundAngle = false) {
  const rq = rope(q, m, theta, roundAngle);
  const rk = rope(k, n, theta, roundAngle);
  let acc = 0;
  for (let i = 0; i < rq.length; i++) acc += rq[i] * rk[i];
  return acc;
}

const q = normals(D, 0);
const k = normals(D, 12345);

// The plane the canvas draws, in polar form: a 2-D dot product is |q||k| cos(angle
// between), and each rotation only moves an angle.
const qp = [q[2 * PLANE], q[2 * PLANE + 1]];
const kp = [k[2 * PLANE], k[2 * PLANE + 1]];
const rQ = Math.hypot(qp[0], qp[1]);
const rK = Math.hypot(kp[0], kp[1]);
const phiQ = Math.atan2(qp[1], qp[0]);
const phiK = Math.atan2(kp[1], kp[0]);
const th0 = THETA[PLANE];

// The closed forms, evaluated once. Neither names an absolute position.
const planeClosed = rQ * rK * Math.cos(phiK - phiQ - GAP * th0);
let headClosed = 0;
for (let i = 0; i < D / 2; i++) {
  const a = Math.hypot(q[2 * i], q[2 * i + 1]);
  const b = Math.hypot(k[2 * i], k[2 * i + 1]);
  const dphi = Math.atan2(k[2 * i + 1], k[2 * i]) - Math.atan2(q[2 * i + 1], q[2 * i]);
  headClosed += a * b * Math.cos(dphi - GAP * THETA[i]);
}

const plane64 = [];
const head64 = [];
const head32a = [];
for (let s = 0; s < SHIFTS; s++) {
  plane64.push(headDot(qp, kp, s + GAP, s, [th0]));
  head64.push(headDot(q, k, s + GAP, s, THETA));
  head32a.push(headDot(q, k, s + GAP, s, THETA, true));
}
const dev = (xs, ref) => xs.reduce((mx, v) => Math.max(mx, Math.abs(v - ref)), 0);

// Every plane on its own, so the sum's invariance is not the only evidence: the head is
// 64 independent 2-D rotations, and the claim is made plane by plane.
let perPlaneMaxDev = 0;
let worstPlane = -1;
for (let i = 0; i < D / 2; i++) {
  const qi = [q[2 * i], q[2 * i + 1]];
  const ki = [k[2 * i], k[2 * i + 1]];
  const dphi = Math.atan2(ki[1], ki[0]) - Math.atan2(qi[1], qi[0]);
  const closed = Math.hypot(...qi) * Math.hypot(...ki) * Math.cos(dphi - GAP * THETA[i]);
  for (let s = 0; s < SHIFTS; s += 1) {
    const e = Math.abs(headDot(qi, ki, s + GAP, s, [THETA[i]]) - closed);
    if (e > perPlaneMaxDev) { perPlaneMaxDev = e; worstPlane = i; }
  }
}

const gaps = Array.from({ length: 65 }, (_, i) => i - 32);
const gapAt = (anchor) => gaps.map((g) => headDot(q, k, anchor + g, anchor, THETA));
const gapA = gapAt(1000);
const gapB = gapAt(37);

// The break: only q carries a position, which is what an absolute scheme does.
const broken = [];
for (let s = 0; s < 512; s++) broken.push(headDot(q, k, s + GAP, 0, THETA));

const pick = (xs, step) => xs.map((v, i) => [i, v]).filter(([i]) => i % step === 0);

const out = {
  config: { d: D, base: BASE, plane: PLANE, theta_plane: th0, gap: GAP, shifts: SHIFTS, seeds: [0, 12345] },
  plane: {
    q: qp, k: kp, r_q: rQ, r_k: rK, phi_q: phiQ, phi_k: phiK,
    closed_form: planeClosed,
    dot_at_s0: plane64[0], dot_at_s4095: plane64[SHIFTS - 1],
    max_abs_dev_f64: dev(plane64, planeClosed),
    range_f64: Math.max(...plane64) - Math.min(...plane64),
  },
  head: {
    closed_form: headClosed,
    dot_at_s0: head64[0], dot_at_s4095: head64[SHIFTS - 1],
    range_f64: Math.max(...head64) - Math.min(...head64),
    max_abs_dev_f64: dev(head64, headClosed),
    max_abs_dev_f32angle: dev(head32a, headClosed),
    per_plane_max_dev_f64: perPlaneMaxDev,
    per_plane_worst: worstPlane,
    f32angle_dev_at: Object.fromEntries([0, 64, 256, 1024, 4095].map((s) => [s, Math.abs(head32a[s] - headClosed)])),
    f32angle_dev_curve: pick(head32a.map((v) => Math.abs(v - headClosed)), 64),
  },
  gap_curve: {
    gaps, at_anchor_1000: gapA, at_anchor_37: gapB,
    max_anchor_disagreement: Math.max(...gaps.map((_, i) => Math.abs(gapA[i] - gapB[i]))),
  },
  broken: {
    range: Math.max(...broken) - Math.min(...broken),
    min: Math.min(...broken), max: Math.max(...broken), first: broken[0],
    curve: pick(broken, 8),
  },
  slide_rows: [0, 1, 7, 64, 256, 1024, 4095].map((s) => [s, plane64[s], head64[s], head32a[s]]),
};
process.stdout.write(JSON.stringify(out));
