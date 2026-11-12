/**
 * Exact small-sample arithmetic for seed-count divergence evidence.
 *
 * Written for Belief#140 ("A 2k-step warmup prevents early divergence at 1.3B"),
 * whose stated evidence is 0 divergences in 10 seeds at >=2,000 warmup steps
 * against 3 in 10 at 500. Three questions, three exported entry points:
 *
 *   1. Could one divergence rate have produced that split?   `fisherExact`
 *   2. How many seeds per arm would make it decisive?        `smallestDecisivePerArm`
 *                                                           `smallestPerArmForPower`
 *   3. What rate is still compatible with N clean seeds?     `clopperPearsonUpper`
 *
 * Counts here are tiny (tens of seeds), so every binomial coefficient is an exact
 * BigInt and no log-gamma approximation is used. Probabilities are ratios of those
 * exact integers, converted to Number only at the end.
 */

/** Exact binomial coefficient. */
export function choose(n: number, k: number): bigint {
  if (!Number.isInteger(n) || !Number.isInteger(k)) throw new Error("choose needs integers");
  if (k < 0 || n < 0 || k > n) return 0n;
  const kk = BigInt(Math.min(k, n - k));
  let num = 1n;
  let den = 1n;
  for (let i = 0n; i < kk; i += 1n) {
    num *= BigInt(n) - i;
    den *= i + 1n;
  }
  return num / den;
}

/** A 2x2 table of seeds: one row per warmup arm. */
export interface Arm {
  /** Seeds that diverged. */
  readonly diverged: number;
  /** Seeds run. */
  readonly seeds: number;
}

const clean = (arm: Arm): number => arm.seeds - arm.diverged;

/** Hypergeometric probability of `a` divergences landing in the first arm. */
function tableProbability(a: number, treat: Arm, control: Arm): number {
  const total = treat.seeds + control.seeds;
  const divergences = treat.diverged + control.diverged;
  const num = choose(divergences, a) * choose(total - divergences, treat.seeds - a);
  const den = choose(total, treat.seeds);
  if (den === 0n) return 0;
  return Number(num) / Number(den);
}

export interface FisherResult {
  /** P(as few or fewer divergences in the treatment arm), conditioning on the margins. */
  readonly oneSided: number;
  /** Sum of every table at most as probable as the observed one. */
  readonly twoSided: number;
  /** The observed table's own conditional probability. */
  readonly observed: number;
}

/**
 * Fisher's exact test on the 2x2 seed table. The null is a single divergence rate
 * shared by both arms; conditioning on the margins leaves a hypergeometric, which
 * is enumerated in full rather than approximated.
 */
export function fisherExact(treat: Arm, control: Arm): FisherResult {
  for (const arm of [treat, control]) {
    if (arm.diverged < 0 || arm.diverged > arm.seeds) throw new Error("diverged must be in [0, seeds]");
  }
  const divergences = treat.diverged + control.diverged;
  const lo = Math.max(0, divergences - control.seeds);
  const hi = Math.min(divergences, treat.seeds);
  const observed = tableProbability(treat.diverged, treat, control);

  let oneSided = 0;
  let twoSided = 0;
  for (let a = lo; a <= hi; a += 1) {
    const p = tableProbability(a, treat, control);
    if (a <= treat.diverged) oneSided += p;
    // The 1 + 1e-12 slack keeps a table that is exactly as probable as the observed
    // one inside the two-sided sum despite float rounding in the ratio above.
    if (p <= observed * (1 + 1e-12)) twoSided += p;
  }
  return { oneSided, twoSided, observed: Math.min(observed, 1) };
}

/** Exact binomial pmf, P(X = k) for X ~ Binomial(n, p). */
export function binomialPmf(k: number, n: number, p: number): number {
  if (k < 0 || k > n) return 0;
  if (p === 0) return k === 0 ? 1 : 0;
  if (p === 1) return k === n ? 1 : 0;
  return Number(choose(n, k)) * p ** k * (1 - p) ** (n - k);
}

/** Exact binomial cdf, P(X <= k). */
export function binomialCdf(k: number, n: number, p: number): number {
  let total = 0;
  for (let i = 0; i <= Math.min(k, n); i += 1) total += binomialPmf(i, n, p);
  return Math.min(total, 1);
}

/**
 * Clopper-Pearson one-sided upper confidence limit: the largest true rate under
 * which `k` or fewer events in `n` trials still has probability `alpha`.
 *
 * For k = 0 this has the closed form 1 - alpha^(1/n); the bisection below is run
 * for every k and the closed form is what the test checks it against.
 */
export function clopperPearsonUpper(k: number, n: number, alpha = 0.05): number {
  if (n <= 0) throw new Error("n must be positive");
  if (k < 0 || k > n) throw new Error("k must be in [0, n]");
  if (k === n) return 1;
  let lo = k / n;
  let hi = 1;
  for (let i = 0; i < 200; i += 1) {
    const mid = (lo + hi) / 2;
    if (binomialCdf(k, n, mid) > alpha) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Clopper-Pearson one-sided lower confidence limit. */
export function clopperPearsonLower(k: number, n: number, alpha = 0.05): number {
  if (k === 0) return 0;
  let lo = 0;
  let hi = k / n;
  for (let i = 0; i < 200; i += 1) {
    const mid = (lo + hi) / 2;
    // P(X >= k) = 1 - P(X <= k - 1)
    if (1 - binomialCdf(k - 1, n, mid) > alpha) hi = mid;
    else lo = mid;
  }
  return (lo + hi) / 2;
}

/** The two-sided Clopper-Pearson interval at level 1 - 2*alpha per tail. */
export function clopperPearsonInterval(k: number, n: number, alpha = 0.05): readonly [number, number] {
  return [clopperPearsonLower(k, n, alpha / 2), clopperPearsonUpper(k, n, alpha / 2)];
}

/**
 * The smallest clean-seed count whose Clopper-Pearson upper limit falls below
 * `rate` -- how many consecutive clean runs it takes to retire a residual rate.
 *
 * Closed form: n > ln(alpha) / ln(1 - rate), so the loop only confirms the ceiling.
 */
export function seedsToRetireRate(rate: number, alpha = 0.05, maxN = 100_000): number {
  if (rate <= 0 || rate >= 1) throw new Error("rate must be in (0, 1)");
  const exact = Math.log(alpha) / Math.log(1 - rate);
  const n = Math.max(1, Math.ceil(exact));
  if (n > maxN) throw new Error(`needs more than ${maxN} seeds`);
  // Guard the ceiling against a boundary case landing exactly on an integer.
  return clopperPearsonUpper(0, n, alpha) < rate ? n : n + 1;
}

export interface DecisiveResult {
  /** Seeds per arm. */
  readonly perArm: number;
  /** Divergences the control arm shows at that size, at the reported rate. */
  readonly controlDiverged: number;
  readonly oneSided: number;
  readonly twoSided: number;
}

/**
 * Hold the reported rates fixed and grow both arms until Fisher's exact test
 * clears `alpha`. The control arm's count is `round(controlRate * n)`, so the
 * p-value is not monotone in n -- it jumps whenever that rounding ticks over,
 * which is why this scans rather than solving.
 */
export function smallestDecisivePerArm(options: {
  readonly controlRate: number;
  readonly treatRate: number;
  readonly alpha?: number;
  readonly sided?: "one" | "two";
  readonly maxPerArm?: number;
}): DecisiveResult {
  const { controlRate, treatRate, alpha = 0.05, sided = "one", maxPerArm = 400 } = options;
  for (let n = 2; n <= maxPerArm; n += 1) {
    const control: Arm = { diverged: Math.round(controlRate * n), seeds: n };
    const treat: Arm = { diverged: Math.round(treatRate * n), seeds: n };
    const { oneSided, twoSided } = fisherExact(treat, control);
    if ((sided === "one" ? oneSided : twoSided) < alpha) {
      return { perArm: n, controlDiverged: control.diverged, oneSided, twoSided };
    }
  }
  throw new Error(`no decisive size at or below ${maxPerArm} seeds per arm`);
}

/**
 * Probability that a trial of `perArm` seeds per arm returns a significant Fisher
 * result, when the true rates really are `controlRate` and `treatRate`. The full
 * joint binomial over both arms' outcomes is enumerated; nothing is simulated.
 */
export function power(options: {
  readonly perArm: number;
  readonly controlRate: number;
  readonly treatRate: number;
  readonly alpha?: number;
  readonly sided?: "one" | "two";
}): number {
  const { perArm, controlRate, treatRate, alpha = 0.05, sided = "one" } = options;
  let total = 0;
  for (let c = 0; c <= perArm; c += 1) {
    const pc = binomialPmf(c, perArm, controlRate);
    if (pc === 0) continue;
    for (let t = 0; t <= perArm; t += 1) {
      const pt = binomialPmf(t, perArm, treatRate);
      if (pt === 0) continue;
      const r = fisherExact({ diverged: t, seeds: perArm }, { diverged: c, seeds: perArm });
      if ((sided === "one" ? r.oneSided : r.twoSided) < alpha) total += pc * pt;
    }
  }
  return total;
}

/** The smallest per-arm seed count reaching `target` power at the given true rates. */
export function smallestPerArmForPower(options: {
  readonly controlRate: number;
  readonly treatRate: number;
  readonly target?: number;
  readonly alpha?: number;
  readonly sided?: "one" | "two";
  readonly maxPerArm?: number;
}): { readonly perArm: number; readonly power: number } {
  const { target = 0.8, maxPerArm = 200, ...rest } = options;
  for (let n = 2; n <= maxPerArm; n += 1) {
    const p = power({ perArm: n, ...rest });
    if (p >= target) return { perArm: n, power: p };
  }
  throw new Error(`no size at or below ${maxPerArm} reaches power ${target}`);
}

/**
 * P(rate_treat < rate_control) under independent uniform Beta(1,1) priors --
 * the posterior probability that the longer warmup helps at all, in whatever
 * amount. Integrated by composite Simpson over the treatment posterior's density
 * times the control posterior's survival function, both exact Beta forms.
 */
export function posteriorProbLower(treat: Arm, control: Arm, panels = 20_000): number {
  if (panels % 2 !== 0) throw new Error("panels must be even");
  // Posterior for an arm with d divergences in s seeds is Beta(d + 1, s - d + 1).
  const aT = treat.diverged + 1;
  const bT = clean(treat) + 1;
  const aC = control.diverged + 1;
  const bC = clean(control) + 1;

  // Beta(a, b) density with integer a, b: x^(a-1)(1-x)^(b-1) / B(a, b), and
  // 1 / B(a, b) = (a + b - 1) * C(a + b - 2, a - 1) exactly.
  const invBeta = (a: number, b: number): number => (a + b - 1) * Number(choose(a + b - 2, a - 1));
  const densityT = (x: number): number => invBeta(aT, bT) * x ** (aT - 1) * (1 - x) ** (bT - 1);
  // P(Beta(aC, bC) > x) = P(Binomial(aC + bC - 1, x) <= aC - 1).
  const survivalC = (x: number): number => binomialCdf(aC - 1, aC + bC - 1, x);

  const f = (x: number): number => densityT(x) * survivalC(x);
  const h = 1 / panels;
  let sum = f(0) + f(1);
  for (let i = 1; i < panels; i += 1) sum += f(i * h) * (i % 2 === 1 ? 4 : 2);
  return (sum * h) / 3;
}

/**
 * P(true rate < `x`) for one arm's own Beta(d + 1, s - d + 1) posterior, under a
 * uniform prior. Exact: P(Beta(a, b) <= x) = P(Binomial(a + b - 1, x) >= a).
 */
export function posteriorRateBelow(arm: Arm, x: number): number {
  if (x < 0 || x > 1) throw new Error("x must be a rate");
  const a = arm.diverged + 1;
  const b = clean(arm) + 1;
  return 1 - binomialCdf(a - 1, a + b - 1, x);
}

/** The arms Belief#140 reports, as the record's description states them. */
export const BELIEF_140 = {
  /** ">= 2,000 warmup steps": no run diverged. */
  treat: { diverged: 0, seeds: 10 } satisfies Arm,
  /** "500 warmup steps": diverged on 3 of 10 seeds. */
  control: { diverged: 3, seeds: 10 } satisfies Arm,
  /** Issue#25: "About one run in five past 1B parameters diverges". */
  baseRate: 0.2,
} as const;
