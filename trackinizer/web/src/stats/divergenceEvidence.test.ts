import { expect, test } from "vitest";

import {
  BELIEF_140,
  binomialCdf,
  binomialPmf,
  choose,
  clopperPearsonInterval,
  clopperPearsonLower,
  clopperPearsonUpper,
  fisherExact,
  posteriorProbLower,
  posteriorRateBelow,
  power,
  seedsToRetireRate,
  smallestDecisivePerArm,
  smallestPerArmForPower,
} from "./divergenceEvidence";

const { treat, control, baseRate } = BELIEF_140;

// --- the primitives, against values that can be checked by hand ------------

test("choose is exact where a float factorial would already have drifted", () => {
  expect(choose(10, 0)).toBe(1n);
  expect(choose(10, 3)).toBe(120n);
  expect(choose(17, 10)).toBe(19448n);
  expect(choose(20, 10)).toBe(184756n);
  expect(choose(4, 5)).toBe(0n);
  // 100!/(50!)^2 -- past 2^53, so an exact BigInt is the point of this function.
  expect(choose(100, 50)).toBe(100891344545564193334812497256n);
});

test("a binomial pmf sums to one and matches a hand-computed term", () => {
  const n = 10;
  const p = 0.3;
  const total = Array.from({ length: n + 1 }, (_, k) => binomialPmf(k, n, p)).reduce((a, b) => a + b);
  expect(total).toBeCloseTo(1, 12);
  // C(10,3) * 0.3^3 * 0.7^7
  expect(binomialPmf(3, 10, 0.3)).toBeCloseTo(120 * 0.027 * 0.7 ** 7, 12);
  // Ten clean seeds when the true rate is the Issue#25 base rate: 0.8^10.
  expect(binomialCdf(0, 10, baseRate)).toBeCloseTo(0.8 ** 10, 12);
});

// --- 1. could one rate have produced the split? ----------------------------

test("Fisher's exact test on 0-of-10 against 3-of-10 is the ratio C(17,10)/C(20,10)", () => {
  const result = fisherExact(treat, control);
  // Conditioning on 3 divergences among 20 seeds, the chance all three miss the
  // 2k arm is C(3,0)*C(17,10)/C(20,10) -- and with 0 observed there is no table
  // below it, so that single term IS the one-sided p-value.
  const byHand = Number(choose(17, 10)) / Number(choose(20, 10));
  expect(result.oneSided).toBeCloseTo(byHand, 12);
  // C(17,10)/C(20,10) reduces to exactly 2/19.
  expect(result.oneSided).toBeCloseTo(2 / 19, 12);
  expect(result.oneSided).toBeCloseTo(0.105263, 6);
  // The mirror table (all three in the 2k arm) is equally probable, so the
  // two-sided value is exactly twice the one-sided one here.
  expect(result.twoSided).toBeCloseTo(2 * byHand, 12);
  // Which is the finding: the claim's own evidence does not clear 0.05.
  expect(result.oneSided).toBeGreaterThan(0.05);
});

test("a table with no difference between the arms returns a p-value of one", () => {
  const even = fisherExact({ diverged: 2, seeds: 10 }, { diverged: 2, seeds: 10 });
  expect(even.twoSided).toBeCloseTo(1, 12);
});

test("every conditional table probability sums to one", () => {
  // Enumerating the 0..3 tables and summing must give the whole distribution.
  const sum = [0, 1, 2, 3]
    .map((a) => fisherExact({ diverged: a, seeds: 10 }, { diverged: 3 - a, seeds: 10 }).observed)
    .reduce((a, b) => a + b);
  expect(sum).toBeCloseTo(1, 12);
});

test("a wider split at the same rates is more significant, not less", () => {
  const ten = fisherExact(treat, control).oneSided;
  const forty = fisherExact({ diverged: 0, seeds: 40 }, { diverged: 12, seeds: 40 }).oneSided;
  expect(forty).toBeLessThan(ten);
});

// --- 2. how many seeds per arm would be decisive? --------------------------

test("12 seeds per arm is the smallest decisive size at the rates the belief reports", () => {
  const found = smallestDecisivePerArm({ controlRate: 0.3, treatRate: 0, alpha: 0.05 });
  expect(found.perArm).toBe(12);
  expect(found.controlDiverged).toBe(4); // round(0.3 * 12)
  expect(found.oneSided).toBeCloseTo(Number(choose(20, 12)) / Number(choose(24, 12)), 12);
  expect(found.oneSided).toBeCloseTo(0.046584, 6);
  // Not monotone: 11 per arm rounds down to 3 divergences and gets WORSE than 10.
  expect(fisherExact({ diverged: 0, seeds: 11 }, { diverged: 3, seeds: 11 }).oneSided).toBeGreaterThan(
    fisherExact(treat, control).oneSided,
  );
});

test("being decisive once is not the same as being likely to decide", () => {
  // At 12 per arm the test clears 0.05 only if the dice fall the reported way.
  const atDecisive = power({ perArm: 12, controlRate: 0.3, treatRate: 0, alpha: 0.05 });
  expect(atDecisive).toBeLessThan(0.6);
  const sized = smallestPerArmForPower({ controlRate: 0.3, treatRate: 0, target: 0.8, alpha: 0.05 });
  expect(sized.perArm).toBeGreaterThan(12);
  expect(sized.power).toBeGreaterThanOrEqual(0.8);
});

test("power rises with seeds and is at most the test level when the arms are identical", () => {
  const rising = [8, 16, 24].map((perArm) => power({ perArm, controlRate: 0.3, treatRate: 0 }));
  expect(rising[1]!).toBeGreaterThan(rising[0]!);
  expect(rising[2]!).toBeGreaterThan(rising[1]!);
  // Under the null the rejection rate must not exceed alpha -- the exact test is
  // conservative, so this is an inequality and not an equality.
  expect(power({ perArm: 20, controlRate: 0.3, treatRate: 0.3, alpha: 0.05 })).toBeLessThanOrEqual(0.05);
});

// --- 3. what rate survives ten clean seeds? --------------------------------

test("ten clean seeds leave a true divergence rate of 25.9% on the table", () => {
  const upper = clopperPearsonUpper(0, 10, 0.05);
  // Closed form for k = 0: the largest p with (1 - p)^10 = 0.05.
  expect(upper).toBeCloseTo(1 - 0.05 ** 0.1, 10);
  expect(upper).toBeCloseTo(0.2589, 4);
  // And that bound sits ABOVE the base rate the warmup is supposed to remove,
  // so ten clean seeds cannot exclude the problem they were run to settle.
  expect(upper).toBeGreaterThan(baseRate);
});

test("the upper limit is the closed form for every k = 0 sample size", () => {
  for (const n of [1, 3, 10, 14, 30, 59]) {
    expect(clopperPearsonUpper(0, n, 0.05)).toBeCloseTo(1 - 0.05 ** (1 / n), 10);
  }
});

test("the 500-step arm's interval straddles the 2k arm's whole plausible range", () => {
  const [lo, hi] = clopperPearsonInterval(3, 10, 0.05);
  expect(lo).toBeCloseTo(0.0667, 4);
  expect(hi).toBeCloseTo(0.6525, 4);
  // The arms' intervals overlap, which is the same verdict Fisher gave.
  expect(lo).toBeLessThan(clopperPearsonUpper(0, 10, 0.05));
  expect(clopperPearsonLower(0, 10)).toBe(0);
  expect(clopperPearsonUpper(10, 10)).toBe(1);
});

test("retiring a 20% residual rate takes 14 clean seeds, and a 5% one takes 59", () => {
  expect(seedsToRetireRate(0.2, 0.05)).toBe(14);
  expect(clopperPearsonUpper(0, 13, 0.05)).toBeGreaterThan(0.2);
  expect(clopperPearsonUpper(0, 14, 0.05)).toBeLessThan(0.2);
  expect(seedsToRetireRate(0.05, 0.05)).toBe(59);
  expect(clopperPearsonUpper(0, 59, 0.05)).toBeLessThan(0.05);
});

// --- the confidence the evidence actually supports -------------------------

test("the posterior integrator reproduces two integrals with closed forms", () => {
  // Two flat posteriors: the arms are exchangeable, so the answer is a half.
  expect(posteriorProbLower({ diverged: 0, seeds: 0 }, { diverged: 0, seeds: 0 })).toBeCloseTo(0.5, 9);
  // Beta(2,1) against Beta(1,1): integral of 2x(1-x) over [0,1] = 1/3.
  expect(posteriorProbLower({ diverged: 1, seeds: 1 }, { diverged: 0, seeds: 0 })).toBeCloseTo(1 / 3, 9);
});

test("the directional claim earns 0.95; 'prevents' earns 0.43", () => {
  // "A longer warmup lowers the rate" -- P(rate at 2k < rate at 500).
  const directional = posteriorProbLower(treat, control);
  // Beta(1,11) against Beta(4,8) integrates to exactly 127/133.
  expect(directional).toBeCloseTo(127 / 133, 9);
  expect(directional).toBeCloseTo(0.954887, 6);
  // "A 2k warmup PREVENTS divergence" -- the residual rate is near zero. Posterior
  // mass below a 5% residual rate, from the same ten clean seeds: 1 - 0.95^11.
  const prevents = posteriorRateBelow(treat, 0.05);
  expect(prevents).toBeCloseTo(1 - 0.95 ** 11, 12);
  expect(prevents).toBeCloseTo(0.4312, 4);
  // The filed confidence of 0.70 sits between the two readings: too low for the
  // claim the evidence supports, too high for the claim the title makes.
  expect(prevents).toBeLessThan(0.7);
  expect(directional).toBeGreaterThan(0.7);
  // A posterior is a distribution, so the two readings must bracket the base rate.
  expect(posteriorRateBelow(treat, baseRate)).toBeCloseTo(1 - 0.8 ** 11, 12);
});
