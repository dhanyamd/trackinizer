"""Derived source reliability: a chi-square test over the proves matrix.

The reliability of a citing Artifact is estimated from the signed-valence
``proves`` citation matrix, jointly with the truth of the claims it cites --
the truth-discovery iteration (Yin et al. 2008 TruthFinder; Li et al. 2014,
SIGMOD) with the long-tail guard that line introduced for sparse sources
(Li et al. 2014, PVLDB, "confidence-aware"): a source's weight is the
chi-square survival function of its own deviation from consensus,

    w_s = P(chi2(L_s) >= D_s),     D_s = sum_j (v_sj - v*_j)^2
    v*_j = sum_s w_s v_sj / sum_s w_s

``L_s`` is how many claims source ``s`` currently speaks on. Under the null
hypothesis that the source's deviations are unit-variance noise, ``D_s`` is
chi-square-distributed with ``L_s`` degrees of freedom, so ``w_s`` is exactly
the p-value of "this source deviates no more than chance": a source that
agrees with the consensus of everything else keeps ``w_s = 1``; one that
deviates far more than its evidence count can explain is driven toward ``0``.
A SPARSE source is bounded softly rather than punished into an extreme: one
claim disagreeing by ``1.0`` lands at ``P(chi2(1) >= 1) ~= 0.32`` -- penalized,
not annihilated -- which is the property the long-tail literature exists for:
naive inverse-deviation weights explode on once-cited sources, and CATD's
chi-square bound is the documented fix. The survival function is that same
chi-square statistic with the significance level left continuous instead of
thresholded at a cut-off, so no alpha constant enters the code.

Iterating the two equations to a fixed point is the whole method: nobody sets
``w_s`` (no votes, no self-annotation, no grade table) -- it is read off how
the source's claims fare against the consensus of every other source. Purely
derived and read-only.

The compute is pure Python over id-keyed dicts, one pass over the proves
matrix per iteration -- linear in edges, seconds at millions, and run off the
request path by the reliability sweep. Cold start is the degenerate case: with
no observed disagreement (every source agrees with consensus, ``D_s = 0``)
every weight is exactly ``1.0``, which makes the reliability-weighted
confidence fold identical to the shipped uniform-weight fold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID


__all__ = [
    "HALF_LIFE_SECONDS",
    "claim_taus",
    "MAX_ITERATIONS",
    "NEUTRAL_TRUTH",
    "TOLERANCE",
    "Citation",
    "chisq_sf",
    "reliability_fixed_point",
    "temporal_weight",
]


HALF_LIFE_SECONDS: float = 365.0 * 24.0 * 3600.0
"""Recency half-life: one year. A citation this old relative to the newest
evidence on its claim contributes half the weight of a fresh one.

The only modelling choice this module makes, and it is exposed rather than
buried: research claims age on the scale of publication and replication cycles,
so a year is the documented default. Time-aware truth discovery is the
published line this follows (Huang & Wang, "Time-Aware Truth Discovery in
Social Sensing", IEEE MASS 2015); exponential forgetting is the standard decay
form. Degeneracy keeps it safe: when every citation on a claim is the same age
(age 0), every temporal weight is exactly 1.0 and the computation is identical
to the time-blind one.
"""

MAX_ITERATIONS: int = 100
"""Hard cap on fixed-point passes; convergence normally stops the loop first."""

TOLERANCE: float = 1e-6
"""L1 change between successive weight vectors below which iteration stops."""

NEUTRAL_TRUTH: float = 0.0
"""Truth assigned to a claim whose every source has collapsed to weight zero:
log-odds neutral, so the claim reads as genuinely undecided rather than
supported or attacked by dead weight."""

_EPS: float = 3e-9
"""Relative series/fraction convergence bound (Numerical Recipes' EPS)."""

_FPMIN: float = 1e-300
"""Smallest representable before inversion, guarding the continued fraction."""

_ITMAX: int = 200
"""Series / continued-fraction iteration cap per chi-square evaluation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation:
    """One signed citation: source ``s`` speaking on claim ``j`` at ``valence``.

    Attributes:
      source: The citing row (any kind; sources are graded).
      claim: The cited Belief/Experiment.
      valence: The signed weight in ``[-1, 1]`` (positive supports, negative
        argues against; magnitude is the evidential weight the recorder
        assigned).
      tau: Temporal weight in ``(0, 1]`` -- how much this citation counts given
        its age relative to the newest evidence on its claim
        (:func:`temporal_weight`). ``1.0`` for the freshest citation, less for
        older ones; ``1.0`` everywhere reproduces the time-blind computation.

    """

    source: UUID
    claim: UUID
    valence: float
    tau: float = 1.0


def temporal_weight(age_seconds: float, *, half_life: float = HALF_LIFE_SECONDS) -> float:
    """Return a citation's recency weight: ``0.5 ** (age / half_life)``.

    Exponential forgetting, the standard decay form: a citation one half-life
    older than the freshest evidence on its claim counts half as much; two
    half-lives, a quarter. Monotone decreasing in age, exactly ``1.0`` at age
    zero (so a claim whose evidence all arrived together is untouched), and
    never zero -- stale evidence is discounted, never erased.

    Args:
      age_seconds: How old this citation is relative to the newest currently-true
        citation on the same claim. Negative ages clamp to 0.
      half_life: Seconds after which a citation's weight halves.

    Returns:
      tau: Temporal weight in ``(0, 1]``.

    """
    if age_seconds <= 0.0:
        return 1.0
    return 0.5 ** (age_seconds / half_life)


def chisq_sf(x: float, df: float) -> float:
    """Return the chi-square survival function ``P(chi2(df) >= x)``.

    The regularized upper incomplete gamma ``Q(df/2, x/2)``, computed by the
    standard series/continued-fraction pair (Numerical Recipes ``gammq``), so
    no distribution table or approximation constant is baked in.

    Args:
      x: Non-negative statistic.
      df: Positive degrees of freedom.

    Returns:
      p: Survival probability in ``[0, 1]``; ``1.0`` at ``x == 0``.

    """
    if x <= 0.0:
        return 1.0
    a = df / 2.0
    nx = x / 2.0
    ln_prefactor = -nx + a * math.log(nx) - math.lgamma(a)
    if nx < a + 1.0:
        # Series for the lower regularized incomplete gamma P(a, nx); Q = 1 - P.
        term = 1.0 / a
        total = term
        for i in range(1, _ITMAX):
            term *= nx / (a + i)
            total += term
            if abs(term) < abs(total) * _EPS:
                return max(0.0, min(1.0, 1.0 - total * math.exp(ln_prefactor)))
        return max(0.0, min(1.0, 1.0 - total * math.exp(ln_prefactor)))
    # Lentz's continued fraction for the upper regularized gamma Q(a, nx).
    b = nx + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return max(0.0, min(1.0, math.exp(ln_prefactor) * h))


def claim_taus(
    citations: Sequence[tuple[float, float]],
    *,
    half_life: float = HALF_LIFE_SECONDS,
) -> list[float]:
    """Per-citation recency weights for ONE claim, cutting only in disputes.

    ``citations`` is ``(age_seconds, valence)`` per citation, age measured
    against the newest evidence on the claim. A citation is decayed ONLY when
    some NEWER citation on the same claim argues the opposite side (the
    product of the valences is negative); otherwise it counts in full:

        tau(c) = 0.5 ** (age(c) / half_life)   if a newer citation contradicts c
               = 1.0                           otherwise

    Recency arbitrates CONFLICT, it does not tax age: corroboration that
    happens to be old counts in full, a claim whose evidence all agrees is
    untouched whatever its age, and the side that spoke last on a dispute
    always counts fully. The conflict predicate is sign opposition -- the
    falsification store's own support/attack semantics -- so no threshold
    constant enters. Neutral valences (0) cannot conflict.

    Provenance, stated honestly: weighting evidence by conflict is the
    Dempster-Shafer lineage (Deng et al. 2004 onward -- credibility weights
    before combination); decaying observations by age is standard time-aware
    truth discovery (Huang & Wang, IEEE MASS 2015). Conditioning the decay on
    conflict is OUR composition of those two published ideas, not a formula
    quoted verbatim from either; it is parameter-free apart from the shared
    half-life, and it degenerates to the time-blind computation whenever a
    claim has no sign opposition (all taus exactly 1.0).

    Args:
      citations: (age, valence) per citation of one claim, any order.
      half_life: Seconds after which a disputed citation's weight halves.

    Returns:
      taus: One temporal weight in ``(0, 1]`` per citation, input order.

    """
    return [
        temporal_weight(age, half_life=half_life)
        if any(
            other_age < age and other_valence * valence < 0
            for other_age, other_valence in citations
        )
        else 1.0
        for age, valence in citations
    ]


def reliability_fixed_point(
    citations: Sequence[Citation],
    *,
    max_iterations: int = MAX_ITERATIONS,
    tolerance: float = TOLERANCE,
) -> dict[UUID, float]:
    """Estimate every source's reliability from the signed citation matrix.

    Args:
      citations: Currently-true ``proves`` edges, one :class:`Citation` per
        edge (``source`` cites ``claim`` at ``valence``). Sources speaking on
        no claim are absent from the result.
      max_iterations: Power-iteration cap.
      tolerance: L1 convergence threshold on the weight vector.

    Returns:
      reliability: ``source -> weight`` in ``[0, 1]``. ``1.0`` is
        agreement-with-consensus (the cold-start uniform prior); ``0`` is
        deviation far beyond chance. Empty when ``citations`` is empty.

    """
    if not citations:
        return {}
    by_source: dict[UUID, list[tuple[UUID, float, float]]] = {}
    by_claim: dict[UUID, list[tuple[UUID, float, float]]] = {}
    for citation in citations:
        by_source.setdefault(citation.source, []).append(
            (citation.claim, citation.valence, citation.tau),
        )
        by_claim.setdefault(citation.claim, []).append(
            (citation.source, citation.valence, citation.tau),
        )
    weights: dict[UUID, float] = dict.fromkeys(by_source, 1.0)
    for _ in range(max_iterations):
        # Reliability- and recency-weighted consensus truth per claim.
        truth: dict[UUID, float] = {}
        for claim, speakers in by_claim.items():
            mass = sum(weights[source] * tau for source, _, tau in speakers)
            truth[claim] = (
                sum(
                    weights[source] * tau * valence
                    for source, valence, tau in speakers
                )
                / mass
                if mass > 1e-12
                else NEUTRAL_TRUTH
            )
        # Chi-square survival of each source's recency-weighted deviation from
        # that consensus. With temporal weights the statistic is a weighted sum
        # of squared deviations, whose null distribution is not chi-square with
        # L degrees of freedom; Satterthwaite's standard moment-matching gives
        # the equivalent scaled chi-square: effective df nu = (sum tau)^2 /
        # sum tau^2 and scale = sum tau^2 / sum tau, so
        # P(sum tau_i z_i^2 >= D) ~= P(chi2(nu) >= D / scale). Uniform tau
        # collapses to nu = L and scale = 1 -- exactly the time-blind form.
        nxt: dict[UUID, float] = {}
        for source, spoken in by_source.items():
            tau_sum = sum(tau for _, _, tau in spoken)
            tau_sq_sum = sum(tau * tau for _, _, tau in spoken)
            deviation = sum(
                tau * (valence - truth[claim]) ** 2 for claim, valence, tau in spoken
            )
            if tau_sum <= 0.0 or tau_sq_sum <= 0.0:
                nxt[source] = 0.0
                continue
            nu = tau_sum * tau_sum / tau_sq_sum
            scale = tau_sq_sum / tau_sum
            weight = chisq_sf(deviation / scale, nu)
            nxt[source] = 0.0 if math.isnan(weight) else weight
        delta = sum(abs(nxt[source] - before) for source, before in weights.items())
        weights = nxt
        if delta < tolerance:
            break
    return weights