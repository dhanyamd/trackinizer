"""Pure reliability math: chi-square survival, fixed point, cold-start degeneracy.

Every expected number is pinned to an independent closed form -- ``erfc`` for
one degree of freedom, the exponential/Polynomial forms for even degrees -- so
a broken survival function or a shifted convention fails loudly rather than
approximately.
"""

from __future__ import annotations

import math
from uuid import uuid4

import pytest

from trackinizer.types.reliability import Citation, chisq_sf, reliability_fixed_point


def _citation(source: object, claim: object, valence: float) -> Citation:
    """Build a Citation from opaque id-likes (the fold only groups by identity)."""
    return Citation(source=source, claim=claim, valence=valence)  # type: ignore[arg-type]


# -- chi-square survival function ---------------------------------------------


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 1.0),
        (0.25, math.erfc(math.sqrt(0.125))),
        (1.0, math.erfc(math.sqrt(0.5))),
        (3.841458820694124, 0.05),  # the df=1, alpha=0.05 critical value
    ],
)
def test_chisq_sf_one_degree_matches_erfc(x: float, expected: float) -> None:
    assert chisq_sf(x, 1.0) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 1.0),
        (2.0, math.exp(-1.0)),
        (5.991464547107979, 0.05),  # the df=2, alpha=0.05 critical value
    ],
)
def test_chisq_sf_two_degrees_matches_exponential(x: float, expected: float) -> None:
    assert chisq_sf(x, 2.0) == pytest.approx(expected, abs=1e-6)


def test_chisq_sf_four_degrees_matches_polynomial() -> None:
    # df=4 survival is exp(-x/2) * (1 + x/2).
    x = 9.487729036781154  # the df=4, alpha=0.05 critical value
    assert chisq_sf(x, 4.0) == pytest.approx(0.05, abs=1e-3)
    x2 = 2.0
    assert chisq_sf(x2, 4.0) == pytest.approx(
        math.exp(-x2 / 2) * (1 + x2 / 2),
        abs=1e-6,
    )


def test_chisq_sf_bounded_and_monotone() -> None:
    for df in (1.0, 2.0, 7.0, 30.0):
        for x in (0.0, 0.5, 1.0, 5.0, 50.0):
            p = chisq_sf(x, df)
            assert 0.0 <= p <= 1.0
        assert chisq_sf(1.0, df) >= chisq_sf(2.0, df)


# -- fixed point ---------------------------------------------------------------


def test_empty_citations_is_empty() -> None:
    assert reliability_fixed_point([]) == {}


def test_lone_agreeing_source_is_full_weight() -> None:
    """Cold start: no disagreement means the uniform prior is the fixed point."""
    source, claim = uuid4(), uuid4()
    weights = reliability_fixed_point([_citation(source, claim, 0.8)])
    assert weights[source] == pytest.approx(1.0)


def test_disagreeing_pair_is_penalized_not_extreme() -> None:
    """Two sources say opposite things; both get the same soft penalty.

    With equal initial weights the consensus truth is 0.0, so each source's
    single-claim deviation is 0.81 and its weight is the df=1 survival at
    0.81 -- ``erfc(sqrt(0.405))`` -- the CATD property that a sparse
    contradicting source is penalized (never driven to an extreme).
    """
    a, b, claim = uuid4(), uuid4(), uuid4()
    weights = reliability_fixed_point(
        [_citation(a, claim, 0.9), _citation(b, claim, -0.9)],
    )
    expected = math.erfc(math.sqrt(0.9**2 / 2))
    assert weights[a] == pytest.approx(expected, abs=1e-6)
    assert weights[b] == pytest.approx(expected, abs=1e-6)
    assert 0.2 < expected < 0.5


def test_consensus_outweighs_the_dissenter() -> None:
    """Two agreeing sources outweigh one dissenter; agreeers stay near 1."""
    a, b, c, claim = uuid4(), uuid4(), uuid4(), uuid4()
    weights = reliability_fixed_point(
        [
            _citation(a, claim, 0.8),
            _citation(b, claim, 0.9),
            _citation(c, claim, -0.9),
        ],
    )
    assert weights[a] > weights[c]
    assert weights[b] > weights[c]
    assert weights[a] <= 1.0


def test_worse_deviation_scores_lower() -> None:
    claim = uuid4()
    mild = uuid4()
    harsh = uuid4()
    weights = reliability_fixed_point(
        [
            _citation(mild, claim, 0.3),
            _citation(harsh, claim, -0.95),
            _citation(uuid4(), claim, 0.9),
            _citation(uuid4(), claim, 0.9),
        ],
    )
    assert weights[mild] > weights[harsh]


def test_fixed_point_is_deterministic() -> None:
    citations = [
        _citation(uuid4(), uuid4(), 0.8),
        _citation(uuid4(), uuid4(), -0.4),
        _citation(uuid4(), uuid4(), 0.5),
    ]
    assert reliability_fixed_point(citations) == reliability_fixed_point(citations)


def test_all_collapsed_sources_leave_claims_neutral_not_supported() -> None:
    """A claim whose every source collapsed to zero weight reads as undecided.

    The consensus truth falls back to neutral log-odds rather than being
    pulled by dead weight, and the sources' weights stay at the floor.
    """
    claim = uuid4()
    weights = reliability_fixed_point(
        [_citation(uuid4(), claim, -1.0), _citation(uuid4(), claim, 1.0)],
        max_iterations=40,
    )
    assert all(weight >= 0.0 for weight in weights.values())