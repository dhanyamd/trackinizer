"""``recompute_reliability`` sweeps the proves matrix, against PGlite.

Exercised against a real engine: the currency rule, the fixed-point write,
and the degeneracy contract -- a graph with no observed disagreement must
leave the derived confidence fold EXACTLY as the pre-reliability build
computed it, because NULL reads as the uniform cold-start prior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from trackinizer.lib.postgres.testing import reset_schema
from trackinizer.server.embedders.stub import StubEmbedder
from trackinizer.server.store.core import Store
from trackinizer.types.belief_confidence import fold_confidence
from trackinizer.types.belief_confidence import NEUTRAL_CONFIDENCE
from trackinizer.wire.bodies import SubmitBelief, SubmitPaper


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from trackinizer.lib.postgres import PGliteEngine


@pytest_asyncio.fixture(loop_scope="session")
async def store(pglite_engine: PGliteEngine) -> AsyncIterator[Store]:
    """Return a bootstrapped Store over the session's shared PGlite engine."""
    await reset_schema(pglite_engine)
    built = Store(pglite_engine, embed=StubEmbedder())
    await built.bootstrap()
    yield built


async def _belief(store: Store, title: str) -> UUID:
    return await store.submit_belief(
        SubmitBelief(account="tester@example.com", title=title),
    )


async def _paper(store: Store, title: str) -> UUID:
    return await store.submit_paper(
        SubmitPaper(account="tester@example.com", title=title),
    )


async def _proves(store: Store, source: UUID, claim: UUID, valence: float) -> None:
    await store.add_edge(
        from_id=source,
        to_id=claim,
        edge_kind="proves",
        actor="tester",
        valence=valence,
    )


async def _reliability_of(store: Store, row_id: UUID) -> float | None:
    async with store.engine.acquire() as conn:
        value = await conn.fetchval(
            "SELECT artifact_reliability FROM inquiries WHERE id = $1",
            row_id,
        )
    return None if value is None else float(value)


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_empty_graph_writes_nothing(store: Store) -> None:
    assert await store.recompute_reliability() == 0


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_single_source_per_claim_is_full_weight(store: Store) -> None:
    """Cold-start degeneracy: no disagreement means every weight is 1.0.

    Each claim has exactly one citing source, so no source deviates from any
    consensus -- the fixed point is the uniform prior, and the confidence fold
    is bit-identical to the pre-reliability build.
    """
    claim_a, claim_b = await _belief(store, "Claim A"), await _belief(store, "Claim B")
    paper_a, paper_b = await _paper(store, "Paper A"), await _paper(store, "Paper B")
    await _proves(store, paper_a, claim_a, 0.8)
    await _proves(store, paper_b, claim_b, -0.4)

    before = await store.confidence_for(claim_a)
    assert await store.recompute_reliability() == 2
    assert await _reliability_of(store, paper_a) == pytest.approx(1.0)
    assert await _reliability_of(store, paper_b) == pytest.approx(1.0)
    assert await store.confidence_for(claim_a) == pytest.approx(before)


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_dissenter_scores_below_consensus(store: Store) -> None:
    claim = await _belief(store, "Contested claim")
    pro_a, pro_b, dissenter = (
        await _paper(store, "Pro A"),
        await _paper(store, "Pro B"),
        await _paper(store, "Dissenter"),
    )
    await _proves(store, pro_a, claim, 0.8)
    await _proves(store, pro_b, claim, 0.9)
    await _proves(store, dissenter, claim, -0.9)

    await store.recompute_reliability()
    pro_weight = await _reliability_of(store, pro_a)
    dissent_weight = await _reliability_of(store, dissenter)
    assert pro_weight is not None and dissent_weight is not None
    assert pro_weight > dissent_weight
    assert 0.0 < dissent_weight < pro_weight


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_sweep_weight_feeds_the_confidence_fold(store: Store) -> None:
    """The column the sweep writes is the exact factor the fold multiplies.

    After the sweep, ``confidence_for`` must equal
    ``fold_confidence(w * citation_confidence * valence)`` with ``w`` read
    straight from the stored column -- the sweep and the fold share one
    number, not two approximations of it.
    """
    claim = await _belief(store, "Weighted claim")
    paper = await _paper(store, "Deviating paper")
    await _proves(store, paper, claim, 0.8)

    await store.recompute_reliability()
    weight = await _reliability_of(store, paper)
    assert weight is not None
    assert await store.confidence_for(claim) == pytest.approx(
        fold_confidence(weight * 0.8),
    )


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_currency_invalidated_source_stops_voting(store: Store) -> None:
    claim = await _belief(store, "Claim with falling source")
    paper = await _paper(store, "Later invalidated")
    await _proves(store, paper, claim, 0.7)
    await store.recompute_reliability()
    assert await _reliability_of(store, paper) is not None

    await store.set_status(paper, "invalid", actor="tester")
    await store.recompute_reliability()
    assert await _reliability_of(store, paper) is None

async def _publish(store: Store, paper: UUID, days_ago: int) -> None:
    """Set a paper's publication date, the recency anchor for its citations."""
    async with store.engine.acquire() as conn:
        await conn.execute(
            "UPDATE inquiries SET paper_publish_date = "
            "clock_timestamp() - make_interval(days => $2) WHERE id = $1",
            paper,
            days_ago,
        )


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_agreement_is_never_taxed_for_its_age(store: Store) -> None:
    """Two agreeing papers of different ages count equally.

    Recency arbitrates DISPUTES, it does not tax age: same-sign citations are
    corroboration, however old, so both decay weights are exactly 1.0 and the
    contributions tie (seq breaks the tie).
    """
    claim = await _belief(store, "Claim with an old and a new supporter")
    old_paper = await _paper(store, "Old survey (2019)")
    new_paper = await _paper(store, "Recent replication (2026)")
    await _proves(store, old_paper, claim, 0.7)
    await _proves(store, new_paper, claim, 0.7)
    await _publish(store, old_paper, 730)  # two years old
    await _publish(store, new_paper, 0)

    report = await store.evidence_for(claim)
    assert report is not None
    assert all(c.decay == 1.0 for c in report.citations)
    assert [c.seq for c in report.citations] == [1, 2]
    assert report.derived_confidence == pytest.approx(
        await store.confidence_for(claim),
    )


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_recency_arbitrates_disputes_only(store: Store) -> None:
    """The fresh side of a dispute counts fully; the stale side decays.

    Old paper supports, newer paper contradicts: the OLD support is decayed by
    its age (a newer opposite-sign citation exists), while the fresh attack
    counts at full weight. Reversing the dates flips the decay.
    """
    claim = await _belief(store, "Contested claim")
    old_supporter = await _paper(store, "Old supporter (2019)")
    fresh_critic = await _paper(store, "Fresh critic (2026)")
    await _proves(store, old_supporter, claim, 0.7)
    await _proves(store, fresh_critic, claim, -0.9)
    await _publish(store, old_supporter, 730)
    await _publish(store, fresh_critic, 0)

    report = await store.evidence_for(claim)
    assert report is not None
    decayed = {c.citer_id: c.decay for c in report.citations}
    assert decayed[old_supporter] < 1.0      # stale side of the dispute
    assert decayed[fresh_critic] == 1.0      # last word counts fully

    # The identical claim with the dates swapped: the attack goes stale and
    # the support counts fully, so confidence rises above neutral.
    claim2 = await _belief(store, "Same dispute, dates swapped")
    old_critic2 = await _paper(store, "Old critic (2019)")
    fresh_supporter2 = await _paper(store, "Fresh supporter (2026)")
    await _proves(store, fresh_supporter2, claim2, 0.7)
    await _proves(store, old_critic2, claim2, -0.9)
    await _publish(store, fresh_supporter2, 0)
    await _publish(store, old_critic2, 730)
    report2 = await store.evidence_for(claim2)
    assert report2 is not None
    assert report2.derived_confidence > NEUTRAL_CONFIDENCE
