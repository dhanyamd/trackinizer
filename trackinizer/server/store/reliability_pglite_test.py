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

async def _backdate(store: Store, source: UUID, claim: UUID, days: int) -> None:
    """Age one citation by ``days`` so recency has something to measure."""
    async with store.engine.acquire() as conn:
        await conn.execute(
            "UPDATE edges SET created = created - make_interval(days => $3) "
            "WHERE from_id = $1 AND to_id = $2",
            source,
            claim,
            days,
        )


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_stale_disagreement_weighs_less_than_fresh_agreement(store: Store) -> None:
    """The store's recency term: a source is judged by its RECENT record.

    Source X agrees on claim A and dissents on claim B. Backdating X's
    agreement (making the agreement stale, the dissent fresh) must lower its
    weight, and backdating the dissent instead must raise it.
    """
    claim_a = await _belief(store, "Claim A")
    claim_b = await _belief(store, "Claim B")
    consenter = await _paper(store, "Consistent paper")
    source_x = await _paper(store, "Mixed-record paper")
    await _proves(store, consenter, claim_a, 0.8)
    await _proves(store, consenter, claim_b, 0.8)
    await _proves(store, source_x, claim_a, 0.75)
    await _proves(store, source_x, claim_b, -0.9)

    # Agreement stale (400 days old), dissent fresh.
    await _backdate(store, source_x, claim_a, 400)
    await store.recompute_reliability()
    stale_agreement = await _reliability_of(store, source_x)

    # Flip: agreement fresh, dissent stale.
    await _backdate(store, source_x, claim_a, -400)
    await _backdate(store, source_x, claim_b, 400)
    await store.recompute_reliability()
    fresh_agreement = await _reliability_of(store, source_x)

    assert stale_agreement is not None and fresh_agreement is not None
    assert fresh_agreement > stale_agreement
