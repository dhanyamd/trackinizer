"""``evidence_for`` ranks the currently-true ``proves`` graph, against PGlite.

Exercised against a real engine rather than a mock: the currency rule, the
contribution arithmetic, and the ranking are what the store promises, and only
a real ``proves`` graph exercises them together. Every expected contribution
is pinned to the fold arithmetic in
:mod:`trackinizer.types.belief_confidence`, and ``derived_confidence`` is
cross-checked against :meth:`Store.confidence_for` so the ranking provably
exposes the accepted model rather than a parallel one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
import pytest_asyncio

from trackinizer.lib.postgres.testing import reset_schema
from trackinizer.server.embedders.stub import StubEmbedder
from trackinizer.server.store.core import Store
from trackinizer.types.belief_confidence import NEUTRAL_CONFIDENCE, fold_confidence
from trackinizer.wire.bodies import (
    SubmitBelief,
    SubmitExperiment,
    SubmitPaper,
)


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


async def _proven_belief(store: Store, title: str) -> UUID:
    """Submit a Belief and mark it proven so it counts as a currently-true citer."""
    belief_id = await store.submit_belief(
        SubmitBelief(account="tester@example.com", title=title),
    )
    await store.set_judgement(belief_id, "proven", actor="tester")
    return belief_id


async def _complete_experiment(store: Store, title: str) -> UUID:
    """Submit an Experiment and complete it so it counts as a currently-true citer."""
    experiment_id = await store.submit_experiment(
        SubmitExperiment(account="tester@example.com", title=title),
    )
    await store.set_status(experiment_id, "complete", actor="tester")
    return experiment_id


async def _paper(store: Store, title: str) -> UUID:
    """Submit a Paper; the default status ``active`` is already currency-true."""
    return await store.submit_paper(
        SubmitPaper(account="tester@example.com", title=title),
    )


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_missing_row_is_none(store: Store) -> None:
    assert await store.evidence_for(uuid4()) is None


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_non_claimable_kind_is_none(store: Store) -> None:
    """A Paper has no inbound proves graph; evidence is undefined, like confidence."""
    paper = await _paper(store, "Not a claim")
    assert await store.evidence_for(paper) is None


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_no_evidence_is_empty_and_neutral(store: Store) -> None:
    claim = await _proven_belief(store, "Unsupported claim")
    report = await store.evidence_for(claim)
    assert report is not None
    assert report.citations == ()
    assert report.derived_confidence == pytest.approx(NEUTRAL_CONFIDENCE)


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_contributions_rank_descending_and_fold_to_confidence(store: Store) -> None:
    claim = await _proven_belief(store, "Mass-matrix adaptation dominates")
    paper_a = await _paper(store, "Survey (0.8)")
    paper_b = await _paper(store, "Ablations (0.6)")
    experiment = await _complete_experiment(store, "HMC run (0.9)")
    await store.add_edge(
        from_id=paper_a, to_id=claim, edge_kind="proves", actor="tester", valence=0.8
    )
    await store.add_edge(
        from_id=paper_b, to_id=claim, edge_kind="proves", actor="tester", valence=0.6
    )
    await store.add_edge(
        from_id=experiment,
        to_id=claim,
        edge_kind="proves",
        actor="tester",
        valence=0.9,
    )

    report = await store.evidence_for(claim)
    assert report is not None
    # A non-claimable citer counts at full weight 1.0; a claimable citer at
    # its own (evidence-free) confidence 0.5. Expected contributions:
    # 0.8, 0.6, 0.45 -- ranked by absolute magnitude.
    contributions = [c.contribution for c in report.citations]
    assert contributions == pytest.approx([0.8, 0.6, 0.45])
    assert [c.citer_confidence for c in report.citations] == pytest.approx(
        [1.0, 1.0, NEUTRAL_CONFIDENCE],
    )
    assert [c.kind for c in report.citations] == ["Paper", "Paper", "Experiment"]
    # The ranking IS the fold: the report's confidence must equal the
    # shipped per-node walk on the same graph.
    assert report.derived_confidence == pytest.approx(
        await store.confidence_for(claim),
    )
    assert report.derived_confidence == pytest.approx(fold_confidence(1.85))


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_a_disproof_ranks_first_and_flips_sign(store: Store) -> None:
    claim = await _proven_belief(store, "Killed claim")
    paper = await _paper(store, "Refuting read")
    await store.add_edge(
        from_id=paper, to_id=claim, edge_kind="proves", actor="tester", valence=-0.9
    )

    report = await store.evidence_for(claim)
    assert report is not None
    assert len(report.citations) == 1
    assert report.citations[0].contribution == pytest.approx(-0.9)
    # Same magnitude rule as authority: the disproof is as load-bearing as a
    # proof, and the fold drops below neutral.
    assert report.derived_confidence == pytest.approx(fold_confidence(-0.9))
    assert report.derived_confidence == pytest.approx(await store.confidence_for(claim))


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_currency_invalidated_citer_is_dropped(store: Store) -> None:
    claim = await _proven_belief(store, "Claim with falling evidence")
    paper = await _paper(store, "Later retracted")
    await store.add_edge(
        from_id=paper, to_id=claim, edge_kind="proves", actor="tester", valence=0.7
    )
    before = await store.evidence_for(claim)
    assert before is not None and len(before.citations) == 1

    # An ``invalid`` Artifact is no longer currently-true, so its citation
    # leaves the ranking -- the same rule the fold applies.
    await store.set_status(paper, "invalid", actor="tester")
    after = await store.evidence_for(claim)
    assert after is not None
    assert after.citations == ()
    assert after.derived_confidence == pytest.approx(NEUTRAL_CONFIDENCE)


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_belief_citer_contributes_its_own_folded_confidence(store: Store) -> None:
    claim = await _proven_belief(store, "The claim")
    mid = await _proven_belief(store, "Proven mid-belief")
    experiment = await _complete_experiment(store, "Mid-belief's evidence")
    await store.add_edge(
        from_id=experiment,
        to_id=mid,
        edge_kind="proves",
        actor="tester",
        valence=0.9,
    )
    await store.add_edge(
        from_id=mid, to_id=claim, edge_kind="proves", actor="tester", valence=0.8
    )

    mid_confidence = await store.confidence_for(mid)
    report = await store.evidence_for(claim)
    assert report is not None
    assert len(report.citations) == 1
    assert report.citations[0].citer_confidence == pytest.approx(mid_confidence)
    assert report.citations[0].contribution == pytest.approx(mid_confidence * 0.8)


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_equal_valence_is_corroboration_and_never_decayed(store: Store) -> None:
    """Same valence, different ages: agreement, so neither is decayed.

    Recency only cuts citations a newer one CONTRADICTS; same-sign citations
    corroborate whatever their age, so both decay weights are 1.0 and the
    contributions tie exactly (seq breaks the tie).
    """
    claim = await _proven_belief(store, "Claim with two equal voices")
    older = await _paper(store, "Older paper")
    newer = await _paper(store, "Newer paper")
    for paper in (older, newer):
        await store.add_edge(
            from_id=paper, to_id=claim, edge_kind="proves", actor="tester",
            valence=0.5,
        )
    async with store.engine.acquire() as conn:
        await conn.execute(
            "UPDATE inquiries SET paper_publish_date = "
            "clock_timestamp() - make_interval(days => 400) WHERE id = ANY($1::uuid[])",
            [older, newer],
        )
    report = await store.evidence_for(claim)
    assert report is not None
    assert [c.seq for c in report.citations] == [1, 2]
    assert all(c.decay == 1.0 for c in report.citations)


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_stale_support_yields_to_a_fresh_attack(store: Store) -> None:
    """A newer contradicting citation decays the older one by its age gap."""
    claim = await _proven_belief(store, "Claim with a dispute")
    old_support = await _paper(store, "Old supporter")
    fresh_attack = await _paper(store, "Fresh attacker")
    await store.add_edge(
        from_id=old_support, to_id=claim, edge_kind="proves", actor="tester",
        valence=0.7,
    )
    await store.add_edge(
        from_id=fresh_attack, to_id=claim, edge_kind="proves", actor="tester",
        valence=-0.9,
    )
    async with store.engine.acquire() as conn:
        await conn.execute(
            "UPDATE inquiries SET paper_publish_date = "
            "clock_timestamp() - make_interval(days => 730) WHERE id = $1",
            old_support,
        )
    report = await store.evidence_for(claim)
    assert report is not None
    by_citer = {c.citer_id: c for c in report.citations}
    assert by_citer[old_support].decay == pytest.approx(0.25, abs=0.01)
    assert by_citer[fresh_attack].decay == 1.0


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_identical_timestamps_tiebreak_by_citer_seq(store: Store) -> None:
    """With identical recording times the contributions tie, and seq breaks it."""
    claim = await _proven_belief(store, "Tied claim")
    first = await _paper(store, "First paper")
    second = await _paper(store, "Second paper")
    for paper in (first, second):
        await store.add_edge(
            from_id=paper, to_id=claim, edge_kind="proves", actor="tester", valence=0.5
        )
    # Equal ARTIFACT dates: the recency anchor is the paper's own date, so
    # giving both the same publication date makes the contributions tie.
    async with store.engine.acquire() as conn:
        await conn.execute(
            "UPDATE inquiries SET paper_publish_date = clock_timestamp() "
            "WHERE id = ANY($1::uuid[])",
            [first, second],
        )
    report = await store.evidence_for(claim)
    assert report is not None
    assert [c.seq for c in report.citations] == [1, 2]
    assert report.citations[0].decay == report.citations[1].decay == 1.0
