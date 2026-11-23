"""Contention semantics for parallel agents sharing one issue queue.

The multi-agent coordination contract under test: an ``active`` row with no
``owner`` is open; with an ``owner`` it is in progress. ``trax next`` and the
claim transition must hold that contract when many agents race.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncio

import pytest
import pytest_asyncio

from trackinizer.lib.postgres import PGliteEngine
from trackinizer.lib.postgres.testing import reset_schema
from trackinizer.server.store.core import Store, StubEmbedder
from trackinizer.types.errors import ConflictError
from trackinizer.wire.bodies import SubmitBelief, SubmitExperiment, SubmitIssue


@pytest_asyncio.fixture(loop_scope="session")
async def store(pglite_engine: PGliteEngine) -> AsyncIterator[Store]:
    """Return a bootstrapped Store over the session's shared PGlite engine."""
    await reset_schema(pglite_engine)
    store = Store(pglite_engine, embed=StubEmbedder())
    await store.bootstrap()
    yield store


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_next_issue_skips_owned(store: Store) -> None:
    """An owned active row is in progress; next must offer unowned work."""
    claimed = await store.submit_issue(
        SubmitIssue(account="tester@example.com", title="claimed"),
    )
    open_ = await store.submit_issue(
        SubmitIssue(account="tester@example.com", title="open"),
    )
    await store.set_owner(claimed, "worker-1", actor="user")
    nxt = await store.next_issue()
    assert nxt is not None
    assert nxt.id == open_
    # Releasing the claim returns the row to the schedulable pool.
    await store.set_owner(claimed, None, actor="user")
    nxt2 = await store.next_issue()
    assert nxt2 is not None
    assert nxt2.id == claimed


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_claim_rejects_stale_expected_owner(store: Store) -> None:
    """A claim carrying a stale guard loses to the committed claimant."""
    task = await store.submit_issue(
        SubmitIssue(account="tester@example.com", title="contended"),
    )
    await store.transition_owner(
        task,
        expected_from=None,
        to="worker-1",
        actor="worker-1",
        reason="claim",
    )
    with pytest.raises(ConflictError):
        await store.transition_owner(
            task,
            expected_from=None,
            to="worker-2",
            actor="worker-2",
        )
    row = await store.get_inquiry(task)
    assert row is not None
    assert row.owner == "worker-1"


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_claim_records_reason_on_audit_row(store: Store) -> None:
    """The claim's --reason lands on the change_log row."""
    task = await store.submit_issue(
        SubmitIssue(account="tester@example.com", title="audited claim"),
    )
    await store.transition_owner(
        task,
        expected_from=None,
        to="worker-1",
        actor="worker-1",
        reason="taking this leaf",
    )
    async with store.engine.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT reason FROM change_log WHERE subject_id = $1 "
            "AND kind = 'owner'",
            task,
        )
    assert row is not None
    assert row["reason"] == "taking this leaf"


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_claim_subscribes_owner(store: Store) -> None:
    """Claiming subscribes the new owner; release keeps the subscription."""
    task = await store.submit_issue(
        SubmitIssue(account="tester@example.com", title="watched claim"),
    )
    await store.transition_owner(
        task,
        expected_from=None,
        to="worker-1",
        actor="worker-1",
    )
    claimed = await store.get_inquiry(task)
    assert claimed is not None
    assert "worker-1" in (claimed.subscribers or ())
    # A second claimant is also subscribed; the first keeps watching.
    await store.transition_owner(
        task,
        expected_from="worker-1",
        to="worker-2",
        actor="worker-2",
    )
    requeued = await store.get_inquiry(task)
    assert requeued is not None
    assert {"worker-1", "worker-2"} <= set(requeued.subscribers or ())
    # Releasing ownership does not unsubscribe the ex-owner.
    await store.transition_owner(
        task,
        expected_from="worker-2",
        to=None,
        actor="worker-2",
    )
    released = await store.get_inquiry(task)
    assert released is not None
    assert {"worker-1", "worker-2"} <= set(released.subscribers or ())


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_many_parallel_claimants_exactly_one_winner(store: Store) -> None:
    """K agents claiming one fresh issue: one winner, K-1 clean 409s."""
    task = await store.submit_issue(
        SubmitIssue(account="tester@example.com", title="hot issue"),
    )
    contenders = 8

    async def claim(name: str) -> str | None:
        try:
            await store.transition_owner(
                task,
                expected_from=None,
                to=name,
                actor=name,
            )
        except ConflictError:
            return None
        return name

    results = await asyncio.gather(
        *(claim(f"worker-{i}") for i in range(contenders)),
    )
    winners = [name for name in results if name is not None]
    assert len(winners) == 1
    row = await store.get_inquiry(task)
    assert row is not None
    assert row.owner == winners[0]
    assert winners[0] in (row.subscribers or ())


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_parallel_workers_drain_shared_queue_exactly_once(
    store: Store,
) -> None:
    """K workers running the trax loop over M issues: no dupes, no losses.

    The acceptance gate for multi-agent coordination: every issue completed
    exactly once, exactly one claim winner per issue, every lost claim a
    clean ConflictError, and the audit trail attributed per completing
    actor -- the same end state one careful agent would have produced.
    """
    issue_count, worker_count = 25, 10
    issue_ids = [
        await store.submit_issue(
            SubmitIssue(account="tester@example.com", title=f"job {i}"),
        )
        for i in range(issue_count)
    ]

    async def worker(name: str) -> tuple[list, list]:
        completed: list = []
        lost: list = []
        while True:
            nxt = await store.next_issue()
            if nxt is None:
                return completed, lost
            try:
                await store.transition_owner(
                    nxt.id,
                    expected_from=None,
                    to=name,
                    actor=name,
                )
            except ConflictError:
                lost.append(nxt.id)
                continue
            await store.transition_status(
                nxt.id,
                expected_from="active",
                to="complete",
                actor=name,
            )
            completed.append(nxt.id)

    outcomes = await asyncio.gather(
        *(worker(f"worker-{i}") for i in range(worker_count)),
    )
    completed = [iid for comp, _ in outcomes for iid in comp]
    lost = [iid for _, race_losses in outcomes for iid in race_losses]

    # Every issue completed exactly once, by exactly one claim winner.
    assert sorted(completed) == sorted(issue_ids)
    assert len(completed) == len(set(completed)) == issue_count
    # Every lost claim was a clean conflict on real work someone else took.
    assert all(iid in issue_ids for iid in lost)

    async with store.engine.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, owner, status, subscribers FROM inquiries "
            "WHERE kind = 'Issue'",
        )
        completions = await conn.fetch(
            "SELECT subject_id, count(*) AS n, max(actor) AS actor "
            "FROM change_log WHERE kind = 'status' "
            "AND new_status = 'complete' GROUP BY subject_id",
        )

    by_id = {row["id"]: row for row in rows}
    assert len(rows) == issue_count
    comp_counts = {row["subject_id"]: row for row in completions}
    for iid in issue_ids:
        row = by_id[iid]
        assert row["status"] == "complete"
        assert row["owner"] is not None
        # The completing actor is the owner, and the owner is subscribed.
        assert comp_counts[iid]["n"] == 1
        assert comp_counts[iid]["actor"] == row["owner"]
        assert row["owner"] in list(row["subscribers"] or [])


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_parallel_workers_write_one_shared_graph(store: Store) -> None:
    """K agents claim issues AND record findings into the shared graph.

    Beyond the claim contract: every completed issue leaves exactly one
    experiment and one finding belief wired by produced/proves edges, every
    worker also cites one shared root belief (the cross-agent convergence
    point), the edges carry their signed valence through the concurrent
    writes, and every write in the storm is attributed to the acting agent
    -- the graph reads back as if one careful agent had recorded the whole
    campaign.
    """
    issue_count, worker_count = 12, 6
    account = "tester@example.com"
    issue_ids = [
        await store.submit_issue(
            SubmitIssue(account=account, title=f"job {i}"),
            actor="coordinator",
        )
        for i in range(issue_count)
    ]
    shared_root = await store.submit_belief(
        SubmitBelief(account=account, title="shared root finding"),
        actor="coordinator",
    )

    async def worker(name: str) -> list[tuple]:
        recorded: list[tuple] = []
        while True:
            nxt = await store.next_issue()
            if nxt is None:
                return recorded
            try:
                await store.transition_owner(
                    nxt.id,
                    expected_from=None,
                    to=name,
                    actor=name,
                )
            except ConflictError:
                continue
            experiment = await store.submit_experiment(
                SubmitExperiment(
                    account=account,
                    title=f"run on job {nxt.seq}",
                    outcome="confirmed",
                ),
                actor=name,
            )
            finding = await store.submit_belief(
                SubmitBelief(
                    account=account,
                    title=f"finding from job {nxt.seq}",
                    judgement="proven",
                ),
                actor=name,
            )
            await store.add_edge(
                from_id=experiment,
                to_id=finding,
                edge_kind="proves",
                valence=0.8,
                actor=name,
            )
            await store.add_edge(
                from_id=experiment,
                to_id=nxt.id,
                edge_kind="produced_by",
                actor=name,
            )
            await store.add_edge(
                from_id=experiment,
                to_id=shared_root,
                edge_kind="favors",
                valence=0.3,
                actor=name,
            )
            await store.transition_status(
                nxt.id,
                expected_from="active",
                to="complete",
                actor=name,
            )
            recorded.append((nxt.id, experiment, finding))

    outcomes = await asyncio.gather(
        *(worker(f"worker-{i}") for i in range(worker_count)),
    )
    recorded = [entry for worker_outcome in outcomes for entry in worker_outcome]

    # Claim invariants: every issue completed exactly once, one winner each.
    assert sorted(issue for issue, _, _ in recorded) == sorted(issue_ids)
    assert len(recorded) == len(set(issue for issue, _, _ in recorded))

    async with store.engine.acquire() as conn:
        experiments = await conn.fetch(
            "SELECT id FROM inquiries WHERE kind = 'Experiment'",
        )
        beliefs = await conn.fetch(
            "SELECT id FROM inquiries WHERE kind = 'Belief'",
        )
        edges = await conn.fetch(
            "SELECT from_id, to_id, edge_kind, valence FROM edges",
        )
        unattributed = await conn.fetch(
            "SELECT count(*) AS n FROM change_log "
            "WHERE actor IN ('system', 'librarian') "
            "AND kind IN ('created', 'edge_added', 'owner', 'status')",
        )

    # Exactly one experiment and one finding per completed issue, plus the
    # shared root: the graph grew by precisely what the campaign produced.
    assert len(experiments) == issue_count
    assert len(beliefs) == issue_count + 1
    exp_ids = {exp for _, exp, _ in recorded}
    finding_ids = {finding for _, _, finding in recorded}
    assert {row["id"] for row in experiments} == exp_ids
    assert {row["id"] for row in beliefs} == finding_ids | {shared_root}

    # Every worker's edges landed, exactly three per experiment (proves its
    # finding, produced_by its issue, favors the shared root); the edge PK
    # makes duplicates impossible, so the count pins completeness too.
    assert len(edges) == 3 * issue_count
    assert len(edges) == len({(r["from_id"], r["to_id"], r["edge_kind"]) for r in edges})

    def has_edge(
        from_id: object,
        to_id: object,
        kind: str,
        valence: float | None = None,
    ) -> bool:
        return any(
            row["from_id"] == from_id
            and row["to_id"] == to_id
            and row["edge_kind"] == kind
            and (valence is None or row["valence"] == valence)
            for row in edges
        )

    for issue_id, experiment, finding in recorded:
        assert has_edge(experiment, finding, "proves", 0.8)
        assert has_edge(experiment, issue_id, "produced_by")
        assert has_edge(experiment, shared_root, "favors", 0.3)

    # Attribution: no campaign write landed under the default actor.
    assert unattributed[0]["n"] == 0
