"""Contention semantics for parallel agents sharing one issue queue.

The multi-agent coordination contract under test: an ``active`` row with no
``owner`` is open; with an ``owner`` it is in progress. ``trax next`` and the
claim transition must hold that contract when many agents race.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from trackinizer.lib.postgres import PGliteEngine
from trackinizer.lib.postgres.testing import reset_schema
from trackinizer.server.store.core import Store, StubEmbedder
from trackinizer.types.errors import ConflictError
from trackinizer.wire.bodies import SubmitIssue


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
