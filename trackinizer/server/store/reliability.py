""":class:`_ReliabilityMixin` -- the truth-discovery reliability sweep.

:meth:`recompute_reliability` loads the whole currently-true ``proves``
citation matrix, iterates the joint truth/reliability fixed point
(:func:`trackinizer.types.reliability.reliability_fixed_point`), and writes
each citing row's weight to the ``artifact_reliability`` column. Global by
nature -- one edge changing any source's agreement profile shifts the
consensus it is measured against -- so it is a full recompute run off the
request path by a periodic sweep, not an incremental per-write update.

A read-only-until-the-final-write leaf like :class:`_AuthorityMixin`: it reads
edges through ``self.engine`` and writes only the derived reliability column,
calling no other mixin. Rows the matrix never reaches are reset to NULL, and
the confidence fold reads NULL as the uniform cold-start prior -- so a graph
with no observed disagreement folds exactly as the pre-reliability build did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datetime import datetime
from uuid import UUID

from trackinizer.server.notify import tx
from trackinizer.server.sql_fragments import RELIABILITY_EDGES_SQL
from trackinizer.server.store.shared import _StoreShared
from trackinizer.server.values import vetted_sql
from trackinizer.types.reliability import (
    Citation,
    reliability_fixed_point,
    temporal_weight,
)


if TYPE_CHECKING:
    from trackinizer.lib.postgres import Conn


__all__ = ["_ReliabilityMixin"]


class _ReliabilityMixin(_StoreShared):
    """The truth-discovery reliability sweep for :class:`Store`."""

    async def recompute_reliability(self, *, conn: Conn | None = None) -> int:
        """Recompute every reliability weight from the current proves matrix.

        Loads the currently-true citation matrix, iterates the fixed point,
        and writes the weights in one transaction so a reader never sees a
        half-written ranking. Rows the matrix never reaches are reset to
        NULL -- the fold's uniform cold-start prior.

        Args:
          conn: Existing connection to reuse, or None to acquire one. PGlite's
            single connection deadlocks on a re-entrant acquire.

        Returns:
          written: Total reliability weights written.

        """
        if conn is not None:
            return await self._recompute_reliability(conn)
        async with self.engine.acquire() as new_conn:
            return await self._recompute_reliability(new_conn)

    # Deliberately NOT named ``_recompute``: ``_AuthorityMixin`` owns that name,
    # and it precedes this mixin in ``Store``'s bases -- a shared private name
    # silently resolved to the authority sweep there, so ``recompute_reliability``
    # ran PageRank and wrote authority columns while returning their count.
    async def _recompute_reliability(self, conn: Conn) -> int:
        """Load the matrix, iterate, and write the column in one transaction."""
        rows = await conn.fetch(RELIABILITY_EDGES_SQL)
        # Recency is measured against the newest citation ON EACH CLAIM, so the
        # temporal weight is deterministic (no wall-clock dependence) and a
        # claim whose evidence all arrived together is untouched: every age is
        # zero and every tau is exactly 1.0.
        newest_by_claim: dict[UUID, datetime] = {}
        for row in rows:
            claim_id = cast(UUID, row["to_id"])
            created = cast(datetime, row["created"])
            if claim_id not in newest_by_claim or created > newest_by_claim[claim_id]:
                newest_by_claim[claim_id] = created
        citations = [
            Citation(
                source=cast(UUID, row["from_id"]),
                claim=cast(UUID, row["to_id"]),
                valence=cast(float, row["valence"]),
                tau=temporal_weight(
                    (newest_by_claim[cast(UUID, row["to_id"])]
                     - cast(datetime, row["created"])).total_seconds(),
                ),
            )
            for row in rows
        ]
        weights = reliability_fixed_point(citations)
        async with tx(conn):
            await conn.execute(
                "UPDATE inquiries SET artifact_reliability = NULL "
                "WHERE artifact_reliability IS NOT NULL",
            )
            if not weights:
                return 0
            ids = list(weights)
            values = [weights[row_id] for row_id in ids]
            await conn.execute(
                vetted_sql(
                    "UPDATE inquiries AS i SET artifact_reliability = v.weight FROM "
                    "(SELECT unnest($1::uuid[]) AS id, "
                    "unnest($2::double precision[]) AS weight) AS v WHERE i.id = v.id",
                ),
                ids,
                values,
            )
            return len(ids)
