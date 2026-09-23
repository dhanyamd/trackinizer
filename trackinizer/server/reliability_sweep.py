"""Background loop that keeps the derived reliability column fresh.

Source reliability (the truth-discovery fixed point over the proves matrix,
see :mod:`trackinizer.types.reliability`) is a global computation, so it is
recomputed wholesale rather than per write. This loop mirrors
:mod:`trackinizer.server.authority_sweep`: it recomputes when at least
:data:`DIRTY_EDGES` edge mutations have landed since the last run OR
:data:`MAX_INTERVAL_SEC` has elapsed, and never more often than
:data:`MIN_INTERVAL_SEC`. An idle graph costs nothing; a busy one folds many
edge changes into one recompute.

Edge mutations are counted from ``change_log`` (the same poll the authority
sweep uses), so the write path gains no new coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import asyncio
import logging

from trackinizer.server.authority_sweep import edge_change_count


if TYPE_CHECKING:
    from trackinizer.server.store.core import Store


__all__ = [
    "DIRTY_EDGES",
    "MAX_INTERVAL_SEC",
    "MIN_INTERVAL_SEC",
    "reliability_sweep_loop",
]


_logger: Final = logging.getLogger(__name__)

DIRTY_EDGES: Final = 50
"""Recompute once this many edge mutations have accrued since the last run."""

MIN_INTERVAL_SEC: Final = 60.0
"""Floor between recomputes: a burst of edits coalesces into one run."""

MAX_INTERVAL_SEC: Final = 600.0
"""Ceiling: recompute at least this often even below the dirty threshold."""


async def reliability_sweep_loop(
    store: Store,
    *,
    min_interval_sec: float = MIN_INTERVAL_SEC,
    max_interval_sec: float = MAX_INTERVAL_SEC,
    dirty_edges: int = DIRTY_EDGES,
) -> None:
    """Recompute reliability forever, coalescing edge-change bursts.

    Sleeps ``min_interval_sec`` between checks. Recomputes when the edge-change
    count has advanced by ``dirty_edges`` since the last run, or when
    ``max_interval_sec`` has elapsed. Runs one recompute at startup so a fresh
    process serves weights immediately. Cancellation (shutdown) propagates.

    Args:
      store: The store to recompute against.
      min_interval_sec: Poll cadence and hard floor between recomputes.
      max_interval_sec: Force a recompute at least this often.
      dirty_edges: Edge-change delta that triggers a recompute early.

    """
    last_count = await edge_change_count(store)
    elapsed = 0.0
    first_pass = True  # Recompute once at startup so a fresh process serves weights.
    while True:
        current = await edge_change_count(store)
        due = current - last_count >= dirty_edges or elapsed >= max_interval_sec
        if first_pass or due:
            try:
                written = await store.recompute_reliability()
            except Exception:
                # A sweep failure must not kill the loop: log and retry next
                # cycle, so a transient DB blip degrades to stale weights, not
                # a permanently dead ranking.
                _logger.exception("reliability sweep failed; retrying next cycle")
            else:
                _logger.info("reliability sweep wrote %d weights", written)
                last_count = current
                elapsed = 0.0
                first_pass = False
        await asyncio.sleep(min_interval_sec)
        elapsed += min_interval_sec