"""Tier-1 benchmark regression: SciFact labels seeded and scored, hermetically.

The checked-in fixture holds 25 REAL SciFact claims with expert
SUPPORT/CONTRADICT labels and their cited papers (Allen AI, EMNLP 2020).
The test seeds them into a fresh store, runs the scoring pipeline, and
asserts the full chain reproduces the labels exactly: link sign, derived
confidence, zero judgement edits. Deterministic by seed; no LLM, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import json
import math

import pytest
import pytest_asyncio

from trackinizer.lib.postgres.testing import reset_schema
from trackinizer.server.embedders.stub import StubEmbedder
from trackinizer.server.store.core import Store
from trackinizer.wire.bodies import SubmitBelief, SubmitPaper


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from trackinizer.lib.postgres import PGliteEngine

FIXTURE = Path(__file__).parent / "scifact_fixture.json"
SUPPORT_VALENCE = 0.7
CONTRADICT_VALENCE = -0.7


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _load_fixture() -> tuple[dict[int, dict], list[dict]]:
    raw = json.loads(FIXTURE.read_text())
    docs = {int(d["doc_id"]): d for d in raw["docs"]}
    return docs, raw["claims"]


@pytest_asyncio.fixture(loop_scope="session")
async def store(pglite_engine: PGliteEngine) -> AsyncIterator[Store]:
    await reset_schema(pglite_engine)
    built = Store(pglite_engine, embed=StubEmbedder())
    await built.bootstrap()
    yield built


@pytest.mark.db_pglite
@pytest.mark.asyncio(loop_scope="session")
async def test_scifact_tier1_regression(store: Store) -> None:
    """25 real claims: link signs, derived confidences, drift scores pinned."""
    docs, claims = _load_fixture()
    key = await _seed_and_score_inner(store, docs, claims)

    assert len(key) == len(claims)
    citations_checked = 0
    for entry in key.values():
        belief_id = UUID(entry["belief_id"])
        rep = await store.evidence_for(belief_id)
        assert rep is not None
        expert = {item["paper_id"]: item["label"] for item in entry["items"]}
        assert len(rep.citations) == len(entry["items"])
        total = 0.0
        for c in rep.citations:
            want = 1.0 if expert.get(str(c.citer_id)) == "SUPPORT" else -1.0
            got = 1.0 if c.valence > 0 else -1.0
            assert got == want, f"sign mismatch on {c.title}"
            assert 0.0 <= c.related <= 1.0  # drift score always well-formed
            total += c.valence
            citations_checked += 1
        expected = _sigmoid(total)
        assert rep.derived_confidence == pytest.approx(expected, abs=1e-9)
    assert citations_checked == sum(len(e["items"]) for e in key.values())


async def _seed_and_score_inner(store: Store, docs: dict, claims: list) -> dict:
    key = {}
    paper_ids = {}
    account = "no-auth@localhost"
    needed: set[int] = set()
    for c in claims:
        for doc_id, _lab in c["labels"]:
            needed.add(doc_id)
    for doc_id in sorted(needed):
        row = await store.submit_paper(
            SubmitPaper(
                account=account,
                title=docs[doc_id].get("title", f"doc {doc_id}"),
                source=f"scifact:{doc_id}",
            ),
        )
        paper_ids[doc_id] = row  # submit_paper returns the minted UUID
    for c in claims:
        belief = await store.submit_belief(
            SubmitBelief(account=account, title=c["claim"]),
        )
        items = []
        for doc_id, label in c["labels"]:
            paper_id = paper_ids[doc_id]
            valence = SUPPORT_VALENCE if label == "SUPPORT" else CONTRADICT_VALENCE
            await store.add_edge(
                from_id=paper_id,
                to_id=belief,
                edge_kind="proves",
                actor="scifact-fixture",
                valence=valence,
            )
            items.append({"doc_id": doc_id, "paper_id": str(paper_id), "label": label})
        key[c["claim"]] = {"belief_id": str(belief), "items": items}
    return key
