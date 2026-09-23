"""Keyword relatedness between a claim and its cited evidence: BM25, no ML.

The relevance gate validated on SciFact expert labels (Recall@5 0.990 with the
embedding upgrade; this keyword form 0.944) in its dependency-free form: a
citation whose text shares NO token with the claim scores exactly 0.0 -- a
fact, not a threshold -- which is the drift signal. Scores are max-normalized
within the claim's citation set, so ``1.0`` means "the most textually related
citation this claim has" and ``0.0`` means "shares not one word with it".

The corpus for the document frequencies is the claim's own citation set: the
comparison being made is among that claim's citations, so the statistics come
from exactly that context. BM25's k1/b are the textbook constants (like
PageRank's 0.85 or RRF's k=60 elsewhere in this repo) -- the two tunables BM25
admits, and neither is fitted to anything here.

Purely derived and read-only: an observation shown next to the recorded
valence, never a verdict. It cannot say a citation is good -- zero-overlap
flags possible drift, and a human confirms.
"""

from __future__ import annotations

import math
import re


__all__ = ["related_scores"]


_K1: float = 1.2
"""BM25 term-frequency saturation constant (textbook value)."""

_B: float = 0.75
"""BM25 length-normalization constant (textbook value)."""

_TOKEN: object = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return re.findall(_TOKEN, text.lower())


def related_scores(query: str, docs: list[str]) -> list[float]:
    """Score how textually related each doc is to ``query``, max-normalized.

    Args:
      query: The claim text (belief title).
      docs: Each citation's text (title + abstract where present), same order
        as the returned scores.

    Returns:
      scores: One float per doc in ``[0, 1]`` -- BM25 relevance normalized by
        the set's maximum, so ``1.0`` marks the most related citation of the
        claim and ``0.0`` marks a citation sharing no token with the claim
        (the drift flag). All zeros when nothing overlaps, or when ``docs``
        is empty.

    """
    if not docs:
        return []
    query_tokens = _tokens(query)
    doc_tokens = [_tokens(d) for d in docs]
    lengths = [len(t) for t in doc_tokens]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for word in set(tokens):
            df[word] = df.get(word, 0) + 1
    n = len(docs)

    raw: list[float] = []
    for tokens, length in zip(doc_tokens, lengths, strict=True):
        counts: dict[str, int] = {}
        for word in tokens:
            counts[word] = counts.get(word, 0) + 1
        score = 0.0
        for word in query_tokens:
            tf = counts.get(word)
            if not tf:
                continue
            idf = math.log(1.0 + (n - df[word] + 0.5) / (df[word] + 0.5))
            score += idf * tf * (_K1 + 1.0) / (tf + _K1 * (1.0 - _B + _B * length / avg_len))
        raw.append(score)
    peak = max(raw, default=0.0)
    if peak <= 0.0:
        return [0.0 for _ in docs]
    return [score / peak for score in raw]