"""Pure BM25 relatedness: normalization, zero-overlap drift, textbook constants."""

from __future__ import annotations

import pytest

from trackinizer.types.textmatch import related_scores


def test_most_related_scores_one_and_others_below() -> None:
    scores = related_scores(
        "Mass-matrix adaptation dominates step-size tuning",
        [
            "Adaptive step-size survey. We study step-size tuning and mass-matrix adaptation.",
            "Wafer-scale inference. Datacenter-scale weight traffic dominates.",
        ],
    )
    assert scores[0] == pytest.approx(1.0)
    assert 0.0 <= scores[1] < 1.0


def test_zero_overlap_is_the_drift_signal() -> None:
    """A citation sharing no token with the claim scores exactly 0.0."""
    scores = related_scores(
        "Batch 768 reproduces the reported ESS",
        ["Quantum wand spells and ritual lattices"],
    )
    assert scores == [0.0]


def test_all_zero_when_nothing_overlaps() -> None:
    scores = related_scores(
        "Zxy qqq ffk",
        ["Alpha beta gamma", "Delta epsilon zeta"],
    )
    assert scores == [0.0, 0.0]


def test_case_insensitive_and_empty_docs() -> None:
    assert related_scores("HMC", ["hmc ablation runs"]) == [pytest.approx(1.0)]
    assert related_scores("anything", []) == []
