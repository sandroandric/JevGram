"""Calls the real Jev API. Runs only when TYPESAFE_API_KEY is available."""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient

from jevgram.labeling import LABELS, VIEWS, label_sentences
from jevgram.pdf_extract import Sentence

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY", "").strip(), reason="TYPESAFE_API_KEY not set"
)


def test_jev_labels_sentences():
    sentences = [
        Sentence(0, "We train on 8 GPUs for 3.5 days using the Adam optimizer.", 0, paragraph=0),
        Sentence(1, "In the ever-evolving landscape of machine learning, attention mechanisms "
                    "play a pivotal role in unlocking unprecedented capabilities.", 0, paragraph=0),
    ]

    async def run():
        async with AsyncTypeSafeClient() as client:
            return await label_sentences(client, sentences)

    result = asyncio.run(run())

    assert [label.index for label in result.labels] == [0, 1]
    for label in result.labels:
        assert label.label in LABELS
        assert 0 <= label.confidence <= 1
        assert set(label.probabilities) == set(LABELS)
        # The API rounds each probability to two decimals, so sums can be off by ~0.01.
        assert sum(label.probabilities.values()) == pytest.approx(1, abs=0.02)
        assert set(label.views) == set(VIEWS)
        for view in label.views.values():
            assert sum(view.values()) == pytest.approx(1, abs=0.02)
    assert result.model.startswith("jev")
