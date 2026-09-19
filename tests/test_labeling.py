import asyncio

import pytest

from jevgram import labeling
from jevgram.labeling import (
    LABELS,
    OPTIONS,
    QUESTION,
    VIEWS,
    WHOLE_TEXT_QUESTION,
    label_document,
    label_from,
    label_sentences,
    label_texts,
    paper_parts,
    summarize,
)
from jevgram.pdf_extract import Sentence
from tests.fakes import FakeTypeSafeClient


def make_sentences(count, per_paragraph=3):
    return [
        Sentence(index=i, text=f"Sentence number {i} is here.", page=0,
                 paragraph=i // per_paragraph)
        for i in range(count)
    ]


def test_views_are_averaged_with_equal_weight():
    views = {
        "alone": {"ai_generated": 0.61, "ai_assisted": 0.22, "human_written": 0.17},
        "neighbors": {"ai_generated": 0.48, "ai_assisted": 0.25, "human_written": 0.27},
        "paragraph": {"ai_generated": 0.55, "ai_assisted": 0.20, "human_written": 0.25},
    }
    label = label_from(7, views)

    assert label.index == 7
    assert label.label == "ai_generated"
    assert label.probabilities["ai_generated"] == pytest.approx(0.5467, abs=1e-4)
    assert label.probabilities["ai_assisted"] == pytest.approx(0.2233, abs=1e-4)
    assert label.probabilities["human_written"] == pytest.approx(0.23)
    assert label.confidence == pytest.approx(label.probabilities["ai_generated"])
    assert label.views == views


def test_context_can_outvote_the_sentence_alone():
    views = {
        "alone": {"ai_generated": 0.6, "ai_assisted": 0.0, "human_written": 0.4},
        "neighbors": {"ai_generated": 0.2, "ai_assisted": 0.0, "human_written": 0.8},
        "paragraph": {"ai_generated": 0.3, "ai_assisted": 0.0, "human_written": 0.7},
    }
    assert label_from(0, views).label == "human_written"


def test_each_sentence_gets_three_bare_questions_with_its_context():
    client = FakeTypeSafeClient()
    sentences = make_sentences(4, per_paragraph=2)

    asyncio.run(label_sentences(client, sentences))

    [(state, questions)] = client.calls
    assert state == {"document_type": "research paper"}
    assert list(questions)[:3] == ["s0_alone", "s0_neighbors", "s0_paragraph"]
    assert len(questions) == 12
    for question in questions.values():
        assert question.instructions["question"] == QUESTION
        # Bare options: the labels only, without descriptions.
        assert dict(question.criteria) == {option: None for option in OPTIONS.values()}

    assert questions["s1_alone"].instructions == {
        "question": QUESTION, "sentence": "Sentence number 1 is here.",
    }
    assert questions["s1_neighbors"].instructions == {
        "question": QUESTION,
        "previous_sentence": "Sentence number 0 is here.",
        "sentence": "Sentence number 1 is here.",
        "next_sentence": "Sentence number 2 is here.",
    }
    assert questions["s2_paragraph"].instructions["paragraph"] == (
        "Sentence number 2 is here. Sentence number 3 is here."
    )
    # The first and last sentences have only one neighbour.
    assert "previous_sentence" not in questions["s0_neighbors"].instructions
    assert "next_sentence" not in questions["s3_neighbors"].instructions


def test_document_type_is_passed_as_state():
    client = FakeTypeSafeClient()
    asyncio.run(label_sentences(client, make_sentences(1), document_type="news article"))
    assert client.calls[0][0] == {"document_type": "news article"}


def test_labels_come_back_in_order_across_batches(monkeypatch):
    monkeypatch.setattr(labeling, "BATCH_SIZE", 4)
    ai = {"ai_generated": 0.8, "ai_assisted": 0.1, "human_written": 0.1}
    client = FakeTypeSafeClient(
        answer_for=lambda view, text: ai if "number 5 " in text else
        {"ai_generated": 0.2, "ai_assisted": 0.1, "human_written": 0.7}
    )

    result = asyncio.run(label_sentences(client, make_sentences(10)))

    assert [len(questions) // len(VIEWS) for _, questions in client.calls] == [4, 4, 2]
    assert [label.index for label in result.labels] == list(range(10))
    assert result.labels[5].label == "ai_generated"
    assert set(result.labels[5].views) == set(VIEWS)
    assert all(label.label == "human_written" for label in result.labels if label.index != 5)
    assert result.model == "jev-test"
    assert result.input_tokens == 300


def test_long_paragraphs_split_requests_by_size(monkeypatch):
    monkeypatch.setattr(labeling, "REQUEST_CHAR_BUDGET", 3_000)
    long_text = "word " * 150
    sentences = [Sentence(index=i, text=long_text.strip() + ".", page=0, paragraph=0)
                 for i in range(4)]
    client = FakeTypeSafeClient()

    result = asyncio.run(label_sentences(client, sentences))

    assert len(client.calls) > 1
    assert [label.index for label in result.labels] == [0, 1, 2, 3]


def test_summary_counts_shares_and_confidence():
    def views(ai, human):
        view = {"ai_generated": ai, "ai_assisted": 0.0, "human_written": human}
        return {name: view for name in VIEWS}

    labels = [
        label_from(0, views(0.9, 0.1)),
        label_from(1, views(0.7, 0.3)),
        label_from(2, views(0.2, 0.8)),
        label_from(3, views(0.4, 0.6)),
    ]
    summary = summarize(labels)

    assert summary["ai_generated"]["count"] == 2
    assert summary["ai_generated"]["share"] == pytest.approx(0.5)
    assert summary["ai_generated"]["mean_confidence"] == pytest.approx(0.8)
    assert summary["human_written"]["mean_confidence"] == pytest.approx(0.7)
    assert summary["ai_assisted"] == {
        "name": "AI assisted", "count": 0, "share": 0.0, "mean_confidence": None,
    }
    assert set(summary) == set(LABELS)


def test_whole_texts_get_one_bare_question_each(monkeypatch):
    monkeypatch.setattr(labeling, "TEXTS_PER_REQUEST", 2)
    ai = {"ai_generated": 0.8, "ai_assisted": 0.1, "human_written": 0.1}
    human = {"ai_generated": 0.1, "ai_assisted": 0.1, "human_written": 0.8}
    client = FakeTypeSafeClient(answer_for=lambda view, text: ai if "robot" in text else human)

    result = asyncio.run(label_texts(client, ["By people.", "By a robot.", "By people too."],
                                     document_type="news article"))

    assert [len(questions) for _, questions in client.calls] == [2, 1]
    assert client.calls[0][0] == {"document_type": "news article"}
    question = client.calls[0][1]["t0"]
    assert question.instructions == {"question": WHOLE_TEXT_QUESTION, "text": "By people."}
    assert dict(question.criteria) == {option: None for option in OPTIONS.values()}
    assert result.distributions == [human, ai, human]
    assert result.models == {"jev-test"}


def test_paper_parts_keep_short_papers_whole_and_balance_long_ones():
    assert paper_parts(["One.", "Two."], limit=100) == ["One.\n\nTwo."]
    paragraphs = ["x" * 40] * 5
    parts = paper_parts(paragraphs, limit=100)
    assert len(parts) == 3
    assert all(len(part) <= 100 for part in parts)
    assert "\n\n".join(parts) == "\n\n".join(paragraphs)


def test_document_overview_averages_sentences_paragraphs_and_paper():
    ai = {"ai_generated": 0.9, "ai_assisted": 0.0, "human_written": 0.1}
    human = {"ai_generated": 0.1, "ai_assisted": 0.0, "human_written": 0.9}

    def answer(view, text):
        if view == "text":
            # Paragraph 0 and the whole paper read as AI, paragraph 1 as human.
            return human if text.startswith("Sentence number 2") else ai
        return ai if text.startswith("Sentence number 0") else human

    client = FakeTypeSafeClient(answer_for=answer)
    result = asyncio.run(label_document(client, make_sentences(4, per_paragraph=2)))
    overview = result.overview

    # Sentences: one AI sentence of four.
    assert overview["sentences"]["probabilities"]["ai_generated"] == pytest.approx(0.3)
    assert overview["sentences"]["count"] == 4
    # Paragraphs: one AI paragraph of two, each counted once.
    assert overview["paragraphs"]["probabilities"]["ai_generated"] == pytest.approx(0.5)
    assert overview["paragraphs"]["count"] == 2
    assert overview["paper"]["probabilities"]["ai_generated"] == pytest.approx(0.9)
    assert overview["paper"]["parts"] == 1
    assert overview["overall"]["probabilities"]["ai_generated"] == pytest.approx((0.3 + 0.5 + 0.9) / 3)
    assert overview["overall"]["label"] == "ai_generated"
    assert overview["sentences"]["label"] == "human_written"
    assert [label.index for label in result.sentences.labels] == [0, 1, 2, 3]
    for entry in overview.values():
        assert sum(entry["probabilities"].values()) == pytest.approx(1)
