import asyncio
import json

import pytest

from benchmarks import detectionai
from benchmarks.detectionai import (
    HUMAN_TEXT_FILE,
    MODEL_FILE,
    MODELS,
    JevCache,
    Passage,
    auroc,
    jev_score,
    load_passages,
    stratified_sample,
    tpr_at_fpr,
)
from tests.fakes import FakeTypeSafeClient


def test_auroc_counts_wins_and_ties():
    assert auroc([0.1, 0.2], [0.8, 0.9]) == 1.0
    assert auroc([0.8, 0.9], [0.1, 0.2]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([0.1, 0.6], [0.5]) == 0.5


def test_tpr_at_fpr_stays_within_the_false_positive_cap():
    human = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    ai = [0.975, 0.985, 0.995, 0.5]
    # 1% of 100 humans may be flagged: only 0.99 lies above the threshold 0.98.
    assert tpr_at_fpr(human, ai, 0.01) == pytest.approx(2 / 4)
    # 5%: threshold 0.94, so three AI texts are flagged.
    assert tpr_at_fpr(human, ai, 0.05) == pytest.approx(3 / 4)


def passage(key, genre):
    return Passage(key=key, genre=genre, texts={"human": key})


def test_stratified_sample_is_balanced_and_reproducible():
    pool = [passage(f"{genre}{i}", genre) for genre in "abc" for i in range(10)]
    first = stratified_sample(pool, 7, seed=1)

    assert [p.genre for p in first].count("a") == 3
    assert [p.genre for p in first].count("b") == 2
    assert [p.genre for p in first].count("c") == 2
    assert [p.key for p in stratified_sample(pool, 7, seed=1)] == [p.key for p in first]
    with pytest.raises(ValueError):
        stratified_sample(pool, 40, seed=1)


def verdict(pangram):
    return {
        "pengram": {"ai_likelihood": pangram},
        "originality": {"confidence": {"AI": pangram}},
        "gptzero": {"average_generated_prob": pangram},
        "roberta-base-detector": 0.5,
    }


def test_load_passages_uses_human_text_with_line_breaks(tmp_path):
    human = "First paragraph here.\nSecond paragraph here."
    (tmp_path / HUMAN_TEXT_FILE).parent.mkdir(parents=True)
    (tmp_path / HUMAN_TEXT_FILE).write_text(json.dumps([{"text": human}]))
    for model, model_id in MODELS.items():
        path = tmp_path / MODEL_FILE.format(model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            # The model files strip line breaks from the human text.
            {"text": human.replace("\n", ""), "genre": "news",
             "ai_generated": {model_id: f"Text by {model}."},
             "human_verdict": verdict(0.1), "ai_verdicts": {model_id: verdict(0.9)}},
            # Missing from the human text file: skipped.
            {"text": "Unmatched.", "genre": "news", "ai_generated": {model_id: "x"},
             "human_verdict": verdict(0.1), "ai_verdicts": {model_id: verdict(0.9)}},
        ]
        path.write_text(json.dumps(records))

    [loaded] = load_passages(tmp_path)

    assert loaded.texts["human"] == human
    assert loaded.texts["claude-opus-4"] == "Text by claude-opus-4."
    assert loaded.detector_scores["human"]["Pangram"] == 0.1
    assert loaded.detector_scores["gpt-4.1"]["Originality"] == 0.9


def test_jev_score_averages_not_human_probability_and_caches(tmp_path):
    ai = {"ai_generated": 0.7, "ai_assisted": 0.1, "human_written": 0.2}
    human = {"ai_generated": 0.2, "ai_assisted": 0.1, "human_written": 0.7}
    client = FakeTypeSafeClient(answer_for=lambda view, text: ai if "robot" in text else human)
    cache = JevCache(tmp_path / "cache.jsonl")
    text = "A plain sentence written by people.\nThe robot wrote this sentence."

    entry = asyncio.run(jev_score(client, cache, "news article", text))

    assert entry["sentences"] == 2
    assert entry["score"] == pytest.approx((0.3 + 0.8) / 2)
    assert entry["flagged_share"] == pytest.approx(0.5)
    assert client.calls[0][0] == {"document_type": "news article"}
    # A second call, even from a fresh cache object, does not query Jev again.
    again = asyncio.run(jev_score(client, JevCache(tmp_path / "cache.jsonl"), "news article", text))
    assert again["score"] == entry["score"]
    assert len(client.calls) == 1


def test_evaluate_compares_human_with_ai_scores():
    passages = []
    for i in range(4):
        p = Passage(key=str(i), genre="news", texts={})
        p.detector_scores = {"human": {"Jevgram (sentences)": 0.1 * i},
                             "gpt-4.1": {"Jevgram (sentences)": 0.9}}
        passages.append(p)
    metrics = detectionai.evaluate(passages, ["gpt-4.1"], "Jevgram (sentences)", seed=1)

    assert metrics["AUROC"] == 1.0
    assert metrics["n human"] == 4 and metrics["n AI"] == 4


def test_whole_passages_are_asked_once_each_grouped_by_genre(tmp_path, monkeypatch):
    class PassageClient(FakeTypeSafeClient):
        async def system_one(self, state, questions):
            self.calls.append((state, questions))
            answers = {}
            for key, question in questions.items():
                ai = "robot" in question.instructions["text"]
                human = 0.1 if ai else 0.9
                probabilities = {"fully AI written": 1 - human - 0.05, "AI assisted": 0.05,
                                 "fully human written": human}
                answers[key] = {"type": "choice", "choice": max(probabilities, key=probabilities.get),
                                "confidence": 0.8, "probabilities": probabilities}
            from typesafe_sdk import SystemOneResponse
            return SystemOneResponse.model_validate({
                "model": "jev-test", "usage": {"input_tokens": 10, "output_tokens": 0},
                "answers": answers})

    client = PassageClient()

    class ClientContext:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(detectionai, "AsyncTypeSafeClient", ClientContext)
    passages = [
        Passage(key="1", genre="news", texts={"human": "People wrote this.", "gpt-4.1": "A robot wrote this."},
                detector_scores={"human": {}, "gpt-4.1": {}}),
        Passage(key="2", genre="novel", texts={"human": "Once upon a time.", "gpt-4.1": "A robot tale."},
                detector_scores={"human": {}, "gpt-4.1": {}}),
    ]
    cache = JevCache(tmp_path / "passages.jsonl")

    asyncio.run(detectionai.score_whole_passages_with_jev(passages, cache))

    assert sorted(state["document_type"] for state, _ in client.calls) == ["news article", "novel excerpt"]
    for _, questions in client.calls:
        assert len(questions) == 2
        for question in questions.values():
            assert question.instructions["question"] == "How was `text` written?"
            assert all(description is None for description in question.criteria.values())
    assert passages[0].detector_scores["human"]["Jev (whole passage)"] == pytest.approx(0.1)
    assert passages[0].detector_scores["gpt-4.1"]["Jev (whole passage)"] == pytest.approx(0.9)
    # Cached: a second run sends nothing.
    asyncio.run(detectionai.score_whole_passages_with_jev(passages, cache))
    assert len(client.calls) == 2
