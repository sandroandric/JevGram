"""Benchmark Jevgram against the detectors in Imas & Jabarian's DetectionAI study.

Data: https://github.com/brianjabarian/DetectionAI (clone it and pass its path).
The study pairs pre-2020 human passages with AI versions written on the same
topic and stores Pangram, Originality, GPTZero and RoBERTa scores for every text.

For a stratified sample of human passages, Jev scores the human text and each AI
version in two ways, told the passage's genre each time:

- "Jevgram (sentences)": the app's three-view method on every sentence; the
  passage score is the mean over its sentences of P(not human written).
- "Jev (whole passage)": the same bare question asked once about the whole
  passage, as the commercial detectors judge whole passages; the score is
  P(not human written).

The detectors' stored scores for exactly the same texts are compared on the same metrics.

Run:
    uv run python -m benchmarks.detectionai --data /path/to/DetectionAI
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jevgram.labeling import label_sentences, label_texts
from jevgram.pdf_extract import sentences_from_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "results" / "detectionai"

# AI versions to test: the newest GPT in the study and the two Claude models.
MODELS = {
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
}
MODEL_FILE = "all_json/100_ai_generated_text_and_detector_result_files/generated_output_{}_all_detectors.json"
# The per-model files strip the human texts' line breaks; this file keeps them.
HUMAN_TEXT_FILE = "all_json/different_threshold_json_files/merged_corpus_withDetectors_defaultProb.json"

DOCUMENT_TYPES = {
    "news": "news article",
    "novel": "novel excerpt",
    "blog": "blog post",
    "resume": "resume",
    "restaurant review": "restaurant review",
    "amazon review": "product review",
}

DETECTORS: dict[str, Callable[[dict], float]] = {
    "Pangram": lambda verdict: verdict["pengram"]["ai_likelihood"],
    "Originality": lambda verdict: verdict["originality"]["confidence"]["AI"],
    "GPTZero": lambda verdict: verdict["gptzero"]["average_generated_prob"],
    "RoBERTa": lambda verdict: verdict["roberta-base-detector"],
}

SENTENCES = "Jevgram (sentences)"
WHOLE_PASSAGE = "Jev (whole passage)"

FPR_CAPS = (0.01, 0.05)
BOOTSTRAP_SAMPLES = 1000
# Each passage runs up to 6 requests of ~18k tokens at once; four passages at a
# time plus patient retries keep a full run within the account's rate limits.
MAX_CONCURRENT_PASSAGES = 4
RETRY = RetryPolicy(max_retries=8, backoff_max=30.0)


@dataclass
class Passage:
    key: str
    genre: str
    # "human" and one entry per model in MODELS.
    texts: dict[str, str]
    detector_scores: dict[str, dict[str, float | None]] = field(default_factory=dict)


def _normalize(text: str) -> str:
    return "".join(text.split())


def _detector_scores(verdict: dict) -> dict[str, float | None]:
    scores = {}
    for name, read in DETECTORS.items():
        try:
            value = read(verdict)
        except (KeyError, TypeError):
            value = None
        scores[name] = float(value) if isinstance(value, (int, float)) else None
    return scores


def load_passages(data_dir: Path) -> list[Passage]:
    """Human passages present in every model file, with their AI versions and detector scores."""
    human_texts = {
        _normalize(record["text"]): record["text"]
        for record in json.loads((data_dir / HUMAN_TEXT_FILE).read_text())
    }
    by_model = {
        model: {_normalize(record["text"]): record
                for record in json.loads((data_dir / MODEL_FILE.format(model_id)).read_text())}
        for model, model_id in MODELS.items()
    }
    passages = []
    for key, record in by_model["gpt-4.1"].items():
        if key not in human_texts or any(key not in records for records in by_model.values()):
            continue
        passage = Passage(key=hashlib.sha256(key.encode()).hexdigest()[:16],
                          genre=record["genre"], texts={"human": human_texts[key]})
        passage.detector_scores["human"] = _detector_scores(record["human_verdict"])
        for model, model_id in MODELS.items():
            model_record = by_model[model][key]
            passage.texts[model] = model_record["ai_generated"][model_id]
            passage.detector_scores[model] = _detector_scores(model_record["ai_verdicts"][model_id])
        passages.append(passage)
    return passages


def stratified_sample(passages: Sequence[Passage], size: int, seed: int) -> list[Passage]:
    """An equal number of passages per genre (remainder to the first genres), reproducible by seed."""
    genres = sorted({passage.genre for passage in passages})
    rng = random.Random(seed)
    sample = []
    for position, genre in enumerate(genres):
        quota = size // len(genres) + (1 if position < size % len(genres) else 0)
        pool = sorted((p for p in passages if p.genre == genre), key=lambda p: p.key)
        if quota > len(pool):
            raise ValueError(f"Genre {genre!r} has {len(pool)} passages, {quota} requested.")
        sample.extend(rng.sample(pool, quota))
    return sample


def auroc(human: Sequence[float], ai: Sequence[float]) -> float:
    """Probability that a random AI text scores above a random human text (ties count half)."""
    wins = sum((a > h) + 0.5 * (a == h) for a in ai for h in human)
    return wins / (len(ai) * len(human))


def tpr_at_fpr(human: Sequence[float], ai: Sequence[float], cap: float) -> float:
    """Share of AI texts flagged at the lowest threshold that flags at most `cap` of human texts."""
    allowed = int(cap * len(human))
    ranked = sorted(human, reverse=True)
    # Flag scores strictly above the (allowed+1)-th highest human score.
    threshold = ranked[allowed] if allowed < len(ranked) else float("-inf")
    return sum(score > threshold for score in ai) / len(ai)


def bootstrap_auroc(human: Sequence[float], ai: Sequence[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    values = sorted(
        auroc(rng.choices(human, k=len(human)), rng.choices(ai, k=len(ai)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return values[int(0.025 * BOOTSTRAP_SAMPLES)], values[int(0.975 * BOOTSTRAP_SAMPLES) - 1]


class JevCache:
    """Jev results per (document type, text), stored as JSON lines so runs can resume."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                self.entries[entry["key"]] = entry

    @staticmethod
    def key(document_type: str, text: str) -> str:
        return hashlib.sha256(f"{document_type}\n{text}".encode()).hexdigest()

    def get(self, document_type: str, text: str) -> dict | None:
        return self.entries.get(self.key(document_type, text))

    def put(self, document_type: str, text: str, entry: dict) -> None:
        entry = {"key": self.key(document_type, text), **entry}
        self.entries[entry["key"]] = entry
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")


async def jev_score(client: AsyncTypeSafeClient, cache: JevCache, document_type: str,
                    text: str) -> dict:
    cached = cache.get(document_type, text)
    if cached is not None:
        return cached
    sentences = sentences_from_text(text)
    if not sentences:
        entry = {"score": None, "sentences": 0, "flagged_share": None, "model": None}
    else:
        result = await label_sentences(client, sentences, document_type=document_type)
        entry = {
            "score": sum(1 - label.probabilities["human_written"] for label in result.labels)
            / len(result.labels),
            "sentences": len(result.labels),
            "flagged_share": sum(label.label != "human_written" for label in result.labels)
            / len(result.labels),
            "model": result.model,
            "input_tokens": result.input_tokens,
        }
    cache.put(document_type, text, entry)
    return entry


async def score_with_jev(passages: Sequence[Passage], cache: JevCache) -> None:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PASSAGES)
    done = 0
    total = sum(len(passage.texts) for passage in passages)

    async with AsyncTypeSafeClient(timeout=120.0, retry=RETRY) as client:
        async def one(passage: Passage, source: str) -> None:
            nonlocal done
            async with semaphore:
                entry = await jev_score(client, cache, DOCUMENT_TYPES[passage.genre],
                                        passage.texts[source])
            passage.detector_scores[source][SENTENCES] = entry["score"]
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  Jev scored {done}/{total} texts", flush=True)

        await asyncio.gather(*(one(passage, source)
                               for passage in passages for source in passage.texts))


async def score_whole_passages_with_jev(passages: Sequence[Passage], cache: JevCache) -> None:
    """Ask Jev once per text with the app's whole-text question, grouped by genre."""
    by_type: dict[str, dict[str, str]] = {}
    for passage in passages:
        document_type = DOCUMENT_TYPES[passage.genre]
        for text in passage.texts.values():
            if text.strip() and cache.get(document_type, text) is None:
                by_type.setdefault(document_type, {})[JevCache.key(document_type, text)] = text

    async with AsyncTypeSafeClient(timeout=120.0, retry=RETRY) as client:
        results = await asyncio.gather(*(
            label_texts(client, list(texts.values()), document_type)
            for document_type, texts in by_type.items()
        ))
    for (document_type, texts), result in zip(by_type.items(), results):
        for text, probabilities in zip(texts.values(), result.distributions):
            cache.put(document_type, text, {
                "score": 1 - probabilities["human_written"],
                "label": max(probabilities, key=probabilities.__getitem__),
                "probabilities": probabilities,
                "model": ", ".join(sorted(result.models)),
                "input_tokens": result.input_tokens / len(texts),
            })

    for passage in passages:
        document_type = DOCUMENT_TYPES[passage.genre]
        for source, text in passage.texts.items():
            entry = cache.get(document_type, text)
            passage.detector_scores[source][WHOLE_PASSAGE] = entry["score"] if entry else None


def evaluate(passages: Sequence[Passage], sources: Sequence[str], detector: str,
             seed: int) -> dict | None:
    """Metrics for one detector: human texts against the AI texts from `sources`."""
    human = [p.detector_scores["human"][detector] for p in passages]
    ai = [p.detector_scores[source][detector] for p in passages for source in sources]
    human = [score for score in human if score is not None]
    ai = [score for score in ai if score is not None]
    if not human or not ai:
        return None
    low, high = bootstrap_auroc(human, ai, seed)
    return {
        "AUROC": auroc(human, ai),
        "AUROC CI low": low,
        "AUROC CI high": high,
        **{f"TPR at {cap:.0%} FPR": tpr_at_fpr(human, ai, cap) for cap in FPR_CAPS},
        "n human": len(human),
        "n AI": len(ai),
    }


def report(passages: Sequence[Passage], seed: int) -> list[dict]:
    detectors = [WHOLE_PASSAGE, SENTENCES, *DETECTORS]
    scopes: list[tuple[str, str, Sequence[Passage], Sequence[str]]] = [
        ("overall", "all models", passages, list(MODELS)),
        *[("model", model, passages, [model]) for model in MODELS],
        *[("genre", genre, [p for p in passages if p.genre == genre], list(MODELS))
          for genre in sorted({p.genre for p in passages})],
    ]
    rows = []
    for scope, name, subset, sources in scopes:
        for detector in detectors:
            metrics = evaluate(subset, sources, detector, seed)
            if metrics is not None:
                rows.append({"scope": scope, "subset": name, "detector": detector, **metrics})
    return rows


def write_outputs(passages: Sequence[Passage], rows: list[dict], output: Path) -> None:
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["passage", "genre", "source", "detector", "score"])
        for passage in passages:
            for source, scores in passage.detector_scores.items():
                for detector, score in scores.items():
                    writer.writerow([passage.key, passage.genre, source, detector, score])


def print_table(rows: list[dict]) -> None:
    header = f"{'subset':18s} {'detector':20s} {'AUROC':>7s} {'95% CI':>15s} {'TPR@1%':>7s} {'TPR@5%':>7s} {'n':>9s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['subset']:18s} {row['detector']:20s} {row['AUROC']:7.3f} "
              f"[{row['AUROC CI low']:.3f}, {row['AUROC CI high']:.3f}] "
              f"{row['TPR at 1% FPR']:7.1%} {row['TPR at 5% FPR']:7.1%} "
              f"{row['n human']:4d}/{row['n AI']:<4d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Path to a DetectionAI clone.")
    parser.add_argument("--sample", type=int, default=200, help="Human passages to sample.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    args.output.mkdir(parents=True, exist_ok=True)
    passages = stratified_sample(load_passages(args.data), args.sample, args.seed)
    print(f"Sampled {len(passages)} human passages with {len(MODELS)} AI versions each.")

    asyncio.run(score_with_jev(passages, JevCache(args.output / "jev_cache.jsonl")))
    asyncio.run(score_whole_passages_with_jev(
        passages, JevCache(args.output / "jev_passage_cache.jsonl")))
    rows = report(passages, args.seed)
    write_outputs(passages, rows, args.output)
    print_table(rows)
    print(f"\nWrote {args.output / 'metrics.csv'} and {args.output / 'scores.csv'}")


if __name__ == "__main__":
    main()
