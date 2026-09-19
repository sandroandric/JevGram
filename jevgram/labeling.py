"""Ask Jev how each sentence, paragraph and whole paper was written.

Every sentence gets the same bare Choice question (fully AI written, fully human
written or AI assisted, with no descriptions of the options) in three views:

- `alone`: the sentence by itself
- `neighbors`: the sentence with the sentence before and the sentence after
- `paragraph`: the sentence with the whole paragraph it belongs to

The three probability distributions are averaged with equal weight; the label
with the highest average wins and its average probability is the confidence.

The shared state holds no text from the paper; each question carries its own
sentence and context, so many sentences can go in one request.

The same bare question is also asked once about each paragraph and once about
the whole paper. The paper overview reports four distributions: the mean over
sentences, the mean over paragraphs (each paragraph counts equally), the whole
paper, and the mean of those three. A paper longer than PAPER_PART_CHARS is
split at paragraph boundaries into the fewest parts that fit Jev's 32k-token
limit, and the parts' answers are averaged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import math
from dataclasses import dataclass

from typesafe_sdk import AsyncTypeSafeClient, Choice

from jevgram.pdf_extract import Sentence

LABELS = {
    "ai_generated": "AI generated",
    "human_written": "Human written",
    "ai_assisted": "AI assisted",
}

# The option text Jev sees for each label; no descriptions are given.
OPTIONS = {
    "ai_generated": "fully AI written",
    "human_written": "fully human written",
    "ai_assisted": "AI assisted",
}

QUESTION = "How was `sentence` written?"
WHOLE_TEXT_QUESTION = "How was `text` written?"
VIEWS = ("alone", "neighbors", "paragraph")
DEFAULT_DOCUMENT_TYPE = "research paper"

# Requests are filled up to BATCH_SIZE sentences or REQUEST_CHAR_BUDGET characters
# of question text, whichever comes first. At roughly 4 characters per token the
# budget keeps a request well below Jev's 64k-token limit even when paragraphs are long.
BATCH_SIZE = 40
REQUEST_CHAR_BUDGET = 120_000
TEXTS_PER_REQUEST = 50
# One question may hold 32k tokens with the state; 90,000 characters stays below
# that even for number-heavy text (about 3.5 characters per token).
PAPER_PART_CHARS = 90_000
MAX_CONCURRENT_REQUESTS = 6


@dataclass
class SentenceLabel:
    index: int
    label: str
    confidence: float
    probabilities: dict[str, float]
    views: dict[str, dict[str, float]]


@dataclass
class LabelingResult:
    model: str
    labels: list[SentenceLabel]
    input_tokens: int


def _choice(instructions: dict) -> Choice:
    return Choice(instructions=instructions,
                  criteria={option: None for option in OPTIONS.values()})


def build_questions(
    sentences: Sequence[Sentence], position: int, paragraphs: dict[int, str]
) -> dict[str, Choice]:
    """The three views of `sentences[position]`, keyed by question id."""
    sentence = sentences[position]
    neighbors: dict[str, str] = {"question": QUESTION}
    if position > 0:
        neighbors["previous_sentence"] = sentences[position - 1].text
    neighbors["sentence"] = sentence.text
    if position + 1 < len(sentences):
        neighbors["next_sentence"] = sentences[position + 1].text
    return {
        f"s{sentence.index}_alone": _choice({"question": QUESTION, "sentence": sentence.text}),
        f"s{sentence.index}_neighbors": _choice(neighbors),
        f"s{sentence.index}_paragraph": _choice({
            "question": QUESTION,
            "sentence": sentence.text,
            "paragraph": paragraphs[sentence.paragraph],
        }),
    }


def paragraph_texts(sentences: Sequence[Sentence]) -> dict[int, str]:
    paragraphs: dict[int, list[str]] = {}
    for sentence in sentences:
        paragraphs.setdefault(sentence.paragraph, []).append(sentence.text)
    return {paragraph: " ".join(texts) for paragraph, texts in paragraphs.items()}


def _question_chars(questions: dict[str, Choice]) -> int:
    return sum(len(str(question.instructions)) for question in questions.values())


def _batches(
    sentences: Sequence[Sentence], paragraphs: dict[int, str]
) -> list[list[tuple[int, dict[str, Choice]]]]:
    batches: list[list[tuple[int, dict[str, Choice]]]] = []
    current: list[tuple[int, dict[str, Choice]]] = []
    chars = 0
    for position in range(len(sentences)):
        questions = build_questions(sentences, position, paragraphs)
        size = _question_chars(questions)
        if current and (len(current) == BATCH_SIZE or chars + size > REQUEST_CHAR_BUDGET):
            batches.append(current)
            current, chars = [], 0
        current.append((position, questions))
        chars += size
    if current:
        batches.append(current)
    return batches


def label_from(index: int, views: dict[str, dict[str, float]]) -> SentenceLabel:
    """Average the views' distributions with equal weight and pick the top label."""
    probabilities = mean_distribution(list(views.values()))
    label = max(probabilities, key=probabilities.__getitem__)
    return SentenceLabel(index=index, label=label, confidence=probabilities[label],
                         probabilities=probabilities, views=views)


def _view(answer_probabilities: dict[str, float]) -> dict[str, float]:
    return {label: answer_probabilities[option] for label, option in OPTIONS.items()}


async def label_sentences(
    client: AsyncTypeSafeClient,
    sentences: Sequence[Sentence],
    document_type: str = DEFAULT_DOCUMENT_TYPE,
) -> LabelingResult:
    """Label every sentence, returning labels in the same order as `sentences`."""
    state = {"document_type": document_type}
    paragraphs = paragraph_texts(sentences)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def run(batch: list[tuple[int, dict[str, Choice]]]):
        questions: dict[str, Choice] = {}
        for _, sentence_questions in batch:
            questions.update(sentence_questions)
        async with semaphore:
            response = await client.system_one(state=state, questions=questions)
        return [position for position, _ in batch], response

    responses = await asyncio.gather(*(
        run(batch) for batch in _batches(sentences, paragraphs)
    ))

    labels: list[SentenceLabel] = []
    models: set[str] = set()
    input_tokens = 0
    for positions, response in responses:
        models.add(response.model)
        input_tokens += response.usage.input_tokens
        for position in positions:
            index = sentences[position].index
            labels.append(label_from(index, {
                view: _view(response.answers[f"s{index}_{view}"].probabilities)
                for view in VIEWS
            }))
    return LabelingResult(model=", ".join(sorted(models)), labels=labels,
                          input_tokens=input_tokens)


def mean_distribution(distributions: Sequence[dict[str, float]]) -> dict[str, float]:
    return {label: sum(d[label] for d in distributions) / len(distributions) for label in LABELS}


def whole_text_question(text: str) -> Choice:
    return _choice({"question": WHOLE_TEXT_QUESTION, "text": text})


def _text_batches(texts: Sequence[str]) -> list[list[int]]:
    """Group text positions into requests by count and total characters."""
    batches: list[list[int]] = []
    current: list[int] = []
    chars = 0
    for position, text in enumerate(texts):
        if current and (len(current) == TEXTS_PER_REQUEST
                        or chars + len(text) > REQUEST_CHAR_BUDGET):
            batches.append(current)
            current, chars = [], 0
        current.append(position)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


@dataclass
class TextsResult:
    distributions: list[dict[str, float]]
    models: set[str]
    input_tokens: int


async def label_texts(
    client: AsyncTypeSafeClient,
    texts: Sequence[str],
    document_type: str = DEFAULT_DOCUMENT_TYPE,
) -> TextsResult:
    """Ask the bare question once about each whole text; distributions follow `texts`."""
    state = {"document_type": document_type}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def run(batch: list[int]):
        async with semaphore:
            response = await client.system_one(
                state=state,
                questions={f"t{position}": whole_text_question(texts[position])
                           for position in batch},
            )
        return batch, response

    responses = await asyncio.gather(*(run(batch) for batch in _text_batches(texts)))
    distributions: list[dict[str, float] | None] = [None] * len(texts)
    models: set[str] = set()
    input_tokens = 0
    for batch, response in responses:
        models.add(response.model)
        input_tokens += response.usage.input_tokens
        for position in batch:
            distributions[position] = _view(response.answers[f"t{position}"].probabilities)
    return TextsResult(distributions=distributions, models=models, input_tokens=input_tokens)


def paper_parts(paragraphs: Sequence[str], limit: int = PAPER_PART_CHARS) -> list[str]:
    """Join paragraphs into the fewest parts of at most about `limit` characters.

    The parts are balanced in size; a single paragraph longer than the target
    stays whole in its own part.
    """
    total = sum(len(paragraph) + 1 for paragraph in paragraphs)
    target = total / max(1, math.ceil(total / limit))
    parts: list[list[str]] = [[]]
    size = 0
    for paragraph in paragraphs:
        if parts[-1] and (size >= target or size + len(paragraph) > limit):
            parts.append([])
            size = 0
        parts[-1].append(paragraph)
        size += len(paragraph) + 1
    return ["\n\n".join(part) for part in parts if part]


@dataclass
class DocumentResult:
    sentences: LabelingResult
    overview: dict
    input_tokens: int
    model: str


async def label_document(
    client: AsyncTypeSafeClient,
    sentences: Sequence[Sentence],
    document_type: str = DEFAULT_DOCUMENT_TYPE,
) -> DocumentResult:
    """Label every sentence and score the paper by sentences, paragraphs and as a whole."""
    paragraphs = list(paragraph_texts(sentences).values())
    parts = paper_parts(paragraphs)
    sentence_result, paragraph_result, paper_result = await asyncio.gather(
        label_sentences(client, sentences, document_type),
        label_texts(client, paragraphs, document_type),
        label_texts(client, parts, document_type),
    )
    by_sentences = mean_distribution([label.probabilities for label in sentence_result.labels])
    by_paragraphs = mean_distribution(paragraph_result.distributions)
    by_paper = mean_distribution(paper_result.distributions)
    overview = {
        "sentences": {"probabilities": by_sentences, "count": len(sentences)},
        "paragraphs": {"probabilities": by_paragraphs, "count": len(paragraphs)},
        "paper": {"probabilities": by_paper, "parts": len(parts)},
        "overall": {"probabilities": mean_distribution([by_sentences, by_paragraphs, by_paper])},
    }
    for entry in overview.values():
        probabilities = entry["probabilities"]
        entry["label"] = max(probabilities, key=probabilities.__getitem__)
    models = {sentence_result.model, *paragraph_result.models, *paper_result.models}
    return DocumentResult(
        sentences=sentence_result,
        overview=overview,
        input_tokens=(sentence_result.input_tokens + paragraph_result.input_tokens
                      + paper_result.input_tokens),
        model=", ".join(sorted(models)),
    )


def summarize(labels: Sequence[SentenceLabel]) -> dict:
    """Count sentences per label and average the confidence for each."""
    summary = {}
    for key, name in LABELS.items():
        matching = [label for label in labels if label.label == key]
        summary[key] = {
            "name": name,
            "count": len(matching),
            "share": len(matching) / len(labels) if labels else 0.0,
            "mean_confidence": (
                sum(label.confidence for label in matching) / len(matching)
                if matching else None
            ),
        }
    return summary
