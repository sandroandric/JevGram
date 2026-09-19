"""Split a PDF's running text into sentences, each with its position on the page.

PyMuPDF gives us words with bounding boxes grouped into blocks. We keep the
blocks that look like prose, join paragraphs that continue across columns,
figures or pages, then split the word stream into sentences. Each sentence keeps
one highlight rectangle per line it occupies, in PDF points relative to the top
left of the displayed page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

# Word flags without TEXT_PRESERVE_LIGATURES so "ﬁ" comes out as "fi".
_WORD_FLAGS = pymupdf.TEXTFLAGS_WORDS & ~pymupdf.TEXT_PRESERVE_LIGATURES

_TERMINAL = re.compile(r"[.?!][)\]\"'”’]*$")
_CLOSERS = ")]\"'”’"
_OPENERS = "([\"'“‘"
_ABBREVIATIONS = {
    "al.", "e.g.", "i.e.", "cf.", "vs.", "etc.", "resp.", "approx.", "viz.",
    "fig.", "figs.", "eq.", "eqs.", "sec.", "secs.", "tab.", "ref.", "refs.",
    "no.", "nos.", "vol.", "pp.", "ch.", "thm.", "def.", "prop.", "lem.",
    "alg.", "dr.", "mr.", "ms.", "prof.", "st.", "jr.", "sr.", "inc.", "ltd.",
    "co.", "corp.", "dept.", "univ.", "ed.", "eds.", "et.",
    # Titles and months, common in news text.
    "sen.", "rep.", "gov.", "gen.", "col.", "lt.", "sgt.", "capt.", "rev.", "hon.",
    "mt.", "ft.", "ave.", "blvd.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.",
    "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
}
# "Figure 1." / "Table 2." inside captions and cross-references.
_NUMBERED_REFERENCES = {"figure", "fig.", "table", "tab.", "section", "sec.",
                        "equation", "eq.", "algorithm", "alg.", "appendix", "step"}
_HEADING_NUMBER = r"(?:[0-9IVX]+[.)]?\s+|[A-Z][.)]\s+)?"
_REFERENCES_HEADING = re.compile(
    rf"^{_HEADING_NUMBER}(references|bibliography|works cited|literature cited)$",
    re.IGNORECASE,
)
_SECTION_NUMBER = re.compile(r"^([0-9]+|[A-Z])(\.[0-9]+)*[.)]?$")
_MINOR_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of",
                "on", "or", "the", "to", "via", "vs", "with"}
_ABSTRACT_HEADING = re.compile(r"^abstract\b[.:—-]?", re.IGNORECASE)

MIN_PROSE_WORDS = 6
MIN_ALPHA_RATIO = 0.6
LONG_BLOCK_WORDS = 20
MIN_SENTENCE_WORDS = 3
MAX_HEADING_WORDS = 8
# How many blocks ahead to look for the rest of a paragraph interrupted by a
# figure, caption or footnote.
CONTINUATION_LOOKAHEAD = 8
# Review copies (ACL, ICML, NeurIPS, CVPR) print line numbers down the margins.
MIN_LINE_NUMBERS_PER_COLUMN = 5
LINE_NUMBER_X_TOLERANCE = 3
# Paragraph starts inside a block: first-line indent, or a short last line.
PARAGRAPH_INDENT = 5
PARAGRAPH_SHORT_LINE = 15


@dataclass
class Word:
    text: str
    page: int
    block: int
    line: int
    rect: tuple[float, float, float, float]


@dataclass
class Block:
    page: int
    number: int
    words: list[Word]
    # Set when the block continues the previous block's sentence across a break.
    continues: bool = False

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


@dataclass
class Sentence:
    index: int
    text: str
    page: int
    # Sentences with the same number belong to the same paragraph.
    paragraph: int = 0
    rects: list[dict] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    title: str | None
    pages: list[dict]
    sentences: list[Sentence]


def extract_sentences(pdf_bytes: bytes) -> ExtractedDocument:
    """Open a PDF from bytes and return its prose sentences with highlight rects."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [{"width": page.rect.width, "height": page.rect.height} for page in doc]
        blocks = _read_blocks(doc)
        title = (doc.metadata or {}).get("title") or None
    prose = _reorder_continuations(_body_prose_blocks(blocks))
    words = _join_paragraphs(prose)
    sentences = [
        _to_sentence(index, paragraph, group)
        for index, (paragraph, group) in enumerate(
            (paragraph, group) for paragraph, group in _split_sentences(words)
            if _is_sentence(group)
        )
    ]
    return ExtractedDocument(title=title.strip() if title else None, pages=pages,
                             sentences=sentences)


def _line_number_column(word: tuple) -> int:
    return round(word[0] / LINE_NUMBER_X_TOLERANCE)


def _line_number_columns(pages_words: list[list[tuple]]) -> set[int]:
    """Horizontal positions where review copies (ACL, ICML, NeurIPS, CVPR) print line numbers.

    Line numbers are stacked at a few fixed positions (margins, sometimes the
    gutter) throughout the document, and most of them are the only word on
    their line. Counting over the whole document also catches positions that
    hold only one or two numbers per page.
    """
    lone_numbers: dict[int, int] = {}
    for words in pages_words:
        words_per_line: dict[tuple[int, int], int] = {}
        for word in words:
            key = (word[5], word[6])
            words_per_line[key] = words_per_line.get(key, 0) + 1
        for word in words:
            if word[4].isdigit() and words_per_line[(word[5], word[6])] == 1:
                column = _line_number_column(word)
                lone_numbers[column] = lone_numbers.get(column, 0) + 1
    return {column for column, count in lone_numbers.items()
            if count >= MIN_LINE_NUMBERS_PER_COLUMN}


def _read_blocks(doc: pymupdf.Document) -> list[Block]:
    pages_words = [page.get_text("words", flags=_WORD_FLAGS) for page in doc]
    line_number_columns = _line_number_columns(pages_words)
    blocks: list[Block] = []
    for page_number, (page, words) in enumerate(zip(doc, pages_words)):
        # Word boxes are in unrotated page space; map them onto the displayed page.
        rotation = page.rotation_matrix
        by_block: dict[int, list[Word]] = {}
        for x0, y0, x1, y1, text, block_no, line_no, _ in words:
            if text.isdigit() and _line_number_column((x0,)) in line_number_columns:
                continue
            rect = pymupdf.Rect(x0, y0, x1, y1) * rotation
            by_block.setdefault(block_no, []).append(
                Word(text, page_number, block_no, line_no,
                     (rect.x0, rect.y0, rect.x1, rect.y1))
            )
        blocks.extend(Block(page_number, number, words) for number, words in by_block.items())
    return blocks


def _is_wordy(block: Block) -> bool:
    """Enough words, mostly letters, and not a heading."""
    words = block.words
    if len(words) < MIN_PROSE_WORDS or _looks_like_heading([word.text for word in words]):
        return False
    text = block.text
    letters = sum(char.isalpha() for char in text)
    visible = sum(not char.isspace() for char in text)
    return visible > 0 and letters / visible >= MIN_ALPHA_RATIO


def _is_prose(block: Block, following: list[Block]) -> bool:
    if not _is_wordy(block):
        return False
    if any(_TERMINAL.search(word.text) for word in block.words):
        return True
    if len(block.words) >= LONG_BLOCK_WORDS:
        return True
    # A short opening of a paragraph broken by a column or page change, such as
    # "Language models are asked to stand in for people: to pilot a survey,".
    next_wordy = next((candidate for candidate in following if _is_wordy(candidate)), None)
    return next_wordy is not None and _continues_text(next_wordy)


def _body_prose_blocks(blocks: list[Block]) -> list[Block]:
    """Keep prose blocks, dropping front matter before the abstract and the reference list."""
    start = 0
    for position, block in enumerate(blocks):
        if block.page > 1:
            break
        if _ABSTRACT_HEADING.match(block.text):
            # Keep the abstract block itself when the heading runs into its text.
            start = position
            break
    kept: list[Block] = []
    in_references = False
    body = blocks[start:]
    for position, block in enumerate(body):
        text = block.text.strip()
        if _REFERENCES_HEADING.match(text):
            in_references = True
            continue
        if in_references and _looks_like_heading([word.text for word in block.words]):
            in_references = False
        following = body[position + 1:position + 1 + CONTINUATION_LOOKAHEAD]
        if not in_references and _is_prose(block, following):
            kept.append(block)
    return kept


def _looks_like_heading(words: list[str]) -> bool:
    """A short title-case line such as "A. Object Detection Baselines"."""
    if not words or len(words) > MAX_HEADING_WORDS or words[0].startswith("["):
        return False
    if words[-1][-1] in ".,;:-":
        return False
    if _SECTION_NUMBER.match(words[0]):
        words = words[1:]
    if not words or not all(any(char.isalpha() for char in word) for word in words):
        return False
    if any(char.isdigit() for word in words for char in word):
        return False
    content = [word for word in words if word.lower() not in _MINOR_WORDS]
    return bool(content) and words[0][0].isupper() and sum(
        word[0].isupper() for word in content
    ) * 2 >= len(content)


def _continues_text(block: Block) -> bool:
    """Whether a block reads as the middle of a sentence ("level features", "8 images of")."""
    words = block.words
    if _starts_lowercase(words[0].text):
        return True
    return (
        len(words) > 1 and words[0].text.isdigit() and _starts_lowercase(words[1].text)
    )


def _reorder_continuations(blocks: list[Block]) -> list[Block]:
    """Move the rest of an interrupted paragraph directly after its first part.

    A paragraph cut off by a figure, caption, footnote or page break ends
    without terminal punctuation. Its remainder is the next nearby block that
    reads as the middle of a sentence, or the first block of the next page
    (which may start with a capitalised word such as a name). The search stops
    at another unfinished paragraph, which has the stronger claim on what follows.
    """
    remaining = list(blocks)
    ordered: list[Block] = []
    while remaining:
        block = remaining.pop(0)
        ordered.append(block)
        if _TERMINAL.search(block.words[-1].text):
            continue
        for position, candidate in enumerate(remaining[:CONTINUATION_LOOKAHEAD]):
            previous_page = (remaining[position - 1] if position else block).page
            starts_next_page = candidate.page > block.page and candidate.page != previous_page
            if _continues_text(candidate) or starts_next_page:
                candidate.continues = True
                remaining.insert(0, remaining.pop(position))
                break
            if not _TERMINAL.search(candidate.words[-1].text):
                break
    return ordered


def _paragraph_starts(block: Block) -> set[int]:
    """Positions of words that begin a new paragraph inside a block.

    PyMuPDF often puts several paragraphs of a column into one block. A line
    starts a paragraph when the previous line ends a sentence and either this
    line is indented or the previous line stops short of the right edge.
    """
    lines: dict[int, list[int]] = {}
    for position, word in enumerate(block.words):
        lines.setdefault(word.line, []).append(position)
    ordered = [positions for _, positions in sorted(lines.items())]
    left = min(block.words[positions[0]].rect[0] for positions in ordered)
    right = max(block.words[positions[-1]].rect[2] for positions in ordered)
    starts = set()
    for previous, current in zip(ordered, ordered[1:]):
        last = block.words[previous[-1]]
        first = block.words[current[0]]
        if not _TERMINAL.search(last.text):
            continue
        indented = first.rect[0] > left + PARAGRAPH_INDENT
        short = last.rect[2] < right - PARAGRAPH_SHORT_LINE
        if indented or short:
            starts.add(current[0])
    return starts


def _join_paragraphs(blocks: list[Block]) -> list[tuple[Word, bool]]:
    """Flatten blocks into (word, starts_paragraph) pairs.

    A paragraph that ends without terminal punctuation continues into the next
    prose block when that block reads as the middle of a sentence (a column,
    figure or page break). Otherwise every block, and every paragraph start
    found inside a block, begins a new paragraph and so a new sentence.
    """
    stream: list[tuple[Word, bool]] = []
    previous: Block | None = None
    for block in blocks:
        continues = (
            previous is not None
            and not _TERMINAL.search(previous.words[-1].text)
            and (block.continues or _continues_text(block))
        )
        starts = _paragraph_starts(block)
        for position, word in enumerate(block.words):
            stream.append((word, (position == 0 and not continues) or position in starts))
        previous = block
    return stream


def _split_sentences(stream: list[tuple[Word, bool]]) -> list[tuple[int, list[Word]]]:
    """Split the word stream into (paragraph number, words) sentences."""
    sentences: list[tuple[int, list[Word]]] = []
    current: list[Word] = []
    paragraph = -1
    for position, (word, starts_paragraph) in enumerate(stream):
        if starts_paragraph:
            if current:
                sentences.append((paragraph, current))
                current = []
            paragraph += 1
        current.append(word)
        following = stream[position + 1] if position + 1 < len(stream) else None
        if following and not following[1] and _ends_sentence(current, following[0]):
            sentences.append((paragraph, current))
            current = []
    if current:
        sentences.append((paragraph, current))
    return sentences


def _ends_sentence(current: list[Word], following: Word) -> bool:
    token = current[-1].text
    if not _TERMINAL.search(token):
        return False
    bare = token.rstrip(_CLOSERS).lstrip(_OPENERS).lower()
    if bare in _ABBREVIATIONS:
        return False
    # Initials such as "J." in "J. Smith".
    if re.fullmatch(r"[a-z]\.", bare):
        return False
    if len(current) >= 2 and re.fullmatch(r"[0-9]+(\.[0-9]+)*\.", bare):
        if current[-2].text.lower() in _NUMBERED_REFERENCES:
            return False
    first = following.text.lstrip(_OPENERS)
    return bool(first) and (first[0].isupper() or first[0].isdigit())


def _starts_lowercase(text: str) -> bool:
    stripped = text.lstrip(_OPENERS)
    return bool(stripped) and stripped[0].islower()


def _is_sentence(words: list[Word]) -> bool:
    """Drop fragments with too few words and headings left over at a block start."""
    texts = [word.text for word in words]
    if sum(any(char.isalpha() for char in text) for text in texts) < MIN_SENTENCE_WORDS:
        return False
    return not _looks_like_heading(texts)


def _sentence_text(words: list[Word]) -> str:
    parts: list[str] = []
    for position, word in enumerate(words):
        text = word.text
        following = words[position + 1] if position + 1 < len(words) else None
        line_ends = following is not None and (
            following.line != word.line or following.block != word.block
            or following.page != word.page
        )
        joins = (
            line_ends and text.endswith("-") and len(text) > 1
            and _starts_lowercase(following.text)
        )
        # "ex-" + "tremely" -> "extremely"; compounds such as "low/mid/high-" + "level"
        # or "English-" + "to-German" keep their hyphen.
        compound = any(mark in text[:-1] for mark in "-/") or (
            following is not None and "-" in following.text
        )
        # A URL wrapped after "/" continues without a space.
        wrapped_url = line_ends and text.endswith("/") and "://" in text
        if joins and not compound:
            parts.append(text[:-1])
        else:
            parts.append(text)
        if following is not None and not joins and not wrapped_url:
            parts.append(" ")
    return "".join(parts)


def _to_sentence(index: int, paragraph: int, words: list[Word]) -> Sentence:
    lines: dict[tuple[int, int, int], list[float]] = {}
    for word in words:
        key = (word.page, word.block, word.line)
        x0, y0, x1, y1 = word.rect
        if key in lines:
            box = lines[key]
            box[0], box[1] = min(box[0], x0), min(box[1], y0)
            box[2], box[3] = max(box[2], x1), max(box[3], y1)
        else:
            lines[key] = [x0, y0, x1, y1]
    rects = [
        {"page": page, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]}
        for (page, _, _), box in lines.items()
    ]
    return Sentence(index=index, text=_sentence_text(words), page=words[0].page,
                    paragraph=paragraph, rects=rects)


def sentences_from_text(text: str) -> list[Sentence]:
    """Split plain text into sentences with the same rules used for PDFs.

    Each non-empty line is a paragraph. Used for inputs that are already text,
    such as benchmark corpora; the sentences carry no highlight rectangles.
    """
    stream: list[tuple[Word, bool]] = []
    paragraphs = [line.split() for line in text.splitlines() if line.strip()]
    for number, tokens in enumerate(paragraphs):
        for position, token in enumerate(tokens):
            stream.append((Word(token, 0, number, 0, (0.0, 0.0, 0.0, 0.0)), position == 0))
    groups = [(paragraph, words) for paragraph, words in _split_sentences(stream)
              if _is_sentence(words)]
    return [
        Sentence(index=index, text=_sentence_text(words), page=0, paragraph=paragraph)
        for index, (paragraph, words) in enumerate(groups)
    ]
