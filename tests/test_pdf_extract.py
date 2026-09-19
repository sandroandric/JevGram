import pymupdf

from jevgram.pdf_extract import extract_sentences


def test_sentences_follow_reading_order_and_skip_non_prose(paper_pdf):
    document = extract_sentences(paper_pdf)

    assert [sentence.text for sentence in document.sentences] == [
        "We study how papers are laid out on the page.",
        "Our method works on real documents, e.g. conference papers, and it is extremely fast.",
        "Results are shown in Fig. 2 and follow Smith et al. closely.",
        "Deep networks are widely used today.",
        "Many other visual recognition tasks in the literature have also "
        "greatly benefited from very deep models.",
        "The batch size is 8 images per device during all of the training runs.",
        "Figure 1. Training error on a small dataset with two models.",
        "We report more experiments in this appendix section here.",
    ]
    assert [sentence.index for sentence in document.sentences] == list(range(8))
    # The paragraph interrupted by the caption stays one paragraph across columns.
    assert [sentence.paragraph for sentence in document.sentences] == [0, 0, 0, 1, 1, 1, 2, 3]
    assert len(document.pages) == 2


def test_sentence_split_across_columns_keeps_both_highlights(paper_pdf):
    document = extract_sentences(paper_pdf)
    interrupted = document.sentences[4]

    assert len(interrupted.rects) == 3
    left = [rect for rect in interrupted.rects if rect["x0"] < 300]
    right = [rect for rect in interrupted.rects if rect["x0"] >= 300]
    assert len(left) == 2 and len(right) == 1


def test_rects_lie_on_their_page(paper_pdf):
    document = extract_sentences(paper_pdf)
    for sentence in document.sentences:
        for rect in sentence.rects:
            page = document.pages[rect["page"]]
            assert 0 <= rect["x0"] < rect["x1"] <= page["width"]
            assert 0 <= rect["y0"] < rect["y1"] <= page["height"]


def test_rotated_page_rects_use_displayed_orientation():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 100), "This rotated page still has a full sentence of text.", fontsize=10)
    page.set_rotation(90)
    document = extract_sentences(doc.tobytes())

    assert document.pages == [{"width": 792, "height": 612}]
    [sentence] = document.sentences
    for rect in sentence.rects:
        assert 0 <= rect["x0"] < rect["x1"] <= 792
        assert 0 <= rect["y0"] < rect["y1"] <= 612
        # Horizontal text on the unrotated page runs vertically once rotated.
        assert rect["y1"] - rect["y0"] > rect["x1"] - rect["x0"]


def test_pdf_without_text_layer_has_no_sentences(scanned_pdf):
    assert extract_sentences(scanned_pdf).sentences == []


def test_review_copy_line_numbers_are_removed():
    """ACL-style review copies number every line in the margins and the gutter."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    lines = [
        "A language model standing in for a popula-",
        "tion must put its probability where answers fall.",
        "We ask what post-training does to that ability",
        "across seven public checkpoints of one family.",
    ]
    for row, line in enumerate(lines * 2):
        y = 100 + 14 * row
        page.insert_text((60, y), line, fontsize=10)
        page.insert_text((12, y), f"{row + 1:03d}", fontsize=8)
    document = extract_sentences(doc.tobytes())

    texts = [sentence.text for sentence in document.sentences]
    assert texts[:2] == [
        "A language model standing in for a population must put its probability where answers fall.",
        "We ask what post-training does to that ability across seven public checkpoints of one family.",
    ]
    assert not any(char.isdigit() for text in texts for char in text)


def test_gutter_line_numbers_found_across_pages_are_removed():
    """ACL prints one or two gutter numbers per page; counting over the document finds them."""
    doc = pymupdf.open()
    for number in range(6):
        page = doc.new_page(width=595, height=842)
        page.insert_text((60, 100), "Economic games have been run with models against human", fontsize=10)
        page.insert_text((290, 100), f"{400 + number}", fontsize=8)
        page.insert_text((60, 114), "samples in many earlier studies of this kind.", fontsize=10)
    document = extract_sentences(doc.tobytes())

    assert len(document.sentences) == 6
    assert all(not any(char.isdigit() for char in sentence.text)
               for sentence in document.sentences)


def test_paragraphs_inside_one_block_are_separated():
    """An indented line or a short previous line starts a new paragraph in the same block."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 100), "\n".join([
        "Language models as simulators. Silicon sampling showed that",
        "conditioned models recover opinion distributions of survey",
        "panels closely.",
        "    Fidelity measurement followed in later work across many",
        "benchmarks of human answers and several model families.",
        "The test also covers a line that fills the column width.",
    ]), fontsize=10)
    document = extract_sentences(doc.tobytes())

    assert [(sentence.paragraph, sentence.text[:24]) for sentence in document.sentences] == [
        (0, "Language models as simul"),
        (0, "Silicon sampling showed "),
        (1, "Fidelity measurement fol"),
        (1, "The test also covers a l"),
    ]


def test_plain_text_is_split_with_the_same_rules():
    from jevgram.pdf_extract import sentences_from_text

    text = (
        "Sen. Shawn Vedaa said Tuesday, Feb. 14, the bill was amended. Results follow Smith et al. closely.\n"
        "\n"
        "A second paragraph, e.g. this one, has two sentences. It ends right here."
    )
    sentences = sentences_from_text(text)

    assert [(s.paragraph, s.text) for s in sentences] == [
        (0, "Sen. Shawn Vedaa said Tuesday, Feb. 14, the bill was amended."),
        (0, "Results follow Smith et al. closely."),
        (1, "A second paragraph, e.g. this one, has two sentences."),
        (1, "It ends right here."),
    ]
    assert all(s.rects == [] for s in sentences)


def test_sentence_continues_on_next_page_even_when_capitalised():
    """A page break mid-sentence, a footer in between, and another open paragraph later."""
    doc = pymupdf.open()
    first = doc.new_page(width=612, height=792)
    first.insert_text((60, 600), "\n".join([
        "Our method keeps the strengths of earlier approaches. As with related",
        "spectral methods, our method uses the",
    ]), fontsize=10)
    first.insert_text((60, 740), "Proceedings of an imaginary workshop on graphs held in 2025.", fontsize=8)
    second = doc.new_page(width=612, height=792)
    second.insert_text((60, 80), "\n".join([
        "Laplacian eigenvectors to guide exploration towards new regions.",
        "Importantly, the inner product of two vectors is written as",
    ]), fontsize=10)
    second.insert_text((60, 200), "\n".join([
        "the sum of their elementwise products over all states here.",
    ]), fontsize=10)
    texts = [sentence.text for sentence in extract_sentences(doc.tobytes()).sentences]

    assert "As with related spectral methods, our method uses the Laplacian eigenvectors " \
           "to guide exploration towards new regions." in texts
    assert "Importantly, the inner product of two vectors is written as the sum of their " \
           "elementwise products over all states here." in texts
