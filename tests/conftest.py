import pymupdf
import pytest


def _block(page: pymupdf.Page, x: float, y: float, lines: list[str], size: float = 10) -> None:
    page.insert_text((x, y), "\n".join(lines), fontsize=size, fontname="helv")


@pytest.fixture
def paper_pdf() -> bytes:
    """A two-column paper with the layouts the extractor has to handle."""
    doc = pymupdf.open()
    first = doc.new_page(width=612, height=792)
    _block(first, 180, 70, ["A Study of Synthetic Layout Handling"], size=16)
    _block(first, 220, 100, ["Ada Lovelace  Charles Babbage", "Analytical Engine Society"])
    _block(first, 60, 150, ["Abstract"], size=12)
    _block(first, 60, 170, [
        "We study how papers are laid out on the page. Our method",
        "works on real documents, e.g. conference papers, and it is ex-",
        "tremely fast. Results are shown in Fig. 2 and follow Smith",
        "et al. closely.",
    ])
    _block(first, 60, 260, ["1. Introduction"], size=12)
    _block(first, 60, 280, [
        "Deep networks are widely used today. Many other visual",
        "recognition tasks in the literature have also",
    ])
    _block(first, 60, 360, [
        "Figure 1. Training error on a small dataset with two models.",
    ], size=8)
    _block(first, 320, 280, [
        "greatly benefited from very deep models. The batch size is",
    ])
    _block(first, 320, 330, [
        "8 images per device during all of the training runs.",
    ])
    _block(first, 60, 700, ["1"])

    second = doc.new_page(width=612, height=792)
    _block(second, 60, 80, ["References"], size=12)
    _block(second, 60, 100, [
        "[1] A. Author and B. Writer. A paper title. In Proceedings, 2015.",
        "[2] C. Person. Another paper title. Journal of Things, 2019.",
    ])
    _block(second, 60, 200, ["A. Additional Experiments"], size=12)
    _block(second, 60, 220, [
        "We report more experiments in this appendix section here.",
    ])
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def scanned_pdf() -> bytes:
    """A PDF page with graphics but no text layer."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0.2, 0.2, 0.2))
    data = doc.tobytes()
    doc.close()
    return data
