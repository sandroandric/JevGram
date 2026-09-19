import pytest
from fastapi.testclient import TestClient

from jevgram.app import app, get_client
from tests.fakes import FakeTypeSafeClient


@pytest.fixture
def client():
    assisted = {"ai_generated": 0.1, "ai_assisted": 0.8, "human_written": 0.1}
    human = {"ai_generated": 0.2, "ai_assisted": 0.1, "human_written": 0.7}
    fake = FakeTypeSafeClient(
        answer_for=lambda view, text: assisted if text.startswith("Deep") else human
    )
    app.dependency_overrides[get_client] = lambda: fake
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(client, data, name="paper.pdf"):
    return client.post("/api/analyze", files={"file": (name, data, "application/pdf")})


def test_index_serves_the_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Jevgram" in response.text


def test_analyze_returns_labeled_sentences(client, paper_pdf):
    response = upload(client, paper_pdf)

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "paper.pdf"
    assert body["model"] == "jev-test"
    assert len(body["pages"]) == 2
    assert len(body["sentences"]) == 8
    deep = next(s for s in body["sentences"] if s["text"].startswith("Deep"))
    assert deep["label"] == "ai_assisted"
    assert deep["confidence"] == pytest.approx(0.8)
    assert set(deep["views"]) == {"alone", "neighbors", "paragraph"}
    assert deep["views"]["paragraph"]["ai_assisted"] == pytest.approx(0.8)
    assert deep["paragraph"] == 1
    assert deep["rects"] and deep["rects"][0]["page"] == 0
    assert body["summary"]["ai_assisted"]["count"] == 1
    overview = body["overview"]
    assert set(overview) == {"sentences", "paragraphs", "paper", "overall"}
    assert overview["sentences"]["count"] == 8
    assert overview["paper"]["parts"] == 1
    assert overview["overall"]["label"] == "human_written"
    assert body["summary"]["human_written"]["count"] == 7


def test_rejects_non_pdf(client):
    response = upload(client, b"hello, not a pdf", name="notes.txt")
    assert response.status_code == 400


def test_rejects_corrupt_pdf(client):
    response = upload(client, b"%PDF-1.7 truncated garbage")
    assert response.status_code == 400


def test_scanned_pdf_explains_missing_text(client, scanned_pdf):
    response = upload(client, scanned_pdf)
    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_missing_api_key_is_reported(paper_pdf):
    with TestClient(app) as test_client:
        app.state.typesafe = None
        response = upload(test_client, paper_pdf)
    assert response.status_code == 503
    assert "TYPESAFE_API_KEY" in response.json()["detail"]
