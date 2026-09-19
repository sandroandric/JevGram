"""Jevgram web app: upload a paper, get every sentence labeled by Jev."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeAPIError, TypeSafeError

from jevgram.labeling import LABELS, label_document, summarize
from jevgram.pdf_extract import extract_sentences

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

load_dotenv(PROJECT_ROOT / ".env")
logger = logging.getLogger("jevgram")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if os.environ.get("TYPESAFE_API_KEY", "").strip():
        async with AsyncTypeSafeClient(model=os.environ.get("JEVGRAM_MODEL", "jev-latest"),
                                       timeout=120.0) as client:
            app.state.typesafe = client
            yield
    else:
        logger.error("TYPESAFE_API_KEY is not set; add it to %s", PROJECT_ROOT / ".env")
        app.state.typesafe = None
        yield


app = FastAPI(title="Jevgram", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_client(request: Request) -> AsyncTypeSafeClient:
    client = request.app.state.typesafe
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="TYPESAFE_API_KEY is not configured on the server. Add it to .env and restart.",
        )
    return client


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
async def analyze(file: UploadFile, client: AsyncTypeSafeClient = Depends(get_client)) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The PDF is larger than 50 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a PDF.")
    try:
        document = extract_sentences(data)
    except pymupdf.FileDataError as error:
        raise HTTPException(status_code=400, detail=f"Could not read the PDF: {error}") from error
    if not document.sentences:
        raise HTTPException(
            status_code=422,
            detail="No selectable text was found. Scanned PDFs need OCR before Jev can read them.",
        )

    try:
        result = await label_document(client, document.sentences)
    except TypeSafeAPIError as error:
        logger.exception("Jev request failed")
        raise HTTPException(status_code=502,
                            detail=f"Jev request failed ({error.status}): {error}") from error
    except TypeSafeError as error:
        logger.exception("Jev request failed")
        raise HTTPException(status_code=502, detail=f"Jev request failed: {error}") from error

    labels = {label.index: label for label in result.sentences.labels}
    return {
        "filename": file.filename,
        "model": result.model,
        "labels": LABELS,
        "pages": document.pages,
        "sentences": [
            {
                "index": sentence.index,
                "text": sentence.text,
                "page": sentence.page,
                "rects": sentence.rects,
                "label": labels[sentence.index].label,
                "confidence": labels[sentence.index].confidence,
                "probabilities": labels[sentence.index].probabilities,
                "paragraph": sentence.paragraph,
                "views": labels[sentence.index].views,
            }
            for sentence in document.sentences
        ],
        "summary": summarize(result.sentences.labels),
        "overview": result.overview,
        "usage": {"input_tokens": result.input_tokens},
    }
