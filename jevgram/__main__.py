"""Run Jevgram locally: `uv run python -m jevgram`."""

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    host = os.environ.get("JEVGRAM_HOST", "127.0.0.1")
    port = int(os.environ.get("JEVGRAM_PORT", "8000"))
    print(f"Jevgram is running at http://{host}:{port}")
    uvicorn.run("jevgram.app:app", host=host, port=port)
