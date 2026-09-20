"""FastAPI: one endpoint.  uvicorn clausecheck.api:app --reload"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import Settings
from .pipeline import review_contract
from .tracing import flush

app = FastAPI(title="Clause Check", version="0.1.0")


@app.get("/health")
def health() -> dict:
    s = Settings()
    return {"ok": True, "mode": s.mode, "model": s.model}


@app.post("/review")
async def review(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "contract.txt").suffix.lower() or ".txt"
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(400, "upload a .pdf or .txt contract")
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        report = review_contract(tmp_path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)
        flush()
    report.contract_name = Path(file.filename or "contract").stem
    return report.model_dump()
