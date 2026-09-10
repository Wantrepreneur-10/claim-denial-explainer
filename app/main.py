"""
Claim Denial Explainer API

IMPORTANT: This is a public demo using entirely SYNTHETIC data (8 made-up
payer rules in data/payer_rules.json). It does not process real claims,
real PHI, or connect to any real payer system. It is not medical or legal
advice. It exists to demonstrate a retrieval + grounded-explanation pattern
for denial-reason matching.

Run: uvicorn app.main:app --reload --port 8002
"""
import time
import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rules_engine import DenialExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("denial-explainer")

app = FastAPI(
    title="Claim Denial Explainer (Demo)",
    description="Synthetic-data demo: retrieves the payer rule behind a denial and explains it in plain language.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_explainer = None


def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = DenialExplainer()
    return _explainer


class ExplainRequest(BaseModel):
    denial_text: str = Field(..., min_length=5, max_length=3000, description="Synthetic denial letter or description")


class ExplainResponse(BaseModel):
    trace_id: str
    matched_rule_id: str | None
    matched_category: str | None
    matched_rule_text: str | None
    similarity: float
    explanation: str
    appeal_guidance: str
    mode: str
    latency_ms: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rules")
def list_rules():
    """Expose the synthetic rule set so visitors can see exactly what's being matched against."""
    explainer = get_explainer()
    return {"rules": explainer.retriever.rules, "disclaimer": "Synthetic demo data only."}


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    trace_id = str(uuid.uuid4())
    start = time.perf_counter()

    explainer = get_explainer()
    try:
        result = explainer.explain(req.denial_text)
    except Exception as e:
        logger.exception("trace=%s explanation failed", trace_id)
        raise HTTPException(status_code=500, detail="explanation failed") from e

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info("trace=%s mode=%s matched=%s latency=%dms", trace_id, result.mode,
                result.matched_rule.rule_id if result.matched_rule else None, latency_ms)

    return ExplainResponse(
        trace_id=trace_id,
        matched_rule_id=result.matched_rule.rule_id if result.matched_rule else None,
        matched_category=result.matched_rule.category if result.matched_rule else None,
        matched_rule_text=result.matched_rule.text if result.matched_rule else None,
        similarity=result.confidence,
        explanation=result.explanation,
        appeal_guidance=result.appeal_guidance,
        mode=result.mode,
        latency_ms=latency_ms,
    )
