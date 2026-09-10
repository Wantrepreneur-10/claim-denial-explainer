# Claim Denial Explainer

Paste a claim denial description → the system retrieves the specific payer
rule that likely caused it, explains why in plain language, and drafts
appeal guidance — grounded only in the retrieved rule, never invented.

**🔴 Live demo:** https://wantrepreneur-10.github.io/claim-denial-explainer/

> ⚠️ **This is a public demo built entirely on synthetic data.** The 8 payer
> rules in `data/payer_rules.json` are made up for demonstration purposes.
> No real claims, no real PHI, no connection to any real insurer. Not
> medical or legal advice. It demonstrates a retrieval + grounded-explanation
> pattern for a real, common healthcare-ops problem.

## Why this project

This is a generic, public-safe rebuild of the reasoning pattern behind a
production system I built and operated: a payer-rule agent that reduced
claim denial rate from 34% to 6% by grounding every decision in the actual
rule text and gating releases behind an evaluation suite, not manual review.

This repo shows that reasoning pattern end-to-end, from anyone's browser,
without requiring access to a real production system or real patient data.

## How it works

1. **Retrieval** (`app/rules_engine.py::RuleRetriever`) — dependency-free
   TF-IDF + cosine similarity over the rule set. No vector DB required for
   a demo at this scale; swap for Pinecone/pgvector + real embeddings for
   production volume.
2. **Grounded explanation** — the LLM (when configured) is shown *only* the
   one retrieved rule and asked to explain why it applies — it is never
   asked to invent a rule, which is how hallucinated policy explanations
   get avoided.
3. **Fallback mode** — with no API key configured, the app still works:
   it returns the matched rule and a templated explanation instead of an
   LLM-generated one. This is why the public demo can stay live without
   burning API credits per visitor, and it's also what CI runs against.

## Retrieval accuracy (reproducible)

```bash
python evals/run_evals.py
```

```
Accuracy: 8/8 = 100.0% (threshold: 85%)
EVAL GATE PASSED.
```

8 test cases against the 8 synthetic rules. Small on purpose — it's the
demo rule set. The eval script and rule set are both in this repo, so
the number is checkable, not just claimed. Wired into
`.github/workflows/ci.yml` so every PR is gated on retrieval accuracy,
the same gating instinct as a RAGAS-checked release pipeline.

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open `frontend/index.html` in a browser, point it at
`http://localhost:8000`. Works immediately with no API key (fallback mode).
Optionally `cp .env.example .env` and add `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` for LLM-generated explanations.

## Deploy (free tier)

**Backend — Render.com:**
1. Push to GitHub → Render → New Web Service → connect repo.
2. Environment: Docker (auto-detects `Dockerfile`).
3. (Optional) add `OPENAI_API_KEY` env var for LLM mode.
4. Deploy, note the public URL.

**Frontend — GitHub Pages:**
1. Settings → Pages → Deploy from branch → `main` → `/frontend`.
2. Paste your Render backend URL into the input box on the page.

## Endpoints

- `POST /explain` — `{"denial_text": "..."}` → matched rule + explanation + appeal guidance
- `GET /rules` — returns the full synthetic rule set (transparency: see exactly what's being matched against)
- `GET /health` — health check

## Extending it

- Swap the TF-IDF retriever for real embeddings (OpenAI `text-embedding-3-small`
  or a local sentence-transformer) for better semantic matching beyond keyword overlap.
- Replace the synthetic rule set with a real, de-identified rule corpus for
  an internal (non-public) version.
- Add a confidence-based human-review flag: below a similarity threshold,
  route to a person instead of auto-explaining — mirrors the audit-trail
  checkpoint pattern from production agent ops.

## Stack

`Python` · `FastAPI` · `Pydantic` · dependency-free TF-IDF retrieval ·
optional OpenAI/Anthropic for grounded explanation generation
