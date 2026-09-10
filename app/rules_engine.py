"""
Rules engine for the Claim Denial Explainer.

Two-stage design, mirroring how you'd actually build this in production:
  1. Retrieval: find the most relevant payer rule(s) for the denial text.
     Implemented here with a dependency-free TF-IDF + cosine similarity
     (swap for a real embedding model / Pinecone for production scale).
  2. Reasoning: an LLM explains *why* that rule applies in plain English
     and drafts appeal guidance grounded in the retrieved rule text
     (never inventing a rule that wasn't retrieved).

Runs in two modes:
  - LLM mode (OPENAI_API_KEY or ANTHROPIC_API_KEY set): full natural-language
    explanation grounded in the retrieved rule.
  - Fallback mode (no key): pure retrieval + templated explanation, so the
    live public demo works without requiring visitors to supply API keys
    or costing you money per page view.
"""
import os
import re
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RULES_PATH = Path(__file__).parent.parent / "data" / "payer_rules.json"


@dataclass
class MatchedRule:
    rule_id: str
    category: str
    text: str
    appeal_hint: str
    similarity: float


@dataclass
class DenialExplanation:
    matched_rule: MatchedRule | None
    explanation: str
    appeal_guidance: str
    mode: str  # "llm" or "fallback"
    confidence: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class RuleRetriever:
    def __init__(self, rules_path: Path = RULES_PATH):
        self.rules = json.loads(rules_path.read_text())
        self._doc_tokens = [_tokenize(r["text"] + " " + r["category"]) for r in self.rules]
        self._df = Counter()
        for tokens in self._doc_tokens:
            for t in set(tokens):
                self._df[t] += 1
        self._n_docs = len(self.rules)

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            idf = math.log((self._n_docs + 1) / (self._df.get(term, 0) + 1)) + 1
            vec[term] = count * idf
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        denom_a = math.sqrt(sum(v * v for v in a.values()))
        denom_b = math.sqrt(sum(v * v for v in b.values()))
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return num / (denom_a * denom_b)

    def retrieve(self, query_text: str, top_k: int = 1) -> list[MatchedRule]:
        query_tokens = _tokenize(query_text)
        query_vec = self._tfidf_vector(query_tokens)

        scored = []
        for rule, doc_tokens in zip(self.rules, self._doc_tokens):
            doc_vec = self._tfidf_vector(doc_tokens)
            sim = self._cosine(query_vec, doc_vec)
            scored.append((sim, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, rule in scored[:top_k]:
            results.append(MatchedRule(
                rule_id=rule["rule_id"],
                category=rule["category"],
                text=rule["text"],
                appeal_hint=rule["appeal_hint"],
                similarity=round(sim, 3),
            ))
        return results


def _fallback_explain(denial_text: str, matched: MatchedRule) -> DenialExplanation:
    """Templated explanation used when no LLM key is configured."""
    explanation = (
        f"Based on keyword and pattern matching against your payer's rule set, this denial most closely "
        f"matches rule {matched.rule_id} ({matched.category}), with a similarity score of {matched.similarity}. "
        f"That rule states: \"{matched.text}\""
    )
    return DenialExplanation(
        matched_rule=matched,
        explanation=explanation,
        appeal_guidance=matched.appeal_hint,
        mode="fallback",
        confidence=matched.similarity,
    )


def _llm_explain(denial_text: str, matched: MatchedRule) -> DenialExplanation:
    """Grounded LLM explanation — the LLM is only ever shown the retrieved rule,
    never asked to invent one, so it can't hallucinate a rule that doesn't exist."""
    system = (
        "You are a claims denial explainer. You are given a denial letter/description and ONE "
        "retrieved payer rule. Explain in plain, empathetic language why this specific rule likely "
        "caused the denial, referencing only the rule text provided — never invent policy details "
        "not present in the rule. Keep it to 3-4 sentences. This is for a public demo using synthetic "
        "data only; do not give medical or legal advice, just explain the rule match."
    )
    user = (
        f"Denial description:\n{denial_text}\n\n"
        f"Retrieved rule ({matched.rule_id} - {matched.category}):\n{matched.text}\n\n"
        f"Explain why this rule likely applies."
    )

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
        )
        explanation = resp.choices[0].message.content
    elif os.getenv("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        explanation = resp.content[0].text
    else:
        return _fallback_explain(denial_text, matched)

    return DenialExplanation(
        matched_rule=matched,
        explanation=explanation,
        appeal_guidance=matched.appeal_hint,
        mode="llm",
        confidence=matched.similarity,
    )


class DenialExplainer:
    def __init__(self):
        self.retriever = RuleRetriever()

    def explain(self, denial_text: str) -> DenialExplanation:
        matches = self.retriever.retrieve(denial_text, top_k=1)
        if not matches or matches[0].similarity < 0.05:
            return DenialExplanation(
                matched_rule=None,
                explanation=(
                    "No confident rule match was found for this denial text in the demo rule set "
                    "(8 synthetic rules). In production this would fall back to a broader rule "
                    "library or flag for human review."
                ),
                appeal_guidance="Consider requesting the specific denial reason code from the payer directly.",
                mode="fallback",
                confidence=0.0,
            )

        matched = matches[0]
        if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
            return _llm_explain(denial_text, matched)
        return _fallback_explain(denial_text, matched)
