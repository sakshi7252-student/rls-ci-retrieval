"""
Search Pipeline — Stage 6: LLM Verifier
=========================================
Asks Bedrock Claude for a YES/NO/MAYBE verdict on each top-ranked candidate.

Prompt returns structured JSON: { "verdict": "YES"|"NO"|"MAYBE", "reason": str,
                                   "confidence": float 0-1 }

Input:  re-ranked search request  (must have "ranked_candidates")
Appends: "verified_candidates": list[VerifiedCandidate]

VerifiedCandidate = RankedCandidate + { "verdict": str, "reason": str,
                                         "confidence": float }
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BEDROCK_REGION       = os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1"))
BEDROCK_MODEL        = os.environ.get("VERIFIER_MODEL", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
MIN_RERANK_SCORE     = float(os.environ.get("MIN_RERANK_SCORE", "0.0"))

_aws: dict = {}

def _get(service: str, region: str | None = None):
    key = f"{service}:{region or ''}"
    if key not in _aws:
        import boto3
        _aws[key] = boto3.client(service, region_name=region) if region else boto3.client(service)
    return _aws[key]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[LLM Verifier] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[LLM Verifier] failed search_id=%s error=%s", search_id, exc)
        raise
    verified_count = sum(1 for c in result["verified_candidates"]
                         if c.get("verdict") == "YES")
    logger.info("[LLM Verifier] done search_id=%s verified=%d", search_id, verified_count)

    return result


def _process(req: dict) -> dict:
    ci_text    = req["ci"].get("knownCI", "")
    ci_assets  = req["ci"].get("assets", [])
    doc_ctx    = req.get("document_context", {})
    ranked     = req.get("ranked_candidates", [])

    # Verify all candidates that cleared the reranker score threshold (no position cap)
    to_verify  = [c for c in ranked]
    skip       = []

    verified = _verify_batch(ci_text, to_verify, doc_ctx, ci_assets)

    # Candidates below threshold are marked SKIP without an LLM call
    for cand in skip:
        verified.append({**cand, "verdict": "SKIP", "reason": "below reranker threshold",
                         "confidence": 0.0})

    return {
        **req,
        "verified_candidates": verified,
    }


_MAX_VERIFY_BATCH = int(os.environ.get("MAX_VERIFY_BATCH", "20"))  # smaller batches reduce cross-candidate copy-paste artifacts


def _verify_batch(
    ci_text: str,
    candidates: list[dict],
    doc_ctx: dict | None = None,
    ci_assets: list | None = None,
) -> list[dict]:
    """Verify all candidates in a single Bedrock call; falls back to sequential on error."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return [_verify(ci_text, candidates[0], doc_ctx, ci_assets)]
    # Split oversized batches to avoid hitting the 8000-token output cap
    if len(candidates) > _MAX_VERIFY_BATCH:
        results = []
        for i in range(0, len(candidates), _MAX_VERIFY_BATCH):
            results.extend(_verify_batch(ci_text, candidates[i:i+_MAX_VERIFY_BATCH], doc_ctx, ci_assets))
        return results

    import re as _re

    # Build shared header (doc profile + drug note)
    doc_profile = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        disease = ", ".join(doc_ctx.get("disease", [])[:1])
        phase   = ", ".join(doc_ctx.get("phase", []))
        doc_profile = (f"DOCUMENT PROFILE:\n  Drug(s): {drugs}\n  Study:   {studies}\n"
                       f"  Disease: {disease}\n  Phase:   {phase}\n\n")

    ci_drug_note = ""
    asset_ctx    = ""
    if ci_assets:
        ci_drug_names = [a.get("name") or a.get("genericName") or a.get("code", "")
                         for a in ci_assets if a]
        ci_drug_names = [n for n in ci_drug_names if n]
        desc = next((a.get("description", "") for a in ci_assets if a and a.get("description")), "")
        desc_lower = _re.sub(r"<[^>]+>", " ", desc).lower()
        if ci_drug_names:
            doc_drugs_lower = {d.lower() for d in doc_ctx.get("primary_drugs", [])} if doc_ctx else set()
            ci_drugs_lower  = {n.lower() for n in ci_drug_names}
            name_overlap = bool(doc_drugs_lower & ci_drugs_lower)
            desc_overlap = any(drug in desc_lower for drug in doc_drugs_lower)
            if doc_drugs_lower and not name_overlap and not desc_overlap:
                doc_drug_str = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
                ci_drug_str  = ", ".join(ci_drug_names)
                ci_drug_note = (f"Context: CI linked to [{ci_drug_str}], document covers "
                                f"[{doc_drug_str}]. Score on content.\n\n")
            else:
                ci_drug_note = f"CI Drug: {', '.join(ci_drug_names)}\n\n"
        if desc:
            asset_ctx = f"Drug/Regimen Context: {_re.sub(chr(60)+'[^>]+>','',desc).strip()[:500]}\n\n"

    # Build one block per candidate — label prev/current/next so the model knows
    # which span is the actual retrieved candidate vs. surrounding context only.
    blocks = []
    for i, c in enumerate(candidates, 1):
        ctx = c.get("context", {})
        prev_t, cur_t, next_t = ctx.get("prev_text", ""), ctx.get("current_text", ""), ctx.get("next_text", "")
        excerpt_parts = []
        if prev_t:
            excerpt_parts.append(f"[PRECEDING CONTEXT]\n{prev_t}")
        excerpt_parts.append(f"[CURRENT EXCERPT]\n{cur_t}")
        if next_t:
            excerpt_parts.append(f"[FOLLOWING CONTEXT]\n{next_t}")
        excerpt = "\n\n".join(excerpt_parts)[:2500]
        blocks.append(f"--- CANDIDATE {i} (p{c.get('page_start')}\u2013{c.get('page_end')}) ---\n{excerpt}")

    prompt = (
        f"You are a clinical document reviewer.\n\n"
        f"{doc_profile}{ci_drug_note}{asset_ctx}"
        f'Confidential Information (CI): "{ci_text}"\n\n'
        f"Each candidate's CURRENT EXCERPT is the actual retrieved match being judged; PRECEDING/"
        f"FOLLOWING CONTEXT is given only so you can interpret it correctly (resolve pronouns, "
        f"abbreviations, sentences continued across a chunk boundary, etc.). Two kinds of evidence "
        f"support YES: (1) LITERAL — the CI's exact fact/number/phrase appears verbatim in the "
        f"CURRENT EXCERPT itself; (2) SEMANTIC — the CURRENT EXCERPT, once correctly understood "
        f"using its context, is itself substantively discussing the same clinical topic/fact as the "
        f"CI (not just mentioning a keyword in passing). A CURRENT EXCERPT that is actually about a "
        f"different topic does NOT become a match just because the CI's fact happens to appear "
        f"somewhere in the surrounding context on the same page — that is evidence for a different, "
        f"nearby excerpt, not this one. In that case the verdict must be NO or MAYBE.\n\n"
        f"For each candidate below, decide if its CURRENT EXCERPT (as understood using its context) "
        f"contains or semantically identifies the CI.\n\n"
        f"For each identity dimension answer true or false:\n"
        f"  same_drug       — excerpt discusses the same drug/regimen as the CI\n"
        f"  same_study      — excerpt is from the same trial/study as the CI\n"
        f"  same_objective  — excerpt shares the same primary/secondary objective\n"
        f"  same_endpoint   — excerpt uses the same primary endpoint (PFS, ORR, etc.)\n"
        f"  same_comparator — excerpt uses the same comparator arms/regimens\n\n"
        f"identity_score: fraction of dimensions that are true (0.0–1.0)\n"
        f"semantic_score: how semantically similar excerpt is to the CI (0.0–1.0)\n\n"
        f"Reply ONLY with a JSON ARRAY of {len(candidates)} objects in the same order:\n"
        f'[{{"verdict":"YES"|"NO"|"MAYBE","reason":"<one sentence>",'
        f'"confidence":<0.0-1.0>,"identity":{{"same_drug":<bool>,"same_study":<bool>,'
        f'"same_objective":<bool>,"same_endpoint":<bool>,"same_comparator":<bool>,'
        f'"identity_score":<0.0-1.0>,"semantic_score":<0.0-1.0>}}}}, ...]\n\n'
        + "\n\n".join(blocks)
    )

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": min(200 * len(candidates), 8000),  # ~85 actual/candidate; cap at API limit
            "messages": [{"role": "user", "content": prompt}],
        }
        resp      = _get("bedrock-runtime", BEDROCK_REGION).invoke_model(
            modelId=BEDROCK_MODEL, contentType="application/json",
            accept="application/json", body=json.dumps(body).encode(),
        )
        resp_body = json.loads(resp["body"].read())
        raw       = resp_body["content"][0]["text"].strip()
        # Strip code fences then find the array — NOT _strip_code_fence which seeks {
        import re as _re2
        raw = _re2.sub(r"^```(?:json)?\s*", "", raw)
        raw = _re2.sub(r"\s*```$", "", raw.strip())
        bracket = raw.find("[")
        if bracket > 0:
            raw = raw[bracket:]
        text      = raw
        usage     = resp_body.get("usage", {})
        in_tok    = usage.get("input_tokens", 0)
        out_tok   = usage.get("output_tokens", 0)
        parsed    = json.loads(text)
        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError(f"Expected list of {len(candidates)}, got {len(parsed) if isinstance(parsed,list) else type(parsed)}")
        # If Claude returned fewer items, verify the missed ones individually
        # (don't pad with a placeholder — a dropped YES candidate would be misverified)
        if len(parsed) < len(candidates):
            logger.warning("[LLM Verifier] batch returned %d/%d — verifying %d missed candidates individually",
                           len(parsed), len(candidates), len(candidates) - len(parsed))
            for missed_cand in candidates[len(parsed):]:
                individual = _verify(ci_text, missed_cand, doc_ctx, ci_assets)
                parsed.append({
                    "verdict":    individual.get("verdict", "MAYBE"),
                    "reason":     individual.get("reason", ""),
                    "confidence": individual.get("confidence", 0.5),
                    "identity":   individual.get("identity", {}),
                    "_tokens":    individual.get("_tokens", {}),
                })
        parsed = parsed[:len(candidates)]  # truncate any extra items Claude occasionally adds
        results = []
        per_tok = max(1, in_tok // len(candidates)), max(1, out_tok // len(candidates))
        for cand, item in zip(candidates, parsed):
            results.append({
                **cand,
                "verdict":    item.get("verdict", "MAYBE"),
                "reason":     item.get("reason", ""),
                "confidence": float(item.get("confidence", 0.5)),
                "identity":   item.get("identity", {}),
                "_tokens":    {"input": per_tok[0], "output": per_tok[1]},
            })
        logger.info("[LLM Verifier] batch n=%d in_tok=%d out_tok=%d", len(candidates), in_tok, out_tok)
        return results
    except Exception as exc:
        logger.warning("[LLM Verifier] batch failed (%s) — falling back to sequential", exc)
        return [_verify(ci_text, c, doc_ctx, ci_assets) for c in candidates]


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences that Claude sometimes wraps JSON in."""
    import re
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # If there's still preamble before the first {, trim it
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    return text.strip()


def _verify(ci_text: str, candidate: dict, doc_ctx: dict | None = None,
            ci_assets: list | None = None) -> dict:
    # Label prev/current/next so the model knows which span is the actual
    # retrieved candidate vs. surrounding context only.
    ctx      = candidate.get("context", {})
    prev_t, cur_t, next_t = ctx.get("prev_text", ""), ctx.get("current_text", ""), ctx.get("next_text", "")
    excerpt_parts = []
    if prev_t:
        excerpt_parts.append(f"[PRECEDING CONTEXT]\n{prev_t}")
    excerpt_parts.append(f"[CURRENT EXCERPT]\n{cur_t}")
    if next_t:
        excerpt_parts.append(f"[FOLLOWING CONTEXT]\n{next_t}")
    combined = "\n\n".join(excerpt_parts)[:3000]

    # Document profile header
    doc_profile = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        disease = ", ".join(doc_ctx.get("disease", [])[:1])
        phase   = ", ".join(doc_ctx.get("phase", []))
        doc_profile = (
            f"DOCUMENT PROFILE:\n"
            f"  Drug(s): {drugs}\n"
            f"  Study:   {studies}\n"
            f"  Disease: {disease}\n"
            f"  Phase:   {phase}\n\n"
        )

    # CI drug note — contextual, never a hard gate
    # Drug identity is a ranking signal, not a filter: combination regimens, comparator arms,
    # and mechanism discussions legitimately cross drug boundaries.
    ci_drug_note = ""
    asset_ctx    = ""
    import re as _re
    if ci_assets:
        ci_drug_names = [
            a.get("name") or a.get("genericName") or a.get("code", "")
            for a in ci_assets if a
        ]
        ci_drug_names = [n for n in ci_drug_names if n]

        # Build a single description blob for overlap checking
        desc = next((a.get("description", "") for a in ci_assets if a and a.get("description")), "")
        desc_lower = _re.sub(r"<[^>]+>", " ", desc).lower()

        if ci_drug_names:
            doc_drugs_lower = {d.lower() for d in doc_ctx.get("primary_drugs", [])} if doc_ctx else set()
            ci_drugs_lower  = {n.lower() for n in ci_drug_names}

            # Overlap via name/code OR via description text
            # (e.g. "Tec-Tal" description mentions "talquetamab" → counts as overlap)
            name_overlap = bool(doc_drugs_lower & ci_drugs_lower)
            desc_overlap = any(drug in desc_lower for drug in doc_drugs_lower)

            if doc_drugs_lower and not name_overlap and not desc_overlap:
                # Genuinely different drug families — note it softly, do NOT hard-gate
                doc_drug_str = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
                ci_drug_str  = ", ".join(ci_drug_names)
                ci_drug_note = (
                    f"Context: This CI is linked to [{ci_drug_str}] and the document "
                    f"primarily covers [{doc_drug_str}]. Score based on content — evidence "
                    f"from comparator arms, related mechanisms, or cross-study references "
                    f"is valid supporting evidence.\n\n"
                )
            else:
                ci_drug_note = f"CI Drug: {', '.join(ci_drug_names)}\n\n"

        # Asset description as drug/regimen context
        if desc:
            desc_clean = _re.sub(r"<[^>]+>", " ", desc).strip()[:500]
            asset_ctx = f"Drug/Regimen Context: {desc_clean}\n\n"

    prompt = (
        f"You are a clinical document reviewer.\n\n"
        f"{doc_profile}"
        f"{ci_drug_note}"
        f"{asset_ctx}"
        f"Confidential Information (CI): \"{ci_text}\"\n\n"
        f"Document excerpt (pages {candidate.get('page_start')}–"
        f"{candidate.get('page_end')}):\n{combined}\n\n"
        f"The CURRENT EXCERPT is the actual retrieved candidate being judged; PRECEDING/FOLLOWING "
        f"CONTEXT is given only so you can interpret it correctly (resolve pronouns, abbreviations, "
        f"sentences continued across a chunk boundary, etc.). Two kinds of evidence support YES: "
        f"(1) LITERAL — the CI's exact fact/number/phrase appears verbatim in the CURRENT EXCERPT "
        f"itself; (2) SEMANTIC — the CURRENT EXCERPT, once correctly understood using its context, "
        f"is itself substantively discussing the same clinical topic/fact as the CI (not just "
        f"mentioning a keyword in passing). A CURRENT EXCERPT that is actually about a different "
        f"topic does NOT become a match just because the CI's fact happens to appear somewhere in "
        f"the surrounding context on the same page — that is evidence for a different, nearby "
        f"excerpt, not this one. In that case the verdict must be NO or MAYBE.\n\n"
        f"Does the CURRENT EXCERPT (as understood using its context) contain or semantically "
        f"identify the CI?\n\n"
        f"For each identity dimension answer true or false:\n"
        f"  same_drug       — excerpt discusses the same drug/regimen as the CI\n"
        f"  same_study      — excerpt is from the same trial/study as the CI\n"
        f"  same_objective  — excerpt shares the same primary/secondary objective\n"
        f"  same_endpoint   — excerpt uses the same primary endpoint (PFS, ORR, etc.)\n"
        f"  same_comparator — excerpt uses the same comparator arms/regimens\n\n"
        f"identity_score: fraction of dimensions that are true (0.0–1.0)\n"
        f"semantic_score: how semantically similar excerpt is to the CI (0.0–1.0)\n\n"
        f"Reply ONLY with valid JSON:\n"
        f"{{\"verdict\": \"YES\"|\"NO\"|\"MAYBE\", \"reason\": \"<one sentence>\", "
        f"\"confidence\": <0.0-1.0>, "
        f"\"identity\": {{\"same_drug\": <bool>, \"same_study\": <bool>, "
        f"\"same_objective\": <bool>, \"same_endpoint\": <bool>, "
        f"\"same_comparator\": <bool>, "
        f"\"identity_score\": <0.0-1.0>, \"semantic_score\": <0.0-1.0>}}}}"
    )

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        500,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp     = _get("bedrock-runtime", BEDROCK_REGION).invoke_model(
            modelId     = BEDROCK_MODEL,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps(body).encode(),
        )
        resp_body = json.loads(resp["body"].read())
        text      = resp_body["content"][0]["text"].strip()
        usage     = resp_body.get("usage", {})
        input_tokens  = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        text   = _strip_code_fence(text)
        parsed = json.loads(text)
        verdict  = parsed.get("verdict", "MAYBE")
        reason   = parsed.get("reason", "")
        conf     = float(parsed.get("confidence", 0.5))
        identity = parsed.get("identity", {})
    except json.JSONDecodeError as exc:
        logger.warning("[LLM Verifier] JSON parse failed chunk=%s: %s | raw=%r",
                       candidate.get("chunk_id"), exc, text[:200] if "text" in dir() else "")
        verdict, reason, conf, identity = "MAYBE", "LLM response was not valid JSON", 0.3, {}
        input_tokens, output_tokens = 0, 0
    except Exception as exc:
        logger.warning("[LLM Verifier] call failed chunk=%s error=%s",
                       candidate.get("chunk_id"), exc)
        verdict, reason, conf, identity = "MAYBE", str(exc), 0.0, {}
        input_tokens, output_tokens = 0, 0

    return {
        **candidate,
        "verdict":    verdict,
        "reason":     reason,
        "confidence": conf,
        "identity":   identity,
        "_tokens":    {"input": input_tokens, "output": output_tokens},
    }
