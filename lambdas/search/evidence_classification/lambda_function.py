
import os
import json
import logging

EC_MAX_BATCH = int(os.environ.get("EC_MAX_BATCH", "30"))
_TOK_PER_CAND = int(os.environ.get("VERIFY_TOKENS_PER_CAND", "200"))
_MAX_OUT_TOKENS   = int(os.environ.get("VERIFY_MAX_OUT_TOKENS", "8000"))


def _is_haiku_55(model: str) -> bool:
    return "claude-haiku-5" in model


def _build_body_haiku_45(prompt: str, max_tokens: int) -> dict:
    """Haiku 4.5: single text block, deterministic via temperature=0."""
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }


def _build_body_haiku_55(prompt: str, max_tokens: int) -> dict:
    """Haiku 5.5: `temperature` is rejected, and a hidden thinking block eats into
    max_tokens before the real answer — give it extra budget."""
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _build_body(model: str, prompt: str, max_tokens: int) -> dict:
    return (_build_body_haiku_55(prompt, max_tokens) if _is_haiku_55(model)
            else _build_body_haiku_45(prompt, max_tokens))


def _extract_text(resp_body: dict) -> str:
    """First `text` content block — same call for every model, 4.5 just has one block
    while 5.5 may prepend a `thinking` block ahead of it."""
    for block in resp_body.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "")
    raise KeyError("no text block in response content")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_REGION         = os.environ.get("BEDROCK_REGION", AWS_REGION)
VERIFIER_MODEL         = os.environ.get("VERIFIER_MODEL",
                                         "eu.anthropic.claude-haiku-4-5-20251001-v1:0")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _classify_evidence(ci_text: str, hit: dict, doc_ctx: dict) -> dict:
    """Classify a verified YES/MAYBE hit into evidence tiers.

    DIRECT      — excerpt explicitly states the CI (same facts, same drug, same data)
    SUPPORTING  — excerpt provides data/context that supports the CI concept
    RELATED_OBJECTIVE   — same study objective or endpoint structure, different drug/arm
    RELATED_PROTOCOL    — same study design / part / phase reference, no direct CI match
    RELATED_DOSE        — dose levels, RP2D, dose escalation context; not the CI itself
    RELATED_POPULATION  — patient demographics, enrollment criteria, baseline characteristics
    RELATED_SAFETY      — adverse events, toxicity, safety profile data
    RELATED_EFFICACY    — efficacy outcomes (ORR, DOR, PFS, OS) not directly addressing the CI
    RELATED_DEFINITION  — abbreviation legends, glossary entries, figure keys
    """
    span    = (hit.get("match_span") or hit.get("text", ""))[:400]
    doc_tag = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        if drugs or studies:
            doc_tag = f"Document — Drug: {drugs} | Study: {studies}\n\n"

    prompt = (
        f"You are a clinical evidence analyst.\n\n"
        f"{doc_tag}"
        f"Confidential Information (CI):\n\"{ci_text}\"\n\n"
        f"Matched excerpt (page {hit.get('match_page', '?')}):\n\"{span}\"\n\n"
        f"Classify the evidence relationship using exactly one of these labels:\n"
        f"- DIRECT: excerpt explicitly states or reproduces the CI (same facts, same drug, same numbers)\n"
        f"- SUPPORTING: excerpt provides data or context that supports the CI concept\n"
        f"- RELATED_OBJECTIVE: related through a shared study objective or endpoint, different drug/arm\n"
        f"- RELATED_PROTOCOL: related through study design, part/phase structure, or protocol reference\n"
        f"- RELATED_DOSE: related through dose levels, RP2D selection, or dose escalation context\n"
        f"- RELATED_POPULATION: related through patient demographics, eligibility, or baseline data\n"
        f"- RELATED_SAFETY: related through adverse events, toxicity, or safety profile\n"
        f"- RELATED_EFFICACY: related through efficacy outcomes (ORR, DOR, PFS, OS) not directly addressing the CI\n"
        f"- RELATED_DEFINITION: abbreviation legend, glossary entry, figure key, or acronym definition\n\n"
        f"Reply ONLY with valid JSON:\n"
        f"{{\"evidence_type\": \"DIRECT\"|\"SUPPORTING\"|\"RELATED_OBJECTIVE\"|\"RELATED_PROTOCOL\"|"
        f"\"RELATED_DOSE\"|\"RELATED_POPULATION\"|\"RELATED_SAFETY\"|\"RELATED_EFFICACY\"|"
        f"\"RELATED_DEFINITION\", "
        f"\"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}}"
    )
    try:
        import boto3 as _boto3
        br   = _boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        body = _build_body(VERIFIER_MODEL, prompt, _TOK_PER_CAND)
        resp      = br.invoke_model(
            modelId=VERIFIER_MODEL, contentType="application/json",
            accept="application/json", body=json.dumps(body).encode()
        )
        import re as _re
        resp_body    = json.loads(resp["body"].read())
        text         = _extract_text(resp_body).strip()
        ec_usage     = resp_body.get("usage", {})
        ec_in_tok    = ec_usage.get("input_tokens", 0)
        ec_out_tok   = ec_usage.get("output_tokens", 0)
        text = _re.sub(r"^```(?:json)?\s*", "", text)
        text = _re.sub(r"\s*```$", "", text.strip())
        brace = text.find("{")
        if brace > 0:
            text = text[brace:]
        parsed = json.loads(text)
        ev = parsed.get("evidence_type", "RELATED_EFFICACY")
        # Normalise any plain RELATED to RELATED_EFFICACY as safe default
        if ev == "RELATED":
            ev = "RELATED_EFFICACY"
        logger.info("[Evidence Classifier] single in_tok=%d out_tok=%d", ec_in_tok, ec_out_tok)
        return {
            "evidence_type":       ev,
            "evidence_confidence": float(parsed.get("confidence", 0.5)),
            "evidence_reason":     parsed.get("reason", ""),
            "_ec_tokens":          {"input": ec_in_tok, "output": ec_out_tok},
        }
    except Exception as exc:
        return {
            "evidence_type":       "RELATED_EFFICACY",
            "evidence_confidence": 0.0,
            "evidence_reason":     f"classification failed: {exc}",
            "_ec_tokens":          {"input": 0, "output": 0},
        }


def _classify_evidence_batch(ci_text: str, hits: list[dict], doc_ctx: dict) -> list[dict]:
    """Classify all YES/MAYBE hits in one Bedrock call; falls back to sequential on error."""
    if not hits:
        return []
    if len(hits) == 1:
        return [_classify_evidence(ci_text, hits[0], doc_ctx)]
    # Split large batches so we never approach the 8000-token output cap
    if len(hits) > EC_MAX_BATCH:
        results = []
        for i in range(0, len(hits), EC_MAX_BATCH):
            results.extend(_classify_evidence_batch(ci_text, hits[i:i+EC_MAX_BATCH], doc_ctx))
        return results

    doc_tag = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        if drugs or studies:
            doc_tag = f"Document \u2014 Drug: {drugs} | Study: {studies}\n\n"

    blocks = []
    for i, h in enumerate(hits, 1):
        span = (h.get("match_span") or h.get("text", ""))[:300]
        blocks.append(f"--- HIT {i} (p{h.get('match_page','?')}) ---\n\"{span}\"")

    labels = ("DIRECT|SUPPORTING|RELATED_OBJECTIVE|RELATED_PROTOCOL|RELATED_DOSE|"
               "RELATED_POPULATION|RELATED_SAFETY|RELATED_EFFICACY|RELATED_DEFINITION")
    prompt = (
        f"You are a clinical evidence analyst.\n\n{doc_tag}"
        f'CI: "{ci_text}"\n\n'
        f"For each excerpt below, classify the evidence relationship.\n"
        f"Valid labels: {labels}\n"
        f"Reply ONLY with a JSON ARRAY of {len(hits)} objects in order:\n"
        f'[{{"evidence_type":"LABEL","confidence":<0.0-1.0>,"reason":"<one sentence>"}}, ...]\n\n'
        + "\n\n".join(blocks)
    )
    try:
        import boto3 as _boto3
        br   = _boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        # 150 tokens/hit is the measured actual usage (was 100 — caused truncation)
        body = _build_body(VERIFIER_MODEL, prompt, min(_TOK_PER_CAND * len(hits), _MAX_OUT_TOKENS))
        resp      = br.invoke_model(modelId=VERIFIER_MODEL, contentType="application/json",
                                    accept="application/json", body=json.dumps(body).encode())
        resp_body = json.loads(resp["body"].read())
        raw       = _extract_text(resp_body).strip()
        import re as _re
        raw = _re.sub(r"^```(?:json)?\s*", "", raw); raw = _re.sub(r"\s*```$", "", raw.strip())
        brace = raw.find("["); raw = raw[brace:] if brace >= 0 else raw
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError(f"Expected {len(hits)} items, got {len(parsed)}")
        # If Claude still returned fewer items, classify the missed ones individually
        # (don't pad with a placeholder — a dropped DIRECT hit would be misclassified)
        n_missing = max(0, len(hits) - len(parsed))
        if len(parsed) < len(hits):
            logger.warning("[EC] batch returned %d/%d — classifying %d missed hits individually",
                           len(parsed), len(hits), len(hits) - len(parsed))
            for missed_hit in hits[len(parsed):]:
                individual = _classify_evidence(ci_text, missed_hit, doc_ctx)
                parsed.append({
                    "evidence_type": individual.get("evidence_type", "RELATED_EFFICACY"),
                    "confidence":    individual.get("evidence_confidence", 0.5),
                    "reason":        individual.get("evidence_reason", ""),
                    "_ec_tokens":    individual.get("_ec_tokens", {}),
                })
        parsed = parsed[:len(hits)]
        usage = resp_body.get("usage", {})
        in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        per = max(1, in_tok // len(hits)), max(1, out_tok // len(hits))
        results = []
        for h, item in zip(hits, parsed):
            ev = item.get("evidence_type", "RELATED_EFFICACY")
            if ev == "RELATED": ev = "RELATED_EFFICACY"
            results.append({**h,
                "evidence_type":       ev,
                "evidence_confidence": float(item.get("confidence", 0.5)),
                "evidence_reason":     item.get("reason", ""),
                "_ec_tokens":          {"input": per[0], "output": per[1]},
            })
        logger.info("[Evidence Classifier] batch n=%d in_tok=%d out_tok=%d missing=%d",
                    len(hits), in_tok, out_tok, n_missing)
        return results
    except Exception as exc:
        logger.warning("[EC] batch failed (%s) \u2014 falling back to sequential", exc)
        return [_classify_evidence(ci_text, h, doc_ctx) for h in hits]
