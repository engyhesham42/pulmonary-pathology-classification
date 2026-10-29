"""Groq-powered explanation layer for the chest X-ray API.

Design rules (kept deliberately strict):
  * The LLM only ever sees structured model outputs: class names, scores, the
    decision threshold, the selected Grad-CAM class, and the specialty names
    produced by backend.guidance. It never sees the image or any patient data.
  * The LLM cannot change scores, labels, or referrals. Its output is plain
    text that is validated and displayed next to the untouched predictions.
  * Any failure (no key, timeout, rate limit, bad JSON, failed validation)
    returns {"status": "unavailable"} and the rest of the response is unaffected.

Config (environment variables):
  GROQ_API_KEY   required for explanations, never logged or returned
  GROQ_MODEL     chat model id (default: openai/gpt-oss-120b)
  GROQ_TIMEOUT   seconds per request (default: 20)
"""
import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    import groq
    from groq import Groq
except ImportError:  # keep the API usable even if the SDK is missing
    groq = None
    Groq = None

log = logging.getLogger("chestai.llm")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT", "20"))
MAX_ATTEMPTS = 2  # one retry when the output fails validation
CACHE_SIZE = 128

SYSTEM_PROMPT = """You are an AI explanation assistant for a research-only chest X-ray classification application.

You do not diagnose diseases.

You must explain only the structured prediction data provided by the backend.

Do not invent findings, symptoms, image features, patient information, or medical conclusions.

Do not claim to have independently analyzed the radiograph.

Model scores are not necessarily calibrated probabilities.

Grad-CAM visualizations show regions influencing model predictions and do not prove the anatomical location of disease.

Use cautious, understandable medical language.

Do not recommend medication or treatment.

Only describe medical specialty suggestions explicitly supplied by the backend.

If information is insufficient, state the limitation clearly.

Never present the AI result as a substitute for professional medical evaluation.

Output rules:
- Respond with a single JSON object and nothing else.
- Quote numbers only exactly as given in the input (scores are on a 0-1 scale). Do not add any other numbers.
- Do not assign urgency or severity.
- Do not mention any medical specialty unless it appears in allowed_follow_up_specialties.
- Keep each field to 1-3 short sentences in plain language."""

USER_TEMPLATE = """Explain the following model output for a non-expert reader.

INPUT (the only source of truth):
{context}

Return a JSON object with exactly these string fields:
- "condition_overview": what the selected condition generally means, in general terms only.
- "score_interpretation": what the selected score means for this model and whether it met the decision threshold; say it is not a calibrated probability.
- "gradcam_note": what a Grad-CAM heatmap is and that it highlights regions that influenced the model, without proving where disease is.
- "uncertainty_note": the model's uncertainty and limits, and that a clinician must interpret the result.
- "follow_up_note": explain the supplied follow-up suggestion only. If follow_up_is_generic is true, say that clinical review is advised without naming a specialty."""


# ------------------------------------------------------------------ schema
class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_overview: str = Field(min_length=20, max_length=600)
    score_interpretation: str = Field(min_length=20, max_length=600)
    gradcam_note: str = Field(min_length=20, max_length=600)
    uncertainty_note: str = Field(min_length=20, max_length=600)
    follow_up_note: str = Field(min_length=20, max_length=600)

    @field_validator("*", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


class ExplanationResult(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: Optional[
        Literal["not_configured", "rate_limited", "timeout", "service_error", "invalid_response"]
    ] = None
    data: Optional[Explanation] = None


def _unavailable(reason: str) -> dict:
    return ExplanationResult(status="unavailable", reason=reason).model_dump()


# ------------------------------------------------------------------ context
def _display(name: str) -> str:
    return name.replace("_", " ")


def build_context(
    predictions: list[dict],
    threshold: float,
    selected_class: str,
    gradcam_available: bool,
    follow_up: dict,
) -> dict:
    """Assemble the grounded payload sent to the LLM (no image, no identifiers)."""
    by_name = {p["class"]: p for p in predictions}
    sel = by_name[selected_class]
    positives = [p for p in predictions if p["positive"]]
    # Per-class thresholds: use the selected class's own threshold when provided
    sel_threshold = sel.get("threshold", threshold)
    return {
        "selected_class": {
            "name": _display(selected_class),
            "score": round(sel["probability"], 2),
            "meets_decision_threshold": sel["positive"],
        },
        "decision_threshold": round(sel_threshold, 2),
        "classes_meeting_threshold": [
            {"name": _display(p["class"]), "score": round(p["probability"], 2)} for p in positives
        ],
        "top_scores": [
            {"name": _display(p["class"]), "score": round(p["probability"], 2)}
            for p in predictions[:5]
        ],
        "gradcam_available_for_selected_class": gradcam_available,
        "allowed_follow_up_specialties": [s["name"] for s in follow_up["specialties"]],
        "follow_up_is_generic": follow_up["generic"],
        "_specialty_ids": [s["id"] for s in follow_up["specialties"]],  # stripped before sending
    }


# ------------------------------------------------------------------ guardrails
_BANNED = [
    (r"\b(prescrib\w*|medication\w*|antibiotic\w*|steroid\w*|dosage|treatment plan|chemotherap\w*|radiotherap\w*|surger\w*|surgical|biops\w*)\b",
     "mentions medication or treatment"),
    (r"\b(urgent\w*|emergency|immediately|asap|life-threatening|severe|critical)\b",
     "assigns urgency or severity"),
    (r"\b(you (likely |probably )?have|you are suffering|the patient has|diagnosed with)\b",
     "makes a diagnostic statement"),
    (r"\b(confirms|proves)\b", "overclaims certainty"),
    (r"\bI (have |'ve )?(analy[sz]ed|examined|reviewed|looked|inspected|viewed|saw|see|detected)\b",
     "claims independent image analysis"),
    (r"\b(the|this) (image|radiograph|x-ray|scan) (shows|reveals|displays|contains|demonstrates)\b",
     "describes image findings"),
]
_BANNED = [(re.compile(p, re.I), msg) for p, msg in _BANNED]

_SPECIALTY_PATTERNS = [
    (re.compile(r"\bpulmonolog\w*", re.I), "pulm"),
    (re.compile(r"\bcardiolog\w*", re.I), "cardio"),
    (re.compile(r"\b(general practitioner|primary care|family (doctor|physician)|GP)\b", re.I), "gp"),
    (re.compile(
        r"\b(oncolog\w*|radiolog\w*|surgeon\w*|thoracic|rheumatolog\w*|infectious[- ]disease|"
        r"internist|emergency (room|department|physician)|intensiv\w*|critical care)\b", re.I), None),
]
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _violations(exp: Explanation, ctx: dict) -> list[str]:
    problems: list[str] = []
    allowed_ids = set(ctx["_specialty_ids"])
    scores = [ctx["selected_class"]["score"], ctx["decision_threshold"]]
    scores += [x["score"] for x in ctx["classes_meeting_threshold"] + ctx["top_scores"]]

    for field, text in exp.model_dump().items():
        for rx, msg in _BANNED:
            if rx.search(text):
                problems.append(f"{field}: {msg}")
        for rx, sid in _SPECIALTY_PATTERNS:
            if rx.search(text) and (sid is None or sid not in allowed_ids):
                problems.append(f"{field}: mentions a specialty that was not supplied")
        for n in _NUM.findall(text):
            x = float(n)
            ok = x in (0.0, 1.0) or any(
                abs(x - s) <= 0.0051 or abs(x - s * 100) <= 0.51 for s in scores
            )
            if not ok:
                problems.append(f"{field}: contains number {n} that was not in the input")
    return problems


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


# ------------------------------------------------------------------ client + cache
_client = None
_client_lock = threading.Lock()
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def _get_client():
    global _client
    key = os.getenv("GROQ_API_KEY")
    if not key or Groq is None:
        return None
    with _client_lock:
        if _client is None:
            _client = Groq(api_key=key, timeout=GROQ_TIMEOUT, max_retries=1)
        return _client


def _call_groq(client, messages: list[dict]) -> str:
    kwargs = dict(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0,
        seed=7,
        max_completion_tokens=2000,  # reasoning models spend part of this on thinking
        response_format={"type": "json_object"},
    )
    if "gpt-oss" in GROQ_MODEL:
        kwargs["reasoning_effort"] = "low"
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


# ------------------------------------------------------------------ public API
def generate_explanation(ctx: dict) -> dict:
    """Return an ExplanationResult dict. Never raises."""
    client = _get_client()
    if client is None:
        return _unavailable("not_configured")

    public_ctx = {k: v for k, v in ctx.items() if not k.startswith("_")}
    ctx_json = json.dumps(public_ctx, indent=2, sort_keys=True)
    cache_key = hashlib.sha256(f"{GROQ_MODEL}|{ctx_json}".encode()).hexdigest()
    with _cache_lock:
        if cache_key in _cache:
            _cache.move_to_end(cache_key)
            return _cache[cache_key]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=ctx_json)},
    ]
    reason = "invalid_response"
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = _call_groq(client, messages)
            exp = Explanation.model_validate(_parse_json(raw))
            problems = _violations(exp, ctx)
            if not problems:
                result = ExplanationResult(status="ok", data=exp).model_dump()
                with _cache_lock:
                    _cache[cache_key] = result
                    while len(_cache) > CACHE_SIZE:
                        _cache.popitem(last=False)
                return result
            log.warning("Explanation rejected (attempt %d): %s", attempt + 1, problems)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Rejected for: " + "; ".join(problems)
                 + ". Return corrected JSON that follows every rule."},
            ]
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            log.warning("Invalid LLM output (attempt %d): %s", attempt + 1, type(e).__name__)
            messages.append({"role": "user", "content": "Your reply was not valid JSON for the required fields. Return only the JSON object."})
        except groq.RateLimitError:
            log.warning("Groq rate limit hit")
            return _unavailable("rate_limited")
        except groq.APITimeoutError:
            log.warning("Groq request timed out")
            return _unavailable("timeout")
        except (groq.APIConnectionError, groq.APIStatusError, groq.GroqError) as e:
            log.warning("Groq error: %s", type(e).__name__)  # no message: avoid leaking details
            return _unavailable("service_error")
        except Exception:  # noqa: BLE001 - the LLM must never break /predict
            log.exception("Unexpected LLM failure")
            return _unavailable("service_error")
    return _unavailable(reason)
