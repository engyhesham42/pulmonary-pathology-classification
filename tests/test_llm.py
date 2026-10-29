"""Tests for the Groq explanation layer and follow-up logic (no real API calls)."""
import sys

sys.path.insert(0, "api")

import guidance
import llm_service as L


def _preds():
    return [
        {"class": "Effusion", "probability": 0.62, "positive": True, "threshold": 0.60},
        {"class": "Atelectasis", "probability": 0.41, "positive": False, "threshold": 0.64},
        {"class": "No Finding", "probability": 0.20, "positive": False, "threshold": 0.37},
    ]


def _ctx():
    fu = guidance.build_follow_up(["Effusion"])
    return L.build_context(_preds(), 0.5, "Effusion", True, fu)


GOOD = ('{"condition_overview":"Effusion generally refers to fluid around the lungs.",'
        '"score_interpretation":"The score 0.62 met the threshold 0.6 but is not a calibrated probability.",'
        '"gradcam_note":"The heatmap highlights regions that influenced the model score.",'
        '"uncertainty_note":"The model can be wrong and a clinician must interpret the result.",'
        '"follow_up_note":"A General Practitioner / Primary Care Physician or a Pulmonologist may review this."}')


def test_follow_up_mapping():
    fu = guidance.build_follow_up(["Edema"])
    assert [s["id"] for s in fu["specialties"]] == ["gp", "pulm", "cardio"]
    assert guidance.build_follow_up(["No Finding"])["generic"] is True


def test_context_uses_per_class_threshold():
    assert _ctx()["decision_threshold"] == 0.6


def test_not_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    L._client = None
    r = L.generate_explanation(_ctx())
    assert r["status"] == "unavailable" and r["reason"] == "not_configured"


def test_valid_output_accepted(monkeypatch):
    monkeypatch.setattr(L, "_get_client", lambda: object())
    monkeypatch.setattr(L, "_call_groq", lambda c, m: GOOD)
    L._cache.clear()
    r = L.generate_explanation(_ctx())
    assert r["status"] == "ok"


def test_guardrails_reject_bad_output(monkeypatch):
    bad = GOOD.replace("may review this", "should start antibiotics immediately")
    monkeypatch.setattr(L, "_get_client", lambda: object())
    monkeypatch.setattr(L, "_call_groq", lambda c, m: bad)
    L._cache.clear()
    r = L.generate_explanation(_ctx())
    assert r["status"] == "unavailable" and r["reason"] == "invalid_response"


def test_unsupplied_specialty_and_invented_number_rejected(monkeypatch):
    bad = GOOD.replace("a Pulmonologist", "an Oncologist").replace("0.62", "0.97")
    monkeypatch.setattr(L, "_get_client", lambda: object())
    monkeypatch.setattr(L, "_call_groq", lambda c, m: bad)
    L._cache.clear()
    assert L.generate_explanation(_ctx())["status"] == "unavailable"
