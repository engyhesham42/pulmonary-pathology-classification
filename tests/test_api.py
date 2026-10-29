"""Run from project root with a model in models/:  pytest tests/ -q
Requires: pip install -r api/requirements.txt pytest httpx
"""
import io
import os
import sys

sys.path.insert(0, "api")

from fastapi.testclient import TestClient
from PIL import Image

from main import app


def png_bytes(size=(300, 300)):
    buf = io.BytesIO()
    Image.new("L", size, 128).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_classes():
    with TestClient(app) as c:
        cl = c.get("/classes").json()["classes"]
        assert "No Finding" in cl and len(cl) == 14


def test_predict_ok():
    with TestClient(app) as c:
        r = c.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200
        d = r.json()
        assert len(d["findings"]) == 14
        assert d["review_priority"] in {"High", "Medium", "Low"}
        assert d["heatmap_base64"] and d["explained_class"] != "No Finding"


def test_explain_specific_class_and_no_explain():
    with TestClient(app) as c:
        f = {"file": ("x.png", png_bytes(), "image/png")}
        r = c.post("/predict?explain_class=Effusion", files=f)
        assert r.status_code == 200 and r.json()["explained_class"] == "Effusion"
        r = c.post("/predict?explain=false", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200 and r.json()["heatmap_base64"] is None


def test_unknown_class():
    with TestClient(app) as c:
        r = c.post("/predict?explain_class=Foo", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 400


def test_bad_type_and_corrupt():
    with TestClient(app) as c:
        assert c.post("/predict", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 400
        assert c.post("/predict", files={"file": ("a.png", b"junk", "image/png")}).status_code == 400


def test_llm_explain_not_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import llm_service
    llm_service._client = None
    with TestClient(app) as c:
        r = c.post("/predict?llm_explain=true", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 200
        d = r.json()
        assert d["follow_up"] is not None and "note" in d["follow_up"]
        assert d["explanation"]["status"] == "unavailable"
        assert d["explanation"]["reason"] == "not_configured"
        assert len(d["findings"]) == 14   # predictions unaffected


def test_llm_explain_ok(monkeypatch):
    import main
    canned = {"status": "ok", "reason": None, "data": {
        "condition_overview": "x" * 25, "score_interpretation": "x" * 25,
        "gradcam_note": "x" * 25, "uncertainty_note": "x" * 25, "follow_up_note": "x" * 25}}
    monkeypatch.setattr(main, "generate_explanation", lambda ctx: canned)
    with TestClient(app) as c:
        r = c.post("/predict?llm_explain=true", files={"file": ("x.png", png_bytes(), "image/png")})
        d = r.json()
        assert d["explanation"]["status"] == "ok" and d["explanation"]["class"] == d["explained_class"]


def test_default_has_no_llm_fields():
    with TestClient(app) as c:
        d = c.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")}).json()
        assert d["follow_up"] is None and d["explanation"] is None
