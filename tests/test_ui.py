"""UI test with a mocked API (Streamlit AppTest). Run from project root: pytest tests/test_ui.py -q"""
import base64, io, sys
from unittest import mock
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

buf = io.BytesIO(); Image.new("RGB", (64, 64), (90, 90, 90)).save(buf, "PNG")
PNG = buf.getvalue()
HEAT = base64.b64encode(PNG).decode()

class FakeFile:
    name, type = "x.png", "image/png"
    def getvalue(self): return PNG

classes = ["Effusion", "Atelectasis", "No Finding"]
def resp(cls):
    return {
        "findings": [
            {"disease": "Effusion", "probability": 0.62, "threshold": 0.60, "detected": True},
            {"disease": "Atelectasis", "probability": 0.41, "threshold": 0.64, "detected": False},
            {"disease": "No Finding", "probability": 0.20, "threshold": 0.37, "detected": False}],
        "detected": ["Effusion"], "review_priority": "Medium", "explained_class": cls,
        "heatmap_base64": HEAT, "model_file": "run3_finetuned.keras",
        "follow_up": {"generic": False, "generic_message": None, "note": "A clinician must interpret.",
                      "specialties": [{"id": "pulm", "name": "Pulmonologist", "reason": "Lung specialist."}]},
        "explanation": {"class": cls, "status": "ok", "reason": None, "data": {
            "condition_overview": "COND_TEXT", "score_interpretation": "SCORE_TEXT",
            "gradcam_note": "CAM_TEXT", "uncertainty_note": "UNC_TEXT", "follow_up_note": "FU_TEXT"}}}



def test_ui_flow():
    calls = []
    class R:
        status_code = 200
        def __init__(self, cls): self._c = cls
        def json(self): return resp(self._c)
    def fake_post(url, params=None, files=None, timeout=None):
        calls.append(params); return R(params.get("explain_class", "Effusion"))

    st.file_uploader = lambda *a, **k: FakeFile()
    with mock.patch("requests.post", fake_post):
        at = AppTest.from_file("ui/app.py", default_timeout=30).run()
        assert not at.exception, at.exception
        at.button[0].click().run()
        assert not at.exception, at.exception
        html = " ".join(m.value for m in at.markdown)
        for needle in ["Review priority: Medium", "cx-chip", "class=\"th\"", "COND_TEXT", "SCORE_TEXT",
                       "CAM_TEXT", "UNC_TEXT", "Pulmonologist", "FU_TEXT", "Grad-CAM · Effusion"]:
            assert needle in html, needle
        print("analyze OK; request params:", calls[-1])
        # change heatmap class
        at.session_state["cam_class"] = "Atelectasis"; at.run()
        assert not at.exception, at.exception
        print("class change OK; request params:", calls[-1])
        assert "Grad-CAM · Atelectasis" in " ".join(m.value for m in at.markdown)

        # LLM unavailable path
        def unavailable(url, params=None, files=None, timeout=None):
            r = R("Effusion"); d = resp("Effusion"); d["explanation"] = {"class": "Effusion", "status": "unavailable", "reason": "not_configured", "data": None}
            r.json = lambda: d; return r
    with mock.patch("requests.post", unavailable):
        at = AppTest.from_file("ui/app.py", default_timeout=30).run()
        at.button[0].click().run()
        assert not at.exception, at.exception
        assert "AI explanation unavailable" in " ".join(m.value for m in at.markdown)
        print("unavailable path OK")
