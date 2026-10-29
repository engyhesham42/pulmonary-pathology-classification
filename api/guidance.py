"""Predefined medical-specialty mapping for follow-up suggestions.

IMPORTANT: this mapping is a DRAFT and must be reviewed and approved by a
qualified clinician before any real-world use. The LLM never creates or edits
entries here; it can only explain what this module returns.

The mapping depends only on which classes the model scored at/above the
decision threshold. It does not infer symptoms, severity, urgency, history,
treatment, or medication.
"""

SPECIALTIES = {
    "gp": {
        "name": "General Practitioner / Primary Care Physician",
        "reason": (
            "A primary care physician can review this result together with your "
            "medical history and decide whether further evaluation is appropriate."
        ),
    },
    "pulm": {
        "name": "Pulmonologist",
        "reason": (
            "Pulmonologists specialise in the lungs and airways and can help "
            "interpret lung-related findings."
        ),
    },
    "cardio": {
        "name": "Cardiologist",
        "reason": (
            "Cardiologists specialise in the heart and circulation and can help "
            "interpret heart-related findings."
        ),
    },
}

# Class -> specialty ids (reviewed list). "No Finding" intentionally maps to
# nothing so it falls back to the generic clinical-review message.
CLASS_TO_SPECIALTIES = {
    "Atelectasis": ["gp", "pulm"],
    "Cardiomegaly": ["gp", "cardio"],
    "Consolidation": ["gp", "pulm"],
    "Edema": ["gp", "pulm", "cardio"],
    "Effusion": ["gp", "pulm"],
    "Emphysema": ["gp", "pulm"],
    "Fibrosis": ["gp", "pulm"],
    "Infiltration": ["gp", "pulm"],
    "Mass": ["gp", "pulm"],
    "No Finding": [],
    "Nodule": ["gp", "pulm"],
    "Pleural_Thickening": ["gp", "pulm"],
    "Pneumonia": ["gp", "pulm"],
    "Pneumothorax": ["gp", "pulm"],
}

_ORDER = ["gp", "pulm", "cardio"]

GENERIC_MESSAGE = (
    "The result does not point to a specific specialty. Please have it reviewed "
    "by a qualified clinician who can interpret it in your clinical context."
)
CLINICIAN_NOTE = (
    "A qualified clinician must interpret these results. This suggestion is "
    "general information, not a diagnosis, and not a mandatory referral."
)


def build_follow_up(positive_classes: list[str]) -> dict:
    """Return the validated follow-up suggestion for the positive classes."""
    ids = {s for c in positive_classes for s in CLASS_TO_SPECIALTIES.get(c, [])}
    specialties = [
        {"id": i, "name": SPECIALTIES[i]["name"], "reason": SPECIALTIES[i]["reason"]}
        for i in _ORDER
        if i in ids
    ]
    return {
        "generic": not specialties,
        "specialties": specialties,
        "generic_message": GENERIC_MESSAGE if not specialties else None,
        "note": CLINICIAN_NOTE,
    }
