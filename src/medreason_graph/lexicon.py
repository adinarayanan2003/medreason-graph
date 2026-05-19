from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    canonical: str
    kind: str
    synonyms: tuple[str, ...]
    semantic_type: str | None = None


CONCEPTS: dict[str, Concept] = {
    "acute coronary syndrome": Concept(
        "acute coronary syndrome",
        "condition",
        (
            "acs",
            "acute coronary syndrome",
            "unstable angina",
            "myocardial infarction",
            "myocardial ischemia",
            "acute myocardial ischemia",
            "heart attack",
            "mi",
        ),
    ),
    "pulmonary embolism": Concept(
        "pulmonary embolism",
        "condition",
        ("pe", "pulmonary embolism", "blood clot in lung"),
    ),
    "aortic dissection": Concept(
        "aortic dissection",
        "condition",
        ("aortic dissection", "tearing chest pain", "ripping chest pain"),
    ),
    "gastroesophageal reflux disease": Concept(
        "gastroesophageal reflux disease",
        "condition",
        ("gerd", "acid reflux", "heartburn", "gastroesophageal reflux"),
    ),
    "migraine": Concept("migraine", "condition", ("migraine", "recurrent headache", "photophobia")),
    "meningitis": Concept("meningitis", "condition", ("meningitis", "neck stiffness", "fever with headache")),
    "pneumonia": Concept(
        "pneumonia",
        "condition",
        ("pneumonia", "community-acquired pneumonia", "cap", "typical bacterial pneumonia"),
    ),
    "asthma": Concept("asthma", "condition", ("asthma", "asthma exacerbation", "bronchial asthma", "reactive airway disease")),
    "chest pain": Concept("chest pain", "symptom", ("chest pain", "chest discomfort", "chest pressure")),
    "left arm radiation": Concept(
        "left arm radiation",
        "symptom",
        ("left arm radiation", "radiates to left arm", "arm radiation", "radiating to the arm", "upper extremity pain"),
    ),
    "diaphoresis": Concept("diaphoresis", "symptom", ("diaphoresis", "sweating", "sweatiness")),
    "nausea": Concept("nausea", "symptom", ("nausea", "vomiting", "nauseated")),
    "dyspnea": Concept("dyspnea", "symptom", ("dyspnea", "shortness of breath", "sob", "breathlessness")),
    "pleuritic pain": Concept("pleuritic pain", "symptom", ("pleuritic pain", "pleuritic chest pain", "worse with breathing")),
    "tearing chest pain": Concept("tearing chest pain", "symptom", ("tearing chest pain", "ripping chest pain")),
    "headache": Concept("headache", "symptom", ("headache", "head pain")),
    "fever": Concept("fever", "symptom", ("fever", "febrile", "high temperature")),
    "neck stiffness": Concept("neck stiffness", "symptom", ("neck stiffness", "stiff neck", "meningismus")),
    "cough": Concept("cough", "symptom", ("cough", "coughing")),
    "wheezing": Concept("wheezing", "symptom", ("wheezing", "wheeze", "wheezes")),
    "sputum production": Concept("sputum production", "symptom", ("sputum production", "sputum", "productive cough")),
    "ecg": Concept("ecg", "test", ("ecg", "ekg", "electrocardiogram", "12-lead ecg")),
    "troponin": Concept("troponin", "test", ("troponin", "cardiac troponin", "serial troponin")),
    "ct pulmonary angiography": Concept(
        "ct pulmonary angiography",
        "test",
        ("ct pulmonary angiography", "ctpa", "pulmonary angiography"),
    ),
    "ct angiography": Concept("ct angiography", "test", ("ct angiography", "cta", "aortic imaging")),
    "chest x-ray": Concept("chest x-ray", "test", ("chest x-ray", "chest xray", "chest radiograph", "radiography")),
    "pulse oximetry": Concept("pulse oximetry", "test", ("pulse oximetry", "oxygen saturation", "spo2")),
    "complete blood count": Concept(
        "complete blood count",
        "test",
        ("complete blood count", "cbc", "white blood cell count", "wbc"),
    ),
    "spirometry": Concept("spirometry", "test", ("spirometry", "pulmonary function test", "pulmonary function testing")),
}


SOURCE_TYPE_WEIGHT = {
    "guideline": 1.0,
    "drug_label": 0.95,
    "systematic_review": 0.9,
    "textbook": 0.75,
    "primary_study": 0.6,
    "case_report": 0.35,
    "unknown": 0.5,
}


STRENGTH_WEIGHT = {
    "strong": 1.0,
    "moderate": 0.7,
    "weak": 0.4,
}


HIGH_RISK_CONDITIONS = {
    "acute coronary syndrome": "emergent",
    "pulmonary embolism": "emergent",
    "aortic dissection": "emergent",
    "meningitis": "emergent",
    "pneumonia": "urgent",
    "asthma": "urgent",
}


DANGEROUS_ALTERNATIVES = {
    "chest pain": ("acute coronary syndrome", "pulmonary embolism", "aortic dissection"),
    "dyspnea": ("pulmonary embolism", "acute coronary syndrome", "pneumonia", "asthma"),
    "headache": ("meningitis",),
}


SECTION_KEYWORDS = {
    "diagnostic_criteria": ("diagnosis", "diagnostic", "criteria", "evaluation"),
    "symptoms": ("symptom", "presentation", "history", "manifestation"),
    "physical_exam": ("exam", "physical"),
    "tests": ("test", "laboratory", "imaging", "ecg", "troponin"),
    "differential": ("differential", "mimic", "alternative"),
    "red_flags": ("red flag", "emergency", "urgent", "danger"),
    "treatment": ("treatment", "management", "therapy"),
    "contraindications": ("contraindication", "avoid"),
    "risk_factors": ("risk factor", "risk"),
    "definition": ("definition", "overview"),
}


AMBIGUOUS_ABBREVIATIONS = {
    "pe": ("pulmonary embolism", "physical exam"),
    "mi": ("acute coronary syndrome", "mitral insufficiency"),
}


ABBREVIATION_CONTEXT = {
    "pe": {
        "pulmonary embolism": ("dyspnea", "pleuritic", "chest pain", "ctpa", "embolism", "tachycardia"),
        "physical exam": ("normal", "exam", "inspection", "palpation", "auscultation"),
    },
    "mi": {
        "acute coronary syndrome": ("chest pain", "troponin", "ecg", "infarction", "ischemia"),
        "mitral insufficiency": ("murmur", "mitral", "valve", "regurgitation"),
    },
}
