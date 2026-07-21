import re
from difflib import SequenceMatcher


def normalize_for_matching(text):
    text = str(text or "").lower()

    replacements = {
        "ı": "i",
        "İ": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_for_matching(a),
        normalize_for_matching(b)
    ).ratio()


def contains_any(text, phrases):
    normalized = normalize_for_matching(text)

    return any(
        normalize_for_matching(phrase) in normalized
        for phrase in phrases
    )


def normalize_transcript_for_domain(transcript):
    original = str(transcript or "").strip()
    text = normalize_for_matching(original)

    result = {
        "original_transcript": original,
        "normalized_query": original,
        "correction_applied": False,
        "correction_reason": None,
        "confidence": 0
    }

    if not text:
        return result

    if (
        "surucu" in text
        or "suruculer" in text
        or "seyirci" in text
        or "seyirciler" in text
        or ("yuru" in text and ("et" in text or "ek" in text))
        or text.count("yuru") >= 2
    ):
        result.update({
            "normalized_query": "Ek sürücü ekleyebilir miyim?",
            "correction_applied": True,
            "correction_reason": "additional_driver_domain_correction",
            "confidence": 90
        })
        return result

    if (
        contains_any(text, [
            "gec teslim",
            "gec iade",
            "gec getir",
            "gec getirirsem",
            "gec kalirsam",
            "gecikme",
            "teslim gec",
            "iade gec"
        ])
        or ("gec" in text and ("teslim" in text or "iade" in text or "getir" in text or "kisim" in text))
    ):
        result.update({
            "normalized_query": "Aracı geç teslim edersem ne olur?",
            "correction_applied": True,
            "correction_reason": "late_return_domain_correction",
            "confidence": 90
        })
        return result

    if contains_any(text, ["yakit", "benzin", "depo", "eksik yakit"]):
        result.update({
            "normalized_query": "Aracı eksik yakıtla iade edersem ne olur?",
            "correction_applied": True,
            "correction_reason": "fuel_domain_correction",
            "confidence": 85
        })
        return result

    if contains_any(text, ["teminat", "depozito", "kredi kart", "banka kart", "sanal kart", "odeme"]):
        result.update({
            "normalized_query": "Ödeme ve teminat koşulları nelerdir?",
            "correction_applied": True,
            "correction_reason": "payment_domain_correction",
            "confidence": 85
        })
        return result

    if contains_any(text, ["iptal", "erken iade", "no show", "noshow", "rezervasyon iptal"]):
        result.update({
            "normalized_query": "Rezervasyon iptal ve iade koşulları nelerdir?",
            "correction_applied": True,
            "correction_reason": "cancel_return_domain_correction",
            "confidence": 85
        })
        return result

    if contains_any(text, ["hasar", "kaza", "sigorta", "guvence", "rapor", "alkol raporu", "polis"]):
        result.update({
            "normalized_query": "Kaza veya hasar durumunda ne yapılmalı?",
            "correction_applied": True,
            "correction_reason": "damage_domain_correction",
            "confidence": 85
        })
        return result

    candidates = [
        ("Ek sürücü ekleyebilir miyim?", "fuzzy_additional_driver"),
        ("Aracı geç teslim edersem ne olur?", "fuzzy_late_return"),
        ("Aracı eksik yakıtla iade edersem ne olur?", "fuzzy_fuel"),
        ("Ödeme ve teminat koşulları nelerdir?", "fuzzy_payment"),
        ("Rezervasyon iptal ve iade koşulları nelerdir?", "fuzzy_cancel"),
        ("Kaza veya hasar durumunda ne yapılmalı?", "fuzzy_damage")
    ]

    best_query = original
    best_reason = None
    best_score = 0

    for candidate, reason in candidates:
        score = similarity(text, candidate)

        if score > best_score:
            best_score = score
            best_query = candidate
            best_reason = reason

    if best_score >= 0.58:
        result.update({
            "normalized_query": best_query,
            "correction_applied": True,
            "correction_reason": best_reason,
            "confidence": int(best_score * 100)
        })

    return result