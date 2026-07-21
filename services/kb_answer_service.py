import os
import re
import time


KB_DIRECT_ANSWER_ENABLED = os.getenv("KB_DIRECT_ANSWER_ENABLED", "true").lower() == "true"


def normalize_text(text):
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

    text = re.sub(r"[^a-z0-9\s/%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_any(text, phrases):
    normalized = normalize_text(text)

    for phrase in phrases:
        if normalize_text(phrase) in normalized:
            return True

    return False


def get_top_score(knowledge_result):
    results = knowledge_result.get("results", []) if knowledge_result else []

    if not results:
        return 0

    return max(item.get("score", 0) for item in results)


def get_intents_from_query(user_query):
    text = normalize_text(user_query)
    intents = set()

    if contains_any(text, [
        "ek surucu",
        "surucu ekle",
        "baskasi kullan",
        "araci baskasi kullanabilir mi",
        "araci kim kullanabilir",
        "ek kisi",
        "ikinci surucu"
    ]):
        intents.add("additional_driver")

    if contains_any(text, [
        "gec teslim",
        "gec iade",
        "gec getir",
        "gec getirirsem",
        "gec kalirsam",
        "gecikme",
        "teslim gec",
        "iade gec"
    ]):
        intents.add("late_return")

    if contains_any(text, [
        "eksik yakit",
        "yakit",
        "benzin",
        "depo",
        "depo eksik",
        "yakit eksik"
    ]):
        intents.add("fuel")

    if contains_any(text, [
        "banka kart",
        "sanal kart",
        "kredi kart",
        "kart kullanabilir miyim",
        "odeme",
        "teminat",
        "depozito",
        "nakit"
    ]):
        intents.add("payment")

    if contains_any(text, [
        "iptal",
        "rezervasyon iptal",
        "erken iade",
        "no show",
        "noshow",
        "gelmezsem",
        "araci erken verirsem"
    ]):
        intents.add("cancel_return")

    if contains_any(text, [
        "kaza",
        "hasar",
        "carparsam",
        "sigorta",
        "guvence",
        "rapor",
        "alkol raporu",
        "polis",
        "jandarma",
        "ariza"
    ]):
        intents.add("damage")

    if contains_any(text, [
        "ehliyet",
        "yas",
        "kac yas",
        "genc surucu",
        "minimum yas",
        "minimum ehliyet"
    ]):
        intents.add("age_license")

    if contains_any(text, [
        "kilometre",
        "km",
        "km asim",
        "kilometre asim",
        "limit"
    ]):
        intents.add("kilometer")

    return intents


def build_result(answer, started_at, model_name="direct_kb_fallback"):
    finished_at = time.perf_counter()

    return {
        "success": True,
        "answer": answer,
        "llm_model": model_name,
        "llm_first_token_ms": 0,
        "llm_total_ms": int((finished_at - started_at) * 1000),
        "error": None
    }


def build_direct_kb_answer(user_query, knowledge_result=None):
    started_at = time.perf_counter()

    if not KB_DIRECT_ANSWER_ENABLED:
        return None

    query_intents = get_intents_from_query(user_query)

    kb_intents = set()

    if knowledge_result:
        kb_intents = set(knowledge_result.get("intent_profiles", []))

    intents = query_intents.union(kb_intents)

    top_score = get_top_score(knowledge_result)

    if not intents and top_score < 30:
        return None

    if "additional_driver" in intents:
        return build_result(
            (
                "Evet, ek sürücü ekleyebilirsiniz. Bunun için ek sürücü hizmetinin satın alınması gerekir. "
                "Dokümana göre aracı yalnızca sözleşme ve teslimat formunda belirtilen kişiler kullanabilir "
                "ve bir araç için en fazla 5 ek sürücü tanımlanabilir."
            ),
            started_at
        )

    if "late_return" in intents:
        return build_result(
            (
                "Aracı geç teslim ederseniz gecikme süresine göre ek ücret yansıtılır. "
                "2 saat üzeri gecikmede günlük kira bedelinin üçte biri, 3 saat ve üzeri gecikmede üçte ikisi, "
                "4 saat ve üzeri gecikmede ise bir günlük kira bedeli alınır. "
                "Ofis kapanış saatindeki iadelerde gecikme opsiyonu geçerli değildir."
            ),
            started_at
        )

    if "fuel" in intents:
        return build_result(
            (
                "Aracı teslim aldığınız yakıt seviyesiyle iade etmeniz gerekir. "
                "Eksik yakıtla iade edilirse eksik yakıt bedeli güncel pompa fiyatı üzerinden hesaplanır. "
                "Buna ek olarak eksik yakıt bedelinin yüzde 40'ı kadar hizmet bedeli alınır."
            ),
            started_at
        )

    if "payment" in intents:
        return build_result(
            (
                "Ödeme için banka kartı veya sanal kart kullanılamaz. "
                "Dokümana göre ödeme kiracının şahsına ait kredi kartından alınır. "
                "Aynı kişiye ait birden fazla kredi kartı kullanılabilir ancak başkasına ait kart kabul edilmez."
            ),
            started_at
        )

    if "cancel_return" in intents:
        return build_result(
            (
                "Rezervasyon, rezervasyon saatine 1 saat kalana kadar ücretsiz iptal edilebilir. "
                "Bu koşul dışında iptal edilirse rezervasyon toplam tutarının yüzde 30'u kadar araç kira ücreti alınır. "
                "Erken iadede kullanılmayan günler için yeniden hesaplama ve ceza uygulanabilir."
            ),
            started_at
        )

    if "damage" in intents:
        return build_result(
            (
                "Kaza yaparsanız önce Garenta yol yardım hattına bilgi vermeniz gerekir. "
                "Aracı yerinden oynatmadan polis veya jandarma raporu alınmalı, aracın fotoğrafları çekilmeli "
                "ve ilgili belgeler en geç 72 saat içinde Garenta'ya iletilmelidir."
            ),
            started_at
        )

    if "age_license" in intents:
        return build_result(
            (
                "Araç grubuna göre minimum yaş ve ehliyet yılı şartları bulunur. "
                "Ekonomi için 21 yaş ve 1 yıl ehliyet, Konfor için 23 yaş ve 2 yıl, "
                "Prestij için 25 yaş ve 3 yıl, Lüks için 27 yaş ve 5 yıl ehliyet şartı vardır. "
                "Şartlar sağlanmıyorsa genç sürücü paketi değerlendirilebilir."
            ),
            started_at
        )

    if "kilometer" in intents:
        return build_result(
            (
                "Araç gruplarının günlük ve aylık kilometre limitleri vardır. "
                "Kiralama başladıktan sonra ek kilometre paketi satın alınamaz. "
                "Limit aşılırsa araç grubuna göre kilometre aşım bedeli yansıtılır."
            ),
            started_at
        )

    return None