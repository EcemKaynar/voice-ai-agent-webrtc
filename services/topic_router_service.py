import re


TOPIC_DEFINITIONS = {
    "additional_driver": {
        "label": "Ek sürücü",
        "query_phrases": [
            "ek sürücü",
            "ek surucu",
            "sürücü ekle",
            "surucu ekle",
            "başka sürücü",
            "baska surucu",
            "ikinci sürücü",
            "ikinci surucu",
            "aracı başkası",
            "araci baskasi",
            "arabayı başkası",
            "arabayi baskasi",
            "aracı kim kullanabilir",
            "araci kim kullanabilir"
        ],
        "content_phrases": [
            "ek sürücü",
            "ek surucu",
            "en fazla 5 adet ek sürücü",
            "en fazla 5 adet ek surucu",
            "sözleşme ve teslimat formunda belirtilen",
            "sozlesme ve teslimat formunda belirtilen"
        ],
        "expanded_query": (
            "ek sürücü sözleşme teslimat formu aracı kullanabilir "
            "en fazla 5 adet ek sürücü"
        )
    },

    "late_return": {
        "label": "Geç teslim / geç iade",
        "query_phrases": [
            "geç teslim",
            "gec teslim",
            "geç iade",
            "gec iade",
            "geç kalırsam",
            "gec kalirsam",
            "gecikme",
            "geç getirirsem",
            "gec getirirsem",
            "teslim geç",
            "teslim gec",
            "iade geç",
            "iade gec"
        ],
        "content_phrases": [
            "olası gecikmeler",
            "olasi gecikmeler",
            "2 saat üzeri",
            "2 saat uzeri",
            "3 saat ve üzeri",
            "3 saat ve uzeri",
            "4 saat ve üzeri",
            "4 saat ve uzeri",
            "kira bedelinin 1/3",
            "kira bedelinizin 2/3",
            "bir günlük kira bedeli",
            "bir gunluk kira bedeli"
        ],
        "expanded_query": (
            "olası gecikmeler 2 saat üzeri 3 saat ve üzeri 4 saat ve üzeri "
            "kira bedelinin 1/3 kira bedelinin 2/3 bir günlük kira bedeli"
        )
    },

    "payment": {
        "label": "Ödeme / kart / teminat",
        "query_phrases": [
            "ödeme",
            "odeme",
            "banka kartı",
            "banka kart",
            "sanal kart",
            "debit kart",
            "kredi kartı",
            "kredi kart",
            "kart kullan",
            "teminat",
            "depozito"
        ],
        "content_phrases": [
            "ödeme koşulları",
            "odeme kosullari",
            "banka kartı ve sanal kart geçerli değildir",
            "banka karti ve sanal kart gecerli degildir",
            "şahsi kredi kartı",
            "sahsi kredi karti",
            "kredi kartından tahsil",
            "kredi kartindan tahsil",
            "teminat ücreti",
            "teminat ucreti"
        ],
        "expanded_query": (
            "ödeme koşulları banka kartı sanal kart geçerli değildir "
            "şahsi kredi kartı teminat tahsil"
        )
    },

    "fuel": {
        "label": "Yakıt",
        "query_phrases": [
            "yakıt",
            "yakit",
            "benzin",
            "depo",
            "eksik yakıt",
            "eksik yakit"
        ],
        "content_phrases": [
            "yakıt",
            "yakit",
            "eksik yakıt",
            "eksik yakit",
            "%40",
            "hizmet bedeli",
            "teslim aldığınız yakıt seviyesi",
            "teslim aldiginiz yakit seviyesi"
        ],
        "expanded_query": (
            "eksik yakıt teslim aldığınız yakıt seviyesi yakıt bedeli "
            "%40 hizmet bedeli"
        )
    },

    "accident_damage": {
        "label": "Kaza / hasar",
        "query_phrases": [
            "kaza",
            "hasar",
            "aracım bozuldu",
            "aracim bozuldu",
            "arıza",
            "ariza",
            "polis raporu",
            "jandarma",
            "alkol raporu",
            "yol yardım",
            "yol yardim",
            "acil yardım",
            "acil yardim"
        ],
        "content_phrases": [
            "kaza",
            "hasar",
            "polis",
            "jandarma",
            "alkol raporu",
            "yol yardım",
            "yol yardim",
            "0850",
            "72 saat"
        ],
        "expanded_query": (
            "kaza hasar polis jandarma alkol raporu yol yardım 0850 72 saat"
        )
    },

    "segment_conditions": {
        "label": "Segment / yaş / ehliyet / genç sürücü",
        "query_phrases": [
            "segment",
            "lüks segment",
            "luks segment",
            "ekonomi segment",
            "konfor segment",
            "prestij segment",
            "genç sürücü",
            "genc surucu",
            "yaş şartı",
            "yas sarti",
            "ehliyet yılı",
            "ehliyet yili",
            "findeks",
            "kilometre limiti",
            "km limiti"
        ],
        "content_phrases": [
            "segment koşulları",
            "segment kosullari",
            "minimum sürücü yaşı",
            "minimum surucu yasi",
            "genç sürücü yaşı",
            "genc surucu yasi",
            "minimum ehliyet yılı",
            "minimum ehliyet yili",
            "findeks koşulu",
            "findeks kosulu"
        ],
        "expanded_query": (
            "segment koşulları minimum sürücü yaşı genç sürücü yaşı "
            "minimum ehliyet yılı genç sürücü ehliyet yılı Findeks"
        )
    },

    "cancellation_return": {
        "label": "İptal / iade / no-show",
        "query_phrases": [
            "iptal",
            "iade",
            "erken iade",
            "no show",
            "no-show",
            "rezervasyon iptal"
        ],
        "content_phrases": [
            "iptal ve iade",
            "erken iade",
            "no-show",
            "no show",
            "rezervasyonunuz",
            "%30"
        ],
        "expanded_query": "iptal iade erken iade no-show rezervasyon iptal"
    }
}


def normalize_topic_text(text):
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

    text = re.sub(r"[^a-z0-9\s/%.,:'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def score_phrases(normalized_text, phrases, weight):
    score = 0

    for phrase in phrases:
        normalized_phrase = normalize_topic_text(phrase)

        if normalized_phrase and normalized_phrase in normalized_text:
            score += weight

    return score


def infer_query_topic(query):
    normalized_query = normalize_topic_text(query)

    scores = {}

    for topic, definition in TOPIC_DEFINITIONS.items():
        score = 0
        score += score_phrases(
            normalized_query,
            definition.get("query_phrases", []),
            weight=10
        )

        for word in normalized_query.split():
            if len(word) < 4:
                continue

            for phrase in definition.get("query_phrases", []):
                if word in normalize_topic_text(phrase):
                    score += 1

        scores[topic] = score

    # Karışmayı azaltan özel topic ayırımları.
    # Bu cevap hardcode değil; sadece routing önceliği.
    if "ek surucu" in normalized_query:
        scores["additional_driver"] += 25
        scores["segment_conditions"] -= 20

    if "genc surucu" in normalized_query:
        scores["segment_conditions"] += 25
        scores["additional_driver"] -= 20

    if "gec teslim" in normalized_query or "gec iade" in normalized_query:
        scores["late_return"] += 25
        scores["fuel"] -= 20

    if "eksik yakit" in normalized_query:
        scores["fuel"] += 25
        scores["late_return"] -= 20

    if "banka kart" in normalized_query or "sanal kart" in normalized_query:
        scores["payment"] += 25

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score < 8:
        return {
            "topic": None,
            "score": best_score,
            "scores": scores,
            "expanded_query": ""
        }

    return {
        "topic": best_topic,
        "score": best_score,
        "scores": scores,
        "expanded_query": TOPIC_DEFINITIONS[best_topic].get("expanded_query", "")
    }


def infer_chunk_topic(title, content, structured=False, segment=None):
    if structured or segment:
        return {
            "topic": "segment_conditions",
            "score": 100
        }

    normalized_text = normalize_topic_text(f"{title} {content}")
    scores = {}

    for topic, definition in TOPIC_DEFINITIONS.items():
        score = 0

        score += score_phrases(
            normalized_text,
            definition.get("content_phrases", []),
            weight=10
        )

        score += score_phrases(
            normalized_text,
            definition.get("query_phrases", []),
            weight=4
        )

        scores[topic] = score

    if "ek surucu" in normalized_text:
        scores["additional_driver"] += 30
        scores["segment_conditions"] -= 15

    if "olasi gecikmeler" in normalized_text:
        scores["late_return"] += 30
        scores["fuel"] -= 15

    if "banka karti ve sanal kart" in normalized_text:
        scores["payment"] += 30

    if "eksik yakit" in normalized_text:
        scores["fuel"] += 30
        scores["late_return"] -= 15

    best_topic = max(scores, key=scores.get)
    best_score = scores[best_topic]

    if best_score < 6:
        return {
            "topic": None,
            "score": best_score
        }

    return {
        "topic": best_topic,
        "score": best_score
    }


def should_allow_chunk_for_topic(query_topic, chunk_topic, semantic_score, lexical_score):
    if not query_topic:
        return True

    if not chunk_topic:
        return True

    if query_topic == chunk_topic:
        return True

    # Ek sürücü sorusunda yaş/ehliyet segment chunk'ları çok karıştırıyordu.
    if query_topic == "additional_driver" and chunk_topic == "segment_conditions":
        return False

    # Geç teslim sorusunda yakıt teslimi karışıyordu.
    if query_topic == "late_return" and chunk_topic == "fuel":
        return False

    # Yakıt sorusunda geç teslim karışmasın.
    if query_topic == "fuel" and chunk_topic == "late_return":
        return False

    # Çok güçlü semantic eşleşme varsa yine de tamamen engelleme.
    if semantic_score >= 0.82 or lexical_score >= 0.55:
        return True

    return False