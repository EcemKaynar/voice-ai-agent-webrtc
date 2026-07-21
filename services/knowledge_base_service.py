import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base"))
SUPPORTED_EXTENSIONS = [".txt", ".md", ".docx"]

_cached_chunks = None


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

    text = text.replace("’", "'")
    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = re.sub(r"[^a-z0-9\s/%.,:'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_visible_text(text):
    text = str(text or "")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    normalized = normalize_text(text)

    stopwords = {
        "ve", "veya", "ile", "icin", "gibi", "olan", "olarak", "bir", "bu",
        "su", "da", "de", "mi", "mu", "ne", "nasil", "hangi", "kadar",
        "gerekiyor", "olur", "var", "yok", "ben", "bana", "arac", "araci",
        "kiralama", "kiraladigim", "kiraladiginiz", "edersem", "olursa",
        "oluyor", "olacak", "miyim", "musun", "misin"
    }

    tokens = []

    for token in normalized.split():
        if len(token) < 2:
            continue

        if token in stopwords:
            continue

        tokens.append(token)

    return tokens


def read_txt_file(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx_file(path):
    texts = []

    with zipfile.ZipFile(path) as docx:
        xml_content = docx.read("word/document.xml")

    root = ET.fromstring(xml_content)

    namespace = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    }

    paragraphs = root.findall(".//w:p", namespace)

    for paragraph in paragraphs:
        paragraph_text_parts = []

        for text_node in paragraph.findall(".//w:t", namespace):
            if text_node.text:
                paragraph_text_parts.append(text_node.text)

        paragraph_text = clean_visible_text(" ".join(paragraph_text_parts))

        if paragraph_text:
            texts.append(paragraph_text)

    return "\n".join(texts)


def read_knowledge_file(path):
    suffix = path.suffix.lower()

    if suffix in [".txt", ".md"]:
        return read_txt_file(path)

    if suffix == ".docx":
        return read_docx_file(path)

    return ""


SECTION_HEADINGS = [
    "Kullanıcı Bilgisi",
    "Yaş ve Ehliyet Yılı Koşulları",
    "Kiralama Süresi",
    "Olası gecikmeler durumunda",
    "Yakıt",
    "Araç Teslimi",
    "Ofise İade",
    "Adresten Teslim Alma ve Adrese Teslim Etme",
    "Güvenceler ve Ek Ürünler",
    "GÜVENCELER",
    "EK ÜRÜNLER",
    "Trafik Cezaları",
    "Kiralama Sözleşmesi Genel Koşulları",
    "Araç Grupları Kilometre Sınırları",
    "Ödeme Koşulları ve Güvenlik",
    "Teminat Ücreti",
    "Güvenlik",
    "Rezervasyon, Sözleşme ve Değişiklik",
    "Şimdi Öde",
    "Sonra Öde",
    "Erken İade",
    "Uzatma",
    "No-Show Uygulaması",
    "İptal ve İade Uygulaması",
    "Önemli Uyarılar"
]


def looks_like_heading(line):
    visible = clean_visible_text(line)
    normalized = normalize_text(visible)

    if not visible:
        return False

    for heading in SECTION_HEADINGS:
        if normalize_text(heading) in normalized and len(visible) <= 140:
            return True

    if visible.endswith(":") and len(visible) <= 90:
        return True

    return False


def extract_heading(line):
    visible = clean_visible_text(line)
    normalized = normalize_text(visible)

    for heading in SECTION_HEADINGS:
        if normalize_text(heading) in normalized:
            return heading

    return visible.rstrip(":")


def split_long_chunk(text, max_chars=1800):
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []

    for sentence in sentences:
        current.append(sentence)
        joined = " ".join(current).strip()

        if len(joined) >= max_chars:
            chunks.append(joined)
            current = []

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def split_text_to_chunks(text, source_name):
    lines = [
        clean_visible_text(line)
        for line in str(text or "").splitlines()
        if clean_visible_text(line)
    ]

    chunks = []
    current_title = source_name
    current_text_parts = []

    def flush_chunk():
        nonlocal current_text_parts

        chunk_text = "\n".join(current_text_parts).strip()

        if chunk_text:
            for part in split_long_chunk(chunk_text):
                chunks.append({
                    "source": source_name,
                    "title": current_title,
                    "text": part
                })

        current_text_parts = []

    for line in lines:
        normalized = normalize_text(line)

        if "olasi gecikmeler durumunda" in normalized:
            flush_chunk()
            current_title = "Olası gecikmeler durumunda"
            current_text_parts.append(line)
            continue

        if looks_like_heading(line):
            flush_chunk()
            current_title = extract_heading(line)
            continue

        current_text_parts.append(line)

    flush_chunk()

    final_chunks = []

    for index, chunk in enumerate(chunks, start=1):
        final_chunks.append({
            "id": f"{source_name}_{index}",
            "source": chunk["source"],
            "title": chunk["title"],
            "text": chunk["text"],
            "normalized_title": normalize_text(chunk["title"]),
            "normalized_text": normalize_text(chunk["text"]),
            "tokens": tokenize(chunk["title"] + " " + chunk["text"])
        })

    return final_chunks


def get_intent_profile(query):
    text = normalize_text(query)

    profiles = []

    late_return_triggers = [
        "gec teslim",
        "gec iade",
        "gec getir",
        "gec getirirsem",
        "gec kalirsam",
        "gecikme",
        "teslim gec",
        "iade gec",
        "gec teslim edersem"
    ]

    if any(trigger in text for trigger in late_return_triggers):
        profiles.append({
            "name": "late_return",
            "positive_phrases": [
                "olasi gecikmeler",
                "gecikme durumunda",
                "2 saat uzeri",
                "3 saat ve uzeri",
                "4 saat ve uzeri",
                "kiralama suresi",
                "teslim ya da iade saatiniz",
                "kira bedelinin 1/3",
                "kira bedelinizin 2/3"
            ],
            "positive_tokens": [
                "gecikme", "gec", "saat", "teslim", "iade", "bedel",
                "kiralama", "suresi", "1/3", "2/3"
            ],
            "negative_tokens": [
                "kaza", "hasar", "guvenlik", "odeme", "teminat", "kilometre",
                "yakit", "trafik", "ceza", "polis", "jandarma", "kredi", "kart"
            ]
        })

    fuel_triggers = ["yakit", "yakıt", "benzin", "depo", "eksik yakit", "eksik yakıt"]

    if any(trigger in text for trigger in fuel_triggers):
        profiles.append({
            "name": "fuel",
            "positive_phrases": [
                "araci teslim aldiginiz yakit seviyesi",
                "eksik yakit",
                "%40",
                "hizmet bedeli"
            ],
            "positive_tokens": ["yakit", "eksik", "depo", "hizmet", "bedeli", "%40"],
            "negative_tokens": ["kaza", "hasar", "ehliyet", "yas", "odeme", "teminat"]
        })

    additional_driver_triggers = [
        "ek surucu",
        "ek sürücü",
        "baskasi kullan",
        "başkası kullan",
        "araci kim kullanabilir",
        "aracı kim kullanabilir",
        "surucu ekle",
        "sürücü ekle"
    ]

    if any(trigger in text for trigger in additional_driver_triggers):
        profiles.append({
            "name": "additional_driver",
            "positive_phrases": [
                "ek surucu",
                "sozlesme ve teslimat formunda belirtilen",
                "en fazla 5 adet ek surucu"
            ],
            "positive_tokens": ["ek", "surucu", "kullanici", "sozlesme", "teslimat", "5"],
            "negative_tokens": ["yakit", "gecikme", "odeme", "teminat", "kaza"]
        })

    age_license_triggers = [
        "ehliyet",
        "yas",
        "yaş",
        "genc surucu",
        "genç sürücü",
        "kac yas",
        "kaç yaş"
    ]

    if any(trigger in text for trigger in age_license_triggers):
        profiles.append({
            "name": "age_license",
            "positive_phrases": [
                "yas ve ehliyet",
                "minimum surucu yasi",
                "minimum ehliyet yili",
                "genc surucu"
            ],
            "positive_tokens": ["yas", "ehliyet", "surucu", "genc", "ekonomi", "konfor", "prestij", "luks"],
            "negative_tokens": ["yakit", "gecikme", "kaza", "odeme"]
        })

    payment_triggers = [
        "odeme",
        "ödeme",
        "kredi kart",
        "banka kart",
        "sanal kart",
        "teminat",
        "depozito"
    ]

    if any(trigger in text for trigger in payment_triggers):
        profiles.append({
            "name": "payment",
            "positive_phrases": [
                "odeme kosullari",
                "kredi kart",
                "banka karti ve sanal kart",
                "teminat ucreti",
                "7 is gunu"
            ],
            "positive_tokens": ["odeme", "kredi", "kart", "teminat", "banka", "sanal", "tahsil", "iade"],
            "negative_tokens": ["kaza", "hasar", "yakit", "gecikme"]
        })

    cancel_triggers = [
        "iptal",
        "erken iade",
        "no show",
        "noshow",
        "rezervasyon iptal"
    ]

    if any(trigger in text for trigger in cancel_triggers):
        profiles.append({
            "name": "cancel_return",
            "positive_phrases": [
                "iptal ve iade",
                "erken iade",
                "no-show",
                "rezervasyon saatine 1 saat kalana kadar",
                "%30"
            ],
            "positive_tokens": ["iptal", "iade", "erken", "no", "show", "rezervasyon", "%30"],
            "negative_tokens": ["yakit", "kaza", "ehliyet", "yas"]
        })

    damage_triggers = [
        "hasar",
        "kaza",
        "sigorta",
        "guvence",
        "güvence",
        "polis raporu",
        "alkol raporu",
        "ariza",
        "arıza"
    ]

    if any(trigger in text for trigger in damage_triggers):
        profiles.append({
            "name": "damage",
            "positive_phrases": [
                "kaza halinde",
                "polis raporu",
                "alkol raporu",
                "hasar",
                "guvence"
            ],
            "positive_tokens": ["hasar", "kaza", "rapor", "alkol", "polis", "jandarma", "guvence"],
            "negative_tokens": ["yakit", "gecikme", "odeme", "teminat"]
        })

    return profiles


def score_chunk(query_tokens, query_text, profiles, chunk):
    chunk_tokens = chunk.get("tokens", [])
    chunk_token_set = set(chunk_tokens)
    normalized_title = chunk.get("normalized_title", "")
    normalized_text = chunk.get("normalized_text", "")
    full_text = normalized_title + " " + normalized_text

    if not query_tokens:
        return 0

    score = 0

    for token in query_tokens:
        if token in chunk_token_set:
            score += 3

        if token in normalized_title:
            score += 5

    for profile in profiles:
        for phrase in profile.get("positive_phrases", []):
            if normalize_text(phrase) in full_text:
                score += 18

        for token in profile.get("positive_tokens", []):
            normalized_token = normalize_text(token)

            if normalized_token in chunk_token_set or normalized_token in full_text:
                score += 5

        for token in profile.get("negative_tokens", []):
            normalized_token = normalize_text(token)

            if normalized_token in chunk_token_set:
                score -= 8

    if "gec teslim" in query_text or "gecikme" in query_text:
        if "olasi gecikmeler" in full_text:
            score += 40

        if "kiralama suresi" in full_text:
            score += 25

        if "2 saat" in full_text and "3 saat" in full_text and "4 saat" in full_text:
            score += 40

        if "en gec 72 saat" in full_text:
            score -= 35

        if "kaza" in full_text or "polis" in full_text or "jandarma" in full_text:
            score -= 25

    return max(score, 0)


def load_knowledge_base(force_reload=False):
    global _cached_chunks

    if _cached_chunks is not None and not force_reload:
        return _cached_chunks

    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for path in KNOWLEDGE_BASE_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = read_knowledge_file(path)

        if not text.strip():
            continue

        chunks = split_text_to_chunks(
            text=text,
            source_name=path.name
        )

        all_chunks.extend(chunks)

    _cached_chunks = all_chunks

    print(f"Knowledge base yüklendi. Chunk sayısı: {len(all_chunks)}")

    return _cached_chunks


def search_knowledge(query, top_k=4):
    chunks = load_knowledge_base()
    query_tokens = tokenize(query)
    query_text = normalize_text(query)
    profiles = get_intent_profile(query)

    scored_chunks = []

    for chunk in chunks:
        score = score_chunk(
            query_tokens=query_tokens,
            query_text=query_text,
            profiles=profiles,
            chunk=chunk
        )

        if score > 0:
            scored_chunks.append({
                "score": score,
                "id": chunk["id"],
                "source": chunk["source"],
                "title": chunk["title"],
                "text": chunk["text"]
            })

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    selected = scored_chunks[:top_k]

    context_parts = []

    for index, item in enumerate(selected, start=1):
        context_parts.append(
            f"[Bilgi {index}]\n"
            f"Kaynak: {item['source']}\n"
            f"Başlık: {item['title']}\n"
            f"İçerik:\n{item['text']}"
        )

    return {
        "query": query,
        "found": len(selected) > 0,
        "intent_profiles": [profile["name"] for profile in profiles],
        "chunk_count": len(chunks),
        "results": selected,
        "context": "\n\n---\n\n".join(context_parts)
    }


def get_knowledge_base_status():
    chunks = load_knowledge_base()

    sources = {}

    for chunk in chunks:
        source = chunk.get("source", "unknown")

        if source not in sources:
            sources[source] = 0

        sources[source] += 1

    return {
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "supported_extensions": SUPPORTED_EXTENSIONS,
        "chunk_count": len(chunks),
        "sources": sources
    }


def reload_knowledge_base():
    chunks = load_knowledge_base(force_reload=True)

    return {
        "success": True,
        "status": get_knowledge_base_status(),
        "chunks_reloaded": len(chunks)
    }