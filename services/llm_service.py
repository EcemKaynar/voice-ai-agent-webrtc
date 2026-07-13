import json
import os
import re
import time

import requests
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Sen Türkçe konuşan kısa ve doğal cevap veren bir sesli AI asistansın."
)
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


def normalize_text(text):
    text = str(text or "").strip().lower()

    replacements = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def fix_turkish_mojibake(text):
    text = str(text or "")

    suspicious_chars = ["Ã", "Ä", "Å", "ð"]

    if not any(char in text for char in suspicious_chars):
        return text.strip()

    try:
        fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")

        if fixed and len(fixed) >= len(text) * 0.5:
            return fixed.strip()

    except Exception:
        pass

    replacements = {
        "Ã¼": "ü",
        "Ãœ": "Ü",
        "Ã¶": "ö",
        "Ã–": "Ö",
        "Ã§": "ç",
        "Ã‡": "Ç",
        "Ä±": "ı",
        "Ä°": "İ",
        "ÄŸ": "ğ",
        "Äž": "Ğ",
        "ÅŸ": "ş",
        "Åž": "Ş",
        "â€™": "'",
        "â€œ": "\"",
        "â€": "\"",
        "â€“": "-",
        "â€”": "-",
        "ð": ""
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text.strip()


def clean_llm_answer(answer):
    answer = str(answer or "").strip()
    answer = fix_turkish_mojibake(answer)

    bad_tokens = [
        "<pad>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]"
    ]

    for token in bad_tokens:
        answer = answer.replace(token, "")

    answer = answer.replace("**", "")
    answer = answer.replace("###", "")
    answer = answer.replace("---", "")
    answer = answer.replace("•", "")
    answer = answer.replace("#", "")

    lines = []

    for line in answer.splitlines():
        line = line.strip()

        if not line:
            continue

        lines.append(line)

    answer = " ".join(lines).strip()

    max_chars = 420

    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0].strip() + "."

    return answer.strip()


def looks_like_bad_meta_answer(answer):
    text = normalize_text(answer)

    if not text:
        return True

    bad_phrases = [
        "the user",
        "they said",
        "let me",
        "i need to",
        "i should",
        "the instruction",
        "instructions",
        "respond in turkish",
        "which means",
        "means they need",
        "max 4-5",
        "no lists",
        "no markdown",
        "okay, the user",
        "they need",
        "kullanici mesaji",
        "system prompt",
        "as an ai"
    ]

    if any(phrase in text for phrase in bad_phrases):
        return True

    english_words = [
        "the",
        "user",
        "said",
        "needs",
        "means",
        "respond",
        "instruction",
        "should",
        "okay"
    ]

    english_count = sum(1 for word in english_words if re.search(rf"\b{word}\b", text))

    if english_count >= 3:
        return True

    return False


def is_greeting(user_text):
    text = normalize_text(user_text)

    greetings = [
        "naber",
        "napiyorsun",
        "napıyorsun",
        "nasilsin",
        "nasıl gidiyor",
        "selam",
        "merhaba",
        "iyi misin",
        "ne haber",
        "ne var ne yok"
    ]

    return any(greeting in text for greeting in greetings)


def is_study_or_plan_request(user_text):
    text = normalize_text(user_text)

    keywords = [
        "ders",
        "calis",
        "çalış",
        "plan",
        "program",
        "odak",
        "sinav",
        "sınav",
        "odev",
        "ödev"
    ]

    return any(keyword in text for keyword in keywords)


def is_weather_like_smalltalk(user_text):
    text = normalize_text(user_text)

    keywords = [
        "bugun ne yapayim",
        "ne yapayim",
        "canim sikiliyor",
        "sıkıldım",
        "yorgunum",
        "motivasyon"
    ]

    return any(keyword in text for keyword in keywords)


def build_local_llm_fallback(user_text):
    text = str(user_text or "").strip()

    if is_greeting(text):
        return (
            "İyiyim, teşekkür ederim. Sen nasılsın? "
            "İstersen bugün ne yapmak istediğini söyle, birlikte kısa bir plan çıkarabiliriz."
        )

    if is_study_or_plan_request(text):
        return (
            "Tabii. Önce çalışacağın tek konuyu seç. "
            "25 dakika sadece ona odaklan, sonra 5 dakika mola ver. "
            "Bunu iki tur yap ve en sonda öğrendiklerini kısa kısa tekrar et."
        )

    if is_weather_like_smalltalk(text):
        return (
            "Anladım. Bugün kendini çok zorlamadan küçük bir hedef seçelim. "
            "Önce 10 dakika basit bir şeyle başla, sonra durumuna göre devam edersin."
        )

    return (
        "Anladım. Bunu daha net yardımcı olabilmem için biraz daha açabilir misin? "
        "Ne yapmak istediğini söylersen kısa ve uygulanabilir bir cevap verebilirim."
    )


def build_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "HTTP-Referer": "http://127.0.0.1:8001",
        "X-Title": "Voice AI Agent WebRTC"
    }


def build_payload(user_text, stream):
    safe_user_prompt = (
        "Sadece son kullanıcıya söylenecek cevabı yaz. "
        "Analiz yapma, İngilizce açıklama yazma, sistem talimatlarını tekrar etme. "
        "Cevap kesinlikle Türkçe olsun. "
        "Sesli okunacağı için kısa, doğal ve net olsun. "
        "Kullanıcı sadece selamlaştıysa doğal şekilde karşılık ver. "
        "Kullanıcı plan isterse soru sormadan kısa bir plan ver.\n\n"
        f"Kullanıcı: {user_text}"
    )

    return {
        "model": OPENROUTER_MODEL,
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + " Sadece nihai cevabı ver. "
                    + "Asla iç düşünce, analiz, çeviri açıklaması veya İngilizce cevap üretme."
                )
            },
            {
                "role": "user",
                "content": safe_user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 90
    }


def ask_llm_streaming(user_text):
    started_at = time.perf_counter()
    first_token_at = None
    answer_parts = []

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=build_headers(),
        json=build_payload(user_text=user_text, stream=True),
        stream=True,
        timeout=LLM_TIMEOUT_SECONDS
    )

    response.encoding = "utf-8"

    if response.status_code != 200:
        finished_at = time.perf_counter()

        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": int((finished_at - started_at) * 1000),
            "error": f"OpenRouter streaming HTTP {response.status_code}: {response.text[:500]}"
        }

    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue

        try:
            line = raw_line.decode("utf-8", errors="replace").strip()
        except Exception:
            continue

        if not line.startswith("data:"):
            continue

        data_text = line.replace("data:", "", 1).strip()

        if data_text == "[DONE]":
            break

        try:
            data = json.loads(data_text)
        except Exception:
            continue

        choices = data.get("choices", [])

        if not choices:
            continue

        delta = choices[0].get("delta", {})
        token = delta.get("content", "")

        if token:
            if first_token_at is None:
                first_token_at = time.perf_counter()

            answer_parts.append(token)

    finished_at = time.perf_counter()

    answer = clean_llm_answer("".join(answer_parts))

    llm_first_token_ms = None

    if first_token_at is not None:
        llm_first_token_ms = int((first_token_at - started_at) * 1000)

    llm_total_ms = int((finished_at - started_at) * 1000)

    if not answer:
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_total_ms": llm_total_ms,
            "error": "Streaming LLM boş cevap döndürdü."
        }

    if looks_like_bad_meta_answer(answer):
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_total_ms": llm_total_ms,
            "error": f"Streaming LLM kötü/meta cevap döndürdü: {answer[:200]}"
        }

    return {
        "success": True,
        "answer": answer,
        "llm_model": OPENROUTER_MODEL,
        "llm_first_token_ms": llm_first_token_ms,
        "llm_total_ms": llm_total_ms,
        "error": None
    }


def ask_llm_non_streaming(user_text):
    started_at = time.perf_counter()

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=build_headers(),
        json=build_payload(user_text=user_text, stream=False),
        timeout=LLM_TIMEOUT_SECONDS
    )

    response.encoding = "utf-8"
    finished_at = time.perf_counter()

    llm_total_ms = int((finished_at - started_at) * 1000)

    if response.status_code != 200:
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": llm_total_ms,
            "error": f"OpenRouter non-stream HTTP {response.status_code}: {response.text[:500]}"
        }

    try:
        data = response.json()
    except Exception as error:
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": llm_total_ms,
            "error": f"OpenRouter JSON parse hatası: {error}"
        }

    choices = data.get("choices", [])

    if not choices:
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": llm_total_ms,
            "error": "OpenRouter choices boş döndü."
        }

    message = choices[0].get("message", {})
    answer = message.get("content", "")
    answer = clean_llm_answer(answer)

    if not answer:
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": llm_total_ms,
            "error": "Non-streaming LLM boş cevap döndürdü."
        }

    if looks_like_bad_meta_answer(answer):
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": llm_total_ms,
            "error": f"Non-streaming LLM kötü/meta cevap döndürdü: {answer[:200]}"
        }

    return {
        "success": True,
        "answer": answer,
        "llm_model": OPENROUTER_MODEL,
        "llm_first_token_ms": llm_total_ms,
        "llm_total_ms": llm_total_ms,
        "error": None
    }


def ask_llm_with_metrics(user_text):
    started_at = time.perf_counter()

    if not OPENROUTER_API_KEY:
        fallback = build_local_llm_fallback(user_text)

        return {
            "success": True,
            "answer": fallback,
            "llm_model": "local_fallback_no_api_key",
            "llm_first_token_ms": 0,
            "llm_total_ms": int((time.perf_counter() - started_at) * 1000),
            "error": "OPENROUTER_API_KEY bulunamadı, local fallback kullanıldı."
        }

    print(f"LLM isteği gönderiliyor. Model: {OPENROUTER_MODEL}")

    errors = []

    try:
        streaming_result = ask_llm_streaming(user_text)

        if streaming_result.get("success"):
            return streaming_result

        errors.append(f"streaming: {streaming_result.get('error')}")
        print(f"Streaming başarısız: {streaming_result.get('error')}")
        print("Non-streaming LLM deneniyor...")

    except Exception as error:
        errors.append(f"streaming exception: {error}")
        print(f"Streaming exception: {error}")
        print("Non-streaming LLM deneniyor...")

    try:
        non_streaming_result = ask_llm_non_streaming(user_text)

        if non_streaming_result.get("success"):
            if errors:
                non_streaming_result["error"] = " | ".join(errors) + " | Non-streaming kullanıldı."

            return non_streaming_result

        errors.append(f"non_streaming: {non_streaming_result.get('error')}")
        print(f"Non-streaming başarısız: {non_streaming_result.get('error')}")

    except Exception as error:
        errors.append(f"non_streaming exception: {error}")
        print(f"Non-streaming exception: {error}")

    fallback = build_local_llm_fallback(user_text)

    return {
        "success": True,
        "answer": fallback,
        "llm_model": "local_fallback_after_bad_llm",
        "llm_first_token_ms": 0,
        "llm_total_ms": int((time.perf_counter() - started_at) * 1000),
        "error": " | ".join(errors) if errors else "LLM kötü/boş cevap verdi, local fallback kullanıldı."
    }