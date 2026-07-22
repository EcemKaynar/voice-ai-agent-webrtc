import json
import os
import re
import time

import requests
from dotenv import load_dotenv

from services.prompt_service import build_voice_agent_prompt


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Sen Türkçe konuşan kısa ve doğal cevap veren bir sesli AI asistansın."
)
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))


def normalize_text(text):
    text = str(text or "").strip().lower()

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

    text = re.sub(r"\s+", " ", text).strip()

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

    remove_tokens = [
        "<pad>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "DOĞAL ASİSTAN CEVABI:",
        "Doğal asistan cevabı:",
        "CEVAP:",
        "Cevap:",
        "ANSWER:",
        "Answer:"
    ]

    for token in remove_tokens:
        answer = answer.replace(token, "")

    answer = answer.replace("**", "")
    answer = answer.replace("###", "")
    answer = answer.replace("---", "")

    lines = []

    for line in answer.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.lower().startswith("cevap:"):
            line = line[6:].strip()

        lines.append(line)

    answer = " ".join(lines).strip()
    answer = re.sub(r"\s+", " ", answer).strip()

    max_chars = 620

    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0].strip() + "."

    return answer.strip()


def looks_like_incomplete_answer(answer):
    text = str(answer or "").strip()

    if not text:
        return True

    if text[-1] not in ".?!":
        return True

    normalized = normalize_text(text).rstrip(".?! ")

    dangling_endings = (
        " kullan",
        " kullanabilir",
        " kullanamaz",
        " gerekmektedir ve",
        " gerekir ve",
        " ancak",
        " çünkü",
        " veya",
        " ve",
        " ile",
        " için",
        " olarak",
        " ayrıca",
        " buna göre"
    )

    return normalized.endswith(dangling_endings)


def looks_like_bad_meta_answer(answer):
    text_raw = str(answer or "").strip()
    text = normalize_text(text_raw)

    if not text:
        return True

    bad_phrases = [
        "we need",
        "we must",
        "we should",
        "user asks",
        "the user asks",
        "the user asked",
        "user said",
        "the user said",
        "provided info",
        "the info includes",
        "according to rules",
        "must answer",
        "answer based on",
        "answer according to",
        "using only info",
        "must not copy",
        "must answer directly",
        "summarize payment",
        "important condition",
        "thus",
        "actually",
        "let me",
        "i need to",
        "i should",
        "system prompt",
        "developer message",
        "knowledge base",
        "bilgi tabani",
        "bilgi tabanı",
        "kullanici sorusu",
        "kullanıcı sorusu",
        "dogal asistan cevabi",
        "doğal asistan cevabı",
        "başlık:",
        "baslik:",
        "kaynak:",
        "icerik:",
        "içerik:",
        "[bilgi",
        "as an ai",
        "respond in turkish",
        "in turkish",
        "turkish, short",
        "2-4 sentences",
        "no headings"
    ]

    if any(phrase in text for phrase in bad_phrases):
        return True

    starts_bad = [
        "we ",
        "the user ",
        "user ",
        "must ",
        "according to ",
        "let's ",
        "i need ",
        "i should "
    ]

    if any(text.startswith(item) for item in starts_bad):
        return True

    english_words = [
        "the",
        "user",
        "asks",
        "asked",
        "must",
        "should",
        "answer",
        "according",
        "provided",
        "info",
        "condition",
        "summarize",
        "directly",
        "rules",
        "thus",
        "actually"
    ]

    english_count = sum(
        1
        for word in english_words
        if re.search(rf"\b{word}\b", text)
    )

    if english_count >= 4:
        return True

    if text.endswith(("tes", "tesek", "teş", "teşk", ",")):
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
        "ne var ne yok",
        "gunaydin",
        "günaydın"
    ]

    return any(greeting in text for greeting in greetings)


def build_local_llm_fallback(user_text, knowledge_context=None):
    user_text = str(user_text or "").strip()
    knowledge_context = str(knowledge_context or "").strip()

    if is_greeting(user_text):
        return (
            "Merhaba, yardımcı olabilirim. "
            "Garenta kiralama koşullarıyla ilgili merak ettiğin konuyu sorabilirsin."
        )

    if knowledge_context:
        return "Bu konuda dokümanda bilgi buldum ama cevabı güvenli şekilde oluşturamadım."

    return "Bu konuda dokümanda net bilgi bulamadım."


def build_headers():
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "HTTP-Referer": "http://127.0.0.1:8001",
        "X-Title": "Voice AI Agent WebRTC"
    }


def build_payload(user_text, stream, knowledge_context=None):
    prompt = build_voice_agent_prompt(
        user_text=user_text,
        knowledge_context=knowledge_context
    )

    return {
        "model": OPENROUTER_MODEL,
        "stream": stream,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sen Garenta araç kiralama süreçleri için Türkçe konuşan bir müşteri destek asistanısın. "
                    "Sadece kullanıcıya söylenecek nihai cevabı yaz. "
                    "Asla analiz, iç düşünce, açıklama, prompt yorumu veya İngilizce metin yazma. "
                    "Cevap tamamen Türkçe, kısa, doğal ve sesli okunmaya uygun olmalı. "
                    "Doküman bilgisini aynen kopyalama, konuşma diliyle özetle. "
                    "Bilgi yoksa sadece 'Bu konuda dokümanda net bilgi bulamadım.' de."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 260
    }


def ask_llm_streaming(user_text, knowledge_context=None):
    started_at = time.perf_counter()
    first_token_at = None
    answer_parts = []

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=build_headers(),
        json=build_payload(
            user_text=user_text,
            stream=True,
            knowledge_context=knowledge_context
        ),
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

    if looks_like_bad_meta_answer(answer) or looks_like_incomplete_answer(answer):
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": llm_first_token_ms,
            "llm_total_ms": llm_total_ms,
            "error": f"Streaming LLM eksik veya geçersiz cevap döndürdü: {answer[:250]}"
        }

    return {
        "success": True,
        "answer": answer,
        "llm_model": OPENROUTER_MODEL,
        "llm_first_token_ms": llm_first_token_ms,
        "llm_total_ms": llm_total_ms,
        "error": None
    }


def ask_llm_non_streaming(user_text, knowledge_context=None):
    started_at = time.perf_counter()

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=build_headers(),
        json=build_payload(
            user_text=user_text,
            stream=False,
            knowledge_context=knowledge_context
        ),
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

    if looks_like_bad_meta_answer(answer) or looks_like_incomplete_answer(answer):
        return {
            "success": False,
            "answer": "",
            "llm_model": OPENROUTER_MODEL,
            "llm_first_token_ms": None,
            "llm_total_ms": llm_total_ms,
            "error": f"Non-streaming LLM eksik veya geçersiz cevap döndürdü: {answer[:250]}"
        }

    return {
        "success": True,
        "answer": answer,
        "llm_model": OPENROUTER_MODEL,
        "llm_first_token_ms": llm_total_ms,
        "llm_total_ms": llm_total_ms,
        "error": None
    }


def ask_llm_with_metrics(user_text, knowledge_context=None):
    started_at = time.perf_counter()
    knowledge_context = str(knowledge_context or "").strip()

    if not OPENROUTER_API_KEY:
        fallback = build_local_llm_fallback(
            user_text=user_text,
            knowledge_context=knowledge_context
        )

        return {
            "success": False,
            "answer": fallback,
            "llm_model": "local_fallback_no_api_key",
            "llm_first_token_ms": 0,
            "llm_total_ms": int((time.perf_counter() - started_at) * 1000),
            "error": "OPENROUTER_API_KEY bulunamadı."
        }

    print(f"LLM isteği gönderiliyor. Model: {OPENROUTER_MODEL}")

    if knowledge_context:
        print("Knowledge context LLM promptuna eklendi.")
    else:
        print("Knowledge context bulunamadı, fallback prompt kullanılacak.")

    errors = []

    try:
        streaming_result = ask_llm_streaming(
            user_text=user_text,
            knowledge_context=knowledge_context
        )

        if streaming_result.get("success"):
            return streaming_result

        errors.append(f"streaming: {streaming_result.get('error')}")
        print(f"Streaming başarısız: {streaming_result.get('error')}")
        print("Non-streaming LLM deneniyor.")

    except Exception as error:
        errors.append(f"streaming exception: {error}")
        print(f"Streaming exception: {error}")
        print("Non-streaming LLM deneniyor.")

    try:
        non_streaming_result = ask_llm_non_streaming(
            user_text=user_text,
            knowledge_context=knowledge_context
        )

        if non_streaming_result.get("success"):
            if errors:
                non_streaming_result["error"] = " | ".join(errors) + " | Non-streaming kullanıldı."

            return non_streaming_result

        errors.append(f"non_streaming: {non_streaming_result.get('error')}")
        print(f"Non-streaming başarısız: {non_streaming_result.get('error')}")

    except Exception as error:
        errors.append(f"non_streaming exception: {error}")
        print(f"Non-streaming exception: {error}")

    fallback = build_local_llm_fallback(
        user_text=user_text,
        knowledge_context=knowledge_context
    )

    return {
        "success": False,
        "answer": fallback,
        "llm_model": "local_fallback_after_bad_llm",
        "llm_first_token_ms": 0,
        "llm_total_ms": int((time.perf_counter() - started_at) * 1000),
        "error": " | ".join(errors) if errors else "LLM kötü/boş cevap verdi."
    }