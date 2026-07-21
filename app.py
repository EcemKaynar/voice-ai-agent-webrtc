import asyncio
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiortc import RTCPeerConnection, RTCSessionDescription

from utils.audio_utils import (
    create_audio_resampler,
    audio_frame_to_mono_int16,
    save_pcm_chunks_to_wav,
    calculate_rms_int16
)

from utils.metrics import (
    init_metrics_db,
    save_voice_metric,
    get_recent_voice_metrics
)

from services.stt_service import (
    transcribe_audio_file,
    preload_stt_model,
    get_stt_config,
    is_stt_model_loaded
)

from services.llm_service import ask_llm_with_metrics

from services.tts_service import (
    stream_tts_audio_chunks,
    get_tts_config
)

from services.knowledge_base_service import (
    search_knowledge,
    get_knowledge_base_status,
    reload_knowledge_base
)

from services.transcript_normalizer_service import normalize_transcript_for_domain
from services.kb_answer_service import build_direct_kb_answer


load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Voice AI Agent WebRTC")

AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "data/audio_chunks")
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "data/tts_outputs")

MIN_AUDIO_RMS = float(os.getenv("MIN_AUDIO_RMS", "70"))
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))

SPEECH_RMS_THRESHOLD = float(os.getenv("SPEECH_RMS_THRESHOLD", "150"))
SILENCE_END_SECONDS = float(os.getenv("SILENCE_END_SECONDS", "1.2"))
MIN_UTTERANCE_SECONDS = float(os.getenv("MIN_UTTERANCE_SECONDS", "1.0"))
MAX_UTTERANCE_SECONDS = float(os.getenv("MAX_UTTERANCE_SECONDS", "15"))
PRE_SPEECH_SECONDS = float(os.getenv("PRE_SPEECH_SECONDS", "0.45"))

MIN_SPEECH_SECONDS = float(os.getenv("MIN_SPEECH_SECONDS", "0.45"))
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.10"))

Path(AUDIO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

init_metrics_db()

app = FastAPI(title=APP_NAME)

app.mount(
    "/tts-audio",
    StaticFiles(directory=TTS_OUTPUT_DIR),
    name="tts_audio"
)

pcs = set()

latest_response = {
    "has_response": False
}

pending_tts_streams = {}

client_states = {}

runtime_state = {
    "stt_preload_success": False,
    "stt_preload_error": None,
    "server_started_at": datetime.now().isoformat(timespec="seconds")
}


class OfferRequest(BaseModel):
    sdp: str
    type: str


class ClientStateRequest(BaseModel):
    peer_id: str
    user_speaking: bool = True
    assistant_playing: bool = False
    ignore_audio_ms: int = 0


def make_audio_filename(peer_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]

    return Path(AUDIO_OUTPUT_DIR) / f"{peer_id}_{timestamp}_{short_id}.wav"


def reset_latest_response():
    latest_response.clear()
    latest_response.update({
        "has_response": False,
        "updated_at": datetime.now().isoformat(timespec="milliseconds")
    })


def update_client_state_values(peer_id, **kwargs):
    state = client_states.get(peer_id, {})

    for key, value in kwargs.items():
        state[key] = value

    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    client_states[peer_id] = state

    return state


def normalize_text_for_filter(text):
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

    return text


def should_ignore_transcript(transcript):
    text = normalize_text_for_filter(transcript)

    if not text:
        return True

    if len(text) < 3:
        return True

    ignored_phrases = [
        "abone olmayi unutmayin",
        "abone olmayı unutmayın",
        "altyazi",
        "altyazi m.k",
        "altyazi m k",
        "izlediginiz icin tesekkurler",
        "thank you for watching",
        "thanks for watching",
        "videonun altinda",
        "bu videonun altinda",
        "bir videonun altinda",
        "kanalima abone",
        "yorumlarda bulusalim",
        "bu dizinin betimlemesi",
        "dizinin betimlemesi",
        "sesli betimleme",
        "sesli betimleme dernegi",
        "trt tarafindan",
        "begenirmeyi",
        "soguklarinizi"
    ]

    if any(phrase in text for phrase in ignored_phrases):
        return True

    words = text.split()

    if len(words) >= 12:
        unique_ratio = len(set(words)) / len(words)

        if unique_ratio < 0.35:
            return True

    repeated_bad_chunks = [
        "videonun altinda",
        "bir videonun altinda",
        "bu videonun altinda",
        "begenirmeyi",
        "bu dizinin betimlemesi"
    ]

    for chunk in repeated_bad_chunks:
        if text.count(chunk) >= 2:
            return True

    return False


def get_streaming_config():
    return {
        "input_mode": "auto_turn_taking",
        "push_to_talk_enabled": False,
        "stt_streaming_enabled": False,
        "stt_current_mode": "auto_silence_detected_segment_then_transcribed",
        "llm_streaming_enabled": True,
        "llm_non_streaming_fallback_enabled": True,
        "tts_backend_streaming_enabled": True,
        "tts_frontend_streaming_enabled": True,
        "tts_current_mode": "edge_tts_audio_chunks_streamed_with_http_streaming",
        "frontend_update_mode": "polling_latest_response_endpoint",
        "webrtc_input_audio_enabled": True,
        "webrtc_output_audio_enabled": False,
        "assistant_audio_feedback_protection": True,
        "knowledge_base_enabled": True,
        "prompt_service_enabled": True,
        "note": (
            "This version uses automatic turn taking. "
            "STT is segment based. TTS is streamed with HTTP StreamingResponse. "
            "Knowledge base context is sent to LLM for natural grounded answers. "
            "Direct KB answer is used only as fallback when LLM answer is weak or meta."
        )
    }


def get_client_state(peer_id):
    return client_states.get(peer_id, {
        "user_speaking": True,
        "assistant_playing": False,
        "server_processing": False,
        "ignore_until": 0
    })


def is_assistant_audio_guard_active(peer_id):
    state = get_client_state(peer_id)

    if state.get("assistant_playing"):
        return True

    if state.get("server_processing"):
        return True

    ignore_until = state.get("ignore_until", 0)

    if ignore_until and time.time() < ignore_until:
        return True

    return False


def is_user_speaking_enabled(peer_id):
    state = get_client_state(peer_id)

    return bool(state.get("user_speaking"))


async def handle_transcript_with_llm(
    peer_id,
    transcript,
    stt_result=None,
    audio_input_path=None,
    pipeline_started_at=None
):
    transcript = str(transcript or "").strip()
    stt_result = stt_result or {}

    if should_ignore_transcript(transcript):
        print(f"[{peer_id}] Transcript filtrelendi, LLM'e gönderilmeyecek: {transcript}")

        update_client_state_values(
            peer_id,
            user_speaking=True,
            assistant_playing=False,
            server_processing=False
        )

        return None

    if pipeline_started_at is None:
        pipeline_started_at = time.perf_counter()

    normalization_result = normalize_transcript_for_domain(transcript)
    effective_query = normalization_result.get("normalized_query") or transcript
    display_transcript = effective_query if normalization_result.get("correction_applied") else transcript

    print(f"[{peer_id}] Transcript normalization:")
    print(f"  Original: {transcript}")
    print(f"  Effective query: {effective_query}")
    print(f"  Correction applied: {normalization_result.get('correction_applied')}")
    print(f"  Reason: {normalization_result.get('correction_reason')}")
    print(f"  Confidence: {normalization_result.get('confidence')}")

    print(f"[{peer_id}] Knowledge base aranıyor:")
    print(f"  Query: {effective_query}")

    knowledge_result = search_knowledge(
        query=effective_query,
        top_k=2
    )

    knowledge_context = knowledge_result.get("context", "")

    print(f"[{peer_id}] Knowledge sonucu:")
    print(f"  Found: {knowledge_result.get('found')}")
    print(f"  Intent profiles: {knowledge_result.get('intent_profiles')}")
    print(f"  Result count: {len(knowledge_result.get('results', []))}")

    for item in knowledge_result.get("results", []):
        print(
            f"  - Score: {item.get('score')} | "
            f"Title: {item.get('title')} | "
            f"Source: {item.get('source')}"
        )

    print(f"[{peer_id}] LLM'e gönderiliyor:")
    print(f"  User text: {effective_query}")

    llm_result = await asyncio.to_thread(
        ask_llm_with_metrics,
        effective_query,
        knowledge_context
    )

    direct_kb_result = build_direct_kb_answer(
        user_query=effective_query,
        knowledge_result=knowledge_result
    )

    answer_for_quality_check = str(llm_result.get("answer") or "").strip()
    answer_quality_text = normalize_text_for_filter(answer_for_quality_check)

    bad_answer_markers = [
        "bilgi 1",
        "bilgi tabani",
        "bilgi tabanı",
        "kaynak:",
        "icerik:",
        "içerik:",
        "baslik:",
        "başlık:",
        "dokumana gore -",
        "dokümana göre -",
        "kiralamaya ait tum ucretler",
        "kiralamaya ait tüm ücretler",
        "bu konuda dokumanda net bilgi bulamadim",
        "bu konuda dokümanda net bilgi bulamadım",
        "we need",
        "we must",
        "we should",
        "user asks",
        "the user asks",
        "user asked",
        "the user asked",
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
        "knowledge base",
        "respond in turkish",
        "in turkish",
        "turkish, short",
        "2-4 sentences",
        "no headings"
    ]

    should_use_direct_fallback = False

    if not llm_result.get("success"):
        should_use_direct_fallback = True

    if str(llm_result.get("llm_model") or "").startswith("local_fallback"):
        should_use_direct_fallback = True

    if not answer_for_quality_check:
        should_use_direct_fallback = True

    if any(marker in answer_quality_text for marker in bad_answer_markers):
        should_use_direct_fallback = True

    if len(answer_for_quality_check) < 35 and direct_kb_result:
        should_use_direct_fallback = True

    english_meta_words = [
        "we",
        "user",
        "must",
        "should",
        "answer",
        "provided",
        "info",
        "according",
        "condition",
        "summarize",
        "directly"
    ]

    english_meta_count = sum(
        1
        for word in english_meta_words
        if f" {word} " in f" {answer_quality_text} "
    )

    if english_meta_count >= 3:
        should_use_direct_fallback = True

    if should_use_direct_fallback and direct_kb_result:
        print(f"[{peer_id}] LLM cevabı zayıf/ham/meta bulundu. Direct KB fallback kullanıldı.")
        llm_result = direct_kb_result
    else:
        print(f"[{peer_id}] Natural LLM answer kullanıldı.")

    print(f"[{peer_id}] LLM sonucu:")
    print(f"  Success: {llm_result.get('success')}")
    print(f"  Answer: {llm_result.get('answer')}")
    print(f"  LLM model: {llm_result.get('llm_model')}")
    print(f"  LLM first token ms: {llm_result.get('llm_first_token_ms')}")
    print(f"  LLM total ms: {llm_result.get('llm_total_ms')}")

    if llm_result.get("error"):
        print(f"  LLM error: {llm_result.get('error')}")

    answer = str(llm_result.get("answer") or "").strip()

    if not answer:
        print(f"[{peer_id}] LLM cevabı boş, TTS'e gönderilmeyecek.")

        update_client_state_values(
            peer_id,
            user_speaking=True,
            assistant_playing=False,
            server_processing=False
        )

        return llm_result

    response_id = str(uuid.uuid4())
    audio_url = f"/tts-stream/{response_id}"

    pending_tts_streams[response_id] = {
        "peer_id": peer_id,
        "transcript": display_transcript,
        "raw_transcript": transcript,
        "normalized_query": effective_query,
        "answer": answer,
        "audio_input_path": audio_input_path,
        "stt_result": stt_result,
        "llm_result": llm_result,
        "knowledge_result": knowledge_result,
        "normalization_result": normalization_result,
        "pipeline_started_at": pipeline_started_at,
        "created_at": datetime.now().isoformat(timespec="milliseconds")
    }

    print(f"[{peer_id}] TTS streaming hazırlandı.")
    print(f"  Response ID: {response_id}")
    print(f"  Audio stream URL: {audio_url}")

    latest_response.clear()
    latest_response.update({
        "has_response": True,
        "id": response_id,
        "metric_id": None,
        "peer_id": peer_id,
        "transcript": display_transcript,
        "raw_transcript": transcript,
        "normalized_query": effective_query,
        "transcript_correction_applied": normalization_result.get("correction_applied"),
        "transcript_correction_reason": normalization_result.get("correction_reason"),
        "transcript_correction_confidence": normalization_result.get("confidence"),
        "answer": answer,
        "audio_url": audio_url,
        "stt_llm_tts_status": "tts_stream_pending",
        "stt_latency_ms": stt_result.get("stt_latency_ms"),
        "llm_first_token_ms": llm_result.get("llm_first_token_ms"),
        "llm_total_ms": llm_result.get("llm_total_ms"),
        "tts_first_byte_ms": None,
        "tts_total_ms": None,
        "total_pipeline_ms": None,
        "llm_model": llm_result.get("llm_model"),
        "tts_voice": get_tts_config().get("tts_voice"),
        "streaming_config": get_streaming_config(),
        "knowledge_found": knowledge_result.get("found"),
        "knowledge_result_count": len(knowledge_result.get("results", [])),
        "knowledge_intent_profiles": knowledge_result.get("intent_profiles", []),
        "knowledge_sources": [
            {
                "title": item.get("title"),
                "source": item.get("source"),
                "score": item.get("score")
            }
            for item in knowledge_result.get("results", [])
        ],
        "updated_at": datetime.now().isoformat(timespec="milliseconds")
    })

    llm_result["tts_stream_url"] = audio_url
    llm_result["response_id"] = response_id

    return llm_result


async def process_saved_wav_with_stt_and_llm(peer_id, saved_path, label="STT sonucu"):
    pipeline_started_at = time.perf_counter()

    update_client_state_values(
        peer_id,
        user_speaking=False,
        assistant_playing=False,
        server_processing=True
    )

    stt_result = await asyncio.to_thread(
        transcribe_audio_file,
        saved_path
    )

    print(f"[{peer_id}] {label}:")
    print(f"  Success: {stt_result.get('success')}")
    print(f"  Text: {stt_result.get('text')}")
    print(f"  STT latency ms: {stt_result.get('stt_latency_ms')}")

    if stt_result.get("error"):
        print(f"  STT error: {stt_result.get('error')}")

    transcript = str(stt_result.get("text") or "").strip()

    llm_tts_result = await handle_transcript_with_llm(
        peer_id=peer_id,
        transcript=transcript,
        stt_result=stt_result,
        audio_input_path=saved_path,
        pipeline_started_at=pipeline_started_at
    )

    if not isinstance(llm_tts_result, dict) or not llm_tts_result.get("response_id"):
        update_client_state_values(
            peer_id,
            user_speaking=True,
            assistant_playing=False,
            server_processing=False
        )

    return {
        "stt_result": stt_result,
        "llm_tts_result": llm_tts_result
    }


async def consume_audio_track(track, peer_id):
    frame_count = 0
    current_sample_rate = TARGET_SAMPLE_RATE

    resampler = create_audio_resampler(
        target_sample_rate=TARGET_SAMPLE_RATE
    )

    speech_active = False

    pre_speech_chunks = []
    pre_speech_samples = 0

    utterance_chunks = []
    utterance_samples = 0
    silence_samples = 0
    utterance_speech_samples = 0

    print(f"[{peer_id}] Audio track dinleniyor...")
    print(f"[{peer_id}] Auto turn-taking mod aktif. Kullanıcı konuşunca otomatik algılanacak.")

    def add_to_pre_speech_buffer(pcm):
        nonlocal pre_speech_samples

        pre_speech_chunks.append(pcm)
        pre_speech_samples += len(pcm)

        max_pre_samples = int(PRE_SPEECH_SECONDS * current_sample_rate)

        while pre_speech_samples > max_pre_samples and pre_speech_chunks:
            removed = pre_speech_chunks.pop(0)
            pre_speech_samples -= len(removed)

    def reset_utterance():
        nonlocal speech_active
        nonlocal pre_speech_chunks
        nonlocal pre_speech_samples
        nonlocal utterance_chunks
        nonlocal utterance_samples
        nonlocal silence_samples
        nonlocal utterance_speech_samples

        speech_active = False
        pre_speech_chunks = []
        pre_speech_samples = 0
        utterance_chunks = []
        utterance_samples = 0
        silence_samples = 0
        utterance_speech_samples = 0

    async def process_current_utterance(reason):
        nonlocal utterance_chunks
        nonlocal utterance_samples
        nonlocal silence_samples
        nonlocal utterance_speech_samples

        if not utterance_chunks or utterance_samples <= 0:
            reset_utterance()
            return

        utterance_duration_seconds = utterance_samples / current_sample_rate
        speech_duration_seconds = utterance_speech_samples / current_sample_rate
        speech_ratio = utterance_speech_samples / max(utterance_samples, 1)
        chunk_rms = calculate_rms_int16(utterance_chunks)

        print(
            f"[{peer_id}] Konuşma parçası bitti. "
            f"Reason: {reason}, "
            f"Duration: {round(utterance_duration_seconds, 2)} sn, "
            f"Speech duration: {round(speech_duration_seconds, 2)} sn, "
            f"Speech ratio: {round(speech_ratio, 2)}, "
            f"RMS: {round(chunk_rms, 2)}"
        )

        if utterance_duration_seconds < MIN_UTTERANCE_SECONDS:
            print(f"[{peer_id}] Konuşma çok kısa, atlandı.")
            reset_utterance()
            return

        if speech_duration_seconds < MIN_SPEECH_SECONDS:
            print(f"[{peer_id}] Gerçek konuşma süresi düşük, atlandı.")
            reset_utterance()
            return

        if speech_ratio < MIN_SPEECH_RATIO:
            print(f"[{peer_id}] Konuşma oranı düşük, atlandı.")
            reset_utterance()
            return

        if chunk_rms < MIN_AUDIO_RMS:
            print(f"[{peer_id}] Konuşma RMS düşük, atlandı.")
            reset_utterance()
            return

        output_path = make_audio_filename(peer_id)

        saved_path = save_pcm_chunks_to_wav(
            pcm_chunks=utterance_chunks,
            output_path=output_path,
            sample_rate=current_sample_rate
        )

        print(
            f"[{peer_id}] WAV kaydedildi: {saved_path} "
            f"Audio duration: {round(utterance_duration_seconds, 2)} sn"
        )

        reset_utterance()

        await process_saved_wav_with_stt_and_llm(
            peer_id=peer_id,
            saved_path=saved_path,
            label="STT sonucu"
        )

    try:
        while True:
            frame = await track.recv()
            frame_count += 1

            pcm_arrays = audio_frame_to_mono_int16(
                frame=frame,
                resampler=resampler
            )

            for pcm in pcm_arrays:
                if pcm is None or len(pcm) == 0:
                    continue

                if is_assistant_audio_guard_active(peer_id):
                    if speech_active:
                        print(f"[{peer_id}] Asistan/processing guard aktif, buffer temizlendi.")
                    reset_utterance()
                    continue

                if not is_user_speaking_enabled(peer_id):
                    reset_utterance()
                    continue

                frame_rms = calculate_rms_int16([pcm])
                is_speech = frame_rms >= SPEECH_RMS_THRESHOLD

                if not speech_active:
                    add_to_pre_speech_buffer(pcm)

                    if is_speech:
                        speech_active = True

                        utterance_chunks = list(pre_speech_chunks)
                        utterance_samples = sum(len(item) for item in utterance_chunks)
                        silence_samples = 0
                        utterance_speech_samples = len(pcm)

                        pre_speech_chunks = []
                        pre_speech_samples = 0

                        print(
                            f"[{peer_id}] Konuşma başladı. "
                            f"Frame RMS: {round(frame_rms, 2)}"
                        )

                    continue

                utterance_chunks.append(pcm)
                utterance_samples += len(pcm)

                if is_speech:
                    silence_samples = 0
                    utterance_speech_samples += len(pcm)
                else:
                    silence_samples += len(pcm)

                utterance_duration_seconds = utterance_samples / current_sample_rate
                silence_duration_seconds = silence_samples / current_sample_rate
                speech_duration_seconds = utterance_speech_samples / current_sample_rate
                speech_ratio = utterance_speech_samples / max(utterance_samples, 1)

                if frame_count % 50 == 0:
                    print(
                        f"[{peer_id}] Dinleniyor. "
                        f"Frame count: {frame_count}, "
                        f"Frame RMS: {round(frame_rms, 2)}, "
                        f"Speech active: {speech_active}, "
                        f"Utterance duration: {round(utterance_duration_seconds, 2)} sn, "
                        f"Speech duration: {round(speech_duration_seconds, 2)} sn, "
                        f"Speech ratio: {round(speech_ratio, 2)}, "
                        f"Silence duration: {round(silence_duration_seconds, 2)} sn"
                    )

                if (
                    silence_duration_seconds >= SILENCE_END_SECONDS
                    and utterance_duration_seconds >= MIN_UTTERANCE_SECONDS
                ):
                    await process_current_utterance(reason="silence_detected")
                    continue

                if utterance_duration_seconds >= MAX_UTTERANCE_SECONDS:
                    await process_current_utterance(reason="max_duration")
                    continue

    except Exception as error:
        print(f"[{peer_id}] Audio track kapandı veya hata oluştu: {error}")

        if utterance_chunks and utterance_samples > 0:
            try:
                await process_current_utterance(reason="track_closed")
            except Exception as save_error:
                print(f"[{peer_id}] Son konuşma işlenemedi: {save_error}")


@app.on_event("startup")
async def on_startup():
    print("Application startup başladı.")

    try:
        kb_status = get_knowledge_base_status()
        print(f"Knowledge base status: {kb_status}")
    except Exception as error:
        print(f"Knowledge base yükleme hatası: {error}")

    print("STT modeli preload ediliyor...")

    try:
        preload_result = await asyncio.to_thread(preload_stt_model)

        runtime_state["stt_preload_success"] = preload_result.get("success")
        runtime_state["stt_preload_error"] = preload_result.get("message")

        print(f"STT preload sonucu: {preload_result}")

    except Exception as error:
        runtime_state["stt_preload_success"] = False
        runtime_state["stt_preload_error"] = str(error)

        print(f"STT preload hatası: {error}")

    print("Application startup tamamlandı.")


@app.get("/")
def home():
    html_path = Path("web/index.html")

    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return HTMLResponse("<h1>Voice AI Agent WebRTC</h1>")


@app.get("/health")
def health():
    return JSONResponse({
        "status": "ok",
        "app": APP_NAME,
        "server_started_at": runtime_state.get("server_started_at"),
        "active_peer_connections": len(pcs),
        "stt_model_loaded": is_stt_model_loaded(),
        "stt_preload_success": runtime_state.get("stt_preload_success"),
        "stt_preload_error": runtime_state.get("stt_preload_error"),
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "min_audio_rms": MIN_AUDIO_RMS,
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
        "silence_end_seconds": SILENCE_END_SECONDS,
        "min_speech_seconds": MIN_SPEECH_SECONDS,
        "min_speech_ratio": MIN_SPEECH_RATIO,
        "tts_config": get_tts_config(),
        "pending_tts_stream_count": len(pending_tts_streams),
        "client_states": client_states,
        "knowledge_base": get_knowledge_base_status()
    })


@app.get("/config")
def config():
    return JSONResponse({
        "app": APP_NAME,
        "mode": "webrtc_auto_turn_taking_knowledge_grounded_streaming_tts_metrics_mvp",
        "audio_config": {
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "min_audio_rms": MIN_AUDIO_RMS,
            "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
            "silence_end_seconds": SILENCE_END_SECONDS,
            "min_utterance_seconds": MIN_UTTERANCE_SECONDS,
            "max_utterance_seconds": MAX_UTTERANCE_SECONDS,
            "pre_speech_seconds": PRE_SPEECH_SECONDS,
            "min_speech_seconds": MIN_SPEECH_SECONDS,
            "min_speech_ratio": MIN_SPEECH_RATIO
        },
        "stt_config": get_stt_config(),
        "tts_config": get_tts_config(),
        "streaming_config": get_streaming_config(),
        "knowledge_base": get_knowledge_base_status()
    })


@app.get("/knowledge/status")
def knowledge_status():
    return JSONResponse(get_knowledge_base_status())


@app.get("/knowledge/search")
def knowledge_search(q: str, top_k: int = 4):
    return JSONResponse(
        search_knowledge(
            query=q,
            top_k=top_k
        )
    )


@app.post("/knowledge/reload")
def knowledge_reload():
    return JSONResponse(reload_knowledge_base())


@app.get("/latest-response")
def get_latest_response():
    return JSONResponse(latest_response)


@app.get("/tts-stream/{response_id}")
async def tts_stream(response_id: str):
    stream_data = pending_tts_streams.get(response_id)

    if not stream_data:
        async def empty_stream():
            yield b""

        return StreamingResponse(
            empty_stream(),
            media_type="audio/mpeg"
        )

    peer_id = stream_data.get("peer_id")
    answer = stream_data.get("answer")
    transcript = stream_data.get("transcript")
    stt_result = stream_data.get("stt_result") or {}
    llm_result = stream_data.get("llm_result") or {}
    audio_input_path = stream_data.get("audio_input_path")
    pipeline_started_at = stream_data.get("pipeline_started_at") or time.perf_counter()

    async def audio_generator():
        tts_started_at = time.perf_counter()
        first_byte_at = None
        tts_success = True
        tts_error = None

        try:
            async for audio_chunk in stream_tts_audio_chunks(answer):
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()

                    print(
                        f"[{peer_id}] TTS stream first byte: "
                        f"{int((first_byte_at - tts_started_at) * 1000)} ms"
                    )

                yield audio_chunk

        except Exception as error:
            tts_success = False
            tts_error = str(error)
            print(f"[{peer_id}] TTS stream error: {tts_error}")

        finally:
            tts_finished_at = time.perf_counter()

            tts_first_byte_ms = None

            if first_byte_at is not None:
                tts_first_byte_ms = int((first_byte_at - tts_started_at) * 1000)

            tts_total_ms = int((tts_finished_at - tts_started_at) * 1000)
            total_pipeline_ms = int((tts_finished_at - pipeline_started_at) * 1000)

            errors = {}

            if stt_result.get("error"):
                errors["stt_error"] = stt_result.get("error")

            if llm_result.get("error"):
                errors["llm_error"] = llm_result.get("error")

            if tts_error:
                errors["tts_error"] = tts_error

            metric_id = save_voice_metric(
                peer_id=peer_id,
                transcript=transcript,
                answer=answer,
                audio_input_path=audio_input_path,
                audio_output_path=f"/tts-stream/{response_id}",
                stt_success=stt_result.get("success"),
                llm_success=llm_result.get("success"),
                tts_success=tts_success,
                stt_latency_ms=stt_result.get("stt_latency_ms"),
                llm_first_token_ms=llm_result.get("llm_first_token_ms"),
                llm_total_ms=llm_result.get("llm_total_ms"),
                tts_first_byte_ms=tts_first_byte_ms,
                tts_total_ms=tts_total_ms,
                total_pipeline_ms=total_pipeline_ms,
                llm_model=llm_result.get("llm_model"),
                tts_voice=get_tts_config().get("tts_voice"),
                errors=errors
            )

            print(f"[{peer_id}] Streaming TTS metric kaydedildi. Metric ID: {metric_id}")

            update_client_state_values(
                peer_id,
                server_processing=False
            )

            if latest_response.get("id") == response_id:
                latest_response.update({
                    "metric_id": metric_id,
                    "stt_llm_tts_status": "tts_stream_completed",
                    "tts_first_byte_ms": tts_first_byte_ms,
                    "tts_total_ms": tts_total_ms,
                    "total_pipeline_ms": total_pipeline_ms,
                    "updated_at": datetime.now().isoformat(timespec="milliseconds")
                })

            pending_tts_streams.pop(response_id, None)

    return StreamingResponse(
        audio_generator(),
        media_type="audio/mpeg"
    )


@app.post("/clear-latest-response")
def clear_latest_response():
    reset_latest_response()

    return JSONResponse({
        "success": True,
        "message": "latest_response temizlendi."
    })


@app.post("/client-state")
def update_client_state(request: ClientStateRequest):
    state = client_states.get(request.peer_id, {})

    state["user_speaking"] = request.user_speaking
    state["assistant_playing"] = request.assistant_playing
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")

    if request.ignore_audio_ms and request.ignore_audio_ms > 0:
        state["ignore_until"] = time.time() + (request.ignore_audio_ms / 1000)
    elif not state.get("ignore_until"):
        state["ignore_until"] = 0

    if "server_processing" not in state:
        state["server_processing"] = False

    client_states[request.peer_id] = state

    return JSONResponse({
        "success": True,
        "peer_id": request.peer_id,
        "state": state
    })


@app.get("/metrics")
def metrics(limit: int = 10):
    items = get_recent_voice_metrics(limit=limit)

    return JSONResponse({
        "count": len(items),
        "items": items
    })


@app.post("/offer")
async def offer(request: OfferRequest):
    peer_id = f"peer_{len(pcs) + 1}_{str(uuid.uuid4())[:6]}"

    pc = RTCPeerConnection()
    pcs.add(pc)

    client_states[peer_id] = {
        "user_speaking": True,
        "assistant_playing": False,
        "server_processing": False,
        "ignore_until": 0,
        "created_at": datetime.now().isoformat(timespec="seconds")
    }

    print(f"[{peer_id}] Yeni WebRTC bağlantısı oluşturuldu.")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[{peer_id}] Connection state: {pc.connectionState}")

        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)
            client_states.pop(peer_id, None)
            print(f"[{peer_id}] Bağlantı kapatıldı.")

    @pc.on("track")
    def on_track(track):
        print(f"[{peer_id}] Track geldi: {track.kind}")

        if track.kind == "audio":
            asyncio.create_task(
                consume_audio_track(track, peer_id)
            )

    offer_description = RTCSessionDescription(
        sdp=request.sdp,
        type=request.type
    )

    await pc.setRemoteDescription(offer_description)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    print(f"[{peer_id}] Answer oluşturuldu.")

    return JSONResponse({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "peer_id": peer_id
    })


@app.on_event("shutdown")
async def on_shutdown():
    coroutines = [pc.close() for pc in list(pcs)]
    await asyncio.gather(*coroutines)
    pcs.clear()
    client_states.clear()
    pending_tts_streams.clear()