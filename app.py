import asyncio
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
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
from services.tts_service import synthesize_speech_with_metrics


load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Voice AI Agent WebRTC")

AUDIO_CHUNK_SECONDS = float(os.getenv("AUDIO_CHUNK_SECONDS", "10"))
AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "data/audio_chunks")
MIN_AUDIO_RMS = float(os.getenv("MIN_AUDIO_RMS", "80"))
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))

SPEECH_RMS_THRESHOLD = float(os.getenv("SPEECH_RMS_THRESHOLD", "180"))
SILENCE_END_SECONDS = float(os.getenv("SILENCE_END_SECONDS", "0.9"))
MIN_UTTERANCE_SECONDS = float(os.getenv("MIN_UTTERANCE_SECONDS", "0.7"))
MAX_UTTERANCE_SECONDS = float(os.getenv("MAX_UTTERANCE_SECONDS", "12"))
PRE_SPEECH_SECONDS = float(os.getenv("PRE_SPEECH_SECONDS", "0.25"))

MIN_SPEECH_SECONDS = float(os.getenv("MIN_SPEECH_SECONDS", "0.3"))
MIN_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.12"))

TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "data/tts_outputs")

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
    user_speaking: bool = False
    assistant_playing: bool = False
    ignore_audio_ms: int = 0


def make_audio_filename(peer_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]

    return Path(AUDIO_OUTPUT_DIR) / f"{peer_id}_{timestamp}_{short_id}.wav"


def reset_latest_response():
    latest_response.clear()
    latest_response.update({
        "has_response": False
    })


def normalize_text_for_filter(text):
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
        "altyazi m k.",
        "izlediginiz icin tesekkurler",
        "tesekkurler izlediginiz icin",
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
        "begenirmeyi ve begenirmeyi",
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
        "input_mode": "push_to_talk",
        "stt_streaming_enabled": False,
        "stt_current_mode": "push_to_talk_segment_saved_then_transcribed",
        "llm_streaming_enabled": True,
        "llm_non_streaming_fallback_enabled": True,
        "tts_backend_streaming_enabled": True,
        "tts_frontend_streaming_enabled": False,
        "tts_current_mode": "edge_tts_stream_saved_to_mp3_then_served_over_http",
        "frontend_update_mode": "polling_latest_response_endpoint",
        "webrtc_input_audio_enabled": True,
        "webrtc_output_audio_enabled": False,
        "assistant_audio_feedback_protection": True,
        "note": (
            "Current MVP uses WebRTC for microphone input with push-to-talk. "
            "Backend ignores audio unless user_speaking is true. "
            "Assistant playback also disables backend processing to avoid feedback loops."
        )
    }


def get_client_state(peer_id):
    return client_states.get(peer_id, {
        "user_speaking": False,
        "assistant_playing": False,
        "ignore_until": 0
    })


def is_assistant_audio_guard_active(peer_id):
    state = get_client_state(peer_id)

    if state.get("assistant_playing"):
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
        return None

    if pipeline_started_at is None:
        pipeline_started_at = time.perf_counter()

    print(f"[{peer_id}] LLM'e gönderiliyor:")
    print(f"  User text: {transcript}")

    llm_result = await asyncio.to_thread(
        ask_llm_with_metrics,
        transcript
    )

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
        return llm_result

    print(f"[{peer_id}] TTS'e gönderiliyor:")
    print(f"  Text: {answer}")

    tts_result = await synthesize_speech_with_metrics(
        text=answer,
        peer_id=peer_id
    )

    print(f"[{peer_id}] TTS sonucu:")
    print(f"  Success: {tts_result.get('success')}")
    print(f"  Audio path: {tts_result.get('audio_path')}")
    print(f"  TTS voice: {tts_result.get('tts_voice')}")
    print(f"  TTS first byte ms: {tts_result.get('tts_first_byte_ms')}")
    print(f"  TTS total ms: {tts_result.get('tts_total_ms')}")

    if tts_result.get("error"):
        print(f"  TTS error: {tts_result.get('error')}")

    total_pipeline_ms = int((time.perf_counter() - pipeline_started_at) * 1000)

    audio_url = None

    if tts_result.get("audio_path"):
        audio_url = f"/tts-audio/{Path(tts_result.get('audio_path')).name}"

    errors = {}

    if stt_result.get("error"):
        errors["stt_error"] = stt_result.get("error")

    if llm_result.get("error"):
        errors["llm_error"] = llm_result.get("error")

    if tts_result.get("error"):
        errors["tts_error"] = tts_result.get("error")

    metric_id = save_voice_metric(
        peer_id=peer_id,
        transcript=transcript,
        answer=answer,
        audio_input_path=audio_input_path,
        audio_output_path=tts_result.get("audio_path"),
        stt_success=stt_result.get("success"),
        llm_success=llm_result.get("success"),
        tts_success=tts_result.get("success"),
        stt_latency_ms=stt_result.get("stt_latency_ms"),
        llm_first_token_ms=llm_result.get("llm_first_token_ms"),
        llm_total_ms=llm_result.get("llm_total_ms"),
        tts_first_byte_ms=tts_result.get("tts_first_byte_ms"),
        tts_total_ms=tts_result.get("tts_total_ms"),
        total_pipeline_ms=total_pipeline_ms,
        llm_model=llm_result.get("llm_model"),
        tts_voice=tts_result.get("tts_voice"),
        errors=errors
    )

    print(f"[{peer_id}] Metric kaydedildi. Metric ID: {metric_id}")

    latest_response.clear()
    latest_response.update({
        "has_response": True,
        "id": str(uuid.uuid4()),
        "metric_id": metric_id,
        "peer_id": peer_id,
        "transcript": transcript,
        "answer": answer,
        "audio_url": audio_url,
        "stt_llm_tts_status": "completed",
        "stt_latency_ms": stt_result.get("stt_latency_ms"),
        "llm_first_token_ms": llm_result.get("llm_first_token_ms"),
        "llm_total_ms": llm_result.get("llm_total_ms"),
        "tts_first_byte_ms": tts_result.get("tts_first_byte_ms"),
        "tts_total_ms": tts_result.get("tts_total_ms"),
        "total_pipeline_ms": total_pipeline_ms,
        "llm_model": llm_result.get("llm_model"),
        "tts_voice": tts_result.get("tts_voice"),
        "streaming_config": get_streaming_config()
    })

    llm_result["tts_result"] = tts_result
    llm_result["metric_id"] = metric_id
    llm_result["total_pipeline_ms"] = total_pipeline_ms

    return llm_result


async def process_saved_wav_with_stt_and_llm(peer_id, saved_path, label="STT sonucu"):
    pipeline_started_at = time.perf_counter()

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
    print(f"[{peer_id}] Push-to-talk mod aktif. Konuşmaya Başla butonu bekleniyor.")

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

        await process_saved_wav_with_stt_and_llm(
            peer_id=peer_id,
            saved_path=saved_path,
            label="STT sonucu"
        )

        reset_utterance()

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
                        print(f"[{peer_id}] Asistan sesi/ignore modu aktif, buffer temizlendi.")
                    reset_utterance()
                    continue

                if not is_user_speaking_enabled(peer_id):
                    if speech_active and utterance_samples > 0:
                        await process_current_utterance(reason="push_to_talk_stop")
                    else:
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
                speech_duration_seconds = utterance_speech_samples / current_sample_rate
                speech_ratio = utterance_speech_samples / max(utterance_samples, 1)

                if frame_count % 50 == 0:
                    print(
                        f"[{peer_id}] Dinleniyor. "
                        f"Frame count: {frame_count}, "
                        f"Frame RMS: {round(frame_rms, 2)}, "
                        f"User speaking: {is_user_speaking_enabled(peer_id)}, "
                        f"Speech active: {speech_active}, "
                        f"Utterance duration: {round(utterance_duration_seconds, 2)} sn, "
                        f"Speech duration: {round(speech_duration_seconds, 2)} sn, "
                        f"Speech ratio: {round(speech_ratio, 2)}"
                    )

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
        "min_speech_seconds": MIN_SPEECH_SECONDS,
        "min_speech_ratio": MIN_SPEECH_RATIO,
        "tts_output_dir": TTS_OUTPUT_DIR,
        "client_states": client_states
    })


@app.get("/config")
def config():
    return JSONResponse({
        "app": APP_NAME,
        "mode": "webrtc_push_to_talk_stt_llm_tts_metrics_mvp",
        "audio_config": {
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "min_audio_rms": MIN_AUDIO_RMS,
            "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
            "min_utterance_seconds": MIN_UTTERANCE_SECONDS,
            "max_utterance_seconds": MAX_UTTERANCE_SECONDS,
            "pre_speech_seconds": PRE_SPEECH_SECONDS,
            "min_speech_seconds": MIN_SPEECH_SECONDS,
            "min_speech_ratio": MIN_SPEECH_RATIO
        },
        "stt_config": get_stt_config(),
        "streaming_config": get_streaming_config()
    })


@app.get("/latest-response")
def get_latest_response():
    return JSONResponse(latest_response)


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
        "user_speaking": False,
        "assistant_playing": False,
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