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

from services.stt_service import transcribe_audio_file
from services.llm_service import ask_llm_with_metrics
from services.tts_service import synthesize_speech_with_metrics


load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Voice AI Agent WebRTC")

AUDIO_CHUNK_SECONDS = float(os.getenv("AUDIO_CHUNK_SECONDS", "10"))
AUDIO_OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "data/audio_chunks")
MIN_AUDIO_RMS = float(os.getenv("MIN_AUDIO_RMS", "50"))
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))

SPEECH_RMS_THRESHOLD = float(os.getenv("SPEECH_RMS_THRESHOLD", "120"))
SILENCE_END_SECONDS = float(os.getenv("SILENCE_END_SECONDS", "1.2"))
MIN_UTTERANCE_SECONDS = float(os.getenv("MIN_UTTERANCE_SECONDS", "1.0"))
MAX_UTTERANCE_SECONDS = float(os.getenv("MAX_UTTERANCE_SECONDS", "12"))
PRE_SPEECH_SECONDS = float(os.getenv("PRE_SPEECH_SECONDS", "0.3"))

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


class OfferRequest(BaseModel):
    sdp: str
    type: str


def make_audio_filename(peer_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]

    return Path(AUDIO_OUTPUT_DIR) / f"{peer_id}_{timestamp}_{short_id}.wav"


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
        "thanks for watching"
    ]

    if any(phrase in text for phrase in ignored_phrases):
        return True

    return False


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
        print(f"[{peer_id}] Transcript boş veya filtrelendi, LLM'e gönderilmeyecek.")
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
        "tts_voice": tts_result.get("tts_voice")
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

    print(f"[{peer_id}] Audio track dinleniyor...")
    print(f"[{peer_id}] Konuşmaya başlayabilirsin. Sistem susunca cümleyi işleyecek.")

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

        speech_active = False
        pre_speech_chunks = []
        pre_speech_samples = 0
        utterance_chunks = []
        utterance_samples = 0
        silence_samples = 0

    async def process_current_utterance(reason):
        nonlocal utterance_chunks
        nonlocal utterance_samples
        nonlocal silence_samples

        if not utterance_chunks or utterance_samples <= 0:
            reset_utterance()
            return

        utterance_duration_seconds = utterance_samples / current_sample_rate
        chunk_rms = calculate_rms_int16(utterance_chunks)

        print(
            f"[{peer_id}] Konuşma parçası bitti. "
            f"Reason: {reason}, "
            f"Duration: {round(utterance_duration_seconds, 2)} sn, "
            f"RMS: {round(chunk_rms, 2)}"
        )

        if utterance_duration_seconds < MIN_UTTERANCE_SECONDS:
            print(f"[{peer_id}] Konuşma çok kısa, atlandı.")
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

                frame_rms = calculate_rms_int16([pcm])
                is_speech = frame_rms >= SPEECH_RMS_THRESHOLD

                if not speech_active:
                    add_to_pre_speech_buffer(pcm)

                    if is_speech:
                        speech_active = True

                        utterance_chunks = list(pre_speech_chunks)
                        utterance_samples = sum(len(item) for item in utterance_chunks)
                        silence_samples = 0

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
                else:
                    silence_samples += len(pcm)

                utterance_duration_seconds = utterance_samples / current_sample_rate
                silence_duration_seconds = silence_samples / current_sample_rate

                if frame_count % 50 == 0:
                    print(
                        f"[{peer_id}] Dinleniyor. "
                        f"Frame count: {frame_count}, "
                        f"Frame RMS: {round(frame_rms, 2)}, "
                        f"Speech active: {speech_active}, "
                        f"Utterance duration: {round(utterance_duration_seconds, 2)} sn, "
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
        "active_peer_connections": len(pcs),
        "audio_chunk_seconds": AUDIO_CHUNK_SECONDS,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "min_audio_rms": MIN_AUDIO_RMS,
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
        "silence_end_seconds": SILENCE_END_SECONDS,
        "tts_output_dir": TTS_OUTPUT_DIR
    })


@app.get("/config")
def config():
    return JSONResponse({
        "app": APP_NAME,
        "mode": "webrtc_stt_llm_tts_metrics_mvp",
        "audio_chunk_seconds": AUDIO_CHUNK_SECONDS,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "min_audio_rms": MIN_AUDIO_RMS,
        "speech_rms_threshold": SPEECH_RMS_THRESHOLD,
        "silence_end_seconds": SILENCE_END_SECONDS,
        "min_utterance_seconds": MIN_UTTERANCE_SECONDS,
        "max_utterance_seconds": MAX_UTTERANCE_SECONDS,
        "pre_speech_seconds": PRE_SPEECH_SECONDS
    })


@app.get("/latest-response")
def get_latest_response():
    return JSONResponse(latest_response)


@app.get("/metrics")
def metrics(limit: int = 10):
    items = get_recent_voice_metrics(limit=limit)

    return JSONResponse({
        "count": len(items),
        "items": items
    })


@app.post("/offer")
async def offer(request: OfferRequest):
    peer_id = f"peer_{len(pcs) + 1}"

    pc = RTCPeerConnection()
    pcs.add(pc)

    print(f"[{peer_id}] Yeni WebRTC bağlantısı oluşturuldu.")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[{peer_id}] Connection state: {pc.connectionState}")

        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            pcs.discard(pc)
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
        "type": pc.localDescription.type
    })


@app.on_event("shutdown")
async def on_shutdown():
    coroutines = [pc.close() for pc in list(pcs)]
    await asyncio.gather(*coroutines)
    pcs.clear()