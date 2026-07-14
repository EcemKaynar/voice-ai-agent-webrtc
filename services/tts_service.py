import os
import time
from pathlib import Path

import edge_tts
from dotenv import load_dotenv


load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge_tts")
TTS_VOICE = os.getenv("TTS_VOICE", "tr-TR-EmelNeural")
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "data/tts_outputs")

Path(TTS_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def get_tts_config():
    return {
        "tts_provider": TTS_PROVIDER,
        "tts_voice": TTS_VOICE,
        "tts_backend_streaming_enabled": True,
        "tts_frontend_streaming_enabled": True,
        "tts_mode": "http_streaming_audio_mpeg"
    }


async def stream_tts_audio_chunks(text):
    """
    Gerçek TTS streaming:
    Edge TTS audio chunk ürettikçe frontend'e gönderilir.
    Tamamlanmış MP3 dosyası beklenmez.
    """
    if TTS_PROVIDER != "edge_tts":
        raise RuntimeError(f"Unsupported TTS provider: {TTS_PROVIDER}")

    communicate = edge_tts.Communicate(
        text=str(text or ""),
        voice=TTS_VOICE
    )

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def synthesize_speech_with_metrics(text, peer_id="default"):
    """
    Eski fallback yöntem.
    Debug gerekirse TTS çıktısını MP3 dosyası olarak üretir.
    Ana akışta artık streaming endpoint kullanılıyor.
    """
    started_at = time.perf_counter()
    first_byte_at = None

    try:
        timestamp = int(time.time() * 1000)
        output_path = Path(TTS_OUTPUT_DIR) / f"{peer_id}_{timestamp}.mp3"

        communicate = edge_tts.Communicate(
            text=str(text or ""),
            voice=TTS_VOICE
        )

        with open(output_path, "wb") as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()

                    file.write(chunk["data"])

        finished_at = time.perf_counter()

        tts_first_byte_ms = None

        if first_byte_at is not None:
            tts_first_byte_ms = int((first_byte_at - started_at) * 1000)

        return {
            "success": True,
            "audio_path": str(output_path),
            "tts_provider": TTS_PROVIDER,
            "tts_voice": TTS_VOICE,
            "tts_first_byte_ms": tts_first_byte_ms,
            "tts_total_ms": int((finished_at - started_at) * 1000),
            "error": None
        }

    except Exception as error:
        finished_at = time.perf_counter()

        return {
            "success": False,
            "audio_path": None,
            "tts_provider": TTS_PROVIDER,
            "tts_voice": TTS_VOICE,
            "tts_first_byte_ms": None,
            "tts_total_ms": int((finished_at - started_at) * 1000),
            "error": str(error)
        }