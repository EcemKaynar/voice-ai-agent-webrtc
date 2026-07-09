import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import edge_tts
from dotenv import load_dotenv


load_dotenv()

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge_tts")
TTS_VOICE = os.getenv("TTS_VOICE", "tr-TR-EmelNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")
TTS_VOLUME = os.getenv("TTS_VOLUME", "+0%")
TTS_OUTPUT_DIR = os.getenv("TTS_OUTPUT_DIR", "data/tts_outputs")


def make_tts_filename(peer_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = str(uuid.uuid4())[:8]

    output_dir = Path(TTS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / f"{peer_id}_{timestamp}_{short_id}.mp3"


async def synthesize_speech_with_metrics(text, peer_id="peer"):
    started_at = time.perf_counter()
    first_byte_at = None

    text = str(text or "").strip()

    if not text:
        return {
            "success": False,
            "audio_path": None,
            "tts_provider": TTS_PROVIDER,
            "tts_voice": TTS_VOICE,
            "tts_first_byte_ms": None,
            "tts_total_ms": 0,
            "error": "TTS için boş metin geldi."
        }

    output_path = make_tts_filename(peer_id)

    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=TTS_VOICE,
            rate=TTS_RATE,
            volume=TTS_VOLUME
        )

        with open(output_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    if first_byte_at is None:
                        first_byte_at = time.perf_counter()

                    audio_file.write(chunk["data"])

        finished_at = time.perf_counter()

        tts_first_byte_ms = None

        if first_byte_at is not None:
            tts_first_byte_ms = int((first_byte_at - started_at) * 1000)

        tts_total_ms = int((finished_at - started_at) * 1000)

        return {
            "success": True,
            "audio_path": str(output_path),
            "tts_provider": TTS_PROVIDER,
            "tts_voice": TTS_VOICE,
            "tts_first_byte_ms": tts_first_byte_ms,
            "tts_total_ms": tts_total_ms,
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