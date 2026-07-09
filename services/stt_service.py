import os
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

STT_PROVIDER = os.getenv("STT_PROVIDER", "faster_whisper")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "tr")

_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        from faster_whisper import WhisperModel

        print(f"Whisper modeli yükleniyor: {WHISPER_MODEL_SIZE}")

        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

        print("Whisper modeli hazır.")

    return _whisper_model


def transcribe_audio_file(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {
            "success": False,
            "text": "",
            "provider": STT_PROVIDER,
            "audio_path": str(audio_path),
            "stt_latency_ms": 0,
            "error": "Audio file not found."
        }

    started_at = time.perf_counter()

    try:
        if STT_PROVIDER != "faster_whisper":
            raise ValueError(f"Desteklenmeyen STT_PROVIDER: {STT_PROVIDER}")

        model = get_whisper_model()

        segments, info = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            vad_filter=False,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.2,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0
        )

        texts = []

        for segment in segments:
            segment_text = str(segment.text or "").strip()

            if segment_text:
                texts.append(segment_text)

        transcript = " ".join(texts).strip()

        finished_at = time.perf_counter()
        stt_latency_ms = int((finished_at - started_at) * 1000)

        return {
            "success": True,
            "text": transcript,
            "provider": STT_PROVIDER,
            "model": WHISPER_MODEL_SIZE,
            "language": WHISPER_LANGUAGE,
            "audio_path": str(audio_path),
            "stt_latency_ms": stt_latency_ms,
            "detected_language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None)
        }

    except Exception as error:
        finished_at = time.perf_counter()
        stt_latency_ms = int((finished_at - started_at) * 1000)

        return {
            "success": False,
            "text": "",
            "provider": STT_PROVIDER,
            "model": WHISPER_MODEL_SIZE,
            "language": WHISPER_LANGUAGE,
            "audio_path": str(audio_path),
            "stt_latency_ms": stt_latency_ms,
            "error": str(error)
        }