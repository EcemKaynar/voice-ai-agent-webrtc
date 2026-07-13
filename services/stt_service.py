import os
import time
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel


load_dotenv()

STT_PROVIDER = os.getenv("STT_PROVIDER", "faster_whisper")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "tr")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))

WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.2"))
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.0"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(
    os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4")
)

_model = None
_model_loaded_at = None
_model_load_latency_ms = None


def get_stt_config():
    return {
        "stt_provider": STT_PROVIDER,
        "whisper_model_size": WHISPER_MODEL_SIZE,
        "whisper_language": WHISPER_LANGUAGE,
        "whisper_device": WHISPER_DEVICE,
        "whisper_compute_type": WHISPER_COMPUTE_TYPE,
        "whisper_beam_size": WHISPER_BEAM_SIZE,
        "whisper_best_of": WHISPER_BEST_OF,
        "whisper_vad_filter": WHISPER_VAD_FILTER,
        "stt_streaming_enabled": False,
        "stt_mode": "utterance_based_batch_transcription",
        "stt_model_loaded": is_stt_model_loaded(),
        "stt_model_load_latency_ms": _model_load_latency_ms,
        "stt_model_loaded_at": _model_loaded_at,
    }


def is_stt_model_loaded():
    return _model is not None


def get_stt_model():
    global _model
    global _model_loaded_at
    global _model_load_latency_ms

    if _model is not None:
        return _model

    started_at = time.perf_counter()

    print(
        "Whisper modeli preload ediliyor. "
        f"Model: {WHISPER_MODEL_SIZE}, "
        f"Device: {WHISPER_DEVICE}, "
        f"Compute type: {WHISPER_COMPUTE_TYPE}"
    )

    _model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE
    )

    finished_at = time.perf_counter()

    _model_load_latency_ms = int((finished_at - started_at) * 1000)
    _model_loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    print(
        "Whisper modeli hazır. "
        f"Load latency: {_model_load_latency_ms} ms"
    )

    return _model


def preload_stt_model():
    """
    Server açılırken çağrılır.
    Böylece ilk kullanıcı konuşmasında model yükleme süresi STT latency içine girmez.
    """
    if STT_PROVIDER != "faster_whisper":
        print(f"STT provider faster_whisper değil: {STT_PROVIDER}")
        return {
            "success": False,
            "message": f"Unsupported STT provider: {STT_PROVIDER}"
        }

    model = get_stt_model()

    return {
        "success": model is not None,
        "provider": STT_PROVIDER,
        "model": WHISPER_MODEL_SIZE,
        "language": WHISPER_LANGUAGE,
        "model_load_latency_ms": _model_load_latency_ms
    }


def transcribe_audio_file(audio_path):
    started_at = time.perf_counter()

    audio_path = str(audio_path)

    if not Path(audio_path).exists():
        finished_at = time.perf_counter()

        return {
            "success": False,
            "text": "",
            "provider": STT_PROVIDER,
            "model": WHISPER_MODEL_SIZE,
            "language": WHISPER_LANGUAGE,
            "audio_path": audio_path,
            "stt_latency_ms": int((finished_at - started_at) * 1000),
            "error": f"Audio file not found: {audio_path}"
        }

    if STT_PROVIDER != "faster_whisper":
        finished_at = time.perf_counter()

        return {
            "success": False,
            "text": "",
            "provider": STT_PROVIDER,
            "model": WHISPER_MODEL_SIZE,
            "language": WHISPER_LANGUAGE,
            "audio_path": audio_path,
            "stt_latency_ms": int((finished_at - started_at) * 1000),
            "error": f"Unsupported STT provider: {STT_PROVIDER}"
        }

    try:
        model = get_stt_model()

        segments, info = model.transcribe(
            audio_path,
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            vad_filter=WHISPER_VAD_FILTER,
            condition_on_previous_text=False,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD
        )

        texts = []

        for segment in segments:
            segment_text = str(segment.text or "").strip()

            if segment_text:
                texts.append(segment_text)

        transcript = " ".join(texts).strip()

        finished_at = time.perf_counter()

        return {
            "success": True,
            "text": transcript,
            "provider": STT_PROVIDER,
            "model": WHISPER_MODEL_SIZE,
            "language": WHISPER_LANGUAGE,
            "audio_path": audio_path,
            "duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
            "stt_latency_ms": int((finished_at - started_at) * 1000),
            "error": None
        }

    except Exception as error:
        finished_at = time.perf_counter()

        return {
            "success": False,
            "text": "",
            "provider": STT_PROVIDER,
            "model": WHISPER_MODEL_SIZE,
            "language": WHISPER_LANGUAGE,
            "audio_path": audio_path,
            "stt_latency_ms": int((finished_at - started_at) * 1000),
            "error": str(error)
        }