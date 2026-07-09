import wave
from pathlib import Path

import numpy as np
from av.audio.resampler import AudioResampler


def create_audio_resampler(target_sample_rate=16000):
    """
    WebRTC'den gelen sesi STT için uygun formata çevirir:
    mono, s16, 16kHz
    """
    return AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sample_rate
    )


def audio_frame_to_mono_int16(frame, resampler):
    """
    aiortc/PyAV audio frame'i düzgün şekilde mono int16 PCM'e çevirir.
    Cızırtı problemini önlemek için manuel kanal karıştırmak yerine
    PyAV AudioResampler kullanıyoruz.
    """
    resampled_frames = resampler.resample(frame)

    pcm_arrays = []

    for resampled_frame in resampled_frames:
        array = resampled_frame.to_ndarray()
        array = np.asarray(array)

        # Mono s16 genelde (1, samples) gelir.
        if array.ndim == 2:
            if array.shape[0] == 1:
                array = array[0]
            elif array.shape[1] == 1:
                array = array[:, 0]
            else:
                array = array.mean(axis=0)

        array = np.squeeze(array)

        if array.dtype != np.int16:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(array, -1.0, 1.0)
                array = array * 32767

            array = array.astype(np.int16)

        pcm_arrays.append(array)

    return pcm_arrays


def save_pcm_chunks_to_wav(
    pcm_chunks,
    output_path,
    sample_rate=16000
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pcm_chunks:
        raise ValueError("Kaydedilecek ses chunk'ı bulunamadı.")

    audio = np.concatenate(pcm_chunks).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return str(output_path)


def calculate_rms_int16(pcm_chunks):
    if not pcm_chunks:
        return 0.0

    audio = np.concatenate(pcm_chunks).astype(np.float32)

    if audio.size == 0:
        return 0.0

    rms = np.sqrt(np.mean(audio ** 2))

    return float(rms)