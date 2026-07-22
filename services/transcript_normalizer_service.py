def normalize_transcript_for_domain(transcript):
    transcript = str(transcript or "").strip()

    return {
        "original_transcript": transcript,
        "normalized_query": transcript,
        "correction_applied": False,
        "correction_reason": None,
        "confidence": 0
    }