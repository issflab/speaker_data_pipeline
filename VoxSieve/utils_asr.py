from typing import Optional, Dict, Any, List
from faster_whisper import WhisperModel

def run_asr(
    wav_path: str,
    model: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
    chunksize: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns dict with:
      words: list of {start, end, word, prob}
    """

    wm = WhisperModel(model, device=device, compute_type=compute_type)

    segments, info = wm.transcribe(
        wav_path,
        language=language,
        word_timestamps=True,
        vad_filter=False,
        beam_size=2,
        chunk_length=chunksize
    )

    words: List[Dict[str, Any]] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            # Some words might miss timestamps in edge cases; skip those
            if w.start is None or w.end is None:
                continue
            words.append({
                "start": float(w.start),
                "end": float(w.end),
                "word": w.word,
                "prob": float(w.probability) if w.probability is not None else None,
            })

    words.sort(key=lambda x: x["start"])
    return {
        "detected_language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "words": words,
    }
