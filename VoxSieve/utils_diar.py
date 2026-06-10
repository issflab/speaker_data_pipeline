from typing import Optional, Dict, Any, List

# Lazy imports to avoid torch/pyannote import costs when not used
from pyannote.audio import Pipeline

def run_diarization(
    wav_path: str,
    hf_token: Optional[str],
    num_speakers: Optional[int] = None,
    min_turn: float = 0.6,
) -> Dict[str, Any]:
    """
    Returns:
      turns: list of dicts {start, end, speaker}
      rttm_lines: list of RTTM strings
    """
    if hf_token is None:
        raise RuntimeError("pyannote diarization needs a HuggingFace token. Set HF_TOKEN or pass --hf-token.")

    # pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=hf_token)

    diarization = pipe(wav_path, num_speakers=num_speakers) if num_speakers else pipe(wav_path)

    turns: List[Dict[str, Any]] = []
    rttm_lines: List[str] = []

    # iterate diarization results
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end - start < min_turn:
            continue
        turns.append({"start": start, "end": end, "speaker": speaker})

        # RTTM line format:
        # SPEAKER <file-id> 1 <start> <duration> <ortho> <stype> <name> <conf> <slat>
        duration = end - start
        file_id = "audio"
        rttm_lines.append(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>")

    turns.sort(key=lambda x: x["start"])
    return {"turns": turns, "rttm_lines": rttm_lines}
