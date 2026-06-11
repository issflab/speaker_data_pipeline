from typing import Optional, Dict, Any, List

import torch
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
        raise RuntimeError(
            "pyannote diarization needs a HuggingFace token. "
            "Set HF_TOKEN or pass --hf-token."
        )

    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        pipe.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    output = (
        pipe(wav_path, num_speakers=num_speakers)
        if num_speakers
        else pipe(wav_path)
    )

    # pyannote.audio 4.x returns DiarizeOutput
    diarization = getattr(output, "speaker_diarization", output)

    turns: List[Dict[str, Any]] = []
    rttm_lines: List[str] = []

    for segment, _, speaker in diarization.itertracks(yield_label=True):

        start = float(segment.start)
        end = float(segment.end)

        if (end - start) < min_turn:
            continue

        turns.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
            }
        )

        duration = end - start
        file_id = "audio"
        rttm_lines.append(
            f"SPEAKER {file_id} 1 "
            f"{start:.3f} "
            f"{duration:.3f} "
            f"<NA> <NA> "
            f"{speaker} "
            f"<NA> <NA>"
        )

    turns.sort(key=lambda x: x["start"])

    print(f"Found {len(turns)} speaker turns")

    return {
        "turns": turns,
        "rttm_lines": rttm_lines,
    }