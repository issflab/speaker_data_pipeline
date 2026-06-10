from typing import List, Dict, Any, Optional


def pick_speaker_at_time(
    diar_turns: List[Dict[str, Any]],
    t: float,
) -> Optional[str]:
    """
    Return the speaker label active at time t (seconds).
    """
    for turn in diar_turns:
        if turn["start"] <= t < turn["end"]:
            return turn["speaker"]
    return None


def pick_speaker_by_window(
    diar_turns: List[Dict[str, Any]],
    t: float,
    window: float = 2.0,
) -> Optional[str]:
    """
    Pick the speaker with the maximum overlap with
    the window [t - window/2, t + window/2].
    """
    if not diar_turns:
        return None

    a = max(0.0, t - window / 2)
    b = t + window / 2

    overlap = {}
    for turn in diar_turns:
        s = max(a, turn["start"])
        e = min(b, turn["end"])
        if e > s:
            overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0.0) + (e - s)

    if not overlap:
        return None

    return max(overlap.items(), key=lambda x: x[1])[0]


def filter_turns_to_speaker(
    diar_turns: List[Dict[str, Any]],
    speaker: str,
) -> List[Dict[str, Any]]:
    """
    Keep only diarization turns belonging to `speaker`.
    """
    return [t for t in diar_turns if t["speaker"] == speaker]
