import subprocess
from typing import List, Dict, Any, Optional

def assign_words_to_speakers(
    words: List[Dict[str, Any]],
    diar_turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Assign each word to the speaker whose diarization turn overlaps the word midpoint.
    """
    assigned = []
    j = 0
    for w in words:
        mid = 0.5 * (w["start"] + w["end"])

        while j < len(diar_turns) and diar_turns[j]["end"] <= mid:
            j += 1

        spk = None
        if j < len(diar_turns):
            t = diar_turns[j]
            if t["start"] <= mid < t["end"]:
                spk = t["speaker"]

        assigned.append({**w, "speaker": spk})
    return assigned


def build_segments(
    assigned_words: List[Dict[str, Any]],
    gap_merge: float = 0.35,
    min_seg: float = 1.0,
    max_seg: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Build segments by grouping consecutive words with the same speaker.
    - merge adjacent segments of same speaker if small gap
    - split if segment exceeds max_seg
    """
    # First pass: group consecutive words by speaker (skip words with no speaker)
    chunks = []
    cur = None

    for w in assigned_words:
        if w.get("speaker") is None:
            continue
        if cur is None:
            cur = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [w["word"]],
                "num_words": 1,
            }
            continue

        if w["speaker"] == cur["speaker"] and (w["start"] - cur["end"]) <= gap_merge:
            cur["end"] = max(cur["end"], w["end"])
            cur["words"].append(w["word"])
            cur["num_words"] += 1
        else:
            chunks.append(cur)
            cur = {
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "words": [w["word"]],
                "num_words": 1,
            }

    if cur is not None:
        chunks.append(cur)

    # Second pass: enforce min/max length
    segments = []
    for c in chunks:
        dur = c["end"] - c["start"]
        if dur < min_seg:
            continue

        text = "".join(c["words"]).strip()

        if dur <= max_seg:
            segments.append({
                "speaker": c["speaker"],
                "start": c["start"],
                "end": c["end"],
                "text": text,
                "num_words": c["num_words"],
            })
        else:
            # Split long segment into windows of <= max_seg
            s = c["start"]
            e = c["end"]
            idx = 0
            while s < e:
                ee = min(s + max_seg, e)
                segments.append({
                    "speaker": c["speaker"],
                    "start": s,
                    "end": ee,
                    "text": text,  # (simple: same text; if you want, we can slice words by time)
                    "num_words": c["num_words"],
                    "split_index": idx,
                })
                s = ee
                idx += 1

    return segments


def cut_segments_ffmpeg(wav_path: str, start: float, end: float, out_path: str, sr: int = 16000):
    """
    Accurate cutting: use -ss before -i for speed or after -i for accuracy.
    Here we prioritize accuracy.
    """
    dur = max(0.0, end - start)
    if dur <= 0.0:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-ac", "1",
        "-ar", str(sr),
        out_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr}")
    
from typing import List, Dict, Any


def build_segments_from_diarization(
    diar_turns: List[Dict[str, Any]],
    gap_merge: float = 0.35,
    min_seg: float = 1.0,
    max_seg: float = 20.0,
) -> List[Dict[str, Any]]:
    """
    Create segments from diarization turns only.
    - merges adjacent turns of same speaker if gap <= gap_merge
    - enforces min/max segment duration
    """
    if not diar_turns:
        return []

    diar_turns = sorted(diar_turns, key=lambda x: x["start"])

    merged = []
    cur = diar_turns[0].copy()

    for t in diar_turns[1:]:
        same = (t["speaker"] == cur["speaker"])
        gap = t["start"] - cur["end"]

        if same and gap <= gap_merge:
            cur["end"] = max(cur["end"], t["end"])
        else:
            merged.append(cur)
            cur = t.copy()
    merged.append(cur)

    segments: List[Dict[str, Any]] = []
    for m in merged:
        dur = m["end"] - m["start"]
        if dur < min_seg:
            continue

        if dur <= max_seg:
            segments.append({
                "speaker": m["speaker"],
                "start": m["start"],
                "end": m["end"],
                "text": "",        # no ASR
                "num_words": 0,
            })
        else:
            # split long segments
            s = m["start"]
            e = m["end"]
            while s < e:
                ee = min(s + max_seg, e)
                if ee - s >= min_seg:
                    segments.append({
                        "speaker": m["speaker"],
                        "start": s,
                        "end": ee,
                        "text": "",
                        "num_words": 0,
                    })
                s = ee

    return segments


def build_fixed_window_segments(
    audio_duration: float,
    window: float = 5.0,
    overlap: float = 0.25,
    min_last: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Build overlapping fixed window segments covering [0, audio_duration].
    step = window - overlap
    min_last: minimum duration for the final partial window (otherwise drop it).
    """
    if window <= 0:
        raise ValueError("window must be > 0")
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must be in [0, window)")

    step = window - overlap
    segments: List[Dict[str, Any]] = []

    t = 0.0
    idx = 0
    while t < audio_duration:
        end = min(t + window, audio_duration)
        dur = end - t
        if dur >= min_last:
            segments.append({
                "segment_index": idx,
                "start": t,
                "end": end,
                "duration": dur,
            })
            idx += 1
        t += step

    return segments