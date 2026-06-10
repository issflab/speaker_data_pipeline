"""
YouTube Speaker Audio Scraper — v3 (fixed)
Usage: python scraper.py "Speaker Name"

Requires:
  pip install yt-dlp groq torch torchaudio soundfile python-dotenv
  ffmpeg on PATH
  GROQ_API_KEY — set in a .env file or as an environment variable
                 (free key at https://console.groq.com)
"""

import asyncio
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, set_key
from groq import Groq

import sys
print("PYTHON:", sys.executable)

load_dotenv()  # load .env from current directory if it exists

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_DURATION = 12000        # 200 min per video
MAX_DURATION = 15000        # 250 min per video
TOTAL_MIN_SECONDS = 12000   # 200 min total bucket
TOTAL_MAX_SECONDS = 15000   # 250 min total bucket
TOP_N_FOR_VAD = 20
GROQ_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Groq setup + helpers
# ---------------------------------------------------------------------------
def setup_groq() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("No GROQ_API_KEY found.")
        print("Get a free key at: https://console.groq.com\n")
        key = input("Paste your Groq API key here: ").strip()
        if not key:
            print("ERROR: No key entered. Exiting.")
            sys.exit(1)
        env_path = Path(".env")
        set_key(str(env_path), "GROQ_API_KEY", key)
        os.environ["GROQ_API_KEY"] = key
        print(f"Key saved to {env_path.resolve()} — won't be asked again.\n")
    return Groq(api_key=key)


def call_groq(client: Groq, prompt: str, retries: int = 3) -> str:
    """Call Groq with automatic retry on failure."""
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)  # exponential backoff
    raise RuntimeError(f"Groq call failed after {retries} attempts: {last_err}")


def robust_parse_json(text: str, kind: str = "object"):
    """
    Robustly extract and parse a JSON object or array from text.
    Handles: markdown fences, leading prose, trailing prose,
    smart quotes, and common Groq formatting quirks.
    """
    # 1. Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # 2. Replace smart/curly quotes that break JSON
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # 3. Find the first JSON structure of the right kind
    if kind == "array":
        # Find the outermost [...] block
        start = text.find("[")
        if start == -1:
            raise ValueError(f"No JSON array found in Groq response:\n{text[:300]}")
        # Walk to find the matching closing bracket
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise ValueError(f"Unmatched '[' in Groq response:\n{text[:300]}")
        json_str = text[start:end]
    else:
        # Find the outermost {...} block
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in Groq response:\n{text[:300]}")
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise ValueError(f"Unmatched '{{' in Groq response:\n{text[:300]}")
        json_str = text[start:end]

    # 4. Parse — if it fails, try to fix trailing commas
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Remove trailing commas before ] or }
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(fixed)


# ---------------------------------------------------------------------------
# Step 1 — Groq Query Generator
# ---------------------------------------------------------------------------
def generate_queries(speaker_name: str, client: Groq) -> list[str]:
    prompt = (
        f"Generate exactly 6 diverse YouTube search queries to find long-form videos "
        f"(ideally 40-70 minutes) where {speaker_name} is the SOLE or PRIMARY speaker.\n\n"
        f"Target video types: full podcast episodes, solo lectures, keynote talks, "
        f"long-form interviews where they are the HOST (not a guest).\n\n"
        f"Rules:\n"
        f'- Put the speaker name in double quotes inside each query string\n'
        f"- Vary the keywords: use podcast, lecture, talk, keynote, episode, solo, full\n"
        f"- Do NOT include queries for short clips, highlights, or panels\n\n"
        f"Return ONLY a valid JSON array of exactly 6 strings. "
        f"No explanation, no markdown, no extra text before or after.\n"
        f'Format: ["query 1", "query 2", "query 3", "query 4", "query 5", "query 6"]'
    )

    for attempt in range(3):
        try:
            text = call_groq(client, prompt)
            queries = robust_parse_json(text, kind="array")
            if not isinstance(queries, list):
                raise ValueError("Expected a list")
            # Clean up each query — ensure they are strings
            queries = [str(q).strip() for q in queries if q]
            if len(queries) < 3:
                raise ValueError(f"Too few queries returned: {queries}")
            return queries[:6]
        except Exception as e:
            print(f"      Query generation attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                # Fallback: hard-coded templates
                print("      Falling back to template queries.")
                return [
                    f'"{speaker_name}" full podcast',
                    f'"{speaker_name}" full lecture',
                    f'"{speaker_name}" keynote talk full',
                    f'"{speaker_name}" solo episode long',
                    f'"{speaker_name}" full interview host',
                    f'"{speaker_name}" talk full length',
                ]


# ---------------------------------------------------------------------------
# Step 2 — YouTube Search with yt-dlp
# ---------------------------------------------------------------------------
def search_youtube(query: str, max_results: int = 20) -> list[dict]:
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(
                f"ytsearch{max_results}:{query}", download=False
            )
            return results.get("entries", []) if results else []
    except Exception as e:
        print(f"      Search failed for query '{query}': {e}")
        return []


def fetch_full_metadata(video_id: str) -> Optional[dict]:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
        return None


def collect_candidates(queries: list[str]) -> list[dict]:
    raw: list[dict] = []
    for q in queries:
        results = search_youtube(q, max_results=20)
        raw.extend(results)
    return raw


# ---------------------------------------------------------------------------
# Step 3 — Deduplicate
# ---------------------------------------------------------------------------
def deduplicate(videos: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for v in videos:
        vid_id = v.get("id") or v.get("url", "")
        if vid_id and vid_id not in seen:
            seen.add(vid_id)
            unique.append(v)
    return unique


# ---------------------------------------------------------------------------
# Step 4 — Duration Filter (200–250 min per video)
# ---------------------------------------------------------------------------
def fast_filter_by_flat_duration(videos: list[dict]) -> list[dict]:
    """
    First pass using flat search data (no extra network calls).
    Videos with no duration info are kept for the enrich pass.
    """
    filtered, unknown = [], []
    for v in videos:
        dur = v.get("duration")
        if dur is None:
            unknown.append(v)
        elif MIN_DURATION <= dur <= MAX_DURATION:
            filtered.append(v)
    return filtered + unknown


def enrich_and_filter_duration(videos: list[dict]) -> list[dict]:
    """
    Second pass: fetch full metadata per video to confirm duration.
    Only keeps videos firmly in the 40–70 min range.
    """
    enriched = []
    for v in videos:
        vid_id = v.get("id")
        if not vid_id:
            continue
        meta = fetch_full_metadata(vid_id)
        if meta is None:
            continue
        duration = meta.get("duration") or 0
        if MIN_DURATION <= duration <= MAX_DURATION:
            meta["webpage_url"] = (
                meta.get("webpage_url")
                or f"https://www.youtube.com/watch?v={vid_id}"
            )
            enriched.append(meta)
    return enriched


# ---------------------------------------------------------------------------
# Step 5 — Groq Relevance Scorer (async, semaphore = 5)
# ---------------------------------------------------------------------------
async def score_single_video(
    video: dict,
    speaker_name: str,
    client: Groq,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        title = (video.get("title") or "")[:120]
        channel = (video.get("channel") or video.get("uploader", ""))[:60]
        description = (video.get("description") or "")[:300]

        prompt = (
            f"You are evaluating YouTube videos for a speaker audio dataset.\n"
            f"Speaker we want: {speaker_name}\n\n"
            f"Video title: {title}\n"
            f"Channel: {channel}\n"
            f"Description (first 300 chars): {description}\n\n"
            f"Score this video from 0 to 10 using these rules:\n"
            f"+3 if {speaker_name} is clearly the SOLE or PRIMARY speaker throughout\n"
            f"+3 if it is a full-length solo talk, lecture, or podcast episode\n"
            f"+2 if title/channel suggest clean studio or professional audio\n"
            f"-4 if this is a panel or {speaker_name} is only a brief guest\n"
            f"-3 if this is a live event with a crowd\n"
            f'-3 if "highlights", "clip", "short", or "compilation" is in the title\n'
            f"-2 if it is a reaction, review, or commentary about them\n\n"
            f"Return ONLY valid JSON with no markdown, no explanation, nothing else:\n"
            f'{{"score": <number 0-10>, "reason": "<one short sentence>"}}'
        )

        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, lambda: call_groq(client, prompt))
            parsed = robust_parse_json(text, kind="object")
            video["groq_score"] = float(parsed.get("score", 5.0))
            video["groq_reason"] = str(parsed.get("reason", ""))
        except Exception as e:
            video["groq_score"] = 5.0
            video["groq_reason"] = f"scoring failed: {e}"

        return video


async def score_all_videos(
    videos: list[dict], speaker_name: str, client: Groq
) -> list[dict]:
    semaphore = asyncio.Semaphore(5)
    tasks = [
        score_single_video(v, speaker_name, client, semaphore) for v in videos
    ]
    return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# Step 6 — Quality Heuristic Scorer
# ---------------------------------------------------------------------------
def heuristic_score(video: dict) -> float:
    score = 5.0
    title = (video.get("title") or "").lower()
    channel = (video.get("channel") or video.get("uploader", "")).lower()
    views = video.get("view_count") or 0

    # Positive signals
    if any(w in channel for w in ["podcast", "lecture", "lab", "talks", "university"]):
        score += 2
    if any(w in title for w in ["studio", "full episode", "full podcast", "full talk"]):
        score += 2
    if any(w in title for w in ["lecture", "talk", "keynote", "conversation"]):
        score += 1
    if views > 100_000:
        score += 2
    elif views > 50_000:
        score += 1

    # Negative signals
    if any(w in title for w in ["live", "panel", "q&a", "debate", "vs ", "rally"]):
        score -= 3
    if any(w in title for w in ["crowd", "audience", "applause", "arena", "stadium"]):
        score -= 3
    if any(w in title for w in ["highlights", "clip", "short", "compilation", "best of"]):
        score -= 4
    if views < 5_000:
        score -= 1

    return max(0.0, min(10.0, score))


# ---------------------------------------------------------------------------
# Step 7 — Combined Rank + Bucket Fill
# ---------------------------------------------------------------------------
def rank_and_shortlist(videos: list[dict], top_n: int = TOP_N_FOR_VAD) -> list[dict]:
    for v in videos:
        v["heuristic_score"] = heuristic_score(v)
        v["combined_score"] = (
            0.6 * v.get("groq_score", 5.0) + 0.4 * v["heuristic_score"]
        )
    ranked = sorted(videos, key=lambda v: v["combined_score"], reverse=True)
    return [v for v in ranked[:top_n] if v["combined_score"] >= 5.0]


def fill_duration_bucket(shortlist: list[dict]) -> list[dict]:
    selected, total = [], 0
    for v in shortlist:
        dur = v.get("duration") or 0
        if total >= TOTAL_MAX_SECONDS:
            break
        if total + dur > TOTAL_MAX_SECONDS + 600:
            # Video would push total more than 10 min over ceiling — skip
            continue
        selected.append(v)
        total += dur
    print(f"      -> {len(selected)} videos selected, total ~{total // 60} min")
    return selected


# ---------------------------------------------------------------------------
# Step 8 — Start_time Detection (Silero-VAD)
# ---------------------------------------------------------------------------
def download_first_n_minutes(url: str, minutes: int = 3) -> str:
    """
    Downloads only the first N minutes of audio using yt-dlp's
    --download-sections flag. Returns path to the downloaded .wav file.
    """
    tmp = tempfile.mktemp(suffix=".wav")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-x",
        "--audio-format", "wav",
        "--download-sections", f"*0-{minutes * 60}",
        "--force-keyframes-at-cuts",
        "-o", tmp,
        "--quiet",
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True, timeout=120, env=os.environ.copy())

    # yt-dlp sometimes appends an extra .wav extension on Windows
    for candidate in [tmp, tmp + ".wav"]:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Downloaded audio file not found. Expected near: {tmp}"
    )


def convert_to_16k_mono(input_path: str) -> str:
    """
    Converts any audio file to 16kHz mono WAV — the exact format
    Silero-VAD requires. Deletes the input file after conversion.
    """
    out_path = input_path.replace(".wav", "_16k.wav")
    # Handle case where input already ends in _16k.wav
    if out_path == input_path:
        out_path = input_path + "_16k.wav"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",   # resample to 16kHz
            "-ac", "1",       # convert to mono
            out_path,
            "-loglevel", "error",
        ],
        check=True,
        timeout=60,
    )
    try:
        os.remove(input_path)
    except OSError:
        pass
    return out_path


def get_first_speech_timestamp(wav_path: str) -> str:
    """
    Runs Silero-VAD on a 16kHz mono WAV file.
    Returns the timestamp of the first detected speech as H:MM:SS.
    Deletes the WAV file after processing.
    """
    import torch
    import soundfile as sf

    # Load model — cached locally after first download (~2MB)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    # Read audio into a float32 tensor
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]   # take first channel if stereo
    wav_tensor = torch.from_numpy(data)

    # Run VAD
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=500,
        min_silence_duration_ms=300,
    )

    # Clean up temp file immediately
    try:
        os.remove(wav_path)
    except OSError:
        pass

    if not speech_timestamps:
        return "0:00:00"

    # Convert sample index → seconds → H:MM:SS
    secs = speech_timestamps[0]["start"] / 16000
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def detect_start_time(url: str) -> str:
    """
    Full VAD pipeline for one URL. Returns H:MM:SS or "0:00:00" on failure.
    Never raises — always returns a string so the CSV row is always written.
    """
    try:
        wav = download_first_n_minutes(url, minutes=3)
        wav_16k = convert_to_16k_mono(wav)
        return get_first_speech_timestamp(wav_16k)
    except Exception as e:
        print(f"  VAD failed: {e}")
        return "0:00:00"


# ---------------------------------------------------------------------------
# Step 9 — CSV Output
# ---------------------------------------------------------------------------
def write_csv(speaker_name: str, results: list[dict]) -> str:
    filename = speaker_name.replace(" ", "_") + ".csv"
    fieldnames = ["Speaker_Name", "Data_type", "Link", "Start_time", "Processed"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Speaker_Name": speaker_name.replace(" ", "_"),
                "Data_type": "",
                "Link": r["url"],
                "Start_time": r["start_time"],
                "Processed": "",
            })
    return filename


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
async def main(speaker_name: str) -> None:
    client = setup_groq()

    print(f'\n[1/9] Generating search queries for "{speaker_name}" with Groq...')
    queries = generate_queries(speaker_name, client)
    for i, q in enumerate(queries, 1):
        print(f"      {i}. {q}")

    print(f"\n[2/9] Searching YouTube ({len(queries)} queries x 20 results)...")
    raw = collect_candidates(queries)
    print(f"      -> {len(raw)} raw results")

    print("\n[3/9] Deduplicating...")
    unique = deduplicate(raw)
    print(f"      -> {len(unique)} unique videos")

    print("\n[4/9] Filtering by duration (40-70 min per video)...")
    pre = fast_filter_by_flat_duration(unique)
    candidates = enrich_and_filter_duration(pre)
    print(f"      -> {len(candidates)} videos in range")

    if not candidates:
        print("No candidates after duration filter. Try a different speaker name.")
        sys.exit(0)

    print(f"\n[5/9] Scoring relevance with Groq ({len(candidates)} videos, async)...")
    candidates = await score_all_videos(candidates, speaker_name, client)
    print("      -> Done")

    print("\n[6/9] Applying quality heuristics...")
    for v in candidates:
        v["heuristic_score"] = heuristic_score(v)
    print("      -> Done")

    print("\n[7/9] Ranking and filling 200-250 min bucket...")
    shortlist = rank_and_shortlist(candidates)
    bucket = fill_duration_bucket(shortlist)

    if not bucket:
        print("No videos passed the score threshold.")
        sys.exit(0)

    total_dur = sum(v.get("duration", 0) for v in bucket)
    print(f"      -> Total: {total_dur // 60} min {total_dur % 60} sec")

    print("\n[8/9] Detecting start times with Silero-VAD...")
    results = []
    for i, video in enumerate(bucket, 1):
        url = (
            video.get("webpage_url")
            or f"https://www.youtube.com/watch?v={video.get('id', '')}"
        )
        dur_min = (video.get("duration") or 0) // 60
        score = round(video.get("combined_score", 0), 1)
        print(
            f"  [{i}/{len(bucket)}] (score {score}) {url} ({dur_min} min)",
            end=" -> ",
            flush=True,
        )
        start_time = detect_start_time(url)
        print(start_time)
        results.append({"url": url, "start_time": start_time})

    print("\n[9/9] Writing CSV...")
    filename = write_csv(speaker_name, results)
    print(f"      -> Saved {len(results)} rows to {filename}")
    print(f"         Total audio: ~{total_dur // 60} min")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scraper.py "Speaker Name"')
        print('Example: python scraper.py "Elon Musk"')
        sys.exit(1)
    speaker = " ".join(sys.argv[1:])
    asyncio.run(main(speaker))
