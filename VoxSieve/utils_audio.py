import subprocess
from pathlib import Path
import json


def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}")
    return p


def download_youtube_audio(url: str, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # yt-dlp will choose best audio and write to the exact file path
    # cmd = [
    #     "yt-dlp",
    #     "-f", "bestaudio/best",
    #     "--no-playlist",
    #     "-o", str(out_path),
    #     url,
    # ]
    cmd = [
    "yt-dlp",
    "--js-runtimes", "node",
    "--cookies", "/home/lkolluru/speaker_data_pipeline/cookies.txt",
    "-f", "bestaudio/best",
    "--no-playlist",
    "-o", str(out_path),
    url,
]
    _run(cmd)

    if not out_path.exists():
        # yt-dlp sometimes changes extension; handle this by searching
        candidates = list(out_path.parent.glob(out_path.name + ".*"))
        if candidates:
            candidates[0].rename(out_path)
        else:
            raise FileNotFoundError(f"Download did not create {out_path}")


def standardize_wav(in_path: Path, out_wav: Path, sr: int = 16000, normalize: bool = False):
    """
    Convert input audio to mono WAV at `sr`.
    If normalize=True, apply ffmpeg loudness normalization (loudnorm).
    """
    in_path = Path(in_path)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Audio filter chain
    # loudnorm in one-pass mode: good practical default

    af_chain = [f"aresample={sr}"]  # forces true resampling
    if normalize:
        af_chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    af = ",".join(af_chain)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vn",
        "-ac", "1",
        "-af", af,
        "-f", "wav",
        str(out_wav),
    ]
    _run(cmd)


def get_audio_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    data = json.loads(p.stdout)
    return float(data["format"]["duration"])
