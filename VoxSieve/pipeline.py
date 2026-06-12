#!/usr/bin/env python3
"""
How to use:
    YouTube URL:
        python3 pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"

    Text file with one YouTube URL per line:
        python3 pipeline.py urls.txt

    CSV manifest with a required Link column:
        python3 pipeline.py manifest.csv

    Local video directory (scanned recursively):
        python3 pipeline.py /path/to/videos
        python3 pipeline.py /path/to/videos --input-mode dir
        # Outputs go into one folder named after the input directory.
        # Each video becomes its own audio/segment files inside that folder.

    Reprocess items even if outputs already exist:
        python3 pipeline.py /path/to/videos --overwrite
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from utils_audio import download_youtube_audio, standardize_wav, get_audio_duration_seconds
from utils_diar import run_diarization
from utils_asr import run_asr
from utils_assign import (
    assign_words_to_speakers,
    build_segments,
    build_segments_from_diarization,
    cut_segments_ffmpeg,
    build_fixed_window_segments,
)
from utils_target import pick_speaker_by_window, filter_turns_to_speaker

from dotenv import load_dotenv
load_dotenv()


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


# ── Output path containers ────────────────────────────────────────────────────

@dataclass
class BatchPaths:
    """Shared output paths when processing a local video directory as a batch."""
    root: Path
    segments_dir: Path
    target_segments_dir: Path
    meta_jsonl: Path
    meta_csv: Path
    target_meta_jsonl: Path
    target_meta_csv: Path


@dataclass
class ItemPaths:
    """Resolved output paths for a single input item."""
    vid_dir: Path
    seg_dir: Path
    target_seg_dir: Path
    meta_jsonl: Path
    meta_csv: Path
    target_meta_jsonl: Path
    target_meta_csv: Path
    wav_path: Path
    raw_audio: Path
    segment_prefix: str
    safe_id: str


# ── Input parsing ─────────────────────────────────────────────────────────────

def parse_time_to_seconds(x) -> float:
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    parts = s.split(":")
    parts = [float(p) for p in parts]

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s

    raise ValueError(f"Invalid time format: {x}")


def load_manifest(path: str) -> pd.DataFrame:
    p = Path(path)

    if not p.exists():
        return pd.DataFrame([{
            "Speaker_Name": "unknown",
            "Data_type": "unknown",
            "Link": path,
            "Start_time": 0.0,
            "Processed": "No",
        }])

    df = pd.read_csv(p)

    missing = {"Link"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    if "Start_time" in df.columns:
        df["Start_time_sec"] = df["Start_time"].apply(parse_time_to_seconds)
    else:
        df["Start_time_sec"] = 0.0

    if "Processed" not in df.columns:
        df["Processed"] = "No"
    df["Processed"] = df["Processed"].fillna("No").astype(str).str.lower()

    if "Speaker_Name" not in df.columns:
        df["Speaker_Name"] = "unknown"

    if "Data_type" not in df.columns:
        df["Data_type"] = "unknown"

    return df


def load_url_file(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        link = line.strip()
        if not link or link.startswith("#"):
            continue
        rows.append({
            "Speaker_Name": "unknown",
            "Data_type": "unknown",
            "Link": link,
            "Start_time_sec": 0.0,
            "Processed": "no",
        })
    return pd.DataFrame(rows)


def scan_video_directory(path: Path) -> pd.DataFrame:
    matches = sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    rows = []
    for video_path in matches:
        relative = video_path.relative_to(path)
        rows.append({
            "Speaker_Name": "unknown",
            "Data_type": "local_video",
            "Link": str(video_path.resolve()),
            "Start_time_sec": 0.0,
            "Processed": "no",
            "Source_Type": "local_video",
            "Source_ID": sanitize_source_id(str(relative.with_suffix(""))),
            "Batch_ID": sanitize_source_id(path.resolve().name),
        })
    return pd.DataFrame(rows)


def build_input_items(input_value: str, input_mode: str = "auto") -> pd.DataFrame:
    p = Path(input_value).expanduser()

    if input_mode == "dir":
        if not p.exists():
            raise ValueError(f"Input directory does not exist: {p}")
        if not p.is_dir():
            raise ValueError(f"Expected a directory for --input-mode dir, got: {p}")
        df = scan_video_directory(p)
        if df.empty:
            raise ValueError(
                f"No supported video files found in directory: {p}. "
                f"Scanned recursively for {sorted(VIDEO_EXTENSIONS)}"
            )
        return df

    if input_mode == "youtube":
        return pd.DataFrame([{
            "Speaker_Name": "unknown",
            "Data_type": "unknown",
            "Link": input_value,
            "Start_time_sec": 0.0,
            "Processed": "no",
            "Source_Type": "youtube",
            "Source_ID": sanitize_source_id(input_value),
        }])

    if not p.exists():
        return pd.DataFrame([{
            "Speaker_Name": "unknown",
            "Data_type": "unknown",
            "Link": input_value,
            "Start_time_sec": 0.0,
            "Processed": "no",
            "Source_Type": "youtube",
            "Source_ID": sanitize_source_id(input_value),
        }])

    if p.is_dir():
        df = scan_video_directory(p)
        if df.empty:
            raise ValueError(
                f"No supported video files found in directory: {p}. "
                f"Scanned recursively for {sorted(VIDEO_EXTENSIONS)}"
            )
        return df

    if p.suffix.lower() == ".csv":
        df = load_manifest(str(p))
    else:
        df = load_url_file(p)
        if df.empty:
            raise ValueError(f"Input file is empty: {p}")

    df["Source_Type"] = "youtube"
    df["Source_ID"] = df["Link"].apply(sanitize_source_id)
    return df


# ── Utilities ─────────────────────────────────────────────────────────────────

def sanitize_source_id(value: str) -> str:
    safe = value.replace("https://", "").replace("http://", "")
    for ch in ("/", "\\", "?", "&", ":", "=", "#", " "):
        safe = safe.replace(ch, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("._") or "item"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_asr_runtime(device: str, compute_type: str) -> tuple[str, str]:
    normalized_device = (device or "").strip().lower()
    normalized_compute = (compute_type or "").strip().lower()

    if normalized_device.startswith("cuda:"):
        print(
            f"[warn] faster-whisper does not accept indexed CUDA devices like '{device}'. "
            "Using 'cuda' instead."
        )
        return "cuda", compute_type

    if normalized_device == "cpu" and normalized_compute == "float16":
        print("[warn] float16 is not supported on CPU. Using int8 instead.")
        return "cpu", "int8"

    return device, compute_type


# ── Output path helpers ───────────────────────────────────────────────────────

def setup_batch_paths(out_root: Path, batch_id: str) -> BatchPaths:
    root = out_root / batch_id
    return BatchPaths(
        root=root,
        segments_dir=root / "segments",
        target_segments_dir=root / "segments_target_speaker",
        meta_jsonl=root / "segments.jsonl",
        meta_csv=root / "segments.csv",
        target_meta_jsonl=root / "segments_target_speaker.jsonl",
        target_meta_csv=root / "segments_target_speaker.csv",
    )


def build_item_paths(item: pd.Series, out_root: Path, batch: Optional[BatchPaths]) -> ItemPaths:
    source_type = item.get("Source_Type", "youtube")
    safe_id = item.get("Source_ID") or sanitize_source_id(item["Link"])

    if source_type == "local_video" and batch is not None:
        prefix = f"{safe_id}_"
        return ItemPaths(
            vid_dir=batch.root,
            seg_dir=batch.segments_dir,
            target_seg_dir=batch.target_segments_dir,
            meta_jsonl=batch.meta_jsonl,
            meta_csv=batch.meta_csv,
            target_meta_jsonl=batch.target_meta_jsonl,
            target_meta_csv=batch.target_meta_csv,
            wav_path=batch.root / f"{prefix}audio_16k_mono.wav",
            raw_audio=batch.root / f"{prefix}raw_audio.m4a",
            segment_prefix=prefix,
            safe_id=safe_id,
        )

    speaker_folder = str(item["Speaker_Name"]).replace(" ", "_")
    vid_dir = out_root / speaker_folder / safe_id
    return ItemPaths(
        vid_dir=vid_dir,
        seg_dir=vid_dir / "segments",
        target_seg_dir=vid_dir / "segments_target_speaker",
        meta_jsonl=vid_dir / "segments.jsonl",
        meta_csv=vid_dir / "segments.csv",
        target_meta_jsonl=vid_dir / "segments_target_speaker.jsonl",
        target_meta_csv=vid_dir / "segments_target_speaker.csv",
        wav_path=vid_dir / "audio_16k_mono.wav",
        raw_audio=vid_dir / "raw_audio.m4a",
        segment_prefix="",
        safe_id=safe_id,
    )


# ── Audio helpers ─────────────────────────────────────────────────────────────

def prepare_wav_for_item(
    item: pd.Series,
    raw_audio: Path,
    wav_path: Path,
    sr: int,
    normalize: bool,
) -> None:
    source_type = item.get("Source_Type", "youtube")
    source_ref = item["Link"]

    if source_type == "local_video":
        standardize_wav(in_path=Path(source_ref), out_wav=wav_path, sr=sr, normalize=normalize)
        return

    download_youtube_audio(url=source_ref, out_path=raw_audio)
    standardize_wav(in_path=raw_audio, out_wav=wav_path, sr=sr, normalize=normalize)


def cut_and_collect_segments(
    segments: List[Dict[str, Any]],
    wav_path: Path,
    out_dir: Path,
    sr: int,
    source_ref: str,
    source_type: str,
    safe_id: str,
    segment_prefix: str = "",
    segment_id_prefix: str = "",
    speaker_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    meta_rows = []
    for i, seg in enumerate(segments):
        seg_name = f"{segment_prefix}segment_{i:06d}_{seg['start']:.2f}_{seg['end']:.2f}.wav"
        seg_path = out_dir / seg_name

        cut_segments_ffmpeg(
            wav_path=str(wav_path),
            start=seg["start"],
            end=seg["end"],
            out_path=str(seg_path),
            sr=sr,
        )

        meta_row = {
            "url": source_ref,
            "source_type": source_type,
            "segment_id": f"{safe_id}:{segment_id_prefix}{i:06d}",
            "source_id": safe_id,
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["end"] - seg["start"],
            "text": seg.get("text", "").strip(),
            "num_words": seg.get("num_words", 0),
            "wav": str(seg_path),
        }
        if speaker_label is not None:
            meta_row["speaker"] = speaker_label
        meta_rows.append(meta_row)

    return meta_rows


# ── Per-item processing ───────────────────────────────────────────────────────

def process_item(
    item: pd.Series,
    paths: ItemPaths,
    args: argparse.Namespace,
    asr_device: str,
    asr_compute_type: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run the full pipeline for one input item.
    Returns (meta_rows, target_meta_rows).
    """
    source_ref = item["Link"]
    source_type = item.get("Source_Type", "youtube")
    speaker_name = item["Speaker_Name"]
    start_time = float(item["Start_time_sec"])

    # 1) Download / extract and standardize
    prepare_wav_for_item(
        item=item,
        raw_audio=paths.raw_audio,
        wav_path=paths.wav_path,
        sr=args.sr,
        normalize=args.normalize,
    )

    # 2) Diarization (optional)
    if args.diarize:
        hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token is None:
            print("WARNING: No HF token provided. pyannote diarization likely requires it.")
            print("Set --hf-token or env HF_TOKEN.")

        diar = run_diarization(
            wav_path=str(paths.wav_path),
            hf_token=hf_token,
            num_speakers=args.num_speakers,
            min_turn=args.min_turn,
        )
        save_json(paths.vid_dir / f"{paths.segment_prefix}diarization.json", diar)
        rttm_path = paths.vid_dir / f"{paths.segment_prefix}diarization.rttm"
        rttm_path.write_text("\n".join(diar["rttm_lines"]) + "\n", encoding="utf-8")
        diar_turns = diar["turns"]
    else:
        diar_turns = [{"start": 0.0, "end": float("inf"), "speaker": "SPK0"}]

    target_speaker: Optional[str] = None
    target_speaker_turns: List[Dict[str, Any]] = []

    if args.diarize:
        target_speaker = pick_speaker_by_window(
            diar_turns, t=start_time, window=args.target_window,
        )
        if target_speaker is None:
            print(
                f"[warn] Could not identify target speaker near "
                f"{start_time:.2f}s for source={source_ref}. "
                "Skipping target-speaker-only outputs."
            )
        else:
            target_speaker_turns = filter_turns_to_speaker(diar_turns, target_speaker)
            save_json(paths.vid_dir / f"{paths.segment_prefix}target_speaker.json", {
                "speaker_name": speaker_name,
                "target_speaker_label": target_speaker,
                "start_time_sec": start_time,
                "window_sec": args.target_window,
            })

    if args.diarize and args.target_only:
        if not target_speaker:
            raise RuntimeError(
                f"Could not identify target speaker near "
                f"{start_time:.2f}s for source={source_ref}"
            )
        if not target_speaker_turns:
            raise RuntimeError(
                f"No diarization turns left after filtering "
                f"(start_time={start_time:.2f}s, source={source_ref})"
            )
        diar_turns = target_speaker_turns

    # 3) ASR
    try:
        asr = run_asr(
            wav_path=str(paths.wav_path),
            model=args.model,
            device=asr_device,
            compute_type=asr_compute_type,
            language=args.language,
        )
    except Exception as exc:
        if asr_device == "cuda" and "unsupported device" in str(exc).lower():
            print("[warn] CUDA ASR is unavailable here. Retrying on CPU with int8.")
            asr = run_asr(
                wav_path=str(paths.wav_path),
                model=args.model,
                device="cpu",
                compute_type="int8",
                language=args.language,
            )
        else:
            raise

    assigned = assign_words_to_speakers(words=asr["words"], diar_turns=diar_turns)
    target_assigned: List[Dict[str, Any]] = []
    if args.diarize and target_speaker_turns:
        target_assigned = assign_words_to_speakers(
            words=asr["words"], diar_turns=target_speaker_turns,
        )

    # 4) Segmentation
    duration = get_audio_duration_seconds(str(paths.wav_path))
    if args.segment_mode == "fixed":
        segments = build_fixed_window_segments(
            audio_duration=duration,
            window=args.window,
            overlap=args.overlap,
            min_last=args.min_last,
        )
    else:
        segments = build_segments(
            assigned_words=assigned,
            gap_merge=args.gap_merge,
            min_seg=args.min_seg,
            max_seg=args.max_seg,
        )

    # 5) Cut all segments
    meta_rows = cut_and_collect_segments(
        segments=segments,
        wav_path=paths.wav_path,
        out_dir=paths.seg_dir,
        sr=args.sr,
        source_ref=source_ref,
        source_type=source_type,
        safe_id=paths.safe_id,
        segment_prefix=paths.segment_prefix,
    )

    target_meta_rows: List[Dict[str, Any]] = []
    if args.diarize and target_speaker_turns:
        if args.no_asr:
            target_segments = build_segments_from_diarization(
                diar_turns=target_speaker_turns,
                gap_merge=args.gap_merge,
                min_seg=args.min_seg,
                max_seg=args.max_seg,
            )
        else:
            target_segments = build_segments(
                assigned_words=target_assigned,
                gap_merge=args.gap_merge,
                min_seg=args.min_seg,
                max_seg=args.max_seg,
            )

        target_meta_rows = cut_and_collect_segments(
            segments=target_segments,
            wav_path=paths.wav_path,
            out_dir=paths.target_seg_dir,
            sr=args.sr,
            source_ref=source_ref,
            source_type=source_type,
            safe_id=paths.safe_id,
            segment_prefix=paths.segment_prefix,
            segment_id_prefix="target_",
            speaker_label=target_speaker,
        )

    # 6) Cleanup intermediate files
    if not args.keep_intermediate and source_type == "youtube":
        try:
            paths.raw_audio.unlink(missing_ok=True)
        except Exception:
            pass

    return meta_rows, target_meta_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    usage_examples = """Examples:
  python3 pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python3 pipeline.py urls.txt
  python3 pipeline.py manifest.csv
  python3 pipeline.py /path/to/videos
  python3 pipeline.py /path/to/videos --input-mode dir
  python3 pipeline.py /path/to/videos --overwrite
"""

    ap = argparse.ArgumentParser(
        description="Audio extraction pipeline for YouTube URLs, URL lists/manifests, or local video directories",
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "input_source",
        help=(
            "A YouTube URL, a CSV manifest, a text file with one URL per line, "
            "or a local directory of videos (scanned recursively)"
        ),
    )
    ap.add_argument(
        "--input-mode",
        choices=["auto", "youtube", "dir"],
        default="auto",
        help="Interpret input_source automatically, as a YouTube URL, or as a local video directory",
    )
    ap.add_argument("--out", default="data_out", help="Output directory")
    ap.add_argument("--device", default="cuda", help="ASR device for faster-whisper (usually 'cuda' or 'cpu')")
    ap.add_argument("--compute-type", default="float16", help="faster-whisper compute type (float16/int8/...)")
    ap.add_argument("--model", default="large-v3-turbo", help="faster-whisper model name")
    ap.add_argument("--language", default=None, help="Force language code (e.g., en, ko). Default: auto-detect.")
    ap.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote (or set HF_TOKEN env var)")
    ap.add_argument("--num-speakers", type=int, default=None, help="Optional fixed number of speakers for diarization")
    ap.add_argument("--min-turn", type=float, default=0.6, help="Minimum diarization turn length to keep (seconds)")
    ap.add_argument("--max-seg", type=float, default=20.0, help="Max output segment length (seconds)")
    ap.add_argument("--min-seg", type=float, default=1.0, help="Min output segment length (seconds)")
    ap.add_argument("--gap-merge", type=float, default=0.35, help="Merge adjacent same-speaker turns if gap <= this (seconds)")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    ap.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files (wav, diar, asr json)")
    ap.add_argument("--diarize", action="store_true", help="Enable speaker diarization (default: off)")
    ap.add_argument("--target-only", action="store_true", help="Keep only the target speaker identified using Start_time from the manifest")
    ap.add_argument("--target-window", type=float, default=4.0, help="Seconds around Start_time used to identify the target speaker")
    ap.add_argument("--no-asr", action="store_true", help="Disable ASR; create segments using diarization turns only")
    ap.add_argument("--normalize", action="store_true", help="Apply loudness normalization during WAV standardization (default: off)")
    ap.add_argument("--segment_mode", choices=["fixed", "asr"], default="fixed", help="Segmentation strategy. fixed=overlapping windows; asr=ASR+diar driven (default: fixed)")
    ap.add_argument("--window", type=float, default=5.0, help="Fixed window size in seconds (for fixed mode)")
    ap.add_argument("--overlap", type=float, default=2.5, help="Overlap in seconds (for fixed mode)")
    ap.add_argument("--min-last", type=float, default=1.0, help="Minimum duration for last partial window")
    ap.add_argument("--save-diar", action="store_true", help="Run diarization and save outputs (independent of segmentation)")
    ap.add_argument("--save-asr", action="store_true", help="Run ASR and save outputs (independent of segmentation)")
    ap.add_argument("--overwrite", action="store_true", help="Reprocess items even if outputs already exist")
    ap.add_argument( "--speaker-name",default=None, help="Speaker folder name")

    args = ap.parse_args()

    if args.target_only and not args.diarize:
        raise ValueError("--target-only requires --diarize")

    links_data = build_input_items(args.input_source, input_mode=args.input_mode)
    asr_device, asr_compute_type = resolve_asr_runtime(args.device, args.compute_type)

    out_root = Path(args.out)
    ensure_dir(out_root)

    processed_count = 0
    skipped_count = 0
    failed_items: List[str] = []
    aggregate_rows: List[Dict[str, Any]] = []
    aggregate_target_rows: List[Dict[str, Any]] = []

    local_batch_mode = (
        not links_data.empty and
        links_data["Source_Type"].eq("local_video").all()
    )
    batch: Optional[BatchPaths] = None

    if local_batch_mode:
        batch_id = links_data.iloc[0].get("Batch_ID") or "local_videos"
        batch = setup_batch_paths(out_root, batch_id)
        ensure_dir(batch.root)
        ensure_dir(batch.segments_dir)
        ensure_dir(batch.target_segments_dir)

        if args.overwrite:
            for p in (batch.meta_jsonl, batch.meta_csv, batch.target_meta_jsonl, batch.target_meta_csv):
                if p.exists():
                    p.unlink()
        elif batch.meta_jsonl.exists() and batch.meta_csv.exists():
            print(f"[skip] Existing outputs found for directory batch {batch.root}")
            print(f"Done. Output: {out_root.resolve()} | processed=0 skipped={len(links_data)} failed=0")
            return

    for idx, row in links_data.iterrows():
        if row["Processed"] in ("yes", "true", "1"):
            skipped_count += 1
            continue

        source_ref = row["Link"]
        source_type = row.get("Source_Type", "youtube")
        paths = build_item_paths(row, out_root, batch)

        ensure_dir(paths.vid_dir)
        ensure_dir(paths.seg_dir)
        if args.diarize:
            ensure_dir(paths.target_seg_dir)

        if (
            source_type != "local_video" and
            not args.overwrite and
            paths.meta_jsonl.exists() and paths.meta_csv.exists()
        ):
            print(f"[skip] Existing outputs found for {source_ref}")
            skipped_count += 1
            continue

        print(f"[{idx + 1}/{len(links_data)}] Processing {source_type}: {source_ref}")

        try:
            meta_rows, target_meta_rows = process_item(row, paths, args, asr_device, asr_compute_type)

            if local_batch_mode:
                append_jsonl(paths.meta_jsonl, meta_rows)
                aggregate_rows.extend(meta_rows)
                pd.DataFrame(aggregate_rows).to_csv(paths.meta_csv, index=False)
                if target_meta_rows:
                    append_jsonl(paths.target_meta_jsonl, target_meta_rows)
                    aggregate_target_rows.extend(target_meta_rows)
                    pd.DataFrame(aggregate_target_rows).to_csv(paths.target_meta_csv, index=False)
            else:
                save_jsonl(paths.meta_jsonl, meta_rows)
                pd.DataFrame(meta_rows).to_csv(paths.meta_csv, index=False)
                if target_meta_rows:
                    save_jsonl(paths.target_meta_jsonl, target_meta_rows)
                    pd.DataFrame(target_meta_rows).to_csv(paths.target_meta_csv, index=False)

            processed_count += 1
            print(f"[done] Wrote {len(meta_rows)} segments to {paths.seg_dir}")
            if target_meta_rows:
                print(f"[done] Wrote {len(target_meta_rows)} target-speaker segments to {paths.target_seg_dir}")

        except Exception as exc:
            failed_items.append(source_ref)
            print(f"[error] Failed to process {source_ref}: {exc}")

    print(
        f"Done. Output: {out_root.resolve()} | "
        f"processed={processed_count} skipped={skipped_count} failed={len(failed_items)}"
    )

    if failed_items:
        print("Failed sources:")
        for source_ref in failed_items:
            print(f" - {source_ref}")


if __name__ == "__main__":
    main()
