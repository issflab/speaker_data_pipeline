# VoxSieve

A pipeline for extracting speaker audio segments from YouTube videos or local video files. Given one or more sources, it downloads audio, runs optional speaker diarization and ASR transcription, then cuts and saves labeled WAV segments with metadata.

## Pipeline

```
Input (YouTube URL / CSV manifest / local video dir)
  → Audio download & WAV standardization (yt-dlp + ffmpeg)
  → Speaker diarization (pyannote)          [optional]
  → ASR transcription (faster-whisper)
  → Speaker–word assignment
  → Segment building (fixed-window or ASR-driven)
  → WAV segment cutting (ffmpeg)
  → Output: segments/*.wav + segments.jsonl + segments.csv
```

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` on PATH
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to `pyannote/speaker-diarization-community-1` (required for diarization)
- A CUDA-capable GPU is recommended for faster-whisper and pyannote

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Set your HuggingFace token in a `.env` file (or pass `--hf-token`):

```
HF_TOKEN=hf_...
```

## Usage

```bash
# Single YouTube URL
python3 pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Text file with one URL per line
python3 pipeline.py urls.txt

# CSV manifest (see below)
python3 pipeline.py manifest.csv

# Local video directory (scanned recursively)
python3 pipeline.py /path/to/videos

# Re-process items even if outputs already exist
python3 pipeline.py manifest.csv --overwrite
```

## CSV Manifest Format

| Column | Required | Description |
|---|---|---|
| `Link` | Yes | YouTube URL or local file path |
| `Speaker_Name` | No | Label written into segment metadata |
| `Data_type` | No | Arbitrary split label (e.g. Train/Test) |
| `Start_time` | No | Timestamp near the target speaker (HH:MM:SS or seconds) |
| `Processed` | No | Set to `yes` to skip a row |

Example:

```csv
Speaker_Name,Data_type,Link,Start_time,Processed
Jane_Smith,Train,https://www.youtube.com/watch?v=...,0:00:23,No
```

## Key Options

| Flag | Default | Description |
|---|---|---|
| `--out` | `data_out` | Output root directory |
| `--diarize` | off | Enable speaker diarization |
| `--target-only` | off | Keep only the target speaker (requires `--diarize`) |
| `--target-window` | `4.0` | Seconds around `Start_time` used to identify the target speaker |
| `--segment-mode` | `fixed` | `fixed` = overlapping windows; `asr` = ASR+diarization-driven |
| `--window` | `5.0` | Window size in seconds (fixed mode) |
| `--overlap` | `2.5` | Overlap in seconds (fixed mode) |
| `--no-asr` | off | Skip ASR; build segments from diarization turns only |
| `--model` | `large-v3-turbo` | faster-whisper model name |
| `--device` | `cuda` | Compute device (`cuda` or `cpu`) |
| `--compute-type` | `float16` | faster-whisper compute type |
| `--language` | auto | Force a language code (e.g. `en`, `ko`) |
| `--num-speakers` | auto | Fixed speaker count for diarization |
| `--normalize` | off | Apply loudness normalization (ffmpeg `loudnorm`) |
| `--keep-intermediate` | off | Keep raw audio and intermediate JSON files |
| `--save-diar` | off | Run diarization and save outputs independently |
| `--save-asr` | off | Run ASR and save outputs independently |
| `--hf-token` | env | HuggingFace token (or set `HF_TOKEN` in `.env`) |

## Output Structure

For YouTube / CSV manifest sources, each item gets its own directory:

```
data_out/
  <source_id>/
    audio_16k_mono.wav
    segments/
      segment_000000_0.00_5.00.wav
      ...
    segments.jsonl
    segments.csv
    segments_target_speaker/    # if --diarize
      ...
    segments_target_speaker.jsonl
    segments_target_speaker.csv
```

For a local video directory, all items are batched under one directory named after the input folder.

Each row in `segments.jsonl` / `segments.csv` contains:

```json
{
  "url": "...",
  "source_type": "youtube",
  "segment_id": "<source_id>:000000",
  "source_id": "...",
  "start": 0.0,
  "end": 5.0,
  "duration": 5.0,
  "text": "transcribed text",
  "num_words": 12,
  "wav": "data_out/.../segments/segment_000000_0.00_5.00.wav"
}
```
