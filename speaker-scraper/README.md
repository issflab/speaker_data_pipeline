# YouTube Speaker Audio Scraper

Automatically finds and collects YouTube videos of a target speaker for audio dataset creation. Uses Groq (LLaMA 3.3) for intelligent query generation and relevance scoring, and Silero-VAD to detect the exact timestamp when speech begins.

## What It Does

Given a speaker name, the pipeline:

1. Uses Groq to generate 6 targeted YouTube search queries
2. Searches YouTube via `yt-dlp` (20 results per query)
3. Deduplicates results
4. Filters videos to a duration range of **200–250 minutes per video** (configurable)
5. Scores each video with Groq for relevance (async, 5 concurrent)
6. Applies quality heuristics (channel type, view count, title signals)
7. Fills a **200–250 minute total audio bucket** from top-ranked videos
8. Detects the first speech timestamp using **Silero-VAD**
9. Outputs a CSV ready for downstream audio processing

## Output

A CSV file named `Speaker_Name.csv` with columns:

| Column | Description |
|---|---|
| `Speaker_Name` | Speaker name (underscored) |
| `Data_type` | Blank — fill in manually (e.g., `interview`, `lecture`) |
| `Link` | YouTube URL |
| `Start_time` | First speech timestamp (`H:MM:SS`) |
| `Processed` | Blank — mark when audio has been processed |

## Requirements

### System Dependencies

- **Python 3.10+**
- **ffmpeg** — must be on your system PATH

  - Windows: download from https://ffmpeg.org/download.html and add `bin/` to PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Python Dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
yt-dlp
groq
torch
torchaudio
soundfile
```

### API Key

A free **Groq API key** is required. Get one at https://console.groq.com

Set it as an environment variable before running:

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "your_key_here"
```

**Windows (Command Prompt):**
```cmd
set GROQ_API_KEY=your_key_here
```

**macOS/Linux:**
```bash
export GROQ_API_KEY="your_key_here"
```

## Usage

```bash
python scraper.py "Speaker Name"
```

**Examples:**

```bash
python scraper.py "Lex Fridman"
python scraper.py "Andrew Huberman"
python scraper.py "Sam Altman"
```

The script prints step-by-step progress and saves the CSV in the current directory.

## Configuration

Edit the constants at the top of `scraper.py` to tune behaviour:

| Constant | Default | Description |
|---|---|---|
| `MIN_DURATION` | `12000` | Min seconds per individual video (200 min) |
| `MAX_DURATION` | `15000` | Max seconds per individual video (250 min) |
| `TOTAL_MIN_SECONDS` | `12000` | Min total seconds for the output bucket |
| `TOTAL_MAX_SECONDS` | `15000` | Max total seconds for the output bucket |
| `TOP_N_FOR_VAD` | `20` | How many top-ranked videos to run VAD on |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model used for scoring |

## Pipeline Overview

```
Speaker Name
    │
    ▼
[1] Groq generates 6 search queries
    │
    ▼
[2] yt-dlp searches YouTube (120 raw results)
    │
    ▼
[3] Deduplicate
    │
    ▼
[4] Duration filter (200–250 min/video) + metadata enrichment
    │
    ▼
[5] Groq relevance scoring (async, 5 concurrent)
    │
    ▼
[6] Heuristic scoring (channel type, title keywords, view count)
    │
    ▼
[7] Rank by combined score → fill 200–250 min total bucket
    │
    ▼
[8] Silero-VAD: detect first speech timestamp per video
    │
    ▼
[9] Write CSV
```

## Notes

- Silero-VAD downloads its model (~2 MB) on first run and caches it locally via `torch.hub`.
- The VAD step downloads only the first 3 minutes of each video — it does not download the full video.
- If a video fails VAD, `Start_time` defaults to `0:00:00` and the row is still written to CSV.
- The script targets videos where the speaker is the **sole or primary voice** — panels, highlight reels, and crowd recordings are penalised.
