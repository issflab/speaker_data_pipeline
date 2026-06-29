# Speaker Data Pipeline

End-to-end pipeline for collecting speaker-centric YouTube content and processing it with VoxSieve for speaker diarization and transcription.

## Setup

1. Clone and configure both repositories:
   - `speaker-scraper`
   - `VoxSieve`

2. Create a `.env` file in each repository.

**speaker-scraper/.env**

```bash
GROQ_API_KEY=your_groq_api_key
```

**VoxSieve/.env**

```bash
HF_TOKEN=your_huggingface_token
```

3. Install dependencies:

```bash
pip install -r speaker-scraper/requirements.txt -r VoxSieve/requirements.txt
```

## Run

```bash
python run_pipeline.py 
```

## Output

```text
VoxSieve/data_out/
```

## YouTube Cookies Setup

Some YouTube videos may require authentication and can return errors such as:

* `Sign in to confirm you're not a bot`
* `Video unavailable`
* `Requested format is not available`

To avoid these issues, export your YouTube cookies and place them in a file named:

```text
cookies.txt
```

Place the file in the project root:

```text
speaker_data_pipeline/
├── cookies.txt
├── run_pipeline.py
├── speaker-scraper/
└── VoxSieve/
```

### Export Cookies

Use a browser extension such as **Get cookies.txt LOCALLY** and export cookies from a browser that is logged into YouTube.

### Update Cookie Path

In `speaker-scraper/scraper.py`, update:

```python
COOKIE_FILE = "/path/to/cookies.txt"
```

Example:

```python
COOKIE_FILE = "/home/lkolluru/speaker_data_pipeline/cookies.txt"
```

### Notes

* Cookies may expire and need to be re-exported periodically.
* The cookies file should not be committed to GitHub.
* Add the following to `.gitignore`:

```text
cookies.txt
*.txt
```
