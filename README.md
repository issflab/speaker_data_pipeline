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
python run_pipeline.py "Speaker Name"
```

Example:

```bash
python run_pipeline.py "Elon Musk"
```

## Output

```text
VoxSieve/data_out/
```
