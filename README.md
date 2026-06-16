# speaker_data_pipeline
Integrating speaker scraper and voxSieve repos

# Speaker Data Pipeline

End-to-end pipeline for collecting speaker-centric YouTube content and processing it with VoxSieve for speaker diarization and transcription.

```

## Setup

1. Clone and configure both repositories:
   - `speaker-scraper`
   - `VoxSieve`

2. Configure the required `.env` files in each repository.

### speaker-scraper

```env
GROQ_API_KEY=<your_groq_api_key>
```

### VoxSieve

```env
HF_TOKEN=<your_huggingface_token>
```

3. Install dependencies:

```bash
pip install -r speaker-scraper/requirements.txt -r VoxSieve/requirements.txt
```

## Run the Complete Pipeline

```bash
python run_pipeline.py "Speaker Name"
```

Example:

```bash
python run_pipeline.py "Elon Musk"
```

## Run Speaker Scraper Only

```bash
cd speaker-scraper
python scraper.py "Speaker Name"
```

## Run VoxSieve Only

```bash
cd VoxSieve
python pipeline.py ../speaker-scraper/Speaker_Name.csv --diarize
```

## Output

Processed outputs are saved under:

```text
VoxSieve/data_out/
```
