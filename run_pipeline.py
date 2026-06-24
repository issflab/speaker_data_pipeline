from pathlib import Path
import pandas as pd
import subprocess
import sys

def process_speaker(speaker_name):
    root = Path(__file__).parent

    scraper_dir = root / "speaker-scraper"
    voxsieve_dir = root / "VoxSieve"

    print(f"\n{'='*60}")
    print(f"Processing: {speaker_name}")
    print(f"{'='*60}")

    subprocess.run(
        [sys.executable, "scraper.py", speaker_name],
        cwd=scraper_dir,
        check=True
    )

    csv_file = scraper_dir / f"{speaker_name.replace(' ', '_')}.csv"

    subprocess.run(
        [
            sys.executable,
            "pipeline.py",
            str(csv_file),
            "--diarize",
            "--speaker-name",
            speaker_name.replace(" ", "_")
        ],
        cwd=voxsieve_dir,
        check=True
    )

def main():

    print("\nSpeaker Dataset Pipeline")
    print("=" * 50)

    mode = input(
        "\nScrape videos automatically? (y/n): "
    ).strip().lower()

    if mode == "y":

        speaker_name = input(
            "\nSpeaker Name: "
        ).strip()

        process_speaker(speaker_name)

    else:

        speaker_name = input(
            "\nSpeaker Name: "
        ).strip()

        rows = []

        print("\nEnter URLs and timestamps.")
        print("Type DONE when finished.\n")

        while True:

            url = input("YouTube URL: ").strip()

            if url.upper() == "DONE":
                break

            start_time = input(
                "Start Time (HH:MM:SS): "
            ).strip()

            rows.append({
                "Speaker_Name": speaker_name,
                "Data_type": "",
                "Link": url,
                "Start_time": start_time,
                "Processed": ""
            })

        if not rows:
            print("No URLs entered.")
            return

        root = Path(__file__).parent

        temp_csv = (
            root /
            f"{speaker_name.replace(' ', '_')}_manual.csv"
        )

        pd.DataFrame(rows).to_csv(
            temp_csv,
            index=False
        )

        voxsieve_dir = root / "VoxSieve"

        subprocess.run(
            [
                sys.executable,
                "pipeline.py",
                str(temp_csv),
                "--diarize",
                "--speaker-name",
                speaker_name.replace(" ", "_"),
                "--manual-mode"      # NEW
            ],
            cwd=voxsieve_dir,
            check=True
        )

if __name__ == "__main__":
    main()