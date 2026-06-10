import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_pipeline.py "Speaker Name"')
        sys.exit(1)

    speaker_name = " ".join(sys.argv[1:])

    root = Path(__file__).parent

    scraper_dir = root / "speaker-scraper"
    voxsieve_dir = root / "VoxSieve"

    print("Current Python:", sys.executable)

    print(f"\n[1/2] Running speaker scraper for {speaker_name}...\n")

    subprocess.run(
        [sys.executable, "scraper.py", speaker_name],
        cwd=scraper_dir,
        check=True
    )

    csv_file = scraper_dir / f"{speaker_name.replace(' ', '_')}.csv"

    print(f"\n[2/2] Running VoxSieve...\n")

    subprocess.run(
        [
            sys.executable,
            "pipeline.py",
            str(csv_file),
            "--diarize"
        ],
        cwd=voxsieve_dir,
        check=True
    )

    print("\nPipeline Complete!")


if __name__ == "__main__":
    main()