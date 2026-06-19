import subprocess
from pathlib import Path

from tqdm import tqdm

SOURCE_ROOT = Path("data/raw/data/Videos")
OUTPUT_ROOT = Path("data/interim/Audio/Videos")
SAMPLE_RATE = 16000


def extract_audio(mp4_path: Path, flac_path: Path) -> None:
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(mp4_path),
            "-vn",  # strip video
            "-ar",
            str(SAMPLE_RATE),  # resample
            "-ac",
            "1",  # mono
            str(flac_path),
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    mp4_files = sorted(SOURCE_ROOT.rglob("*.mp4"))
    print(f"Found {len(mp4_files)} MP4 files")

    for mp4_path in tqdm(mp4_files):
        relative = mp4_path.relative_to(SOURCE_ROOT)
        flac_path = (OUTPUT_ROOT / relative).with_suffix(".flac")

        if flac_path.exists():
            continue

        try:
            extract_audio(mp4_path, flac_path)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR on {relative}:\n{e.stderr.decode()}")


if __name__ == "__main__":
    main()
