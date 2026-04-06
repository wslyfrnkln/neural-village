"""Audio preprocessing utilities for TRIBE v2 neural prediction input.

Converts arbitrary audio formats to 16kHz mono MP4 containers as required
by Meta FAIR's TRIBE v2 brain prediction model.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}


def validate_ffmpeg() -> bool:
    """Check that ffmpeg is available on the system PATH.

    Returns:
        True if ffmpeg is available, False otherwise.
    """
    result = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    )
    return result.returncode == 0


def preprocess_audio(
    input_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Convert audio file to 16kHz mono MP4 container for TRIBE v2 input.

    Args:
        input_path: Path to input audio file (MP3, WAV, FLAC, M4A, OGG, AAC).
        output_dir: Directory to write output MP4. Uses a temp dir if None.

    Returns:
        Path to the output MP4 file.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If input file extension is not a supported audio format.
        RuntimeError: If ffmpeg conversion fails.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: '{input_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    wav_path = output_dir / f"{stem}_16k_mono.wav"
    mp4_path = output_dir / f"{stem}.mp4"

    # Step 1: convert to 16kHz mono WAV
    logger.info("Converting %s to 16kHz mono WAV: %s", input_path, wav_path)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1",
                str(wav_path),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg WAV conversion failed: {e.stderr.decode()}"
        ) from e

    # Step 2: wrap in MP4 container with black video track (required by TRIBE v2)
    logger.info("Wrapping WAV in MP4 container: %s", mp4_path)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1",
                "-shortest",
                "-c:v", "libx264", "-c:a", "aac",
                str(mp4_path),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg MP4 wrapping failed: {e.stderr.decode()}"
        ) from e

    logger.info("Preprocessing complete: %s", mp4_path)
    return mp4_path
