from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from neural_playback import config
from neural_playback.preprocess import preprocess_audio

logger = logging.getLogger(__name__)


def load_model(
    cache_dir: str | Path | None = None,
    device: str | None = None,
):
    """Load the TRIBE v2 brain prediction model from HuggingFace.

    Args:
        cache_dir: Directory to cache model weights. Uses config.CACHE_DIR if None.
        device: Compute device ("cuda", "mps", "cpu"). Auto-detects if None.

    Returns:
        Loaded TribeModel instance.
    """
    from tribev2 import TribeModel

    if cache_dir is None:
        cache_dir = config.CACHE_DIR
    if device is None:
        device = config.get_device()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TRIBE v2 model (device=%s, cache=%s)", device, cache_dir)
    t0 = time.time()
    model = TribeModel.from_pretrained(config.MODEL_ID, cache_folder=str(cache_dir))
    elapsed = time.time() - t0
    logger.info("TRIBE v2 model loaded in %.1fs", elapsed)
    return model


def predict_brain_response(
    model,
    mp4_path: str | Path,
) -> tuple[np.ndarray, list]:
    """Run TRIBE v2 inference on a prepared MP4 file.

    Args:
        model: Loaded TribeModel instance (from load_model).
        mp4_path: Path to MP4 file (16kHz mono audio, black video track).

    Returns:
        Tuple of (preds, segments) where preds is a numpy array of shape
        (n_timesteps, n_vertices) representing predicted brain activation,
        and segments is a list of segment metadata.

    Raises:
        FileNotFoundError: If mp4_path does not exist.
    """
    mp4_path = Path(mp4_path)
    if not mp4_path.exists():
        raise FileNotFoundError(f"MP4 file not found: {mp4_path}")

    logger.info("Running TRIBE v2 inference on %s", mp4_path)
    t0 = time.time()
    df = model.get_events_dataframe(video_path=str(mp4_path))
    preds, segments = model.predict(events=df)
    elapsed = time.time() - t0

    logger.info(
        "Inference complete in %.1fs — output shape: %s, segments: %d",
        elapsed, preds.shape, len(segments)
    )
    return preds, segments


def predict_from_audio(
    audio_path: str | Path,
    model=None,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, list, Path]:
    """Convenience wrapper: preprocess audio then run TRIBE v2 inference.

    Args:
        audio_path: Path to input audio file (MP3, WAV, FLAC, M4A, OGG, AAC).
        model: Pre-loaded TribeModel. Loads a new one if None.
        cache_dir: Model cache directory. Uses config.CACHE_DIR if None.

    Returns:
        Tuple of (preds, segments, mp4_path) where preds is shape
        (n_timesteps, n_vertices), segments is metadata list, and
        mp4_path is the preprocessed MP4 used for inference.
    """
    mp4_path = preprocess_audio(audio_path)

    if model is None:
        model = load_model(cache_dir=cache_dir)

    preds, segments = predict_brain_response(model, mp4_path)
    return preds, segments, mp4_path
