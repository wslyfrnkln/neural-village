from pathlib import Path
import logging
import torch

MODEL_ID = "facebook/tribev2"
CACHE_DIR = Path("./cache")
SAMPLE_RATE = 16000
FSAVERAGE_MESH = "fsaverage5"
N_CORTICAL_VERTICES = 20484  # fsaverage5: 10242 per hemisphere * 2

ROI_MAP_PATH = Path(__file__).parent.parent / "data" / "roi_music_map.json"
DESTRIEUX_SUBSET_PATH = Path(__file__).parent.parent / "data" / "destrieux_subset.json"

REPORT_CARD_DISCLAIMER = (
    "Scores reflect predicted neural engagement patterns from Meta FAIR's TRIBE v2 model. "
    "These are computational predictions, not clinical measurements. "
    "See: github.com/facebookresearch/tribev2"
)


def get_device() -> str:
    """Detect the best available compute device.

    Returns:
        "cuda" if CUDA GPU available, "mps" if Apple Silicon MPS available,
        otherwise "cpu".
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logging.info(f"[neural-village] device selected: {device}")
    return device
