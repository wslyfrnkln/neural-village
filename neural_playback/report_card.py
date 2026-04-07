"""Neural report card generation for neural-village.

Converts TRIBE v2 ROI activation timeseries into 0-10 trait scores and
a formatted plain-text report card.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from neural_playback import config
from neural_playback.roi_mapping import load_roi_map, get_roi_timeseries

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Sigmoid scaling constant — controls how steeply raw activation maps to 0-10
_SIGMOID_K = 1.0


def compute_trait_scores(
    roi_timeseries: pd.DataFrame,
    roi_map: list[dict] | None = None,
) -> dict[str, float]:
    """Compute 0.0-10.0 trait scores from ROI activation timeseries.

    For each trait, collects contributing ROIs, computes weighted mean absolute
    activation, then maps to 0-10 via sigmoid-like transform.

    Args:
        roi_timeseries: DataFrame with columns = roi_names, index = timestep.
        roi_map: List of region dicts. Loads from file if None.

    Returns:
        Dict mapping trait_name -> score (float, 0.0-10.0, one decimal place).
    """
    if roi_map is None:
        roi_map = load_roi_map()

    # Group ROIs by trait
    trait_rois: dict[str, list[tuple[str, float]]] = {}
    for roi in roi_map:
        trait = roi["trait"]
        if trait not in trait_rois:
            trait_rois[trait] = []
        trait_rois[trait].append((roi["roi_name"], roi["weight"]))

    scores: dict[str, float] = {}
    for trait, rois in trait_rois.items():
        raw_total = 0.0
        weight_total = 0.0
        for roi_name, weight in rois:
            if roi_name in roi_timeseries.columns:
                mean_abs = float(np.abs(roi_timeseries[roi_name].values).mean())
                raw_total += mean_abs * weight
                weight_total += weight
            else:
                logger.debug("ROI '%s' not in timeseries columns — skipping", roi_name)

        if weight_total > 0:
            raw_value = raw_total / weight_total
        else:
            raw_value = 0.0

        # Sigmoid-like transform to [0, 10]
        score = 10.0 * (2.0 / (1.0 + np.exp(-_SIGMOID_K * raw_value)) - 1.0)
        score = float(np.clip(score, 0.0, 10.0))
        scores[trait] = round(score, 1)

    return scores


def format_report_card(
    scores: dict[str, float],
    track_name: str = "Unknown Track",
    include_disclaimer: bool = True,
) -> str:
    """Format trait scores as a plain-text neural report card.

    Args:
        scores: Dict of trait_name -> score (0.0-10.0).
        track_name: Track name for the header.
        include_disclaimer: Whether to append the scientific disclaimer.

    Returns:
        Formatted report card string.
    """
    width = 42
    title = f"NEURAL REPORT CARD — {track_name}"
    lines = [
        title,
        "\u2550" * width,
        "",
    ]

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for trait, score in sorted_scores:
        score_str = f"{score:.1f} / 10"
        dots = "." * (width - len(trait) - len(score_str) - 2)
        lines.append(f"{trait} {dots} {score_str}")

    lines.append("")
    lines.append("\u2500" * width)

    if include_disclaimer:
        lines.append("")
        # Wrap disclaimer at width
        disclaimer = config.REPORT_CARD_DISCLAIMER
        words = disclaimer.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line = (current_line + " " + word).strip()
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

    return "\n".join(lines)


def generate_report_card(
    preds: np.ndarray,
    labels: np.ndarray,
    track_name: str = "Unknown Track",
    roi_map: list[dict] | None = None,
) -> tuple[dict[str, float], str]:
    """Convenience wrapper: preds + labels -> scores + formatted report card.

    Args:
        preds: Array of shape (n_timesteps, n_vertices) — TRIBE v2 output.
        labels: Destrieux label array of shape (n_vertices,).
        track_name: Track name for the report card header.
        roi_map: List of region dicts. Loads from file if None.

    Returns:
        Tuple of (scores_dict, formatted_report_string).
    """
    if roi_map is None:
        roi_map = load_roi_map()

    timeseries = get_roi_timeseries(preds, labels, roi_map)
    scores = compute_trait_scores(timeseries, roi_map)
    report = format_report_card(scores, track_name=track_name)
    return scores, report
