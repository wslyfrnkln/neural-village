"""ROI mapping utilities for neural-playback.

Maps vertex-level TRIBE v2 predictions to named brain regions using the
curated ROI-to-music-trait mapping in data/roi_music_map.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from neural_playback import config

logger = logging.getLogger(__name__)


def load_roi_map(path: Path | None = None) -> list[dict]:
    """Load the curated ROI-to-music-trait mapping from JSON.

    Args:
        path: Path to roi_music_map.json. Uses config.ROI_MAP_PATH if None.

    Returns:
        List of region dicts from the JSON "regions" key.
    """
    if path is None:
        path = config.ROI_MAP_PATH
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data["regions"]


def get_trait_names(roi_map: list[dict] | None = None) -> list[str]:
    """Return sorted unique trait names from the ROI map.

    Args:
        roi_map: List of region dicts. Loads from file if None.

    Returns:
        Sorted list of unique trait name strings.
    """
    if roi_map is None:
        roi_map = load_roi_map()
    return sorted(set(r["trait"] for r in roi_map))


def load_destrieux_labels(mesh: str = "fsaverage5") -> np.ndarray:
    """Load Destrieux atlas label array for the fsaverage mesh.

    Fetches the Destrieux 2009 atlas via nilearn and returns a concatenated
    label array covering both hemispheres (left then right).

    Args:
        mesh: Surface mesh name (default "fsaverage5").

    Returns:
        numpy array of shape (N_CORTICAL_VERTICES,) with integer label indices.
    """
    from nilearn import datasets, surface

    logger.info("Loading Destrieux atlas labels for mesh=%s", mesh)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    destrieux = datasets.fetch_atlas_destrieux_2009(legacy_format=False)

    left_labels = surface.load_surf_data(destrieux["map_left"])
    right_labels = surface.load_surf_data(destrieux["map_right"])

    labels = np.concatenate([left_labels, right_labels]).astype(np.int32)
    logger.info(
        "Destrieux labels loaded: shape=%s, unique=%d",
        labels.shape,
        len(np.unique(labels)),
    )
    return labels


def aggregate_vertices_to_rois(
    preds: np.ndarray,
    labels: np.ndarray,
    roi_map: list[dict],
) -> dict[str, np.ndarray]:
    """Map vertex-level predictions to named brain regions.

    For each ROI in roi_map that has destrieux_labels, find matching vertices
    in the label array and compute mean activation across those vertices per
    timestep. ROIs with empty destrieux_labels (subcortical) return zero arrays
    with a log warning.

    Args:
        preds: Array of shape (n_timesteps, n_vertices) — TRIBE v2 output.
        labels: Array of shape (n_vertices,) — Destrieux label index per vertex.
        roi_map: List of region dicts from load_roi_map().

    Returns:
        Dict mapping roi_name -> array of shape (n_timesteps,).
    """
    n_timesteps = preds.shape[0]
    result: dict[str, np.ndarray] = {}

    # Build label-name to index lookup from destrieux_subset.json
    try:
        with open(config.DESTRIEUX_SUBSET_PATH) as f:
            subset = json.load(f)
        label_to_idx: dict[str, int] = subset.get("cortical_labels", {})
    except (FileNotFoundError, KeyError):
        label_to_idx = {}
        logger.warning(
            "Could not load destrieux_subset.json — label mapping unavailable"
        )

    for roi in roi_map:
        roi_name = roi["roi_name"]
        destrieux_labels = roi.get("destrieux_labels", [])

        if not destrieux_labels:
            logger.warning(
                "ROI '%s' has no destrieux_labels (subcortical) — returning zeros",
                roi_name,
            )
            result[roi_name] = np.zeros(n_timesteps, dtype=np.float32)
            continue

        # Collect vertex indices matching any of this ROI's label names
        vertex_mask = np.zeros(labels.shape[0], dtype=bool)
        for label_name in destrieux_labels:
            idx = label_to_idx.get(label_name)
            if idx is not None:
                vertex_mask |= labels == idx

        if not np.any(vertex_mask):
            logger.debug(
                "ROI '%s': no matching vertices in label array — returning zeros",
                roi_name,
            )
            result[roi_name] = np.zeros(n_timesteps, dtype=np.float32)
        else:
            # Only use cortical vertices (first N_CORTICAL_VERTICES columns)
            n_cortical = min(preds.shape[1], config.N_CORTICAL_VERTICES)
            cortical_mask = vertex_mask[:n_cortical]
            if not np.any(cortical_mask):
                result[roi_name] = np.zeros(n_timesteps, dtype=np.float32)
            else:
                result[roi_name] = (
                    preds[:, :n_cortical][:, cortical_mask].mean(axis=1)
                )
                logger.debug(
                    "ROI '%s': %d matching vertices",
                    roi_name,
                    int(np.sum(cortical_mask)),
                )

    return result


def get_roi_timeseries(
    preds: np.ndarray,
    labels: np.ndarray | None = None,
    roi_map: list[dict] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: returns per-ROI activation timeseries as a DataFrame.

    Args:
        preds: Array of shape (n_timesteps, n_vertices) — TRIBE v2 output.
        labels: Destrieux label array of shape (n_vertices,). Loads if None.
        roi_map: List of region dicts. Loads from file if None.

    Returns:
        DataFrame with columns = roi_names, index = timestep index,
        values = mean activation per ROI per timestep.
    """
    if roi_map is None:
        roi_map = load_roi_map()
    if labels is None:
        labels = load_destrieux_labels()

    roi_timeseries = aggregate_vertices_to_rois(preds, labels, roi_map)
    return pd.DataFrame(roi_timeseries)
