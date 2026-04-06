from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless/Colab use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from neural_playback import config
from neural_playback.roi_mapping import aggregate_vertices_to_rois, load_roi_map

logger = logging.getLogger(__name__)

# Dark brand palette
BG_COLOR = "#0a0a0a"
ACCENT_COLOR = "#00e5ff"
TEXT_COLOR = "#ffffff"
SECONDARY_COLOR = "#aaaaaa"
BOX_COLOR = "#111111"

# Color palette for temporal chart traces
TRACE_COLORS = [
    "#00e5ff", "#ff4081", "#ffd740", "#69ff47", "#ff6e40",
    "#e040fb", "#40c4ff", "#ffab40", "#f48fb1", "#b9f6ca",
    "#ea80fc", "#84ffff",
]

# Hardcoded 2D figure-fraction centroids for each ROI on a lateral-left brain
# render. Brain axes occupies [0.0, 0.1, 0.6, 0.8] of the figure. These are
# expressed directly in figure fraction (0-1) for use with the overlay axes.
# Coordinates are approximate for fsaverage5 lateral view — adjust after first
# real render against the actual pixel output.
_ROI_CENTROIDS_LATERAL_LEFT: dict[str, tuple[float, float] | None] = {
    "primary_auditory_cortex":    (0.36, 0.42),
    "superior_temporal_gyrus":    (0.345, 0.39),
    "motor_cortex":               (0.263, 0.60),
    "premotor_cortex":            (0.240, 0.57),
    "supplementary_motor_area":   (0.233, 0.63),
    "insula":                     (0.323, 0.45),
    "anterior_cingulate":         (0.210, 0.53),
    "prefrontal_cortex":          (0.180, 0.60),
    # Subcortical — not visible on cortical surface render
    "amygdala":                   None,
    "nucleus_accumbens":          None,
    "hippocampus":                None,
    "cerebellum":                 None,
}


def get_roi_centroid_2d(
    roi_name: str,
    view: str = "lateral_left",
) -> tuple[float, float] | None:
    """Get approximate 2D figure-fraction centroid of an ROI on the brain render.

    Args:
        roi_name: ROI name matching roi_music_map.json roi_name field.
        view: Render view. Currently only "lateral_left" is supported.

    Returns:
        (x, y) figure-fraction coordinates, or None if unknown/not visible.
    """
    if view != "lateral_left":
        logger.warning(
            "View '%s' not supported — only 'lateral_left' has centroids", view
        )
        return None
    coords = _ROI_CENTROIDS_LATERAL_LEFT.get(roi_name)
    if coords is None:
        logger.debug("No centroid for ROI '%s' in view '%s'", roi_name, view)
    return coords


def render_brain_annotated(
    preds: np.ndarray,
    roi_map: list[dict],
    labels: np.ndarray,
    timestep: int | None = None,
    threshold: float = 0.5,
    output_path: str | Path | None = None,
    track_name: str = "Unknown Track",
) -> Path:
    """Render annotated brain surface with Jarvis-style callout boxes.

    Uses Nilearn to render the brain surface, then overlays matplotlib
    annotation lines and callout boxes for active ROIs above threshold.
    Only the top 6 most active ROIs are annotated to avoid clutter.

    Args:
        preds: Array of shape (n_timesteps, n_vertices) — TRIBE v2 output.
        roi_map: List of region dicts from load_roi_map().
        labels: Destrieux label array of shape (n_vertices,).
        timestep: Specific second to render. Uses mean if None.
        threshold: Minimum normalized activation (0.0-1.0) to show callout.
        output_path: Save path for PNG. Uses temp file if None.
        track_name: Track name shown in top-left corner.

    Returns:
        Path to saved PNG file.
    """
    from nilearn import datasets, plotting

    # --- Step 1: select activation slice ---
    if timestep is None:
        activation = preds.mean(axis=0)
        time_label = "MEAN ACTIVATION"
    else:
        activation = preds[timestep]
        time_label = f"T={timestep}s"

    n_cortical = min(preds.shape[1], config.N_CORTICAL_VERTICES)
    left_activation = activation[: n_cortical // 2].astype(np.float64)

    # Normalize to [0, 1] for threshold comparison
    act_min, act_max = left_activation.min(), left_activation.max()
    act_range = act_max - act_min
    if act_range > 0:
        norm_activation = (left_activation - act_min) / act_range
    else:
        norm_activation = np.zeros_like(left_activation)

    # --- Step 2: fetch mesh ---
    logger.info("Fetching fsaverage5 mesh...")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=config.FSAVERAGE_MESH)

    # --- Step 3: create figure ---
    fig = plt.figure(figsize=(12, 7), facecolor=BG_COLOR)

    # Brain render occupies left ~60% of figure
    ax_brain = fig.add_axes([0.0, 0.1, 0.6, 0.8])
    ax_brain.set_facecolor(BG_COLOR)
    ax_brain.axis("off")

    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.pial_left,
        stat_map=left_activation,
        bg_map=fsaverage.sulc_left,
        cmap="magma",
        colorbar=False,
        view="lateral",
        figure=fig,
        axes=ax_brain,
        darkness=0.7,
    )

    # Full-figure transparent overlay axes — used for annotation so that both
    # xy (brain point) and xytext (callout anchor) can be expressed in figure
    # fraction. matplotlib's ax.annotate supports xycoords/textcoords='figure
    # fraction' only when called on a real Axes object.
    ax_overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor="none")
    ax_overlay.set_xlim(0, 1)
    ax_overlay.set_ylim(0, 1)
    ax_overlay.axis("off")
    ax_overlay.set_zorder(10)

    # --- Step 4: compute active ROIs ---
    roi_activations = aggregate_vertices_to_rois(preds, labels, roi_map)
    active_rois: list[tuple[float, dict, tuple[float, float]]] = []

    for roi in roi_map:
        roi_name = roi["roi_name"]
        act_series = roi_activations.get(roi_name, np.zeros(preds.shape[0]))
        if timestep is None:
            roi_act = float(np.abs(act_series).mean())
        else:
            roi_act = (
                float(abs(act_series[timestep]))
                if len(act_series) > timestep
                else 0.0
            )

        # Normalize against the full left-hemisphere activation range
        if act_range > 0:
            roi_act_norm = (roi_act - act_min) / act_range
        else:
            roi_act_norm = 0.0

        centroid = get_roi_centroid_2d(roi_name)
        if centroid is not None and roi_act_norm >= threshold:
            active_rois.append((roi_act_norm, roi, centroid))

    # Sort by activation descending, keep top 6
    active_rois.sort(key=lambda x: x[0], reverse=True)
    active_rois = active_rois[:6]

    # --- Step 5: draw callout annotations ---
    for i, (act_val, roi, (cx, cy)) in enumerate(active_rois):
        # Callout anchor: right panel, two columns, stacked rows
        anchor_x = 0.65 + (i % 2) * 0.17
        anchor_y = 0.82 - (i // 2) * 0.28

        display_name = roi.get("display_name", roi["roi_name"])
        annotation_text = roi.get("annotation", "")
        timing_text = roi.get("timing_insight_template", "")

        # Build three-line callout text
        callout_text = f"{display_name}\n{annotation_text}\n{timing_text}"

        ax_overlay.annotate(
            callout_text,
            xy=(cx, cy),
            xycoords="axes fraction",
            xytext=(anchor_x, anchor_y),
            textcoords="axes fraction",
            fontsize=8,
            color=TEXT_COLOR,
            fontfamily="DejaVu Sans",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=BOX_COLOR,
                edgecolor=ACCENT_COLOR,
                alpha=0.88,
                linewidth=0.8,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=ACCENT_COLOR,
                lw=1.1,
                connectionstyle="arc3,rad=0.2",
            ),
            multialignment="left",
        )

    # --- Step 6: title + footer ---
    fig.text(
        0.02,
        0.96,
        track_name.upper(),
        color=TEXT_COLOR,
        fontsize=11,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        transform=fig.transFigure,
    )
    fig.text(
        0.58,
        0.96,
        time_label,
        color=SECONDARY_COLOR,
        fontsize=9,
        fontfamily="DejaVu Sans",
        transform=fig.transFigure,
    )
    disclaimer_short = config.REPORT_CARD_DISCLAIMER[:80] + "..."
    fig.text(
        0.02,
        0.02,
        disclaimer_short,
        color=SECONDARY_COLOR,
        fontsize=6,
        fontfamily="DejaVu Sans",
        transform=fig.transFigure,
    )

    # --- Step 7: save ---
    if output_path is None:
        _, tmp_path = tempfile.mkstemp(suffix=".png")
        output_path = Path(tmp_path)
    else:
        output_path = Path(output_path)

    fig.savefig(
        str(output_path), dpi=150, bbox_inches="tight", facecolor=BG_COLOR
    )
    plt.close(fig)
    logger.info("Annotated brain render saved: %s", output_path)
    return output_path


def render_brain_static(
    preds: np.ndarray,
    timestep: int | None = None,
    output_path: str | Path | None = None,
    colormap: str = "magma",
) -> Path:
    """Render unannotated 2x2 brain surface grid (left/right, lateral/medial).

    Simpler version without callout annotations — used for README, quick
    overview, or when annotation centroids aren't calibrated.

    Args:
        preds: Array of shape (n_timesteps, n_vertices).
        timestep: Specific second to render. Uses mean if None.
        output_path: Save path for PNG. Uses temp file if None.
        colormap: Matplotlib colormap name.

    Returns:
        Path to saved PNG file.
    """
    from nilearn import datasets, plotting

    if timestep is None:
        activation = preds.mean(axis=0)
    else:
        activation = preds[timestep]

    n_cortical = min(preds.shape[1], config.N_CORTICAL_VERTICES)
    left_act = activation[: n_cortical // 2].astype(np.float64)
    right_act = activation[n_cortical // 2 : n_cortical].astype(np.float64)

    fsaverage = datasets.fetch_surf_fsaverage(mesh=config.FSAVERAGE_MESH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor=BG_COLOR)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    panels = [
        (axes[0, 0], fsaverage.pial_left,  fsaverage.sulc_left,  left_act,  "lateral", "L lateral"),
        (axes[0, 1], fsaverage.pial_left,  fsaverage.sulc_left,  left_act,  "medial",  "L medial"),
        (axes[1, 0], fsaverage.pial_right, fsaverage.sulc_right, right_act, "lateral", "R lateral"),
        (axes[1, 1], fsaverage.pial_right, fsaverage.sulc_right, right_act, "medial",  "R medial"),
    ]

    for ax, mesh, sulc, act, view, label in panels:
        ax.set_facecolor(BG_COLOR)
        plotting.plot_surf_stat_map(
            surf_mesh=mesh,
            stat_map=act,
            bg_map=sulc,
            cmap=colormap,
            colorbar=False,
            view=view,
            figure=fig,
            axes=ax,
            darkness=0.7,
        )
        ax.set_title(label, color=SECONDARY_COLOR, fontsize=8, pad=2)
        ax.axis("off")

    if output_path is None:
        _, tmp_path = tempfile.mkstemp(suffix=".png")
        output_path = Path(tmp_path)
    else:
        output_path = Path(output_path)

    fig.savefig(
        str(output_path), dpi=150, bbox_inches="tight", facecolor=BG_COLOR
    )
    plt.close(fig)
    logger.info("Static brain render saved: %s", output_path)
    return output_path


def create_temporal_chart(
    roi_timeseries: pd.DataFrame,
    title: str = "Neural Activation Over Time",
    output_path: str | Path | None = None,
) -> go.Figure:
    """Create an interactive Plotly dark-theme temporal activation chart.

    Args:
        roi_timeseries: DataFrame with columns = roi_names, index = timestep.
        title: Chart title.
        output_path: If provided, saves as interactive HTML.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    for i, col in enumerate(roi_timeseries.columns):
        color = TRACE_COLORS[i % len(TRACE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=list(roi_timeseries.index),
                y=roi_timeseries[col].tolist(),
                name=col,
                line=dict(color=color, width=1.5),
                mode="lines",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=13)),
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, Helvetica, sans-serif", color=TEXT_COLOR),
        xaxis=dict(
            title="Time (seconds)",
            gridcolor="#222222",
            zerolinecolor="#333333",
        ),
        yaxis=dict(
            title="Mean Activation (a.u.)",
            gridcolor="#222222",
            zerolinecolor="#333333",
        ),
        legend=dict(
            x=1.01,
            y=1.0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#333333",
        ),
        margin=dict(l=60, r=160, t=60, b=60),
    )

    if output_path is not None:
        output_path = Path(output_path)
        fig.write_html(str(output_path))
        logger.info("Temporal chart saved: %s", output_path)

    return fig
