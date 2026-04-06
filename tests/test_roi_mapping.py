import numpy as np
import pandas as pd
import pytest
from neural_playback.roi_mapping import (
    load_roi_map,
    get_trait_names,
    aggregate_vertices_to_rois,
    get_roi_timeseries,
)


def test_load_roi_map_returns_regions():
    regions = load_roi_map()
    assert isinstance(regions, list)
    assert len(regions) >= 12
    for r in regions:
        assert "roi_name" in r
        assert "trait" in r
        assert "weight" in r


def test_get_trait_names_returns_unique_sorted():
    names = get_trait_names()
    assert isinstance(names, list)
    assert len(names) >= 5
    assert names == sorted(set(names))


def test_aggregate_vertices_to_roi_shape():
    # Fake preds (10 timesteps, 100 vertices) and label array
    preds = np.random.rand(10, 100).astype(np.float32)
    # Assign first 20 vertices to label index 34 (G_temp_sup-G_T_transv)
    labels = np.zeros(100, dtype=np.int32)
    labels[:20] = 34
    roi_map = load_roi_map()
    result = aggregate_vertices_to_rois(preds, labels, roi_map)
    assert isinstance(result, dict)
    # primary_auditory_cortex uses label 34 — should have shape (10,)
    assert "primary_auditory_cortex" in result
    assert result["primary_auditory_cortex"].shape == (10,)


def test_aggregate_handles_missing_labels():
    # Labels array with no matching vertices for any ROI
    preds = np.zeros((5, 50), dtype=np.float32)
    labels = np.full(50, 999, dtype=np.int32)  # label 999 matches nothing
    roi_map = load_roi_map()
    result = aggregate_vertices_to_rois(preds, labels, roi_map)
    # Should return zeros, not KeyError
    for roi in roi_map:
        if roi["destrieux_labels"]:
            assert roi["roi_name"] in result
            assert np.allclose(result[roi["roi_name"]], 0.0)


def test_get_roi_timeseries_returns_dataframe():
    preds = np.random.rand(8, 100).astype(np.float32)
    labels = np.zeros(100, dtype=np.int32)
    labels[:20] = 34
    df = get_roi_timeseries(preds, labels)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 8  # 8 timesteps
    assert "primary_auditory_cortex" in df.columns
