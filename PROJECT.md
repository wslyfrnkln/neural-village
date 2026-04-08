# neural-village — Project Tracker

**Last Updated:** 2026-04-08
**Status:** Phase 6 (visualization tests) next | Phase 7 blocked on Colab GPU quota (reset ~24h from 2026-04-07) | Health: green
**Owner:** WESLEYFRANKLIN
**Goal:** Run a song through Meta FAIR's TRIBE v2 brain prediction model. Output annotated brain activation render + temporal chart + neural report card. Colab-first. Open source. TikTok demo as launch vehicle.
**Stack:** Python 3.10+, tribev2, PyTorch (CUDA/MPS), Nilearn, Plotly, ffmpeg, Jupyter/Colab
**Repo:** github.com/wslyfrnkln/neural-village
**License:** CC BY-NC 4.0 (matches TRIBE v2 upstream)

---

## Blockers

- [ ] Colab free GPU quota hit 2026-04-07 — resets ~24h or $0.35/hr T4 pay-as-you-go

---

## Next Session

### Action Items
- [ ] Write `VISUALIZATION_PLAN.md` and execute Phase 6 (tests + centroid calibration)
- [ ] Re-run Colab demo notebook once GPU quota resets — upload RACING or SPEECHLESS
- [ ] Capture inference time, peak VRAM, output shape from Colab run
- [ ] Screenshot brain render for `assets/example_output.png` (README placeholder)

### Open Questions
- Does TRIBE v2 `get_events_dataframe(video_path=...)` API signature work exactly as documented? (Only confirmed via source, not run)
- Inference time on T4 for 30s clip — if >5min need section-only processing in `inference.py`
- MPS hang root cause — Wav2Vec-BERT op not supported? Which op?

---

## Architecture

```
MP3 → [ffmpeg: 16kHz mono MP4] → [TRIBE v2: brain prediction] → preds (n_timesteps × n_vertices)
                                                                          |
                                                          [ROI mapping: Destrieux + Harvard-Oxford]
                                                           |              |              |
                                                    [Brain render]  [Timeline]  [Report card]
                                                     (Nilearn)      (Plotly)    (JSON lookup)
```

| Module | File | Status |
|--------|------|--------|
| Audio preprocessing | `neural_playback/preprocess.py` | ✅ tested |
| TRIBE v2 inference | `neural_playback/inference.py` | ✅ written, GPU-only |
| ROI mapping | `neural_playback/roi_mapping.py` | ✅ tested |
| Visualization | `neural_playback/visualization.py` | ⚠️ written, untested, centroids uncalibrated |
| Report card | `neural_playback/report_card.py` | ✅ tested |
| Config + device detection | `neural_playback/config.py` | ✅ |
| Colab demo notebook | `notebooks/neural_playback_demo.ipynb` | ⚠️ written, unrun |
| Validation notebook | `notebooks/validation.ipynb` | ✅ VRAM cleared (15.6GB T4) |
| CLI | `cli/main.py` | ⚠️ written, untested |

---

## Key Decisions

| Date | Decision |
|------|----------|
| 2026-04-06 | Colab-first (T4 GPU required). Local dev via MPS on M4 Max. CPU fallback for non-inference tasks. |
| 2026-04-06 | CC BY-NC 4.0 license — must match TRIBE v2 upstream. Not MIT. |
| 2026-04-06 | Released track for public demo only (RACING or SPEECHLESS). No unreleased audio in public notebook — leak risk. |
| 2026-04-06 | ROI-to-trait mapping is a curated JSON table, not live Neurosynth queries. 12 ROIs → 7 traits. |
| 2026-04-06 | WESLEYFRANKLIN brand, not SinAudio. |
| 2026-04-06 | Subcortical regions resolved by name via TRIBE v2's Harvard-Oxford atlas at runtime — not hardcoded indices. |
| 2026-04-06 | Cerebellum visualization excluded (TRIBE v2 `NotImplementedError`). Returns zeros in report card with warning. |
| 2026-04-07 | Repo renamed neural-playback → neural-village. GitHub: wslyfrnkln/neural-village. |
| 2026-04-07 | VRAM gate cleared — T4 has 15.6GB, 14GB threshold confirmed safe. |
| 2026-04-07 | MPS backend confirmed broken for TRIBE v2 inference — Wav2Vec-BERT stalls at "Extracting words from audio". CPU fallback needed for local dev. |
| 2026-04-08 | Project moved to `~/Development/Judo/projects/neural-village/` — was incorrectly placed at `~/Development/neural-playback`. |
| 2026-04-08 | `destrieux_subset.json` TBD indices removed — subcortical resolved by name at runtime, not hardcoded. No blocked work. |
| 2026-04-08 | `claude-village` launcher added to `.zshrc`. Context file at `~/.claude/contexts/neural-village.md`. `.zshrc` restored from backup (was wiped). |

---

## Known Risks

| Risk | Status | Notes |
|------|--------|-------|
| T4 VRAM | ✅ CLEARED | 15.6GB available, 14GB threshold |
| Inference time on T4 | UNKNOWN | Not benchmarked — Phase 7 gates this |
| MPS hang | CONFIRMED | Stalls at Wav2Vec-BERT. CPU fallback needed. |
| Centroid calibration | PENDING | ROI callout positions are approximate until Phase 7 render |
| Cerebellum viz | KNOWN GAP | NotImplementedError in TRIBE v2. Zeros in report card. |
| API signature accuracy | UNVERIFIED | `get_events_dataframe(video_path=...)` confirmed in source, not yet run |
