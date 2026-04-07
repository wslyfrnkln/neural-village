# Roadmap: neural-village

Experimental project. Run a song through a brain prediction model, get back a labeled brain activation render, temporal chart, and neural report card.

**Purpose:** Forward-looking phases only. Completed work lives in `.planning/archive/` — not here.

---

## Completed — Phases 0–5 (Backend Pipeline)

Core pipeline written and unit tested (16/16 passing). Full detail in `.planning/NEURAL-PLAYBACK-PLAN.md`.

---

## Phase 6 — Visualization Tests + Centroid Calibration

**Goal:** `visualization.py` has test coverage and brain render callout lines actually land on the brain surface.

**Why now:** Visualization is the entire point of this project — it's the TikTok moment. It's also completely untested and the ROI centroids are hardcoded approximations that have never been verified against real output.

**Plan:** `VISUALIZATION_PLAN.md` (not yet written)

- [ ] Write `tests/test_visualization.py`
- [ ] Smoke test `render_brain_static()` with synthetic preds
- [ ] Calibrate `_ROI_CENTROIDS_LATERAL_LEFT` against actual render output

---

## Phase 7 — End-to-End Inference (Colab)

**Goal:** Run a real track through the full pipeline on Colab T4. Get actual output.

**Why now:** Nothing has run on real audio. VRAM is cleared (15.6GB on T4), but inference time is unknown and the API behavior is unverified against real music input.

**Blocker:** Colab free GPU quota hit 2026-04-07. Resets ~24h, or ~$0.35/hr T4 pay-as-you-go.

**Plan:** No plan file — one Colab session. Upload RACING or SPEECHLESS (released only), run demo notebook, capture output, calibrate centroids.

---

## Phase 8 — CLI End-to-End + Polish

**Goal:** `python cli/main.py analyze --input RACING.mp3` runs clean on M4 Max and produces output.

**Why:** CLI is the local dev path and the offline demo. MPS hung during validation — needs CPU fallback and proper error handling before it's usable.

**Plan:** `CLI_POLISH_PLAN.md` (not yet written — write after Phase 7 output informs what needs fixing)

---

## Phase 9 — TikTok Demo

**Goal:** Screen record Colab running RACING through the pipeline. Brain lights up. Report card reveals. 60 seconds.

**Why:** This is the whole point. Artist + developer running his own music through a brain prediction model is the content angle. Open source launch follows the post.

**Dependencies:** Phase 7 (real output), Phase 6 (calibrated callouts)

---

## Phase 10 — Open Source Launch

**Goal:** Repo is clean enough for strangers to use and contribute.

**Why:** Cultural capital. GitHub-first. WESLEYFRANKLIN brand on a neuroscience tool is the story.

**Plan:** Not yet written. Trigger: after Phase 9 post drops.
