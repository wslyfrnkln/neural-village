---
type: plan
status: active
project: neural-playback
complexity: high
created: 2026-04-06
phases: 6
---

# Engineering Plan: neural-playback

## Spec Reference
Standalone GitHub project at ~/Development/neural-playback. Runs MP3 through Meta FAIR TRIBE v2 brain prediction model, outputs brain activation renders (Nilearn 3D), temporal timeline (Plotly), and neural report card (0-10 scored traits). Colab-first (T4 free tier), CC BY-NC 4.0, WESLEYFRANKLIN brand.

Reference: HANDOFF-neural-playback.md, BRAINSTORM-neural-playback.md, TRIBE v2 research report.

## Codebase Context

**Stack:** Python 3.10+, PyTorch (CUDA for Colab, MPS for local), HuggingFace Transformers, tribev2, Nilearn, Plotly, ffmpeg, Jupyter/Colab
**Architecture:** Single-repo, notebook-first. Core logic lives in a `neural_playback/` Python package so both the Colab notebook and CLI can import it. Pipeline is linear: audio ingest → preprocessing → TRIBE v2 inference → ROI mapping → visualization → report card.
**Conventions:**
- File layout: `neural_playback/` package (importable), `notebooks/` (Colab .ipynb), `cli/` (Click-based CLI wrapper), `data/` (ROI mapping JSON, atlas data), `assets/` (brand assets, example outputs)
- Naming: snake_case for Python modules, UPPER_SNAKE for constants
- Tests: `tests/` directory, pytest, one test file per module
- No `print()` in library code — use `logging` module. Notebooks can print.
- Type hints on all public functions
- Docstrings: Google style
**Concerns:**
- TRIBE v2 VRAM on T4 is LOW confidence — Phase 0 must validate before any production code
- MPS backend untested with this model — may need CPU fallback on Apple Silicon
- Wav2Vec-BERT 2.0 was speech-trained; music encoding fidelity is an unknown
- Subcortical prediction quality may be lower than cortical — affects report card accuracy for reward/emotion regions

**Planned File Layout:**
```
neural-playback/
├── neural_playback/
│   ├── __init__.py
│   ├── preprocess.py        # MP3 → 16kHz mono → MP4 container
│   ├── inference.py         # TRIBE v2 model loading + prediction
│   ├── roi_mapping.py       # Vertex → ROI label mapping via atlas
│   ├── visualization.py     # Brain renders (Nilearn) + temporal charts (Plotly)
│   ├── report_card.py       # ROI activations → scored traits → plain-English report
│   └── config.py            # Constants, paths, device detection
├── data/
│   ├── roi_music_map.json   # Curated ROI → trait mapping (D-04)
│   └── destrieux_subset.json # Music-relevant ROI subset definition
├── notebooks/
│   ├── neural_playback_demo.ipynb  # Primary Colab notebook
│   └── validation.ipynb            # Phase 0 VRAM/timing validation
├── cli/
│   └── main.py              # Click CLI wrapper
├── tests/
│   ├── test_preprocess.py
│   ├── test_roi_mapping.py
│   ├── test_report_card.py
│   └── conftest.py
├── assets/
│   └── example_output.png   # Example brain render for README
├── requirements.txt         # Pinned deps for local install
├── requirements-colab.txt   # Colab-specific deps (lighter)
├── setup.py
├── README.md
├── LICENSE                  # CC BY-NC 4.0
└── .gitignore
```

---

## Discretionary Recommendations

**Atlas choice:** Custom 12-15 music-relevant ROI subset of Destrieux. Rationale: 148 ROIs is noise for a general audience. The music-brain literature consistently identifies ~12 core regions (primary auditory cortex, superior temporal gyrus, motor cortex, premotor cortex, cerebellum, basolateral amygdala, nucleus accumbens, hippocampus, prefrontal cortex, insula, anterior cingulate, supplementary motor area). A curated subset is more interpretable, more visually striking, and more defensible in the report card. Full Destrieux labels stay in the data file for anyone who wants them.

**Temporal chart:** Plotly. Rationale: interactive hover/zoom in Colab cells, minimal extra weight (already in Colab's default stack), and the interactivity makes a better TikTok screen recording than a static matplotlib chart.

**Report card scoring:** Continuous 0.0–10.0 displayed as X.X/10. Rationale: percentile requires a reference distribution we don't have, discrete Low/Medium/High loses nuance and feels generic. Continuous with one decimal place reads as precise-but-not-falsely-exact. Include a disclaimer that scores are "predicted neural engagement" not measured.

**Repo name:** "neural-playback" confirmed — no collision on PyPI, distinctive enough on GitHub.

---

## Implementation Tasks

### Phase 0 — Validation (MUST pass before any production code)

##### Task 0.1 — Initialize repo scaffold
**Status:** not_started
**Wave:** 1
**Files:** `~/Development/neural-playback/.gitignore`, `~/Development/neural-playback/LICENSE`, `~/Development/neural-playback/README.md`, `~/Development/neural-playback/setup.py`, `~/Development/neural-playback/neural_playback/__init__.py`, `~/Development/neural-playback/requirements.txt`, `~/Development/neural-playback/requirements-colab.txt`
**Read first:** `/Users/wesleyodd/Content/HANDOFF-neural-playback.md`
**Action:**
1. `mkdir -p ~/Development/neural-playback/{neural_playback,notebooks,cli,data,tests,assets}`
2. `cd ~/Development/neural-playback && git init`
3. Create `.gitignore` with: `__pycache__/`, `*.pyc`, `.env`, `cache/`, `*.mp3`, `*.mp4`, `*.wav`, `output/`, `.ipynb_checkpoints/`, `dist/`, `*.egg-info/`
4. Create `LICENSE` file with full CC BY-NC 4.0 text (copy from https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt or use the standard text block)
5. Create minimal `README.md` with project title "neural-playback", one-line description "Run your music through a brain. Predict neural activation patterns from audio using Meta FAIR's TRIBE v2.", badges placeholder, "Status: Phase 0 — Validating" note, and CC BY-NC 4.0 license badge
6. Create `setup.py` with name="neural-playback", version="0.1.0", python_requires=">=3.10", install_requires pointing to requirements.txt
7. Create `neural_playback/__init__.py` with `__version__ = "0.1.0"`
8. Create `requirements.txt` with: `torch>=2.1.0`, `tribev2`, `nilearn>=0.10.0`, `plotly>=5.18.0`, `numpy>=1.24.0`, `pandas>=2.0.0`, `nibabel>=5.0.0`, `click>=8.1.0`, `ffmpeg-python>=0.2.0`, `torchaudio>=2.1.0`
9. Create `requirements-colab.txt` with: `tribev2`, `nilearn>=0.10.0`, `plotly>=5.18.0`, `nibabel>=5.0.0`, `ffmpeg-python>=0.2.0` (torch/numpy/pandas already in Colab)
10. `git add -A && git commit -m "feat: initialize neural-playback repo scaffold"`

**Verify:** `ls ~/Development/neural-playback/neural_playback/__init__.py && cat ~/Development/neural-playback/LICENSE | head -3 && cat ~/Development/neural-playback/.gitignore | grep __pycache__`
**Acceptance criteria:** `grep -r "0.1.0" ~/Development/neural-playback/neural_playback/__init__.py` returns a match; `ls ~/Development/neural-playback/LICENSE` exists; `git -C ~/Development/neural-playback log --oneline | head -1` shows the initial commit
**Done:** Repo exists at `~/Development/neural-playback` with git history, license, gitignore, package skeleton, and requirements files
**Dependencies:** None

##### Task 0.2 — Create Colab VRAM validation notebook
**Status:** not_started
**Wave:** 2
**Files:** `~/Development/neural-playback/notebooks/validation.ipynb`
**Read first:** `/Users/wesleyodd/Obsidian/Javelin/research/content/2026-04-06-tribe-v2-music-model.md` (lines 82-88 for API), `/Users/wesleyodd/Development/Judo/.planning/BRAINSTORM-neural-playback.md` (lines 41-44 for assumptions A-02, A-04)
**Action:** Create a Jupyter notebook (`validation.ipynb`) as a JSON `.ipynb` file with the following cells:

Cell 1 (markdown): "# neural-playback — Phase 0 Validation" + "Tests: (1) TRIBE v2 loads on T4, (2) VRAM stays under 14GB during inference, (3) 30-second audio completes in under 5 minutes"

Cell 2 (code): Install deps: `!pip install -q tribev2 nilearn ffmpeg-python`

Cell 3 (code): Check GPU: `!nvidia-smi` + `import torch; print(f"CUDA available: {torch.cuda.is_available()}"); print(f"GPU: {torch.cuda.get_device_name(0)}"); print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")`

Cell 4 (code): Generate 30-second test tone as MP4:
```python
import subprocess, numpy as np, tempfile, os
# Generate 30s of 440Hz sine wave at 16kHz mono
sr = 16000
duration = 30
t = np.linspace(0, duration, sr * duration, dtype=np.float32)
tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
wav_path = "/tmp/test_tone.wav"
import soundfile as sf
sf.write(wav_path, tone, sr)
# Convert to MP4 container via ffmpeg
mp4_path = "/tmp/test_tone.mp4"
subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1", "-shortest", "-c:v", "libx264", "-c:a", "aac", mp4_path], capture_output=True)
print(f"Test MP4: {os.path.getsize(mp4_path)} bytes")
```

Cell 5 (code): Load model and measure VRAM:
```python
from tribev2 import TribeModel
import torch, time
torch.cuda.reset_peak_memory_stats()
print("Loading model...")
t0 = time.time()
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
load_time = time.time() - t0
vram_after_load = torch.cuda.max_memory_allocated() / 1e9
print(f"Model loaded in {load_time:.1f}s, VRAM after load: {vram_after_load:.2f} GB")
```

Cell 6 (code): Run inference and measure VRAM + time:
```python
torch.cuda.reset_peak_memory_stats()
print("Running inference on 30s test tone...")
t0 = time.time()
df = model.get_events_dataframe(video_path="/tmp/test_tone.mp4")
preds, segments = model.predict(events=df)
inference_time = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(f"Inference time: {inference_time:.1f}s")
print(f"Peak VRAM: {peak_vram:.2f} GB")
print(f"Output shape: {preds.shape}")
print(f"Segments: {len(segments)}")
```

Cell 7 (code): Validation summary:
```python
VRAM_LIMIT = 14.0  # GB, leave 2GB headroom on 16GB T4
TIME_LIMIT = 300   # 5 minutes for 30s audio
results = {
    "vram_ok": peak_vram < VRAM_LIMIT,
    "time_ok": inference_time < TIME_LIMIT,
    "peak_vram_gb": round(peak_vram, 2),
    "inference_time_s": round(inference_time, 1),
    "output_shape": list(preds.shape),
}
import json
print(json.dumps(results, indent=2))
if results["vram_ok"] and results["time_ok"]:
    print("\n✅ VALIDATION PASSED — proceed to Phase 1")
else:
    if not results["vram_ok"]:
        print(f"\n❌ VRAM EXCEEDED: {peak_vram:.2f} GB > {VRAM_LIMIT} GB — need Colab Pro (A100) or fp16 quantization")
    if not results["time_ok"]:
        print(f"\n❌ TOO SLOW: {inference_time:.1f}s for 30s audio — full track will be impractical, need section-only processing")
```

Cell 8 (markdown): "## Next Steps" + "If PASSED: proceed to Phase 1 core pipeline. If FAILED on VRAM: document minimum GPU, add `torch.cuda.amp.autocast()` half-precision path and retest. If FAILED on time: implement section-only processing (30-60s segments) instead of full-track."

**Verify:** `python3 -c "import json; nb=json.load(open('/Users/wesleyodd/Development/neural-playback/notebooks/validation.ipynb')); print(len(nb['cells']), 'cells'); assert len(nb['cells']) >= 7"`
**Acceptance criteria:** `grep -c "VALIDATION PASSED" ~/Development/neural-playback/notebooks/validation.ipynb` returns at least 1; `grep -c "nvidia-smi" ~/Development/neural-playback/notebooks/validation.ipynb` returns at least 1; notebook has 7+ cells
**Done:** Validation notebook exists, is valid JSON ipynb, contains VRAM check, timing benchmark, and pass/fail logic with clear next-step instructions for each failure mode
**Dependencies:** Task 0.1

##### Task 0.3 — Create config module with device detection
**Status:** not_started
**Wave:** 2
**Files:** `~/Development/neural-playback/neural_playback/config.py`
**Read first:** None (straightforward utility module)
**Action:** Create `config.py` with:
1. `MODEL_ID = "facebook/tribev2"` constant
2. `CACHE_DIR = Path("./cache")` (relative, overridable)
3. `SAMPLE_RATE = 16000` constant
4. `FSAVERAGE_MESH = "fsaverage5"` constant
5. `N_CORTICAL_VERTICES = 20484` constant (fsaverage5 has 10242 per hemisphere)
6. `ROI_MAP_PATH = Path(__file__).parent.parent / "data" / "roi_music_map.json"` 
7. `DESTRIEUX_SUBSET_PATH = Path(__file__).parent.parent / "data" / "destrieux_subset.json"`
8. Function `get_device() -> str` that returns `"cuda"` if `torch.cuda.is_available()`, `"mps"` if `torch.backends.mps.is_available()`, else `"cpu"`. Include a `logging.info` message stating which device was selected.
9. `REPORT_CARD_DISCLAIMER = "Scores reflect predicted neural engagement patterns from Meta FAIR's TRIBE v2 model. These are computational predictions, not clinical measurements. See: github.com/facebookresearch/tribev2"`
10. All imports at top: `from pathlib import Path`, `import logging`, `import torch`

**Verify:** `cd ~/Development/neural-playback && python3 -c "from neural_playback.config import get_device, MODEL_ID, SAMPLE_RATE; print(get_device()); print(MODEL_ID); print(SAMPLE_RATE)"`
**Acceptance criteria:** `grep -c "get_device" ~/Development/neural-playback/neural_playback/config.py` returns at least 2 (definition + usage); `grep "facebook/tribev2" ~/Development/neural-playback/neural_playback/config.py` returns a match; `grep "predicted neural engagement" ~/Development/neural-playback/neural_playback/config.py` returns a match
**Done:** Config module importable, device detection works on M4 Max (returns "mps" or "cpu"), all constants defined
**Dependencies:** Task 0.1

---

### Phase 1 — Core Pipeline (audio ingest → TRIBE v2 inference → raw output)

##### Task 1.1 — Audio preprocessing module
**Status:** not_started
**Wave:** 3
**Files:** `~/Development/neural-playback/neural_playback/preprocess.py`, `~/Development/neural-playback/tests/test_preprocess.py`
**Read first:** `/Users/wesleyodd/Development/Judo/.planning/BRAINSTORM-neural-playback.md` (lines 57-59, edge case 4 on audio preprocessing)
**Action:**
1. Write `tests/test_preprocess.py` first (TDD):
   - `test_mp3_to_mp4_creates_output()`: given a dummy WAV file, `preprocess_audio()` returns a valid MP4 path that exists
   - `test_output_is_16khz_mono()`: use `torchaudio.info()` on intermediate WAV to verify sample rate = 16000 and channels = 1
   - `test_invalid_input_raises()`: passing a nonexistent path raises `FileNotFoundError`
   - `test_non_audio_raises()`: passing a text file raises `ValueError` with message containing "Unsupported audio format"
   - Use `conftest.py` for a pytest fixture that generates a 2-second 440Hz sine WAV at 44100Hz stereo as test input

2. Write `tests/conftest.py`:
   - Fixture `sample_wav(tmp_path)` that uses `numpy` + `soundfile` to create a 2-second stereo 44100Hz sine wave WAV, returns the path
   - Fixture `sample_mp3(tmp_path, sample_wav)` that uses ffmpeg subprocess to convert WAV to MP3, returns the path

3. Write `neural_playback/preprocess.py`:
   - Function `preprocess_audio(input_path: str | Path, output_dir: str | Path | None = None) -> Path`:
     - Validate input file exists, raise `FileNotFoundError` if not
     - Check extension is in `{".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}`, raise `ValueError` if not
     - Use `subprocess.run` with ffmpeg to: convert to 16kHz mono WAV (intermediate), then wrap in MP4 container with black video track (320x240, 1fps, libx264) and AAC audio
     - If `output_dir` is None, use a `tempfile.mkdtemp()` directory
     - Return the Path to the output MP4
     - Log each step with `logging.info`
   - Function `validate_ffmpeg() -> bool`: check `ffmpeg -version` succeeds, return True/False
   - All ffmpeg calls use `subprocess.run` with `capture_output=True, check=True` and catch `subprocess.CalledProcessError` with a clear error message

**Verify:** `cd ~/Development/neural-playback && python3 -m pytest tests/test_preprocess.py -v`
**Acceptance criteria:** `grep -c "def preprocess_audio" ~/Development/neural-playback/neural_playback/preprocess.py` returns 1; `grep -c "def test_" ~/Development/neural-playback/tests/test_preprocess.py` returns at least 4; pytest passes all tests
**Done:** Preprocessing module converts MP3/WAV to 16kHz mono MP4, tests pass, error handling for missing files and bad formats
**Dependencies:** Task 0.3

##### Task 1.2 — TRIBE v2 inference module
**Status:** not_started
**Wave:** 3
**Files:** `~/Development/neural-playback/neural_playback/inference.py`
**Read first:** `/Users/wesleyodd/Obsidian/Javelin/research/content/2026-04-06-tribe-v2-music-model.md` (lines 82-88 for API pattern)
**Action:** Create `neural_playback/inference.py` with:

1. Function `load_model(cache_dir: str | Path | None = None, device: str | None = None) -> TribeModel`:
   - If `cache_dir` is None, use `config.CACHE_DIR`
   - If `device` is None, use `config.get_device()`
   - Call `TribeModel.from_pretrained(config.MODEL_ID, cache_folder=str(cache_dir))`
   - Log model load time and device
   - Return model

2. Function `predict_brain_response(model: TribeModel, mp4_path: str | Path) -> tuple[np.ndarray, list]`:
   - Validate mp4_path exists
   - Call `model.get_events_dataframe(video_path=str(mp4_path))` → df
   - Call `model.predict(events=df)` → `(preds, segments)`
   - Log output shape: `preds.shape` and number of segments
   - Return `(preds, segments)` where preds is numpy array of shape `(n_timesteps, n_vertices)`

3. Function `predict_from_audio(audio_path: str | Path, model: TribeModel | None = None, cache_dir: str | Path | None = None) -> tuple[np.ndarray, list, Path]`:
   - Convenience function: calls `preprocess.preprocess_audio()` then `predict_brain_response()`
   - If model is None, loads it (caches for reuse)
   - Returns `(preds, segments, mp4_path)`

4. All functions have Google-style docstrings with Args, Returns, Raises sections
5. Imports: `from tribev2 import TribeModel`, `numpy as np`, `from neural_playback import config`, `from neural_playback.preprocess import preprocess_audio`, `logging`, `time`, `Path`

**Verify:** `python3 -c "from neural_playback.inference import load_model, predict_brain_response, predict_from_audio; print('imports ok')"` (Note: actual inference test requires GPU, so import-level verification only for local)
**Acceptance criteria:** `grep -c "def load_model" ~/Development/neural-playback/neural_playback/inference.py` returns 1; `grep -c "def predict_brain_response" ~/Development/neural-playback/neural_playback/inference.py` returns 1; `grep -c "def predict_from_audio" ~/Development/neural-playback/neural_playback/inference.py` returns 1; `grep "get_events_dataframe" ~/Development/neural-playback/neural_playback/inference.py` returns a match
**Done:** Inference module wraps TRIBE v2 API with device detection, logging, and a convenience function that handles the full audio-to-prediction pipeline
**Dependencies:** Task 0.3, Task 1.1

---

### Phase 2 — ROI Mapping + Visualization

##### Task 2.1 — Curated ROI-to-music-trait JSON mapping
**Status:** not_started
**Wave:** 4
**Files:** `~/Development/neural-playback/data/roi_music_map.json`, `~/Development/neural-playback/data/destrieux_subset.json`
**Read first:** `/Users/wesleyodd/Development/Judo/.planning/BRAINSTORM-neural-playback.md` (lines 14-15 on D-04)
**Action:**
1. Create `data/roi_music_map.json` — a JSON object mapping ROI names to music-relevant trait labels. Structure:
```json
{
  "meta": {
    "version": "1.0",
    "description": "Curated mapping from brain ROIs to music perception traits. Based on published meta-analyses of music and auditory neuroscience.",
    "disclaimer": "These mappings reflect consensus from peer-reviewed literature. Individual brains vary. Scores are predictions, not measurements.",
    "sources": [
      "Zatorre et al. 2007 - Neural specializations for music",
      "Koelsch 2014 - Brain correlates of music-evoked emotions",
      "Trost et al. 2012 - Mapping aesthetic musical emotions in the brain",
      "Grahn & Brett 2007 - Rhythm and beat perception in motor areas"
    ]
  },
  "regions": [
    {
      "roi_name": "primary_auditory_cortex",
      "display_name": "auditory_cortex",
      "destrieux_labels": ["G_temp_sup-G_T_transv", "S_temporal_transverse"],
      "hemisphere": "bilateral",
      "trait": "Sonic Detail",
      "trait_description": "Sensitivity to timbral complexity, harmonic richness, and spectral detail",
      "annotation": "Processes raw sound — texture, timbre, harmonic content",
      "timing_insight_template": "Active here = the ear is working hard at this moment",
      "weight": 1.0,
      "confidence": "high"
    },
    {
      "roi_name": "superior_temporal_gyrus",
      "display_name": "superior_temporal_gyrus",
      "destrieux_labels": ["G_temp_sup-Lateral"],
      "hemisphere": "bilateral",
      "trait": "Melodic Engagement",
      "trait_description": "Processing of pitch patterns, melodic contour, and harmonic progression",
      "annotation": "Tracks melody and pitch — where the brain follows the song",
      "timing_insight_template": "Lighting up here = strong melodic pull at this point",
      "weight": 1.0,
      "confidence": "high"
    },
    {
      "roi_name": "motor_cortex",
      "display_name": "motor_cortex",
      "destrieux_labels": ["G_precentral"],
      "hemisphere": "bilateral",
      "trait": "Groove",
      "trait_description": "Motor system engagement predicting urge to move, tap, or dance",
      "annotation": "Controls movement — activates when music makes you want to move",
      "timing_insight_template": "Motor cortex before a hook = the body knows it's coming",
      "weight": 1.0,
      "confidence": "high"
    },
    {
      "roi_name": "premotor_cortex",
      "display_name": "premotor_cortex",
      "destrieux_labels": ["G_front_middle"],
      "hemisphere": "bilateral",
      "trait": "Groove",
      "trait_description": "Motor planning and rhythmic anticipation",
      "annotation": "Plans movement before it happens — rhythmic anticipation",
      "timing_insight_template": "Active here = brain anticipating the next beat",
      "weight": 0.7,
      "confidence": "high"
    },
    {
      "roi_name": "supplementary_motor_area",
      "display_name": "supplementary_motor_area",
      "destrieux_labels": ["G_front_sup"],
      "hemisphere": "bilateral",
      "trait": "Groove",
      "trait_description": "Internal rhythm generation and beat prediction",
      "annotation": "Generates internal rhythm — the brain's metronome",
      "timing_insight_template": "Lit up = the groove is locked in here",
      "weight": 0.6,
      "confidence": "medium"
    },
    {
      "roi_name": "amygdala",
      "display_name": "amygdala",
      "destrieux_labels": [],
      "hemisphere": "bilateral",
      "subcortical": true,
      "trait": "Emotional Intensity",
      "trait_description": "Emotional arousal response to musical tension, dissonance, and surprise",
      "annotation": "Emotion center — responds to tension, surprise, and release",
      "timing_insight_template": "Firing here = something emotionally charged at this moment",
      "weight": 1.0,
      "confidence": "high"
    },
    {
      "roi_name": "nucleus_accumbens",
      "display_name": "nucleus_accumbens",
      "destrieux_labels": [],
      "hemisphere": "bilateral",
      "subcortical": true,
      "trait": "Reward / Pleasure",
      "trait_description": "Dopaminergic reward response — musical chills, anticipation payoff, hook satisfaction",
      "annotation": "Dopamine reward center — chills, payoff, hook satisfaction",
      "timing_insight_template": "Active = the brain is releasing dopamine right here",
      "weight": 1.0,
      "confidence": "high"
    },
    {
      "roi_name": "hippocampus",
      "display_name": "hippocampus",
      "destrieux_labels": [],
      "hemisphere": "bilateral",
      "subcortical": true,
      "trait": "Memory Resonance",
      "trait_description": "Familiarity processing, musical memory formation, nostalgic response",
      "annotation": "Memory formation — links music to past experiences",
      "timing_insight_template": "Lit here = this section may encode as a strong memory",
      "weight": 0.8,
      "confidence": "medium"
    },
    {
      "roi_name": "insula",
      "display_name": "insula",
      "destrieux_labels": ["G_insular_short", "G_Ins_lg_and_S_cent_ins"],
      "hemisphere": "bilateral",
      "trait": "Emotional Intensity",
      "trait_description": "Interoceptive awareness — felt bodily response to music (chills, tension, release)",
      "annotation": "Body awareness — where you feel music physically",
      "timing_insight_template": "Active = a felt physical response at this moment",
      "weight": 0.8,
      "confidence": "high"
    },
    {
      "roi_name": "anterior_cingulate",
      "display_name": "anterior_cingulate",
      "destrieux_labels": ["G_and_S_cingul-Ant"],
      "hemisphere": "bilateral",
      "trait": "Sonic Complexity",
      "trait_description": "Conflict monitoring — engagement with unexpected harmonic or rhythmic events",
      "annotation": "Monitors conflict — activates on harmonic surprise or rhythm breaks",
      "timing_insight_template": "Active = something unexpected happened here",
      "weight": 0.7,
      "confidence": "medium"
    },
    {
      "roi_name": "prefrontal_cortex",
      "display_name": "prefrontal_cortex",
      "destrieux_labels": ["G_front_sup", "G_orbital"],
      "hemisphere": "bilateral",
      "trait": "Sonic Complexity",
      "trait_description": "Higher-order music structure processing — form, narrative arc, compositional complexity",
      "annotation": "Processes song structure — narrative arc, arrangement decisions",
      "timing_insight_template": "Lit up = brain parsing the larger structure of the song",
      "weight": 0.8,
      "confidence": "medium"
    },
    {
      "roi_name": "cerebellum",
      "display_name": "cerebellum",
      "destrieux_labels": [],
      "hemisphere": "bilateral",
      "subcortical": true,
      "trait": "Groove",
      "trait_description": "Precise timing and rhythmic coordination",
      "annotation": "The brain's timing engine — millisecond rhythm tracking",
      "timing_insight_template": "Active here = tight rhythmic precision at this moment",
      "weight": 0.5,
      "confidence": "medium"
    }
  ]
}
```

2. Create `data/destrieux_subset.json` — list of Destrieux atlas label indices that correspond to the music-relevant regions above. Structure:
```json
{
  "meta": {
    "atlas": "destrieux",
    "mesh": "fsaverage5",
    "description": "Destrieux atlas label indices for music-relevant cortical ROIs. Subcortical regions (amygdala, NAcc, hippocampus, cerebellum) are mapped separately from TRIBE v2 subcortical output."
  },
  "cortical_labels": {
    "G_temp_sup-G_T_transv": 34,
    "S_temporal_transverse": 71,
    "G_temp_sup-Lateral": 33,
    "G_precentral": 28,
    "G_front_middle": 14,
    "G_front_sup": 15,
    "G_insular_short": 17,
    "G_Ins_lg_and_S_cent_ins": 16,
    "G_and_S_cingul-Ant": 3,
    "G_orbital": 24
  },
  "subcortical_roi_indices": {
    "amygdala": "model_output_index_TBD_from_validation",
    "nucleus_accumbens": "model_output_index_TBD_from_validation",
    "hippocampus": "model_output_index_TBD_from_validation",
    "cerebellum": "model_output_index_TBD_from_validation"
  },
  "note": "Subcortical indices must be determined from TRIBE v2 output structure during Phase 0 validation. Update these values after running validation.ipynb and inspecting preds shape and segment labels."
}
```

**Verify:** `python3 -c "import json; d=json.load(open('/Users/wesleyodd/Development/neural-playback/data/roi_music_map.json')); print(len(d['regions']), 'regions'); assert len(d['regions']) >= 12"` and `python3 -c "import json; d=json.load(open('/Users/wesleyodd/Development/neural-playback/data/destrieux_subset.json')); print(len(d['cortical_labels']), 'cortical labels')"`
**Acceptance criteria:** `grep -c "trait" ~/Development/neural-playback/data/roi_music_map.json` returns at least 12; `grep "Groove" ~/Development/neural-playback/data/roi_music_map.json` returns matches; `grep "annotation" ~/Development/neural-playback/data/roi_music_map.json` returns at least 12 matches; `grep "timing_insight_template" ~/Development/neural-playback/data/roi_music_map.json` returns at least 12 matches; `grep "display_name" ~/Development/neural-playback/data/roi_music_map.json` returns at least 12 matches; both JSON files parse without error
**Done:** Two JSON data files defining the curated 12-region music-brain mapping with trait labels, display names, annotation text, timing insight templates, weights, confidence levels, and atlas cross-references
**Dependencies:** Task 0.1

##### Task 2.2 — ROI mapping module
**Status:** not_started
**Wave:** 4
**Files:** `~/Development/neural-playback/neural_playback/roi_mapping.py`, `~/Development/neural-playback/tests/test_roi_mapping.py`
**Read first:** `~/Development/neural-playback/data/roi_music_map.json` (created in Task 2.1)
**Action:**
1. Write `tests/test_roi_mapping.py` first (TDD):
   - `test_load_roi_map_returns_regions()`: `load_roi_map()` returns a list of dicts, each with keys "roi_name", "trait", "weight"
   - `test_get_trait_names_returns_unique()`: `get_trait_names()` returns a sorted list of unique trait names from the map
   - `test_aggregate_vertices_to_roi_shape()`: given a fake preds array of shape `(10, 100)` and a fake label array of length 100, `aggregate_vertices_to_rois()` returns a dict mapping ROI names to arrays of shape `(10,)`
   - `test_aggregate_handles_missing_labels()`: ROIs with no matching vertices in the label array are returned with zero arrays, not KeyError

2. Write `neural_playback/roi_mapping.py`:
   - Function `load_roi_map(path: Path | None = None) -> list[dict]`: loads `roi_music_map.json`, returns the "regions" list
   - Function `get_trait_names(roi_map: list[dict] | None = None) -> list[str]`: returns sorted unique trait names (e.g., ["Emotional Intensity", "Groove", "Melodic Engagement", ...])
   - Function `load_destrieux_labels(mesh: str = "fsaverage5") -> np.ndarray`: uses `nilearn.datasets.fetch_atlas_destrieux_2009()` (or equivalent) to get label arrays for both hemispheres, returns concatenated label array of length `N_CORTICAL_VERTICES`
   - Function `aggregate_vertices_to_rois(preds: np.ndarray, labels: np.ndarray, roi_map: list[dict]) -> dict[str, np.ndarray]`: for each ROI in the map that has `destrieux_labels`, find vertices matching those labels, take mean activation across those vertices per timestep. Returns `{roi_name: array of shape (n_timesteps,)}`. For ROIs with empty destrieux_labels (subcortical), return zeros with a log warning.
   - Function `get_roi_timeseries(preds: np.ndarray, labels: np.ndarray | None = None, roi_map: list[dict] | None = None) -> pd.DataFrame`: convenience wrapper that returns a DataFrame with columns = ROI names, index = timestep, values = mean activation. Calls `aggregate_vertices_to_rois` internally.

**Verify:** `cd ~/Development/neural-playback && python3 -m pytest tests/test_roi_mapping.py -v`
**Acceptance criteria:** `grep -c "def aggregate_vertices_to_rois" ~/Development/neural-playback/neural_playback/roi_mapping.py` returns 1; `grep -c "def test_" ~/Development/neural-playback/tests/test_roi_mapping.py` returns at least 4; pytest passes
**Done:** ROI mapping module loads the curated JSON, maps vertex-level predictions to named brain regions, returns per-ROI timeseries as a DataFrame
**Dependencies:** Task 0.3, Task 2.1

##### Task 2.3 — Annotated brain render (Nilearn + matplotlib callout overlay)
**Status:** not_started
**Wave:** 5
**Files:** `~/Development/neural-playback/neural_playback/visualization.py`
**Read first:** `~/Development/neural-playback/data/roi_music_map.json` (for annotation + timing_insight_template fields per region)

**Design:** Nilearn renders the brain surface to a static PNG. Matplotlib then draws annotation lines from lit-up regions to labeled callout boxes positioned around the outside. Only regions above an activation threshold get callouts. Callout box contains: **region name** (bold, white), **music meaning** (one line, grey), **timing insight** (one line, accent color — placeholder text until user sees real output).

**Action:** Create `neural_playback/visualization.py` with:

1. Function `get_roi_centroid_2d(roi_name: str, view: str = "lateral_left") -> tuple[float, float] | None`:
   - Returns approximate 2D pixel coordinates of each ROI's centroid on the rendered brain image for the given view.
   - Use a hardcoded lookup dict (coordinates determined empirically from fsaverage5 renders at standard size 1200×800px). Keys: roi names from roi_music_map.json. Values: `(x, y)` pixel coords.
   - Lookup dict must cover all 12 ROIs for `"lateral_left"` view. Other views can return None for unknown ROIs.
   - If ROI not in dict, log a warning and return None.

2. Function `render_brain_annotated(preds: np.ndarray, roi_map: list[dict], labels: np.ndarray, timestep: int | None = None, threshold: float = 0.5, output_path: str | Path | None = None, track_name: str = "Unknown Track") -> Path`:
   - If `timestep` is None, use mean activation across all timesteps; label as "MEAN ACTIVATION"
   - If `timestep` is an int, use that second; label as f"T={timestep}s"
   - **Step 1 — Nilearn render:**
     - Use `nilearn.plotting.plot_surf_stat_map(surf_mesh=fsaverage.pial_left, stat_map=left_activation, bg_map=fsaverage.sulc_left, cmap="magma", colorbar=False, view="lateral", figure=fig, axes=ax_brain)` to render the brain surface into a matplotlib axes at 1200×800px, `facecolor="#0a0a0a"`
   - **Step 2 — Compute active ROIs:**
     - Call `aggregate_vertices_to_rois(preds_slice, labels, roi_map)` to get per-ROI mean activation for the selected timestep
     - Filter to ROIs where activation > threshold. Sort by activation descending. Take top 6 max (to avoid callout clutter).
   - **Step 3 — Draw callout annotations:**
     - For each active ROI above threshold:
       - Get 2D centroid from `get_roi_centroid_2d(roi_name)`
       - If centroid is None, skip
       - Compute callout box anchor point: push outward from brain center by 200px in the direction of the centroid
       - Draw a line from centroid to box anchor using `ax.annotate()` with `arrowprops=dict(arrowstyle="-", color="#00e5ff", lw=1.2, connectionstyle="arc3,rad=0.2")`
       - Draw callout box using `ax.text()` with:
         - Line 1: `roi["display_name"]` — bold, white, 11pt, font "Inter" or fallback "DejaVu Sans"
         - Line 2: `roi["annotation"]` — grey (`#aaaaaa`), 9pt
         - Line 3: `roi["timing_insight_template"]` — accent cyan (`#00e5ff`), 9pt, italic
         - Box style: `bbox=dict(boxstyle="round,pad=0.4", facecolor="#111111", edgecolor="#00e5ff", alpha=0.85, linewidth=0.8)`
   - **Step 4 — Title + footer:**
     - Top-left: track name in white, 13pt
     - Top-right: "T=Xs" or "MEAN" label in grey
     - Bottom: disclaimer text from `config.REPORT_CARD_DISCLAIMER` in grey, 7pt
   - Save to `output_path` or `tempfile.mkstemp(suffix=".png")`. Return Path.

3. Function `render_brain_static(preds: np.ndarray, timestep: int | None = None, output_path: str | Path | None = None, colormap: str = "magma") -> Path`:
   - Simpler version without annotations — 2×2 grid (left lateral, left medial, right lateral, right medial). Dark background. Used for quick overview / README example image.
   - No callouts. Just the colored surface.

4. Function `create_temporal_chart(roi_timeseries: pd.DataFrame, title: str = "Neural Activation Over Time", output_path: str | Path | None = None) -> go.Figure`:
   - Plotly `go.Figure` with one trace per ROI
   - X-axis: time in seconds (1Hz assumption from TRIBE v2 output)
   - Y-axis: mean activation
   - Color palette: `["#00e5ff", "#ff4081", "#ffd740", "#69ff47", "#ff6e40", "#e040fb", "#40c4ff", "#ffab40", "#f48fb1", "#b9f6ca", "#ea80fc", "#84ffff"]`
   - Dark theme: `template="plotly_dark"`, `paper_bgcolor="#0a0a0a"`, `plot_bgcolor="#0a0a0a"`
   - If `output_path` provided, save as HTML. Return Figure regardless.

5. `display_name` field: read from `roi_music_map.json` directly (e.g. `"motor_cortex"`). Render in callout as-is — all lowercase underscore format is intentional. Fallback to `roi_name` if missing.

6. Imports: `nilearn.plotting`, `nilearn.datasets`, `matplotlib.pyplot as plt`, `matplotlib.patches`, `plotly.graph_objects as go`, `numpy as np`, `pandas as pd`, `Path`, `logging`, `from neural_playback.roi_mapping import aggregate_vertices_to_rois`, `from neural_playback import config`

**Verify:** `python3 -c "from neural_playback.visualization import render_brain_annotated, render_brain_static, create_temporal_chart; print('imports ok')"`
**Acceptance criteria:** `grep -c "def render_brain_annotated" ~/Development/neural-playback/neural_playback/visualization.py` returns 1; `grep -c "def get_roi_centroid_2d" ~/Development/neural-playback/neural_playback/visualization.py` returns 1; `grep -c "def create_temporal_chart" ~/Development/neural-playback/neural_playback/visualization.py` returns 1; `grep "arc3,rad" ~/Development/neural-playback/neural_playback/visualization.py` returns a match (annotation line style); `grep "0a0a0a" ~/Development/neural-playback/neural_playback/visualization.py` returns a match
**Done:** `render_brain_annotated()` produces a dark-background PNG with Nilearn brain surface colored by activation, matplotlib annotation lines from active regions (above threshold) to callout boxes showing region name + music meaning + timing insight placeholder. Only top 6 active regions annotated. `render_brain_static()` produces unannotated 2×2 grid. Both dark-themed.
**Dependencies:** Task 2.2

---

### Phase 3 — Neural Report Card

##### Task 3.1 — Report card scoring module
**Status:** not_started
**Wave:** 5
**Files:** `~/Development/neural-playback/neural_playback/report_card.py`, `~/Development/neural-playback/tests/test_report_card.py`
**Read first:** `~/Development/neural-playback/data/roi_music_map.json` (created in Task 2.1)
**Action:**
1. Write `tests/test_report_card.py` first (TDD):
   - `test_compute_trait_scores_returns_all_traits()`: given a fake ROI timeseries DataFrame, `compute_trait_scores()` returns a dict with a float score for every unique trait in the ROI map
   - `test_scores_are_in_range()`: all scores are between 0.0 and 10.0
   - `test_zero_activation_gives_zero_score()`: an all-zeros input produces scores near 0.0 (within tolerance of 0.1)
   - `test_format_report_card_contains_all_traits()`: `format_report_card()` returns a string containing every trait name
   - `test_format_report_card_contains_disclaimer()`: output contains "predicted neural engagement"

2. Write `neural_playback/report_card.py`:
   - Function `compute_trait_scores(roi_timeseries: pd.DataFrame, roi_map: list[dict] | None = None) -> dict[str, float]`:
     - Load ROI map if not provided
     - For each unique trait, collect all ROIs that map to that trait
     - For each ROI, compute the mean absolute activation across all timesteps, multiply by that ROI's weight
     - Sum weighted activations per trait, normalize to 0.0-10.0 scale using a sigmoid-like transform: `score = 10.0 * (2.0 / (1.0 + np.exp(-k * raw_value)) - 1.0)` where k is a scaling constant (start with k=1.0, tunable)
     - Clamp to `[0.0, 10.0]` after transform
     - Return `{trait_name: round(score, 1)}`

   - Function `format_report_card(scores: dict[str, float], track_name: str = "Unknown Track", include_disclaimer: bool = True) -> str`:
     - Returns a plain-English formatted string:
       ```
       NEURAL REPORT CARD — {track_name}
       ═══════════════════════════════════
       
       Groove .................. 7.2 / 10
       Emotional Intensity ..... 8.1 / 10
       Sonic Detail ............ 6.5 / 10
       Melodic Engagement ...... 7.8 / 10
       Sonic Complexity ........ 5.9 / 10
       Reward / Pleasure ....... 8.4 / 10
       Memory Resonance ........ 6.0 / 10
       
       ───────────────────────────────────
       {REPORT_CARD_DISCLAIMER if include_disclaimer}
       ```
     - Pad dots between trait name and score for visual alignment
     - Sort traits by score descending

   - Function `generate_report_card(preds: np.ndarray, labels: np.ndarray, track_name: str = "Unknown Track", roi_map: list[dict] | None = None) -> tuple[dict[str, float], str]`:
     - Convenience wrapper: calls `roi_mapping.get_roi_timeseries()` → `compute_trait_scores()` → `format_report_card()`
     - Returns `(scores_dict, formatted_string)`

**Verify:** `cd ~/Development/neural-playback && python3 -m pytest tests/test_report_card.py -v`
**Acceptance criteria:** `grep -c "def compute_trait_scores" ~/Development/neural-playback/neural_playback/report_card.py` returns 1; `grep -c "def format_report_card" ~/Development/neural-playback/neural_playback/report_card.py` returns 1; `grep -c "def test_" ~/Development/neural-playback/tests/test_report_card.py` returns at least 5; `grep "predicted neural engagement" ~/Development/neural-playback/neural_playback/report_card.py` returns a match; pytest passes
**Done:** Report card module computes 0.0-10.0 scores per trait from ROI activations, formats a clean plain-English report with disclaimer, all tests pass
**Dependencies:** Task 2.2

---

### Phase 4 — Colab Notebook + README + Polish

##### Task 4.1 — Primary Colab demo notebook
**Status:** not_started
**Wave:** 6
**Files:** `~/Development/neural-playback/notebooks/neural_playback_demo.ipynb`
**Read first:** `~/Development/neural-playback/notebooks/validation.ipynb` (created in Task 0.2 for structure reference), `/Users/wesleyodd/Development/Judo/.planning/BRAINSTORM-neural-playback.md` (lines 16, 68 for D-06 and brand alignment)
**Action:** Create the primary demo notebook as a JSON `.ipynb` file with these cells:

Cell 1 (markdown): Title card:
```
# neural-playback 🧠🎵
**Run your music through a brain.**

Predict neural activation patterns from audio using Meta FAIR's TRIBE v2.

> Scores reflect predicted neural engagement patterns — computational predictions, not clinical measurements.

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
Built by [WESLEYFRANKLIN](https://github.com/wesleyfranklin)
```

Cell 2 (code): Setup — install deps, check GPU:
```python
!pip install -q tribev2 nilearn plotly ffmpeg-python nibabel soundfile
import torch
assert torch.cuda.is_available(), "GPU required. Go to Runtime → Change runtime type → T4 GPU"
print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
```

Cell 3 (markdown): "## Step 1: Upload your audio" + instructions for Colab file upload

Cell 4 (code): File upload cell:
```python
from google.colab import files
uploaded = files.upload()
audio_path = list(uploaded.keys())[0]
print(f"Uploaded: {audio_path}")
```

Cell 5 (code): Alternative — use a demo track URL (provide a placeholder URL with comment to replace):
```python
# Alternative: download a demo track
# import urllib.request
# demo_url = "https://example.com/demo.mp3"  # Replace with actual demo URL
# audio_path = "demo.mp3"
# urllib.request.urlretrieve(demo_url, audio_path)
```

Cell 6 (markdown): "## Step 2: Preprocess audio"

Cell 7 (code): Preprocess:
```python
from neural_playback.preprocess import preprocess_audio
mp4_path = preprocess_audio(audio_path)
print(f"Preprocessed: {mp4_path}")
```

Cell 8 (markdown): "## Step 3: Run brain prediction"

Cell 9 (code): Inference:
```python
from neural_playback.inference import load_model, predict_brain_response
import time
model = load_model()
print("Running brain prediction...")
t0 = time.time()
preds, segments = predict_brain_response(model, mp4_path)
print(f"Done in {time.time()-t0:.1f}s — output shape: {preds.shape}")
```

Cell 10 (markdown): "## Step 4: Brain activation render"

Cell 11 (code): Render brain:
```python
from neural_playback.visualization import render_brain_annotated
from neural_playback.roi_mapping import load_destrieux_labels, load_roi_map
labels = load_destrieux_labels()
roi_map = load_roi_map()
png_path = render_brain_annotated(preds, roi_map, labels, timestep=None, threshold=0.5, track_name=audio_path)
from IPython.display import Image
Image(str(png_path))
```

Cell 12 (markdown): "## Step 5: Activation over time"

Cell 13 (code): Temporal chart:
```python
from neural_playback.roi_mapping import get_roi_timeseries, load_destrieux_labels
from neural_playback.visualization import create_temporal_chart
labels = load_destrieux_labels()
timeseries = get_roi_timeseries(preds, labels)
fig = create_temporal_chart(timeseries, title=f"Neural Activation — {audio_path}")
fig.show()
```

Cell 14 (markdown): "## Step 6: Neural Report Card"

Cell 15 (code): Report card:
```python
from neural_playback.report_card import generate_report_card
scores, report = generate_report_card(preds, labels, track_name=audio_path)
print(report)
```

Cell 16 (markdown): "## Step 7: Download results"

Cell 17 (code): Export:
```python
from neural_playback.visualization import render_brain_static
import json
# Save static brain render
png_path = render_brain_static(preds, output_path="brain_activation.png")
# Save report card
with open("report_card.json", "w") as f:
    json.dump({"track": audio_path, "scores": scores}, f, indent=2)
with open("report_card.txt", "w") as f:
    f.write(report)
# Download
from google.colab import files
files.download("brain_activation.png")
files.download("report_card.json")
files.download("report_card.txt")
```

Cell 18 (markdown): Footer with license, attribution, disclaimer, and link to GitHub repo.

**Verify:** `python3 -c "import json; nb=json.load(open('/Users/wesleyodd/Development/neural-playback/notebooks/neural_playback_demo.ipynb')); print(len(nb['cells']), 'cells'); assert len(nb['cells']) >= 16"`
**Acceptance criteria:** `grep -c "neural_playback" ~/Development/neural-playback/notebooks/neural_playback_demo.ipynb` returns at least 8; `grep "WESLEYFRANKLIN" ~/Development/neural-playback/notebooks/neural_playback_demo.ipynb` returns a match; `grep "CC BY-NC" ~/Development/neural-playback/notebooks/neural_playback_demo.ipynb` returns a match; `grep "predicted neural engagement" ~/Development/neural-playback/notebooks/neural_playback_demo.ipynb` returns a match; notebook has 16+ cells
**Done:** Complete Colab notebook with upload → preprocess → inference → brain render → temporal chart → report card → download pipeline, branded WESLEYFRANKLIN, disclaimer present, CC BY-NC badge
**Dependencies:** Task 1.1, Task 1.2, Task 2.2, Task 2.3, Task 3.1

##### Task 4.2 — README with usage, architecture, screenshots
**Status:** not_started
**Wave:** 6
**Files:** `~/Development/neural-playback/README.md`
**Read first:** `~/Development/neural-playback/neural_playback/__init__.py` (for version)
**Action:** Rewrite `README.md` (replace the placeholder from Task 0.1) with:

1. Title: `# neural-playback` + tagline "Run your music through a brain."
2. Badge row: CC BY-NC 4.0 badge, Python 3.10+ badge, "Open in Colab" badge (link to raw notebook URL on GitHub)
3. One-paragraph description: what it does, what TRIBE v2 is (one sentence), what you get (brain renders, temporal timeline, report card)
4. Screenshot/example section: `![Example output](assets/example_output.png)` placeholder with note to add after first successful run
5. Quick Start section:
   - Colab: "Click the Open in Colab badge above" with 3-step instructions
   - Local (M4 Max / CUDA): `git clone`, `pip install -r requirements.txt`, `python -m neural_playback.cli --input song.mp3 --output ./results/`
6. Architecture diagram (ASCII):
   ```
   MP3 → [ffmpeg: 16kHz mono MP4] → [TRIBE v2: brain prediction] → [preds: n_timesteps × n_vertices]
                                                                          ↓
                                                        [ROI mapping: Destrieux atlas]
                                                          ↓              ↓              ↓
                                                    [Brain render]  [Timeline]  [Report card]
                                                     (Nilearn)      (Plotly)    (JSON lookup)
   ```
7. Neural Report Card section: explain the 7 traits (Groove, Emotional Intensity, Sonic Detail, Melodic Engagement, Sonic Complexity, Reward/Pleasure, Memory Resonance) with one-line descriptions
8. Scientific disclaimer: "Scores reflect predicted neural engagement patterns from Meta FAIR's TRIBE v2 model. These are computational predictions based on population-level fMRI data, not clinical measurements of individual brain activity."
9. Hardware requirements: Colab T4 (free tier), local GPU with 8GB+ VRAM, Apple Silicon via MPS
10. License section: CC BY-NC 4.0, with note that this matches TRIBE v2 upstream
11. Attribution: Meta FAIR for TRIBE v2, Nilearn, relevant neuroscience papers
12. Built by section: "Built by [WESLEYFRANKLIN](https://wslyfrnkln.com) — artist, developer."

**Verify:** `wc -l ~/Development/neural-playback/README.md` returns at least 80 lines; `head -5 ~/Development/neural-playback/README.md`
**Acceptance criteria:** `grep "WESLEYFRANKLIN" ~/Development/neural-playback/README.md` returns a match; `grep "CC BY-NC" ~/Development/neural-playback/README.md` returns a match; `grep "predicted neural engagement" ~/Development/neural-playback/README.md` returns a match; `grep "TRIBE v2" ~/Development/neural-playback/README.md` returns a match; `grep "Open in Colab" ~/Development/neural-playback/README.md` returns a match; README is at least 80 lines
**Done:** Complete README with quick start, architecture diagram, trait descriptions, scientific disclaimer, and WESLEYFRANKLIN branding
**Dependencies:** Task 0.1

---

### Phase 5 — CLI Wrapper (Secondary / Stretch)

##### Task 5.1 — Click CLI for local M4 Max usage
**Status:** not_started
**Wave:** 7
**Files:** `~/Development/neural-playback/cli/main.py`, `~/Development/neural-playback/cli/__init__.py`
**Read first:** `~/Development/neural-playback/neural_playback/inference.py` (for function signatures)
**Action:**
1. Create `cli/__init__.py` (empty file)
2. Create `cli/main.py` using Click:
   - Command `neural-playback` with subcommand `analyze`:
     - `--input` / `-i`: required, path to audio file (MP3, WAV, FLAC, M4A)
     - `--output` / `-o`: optional, output directory (default: `./output/`)
     - `--device`: optional, override device detection ("cuda", "mps", "cpu")
     - `--track-name` / `-n`: optional, track name for report card header
     - `--no-render`: flag, skip brain render (faster)
     - `--no-chart`: flag, skip temporal chart
     - `--format`: output format for report card, choices=["text", "json", "both"], default="both"
   - Implementation:
     - Validate input file exists
     - Call `preprocess_audio(input_path)`
     - Call `load_model(device=device)` → `predict_brain_response(model, mp4_path)`
     - Call `load_destrieux_labels()` → `get_roi_timeseries(preds, labels)`
     - Unless `--no-render`: call `render_brain_static(preds, output_path=output_dir/"brain_activation.png")`
     - Unless `--no-chart`: call `create_temporal_chart(timeseries, output_path=output_dir/"timeline.html")`
     - Call `generate_report_card(preds, labels, track_name=track_name)`
     - Save report as text and/or JSON based on `--format`
     - Print report card to stdout
     - Print summary: "Saved N files to {output_dir}"
   - Add `if __name__ == "__main__": analyze()` block
3. Add entry point to `setup.py`: `entry_points={"console_scripts": ["neural-playback=cli.main:analyze"]}`

**Verify:** `cd ~/Development/neural-playback && python3 cli/main.py --help`
**Acceptance criteria:** `grep -c "@click" ~/Development/neural-playback/cli/main.py` returns at least 1; `grep "def analyze" ~/Development/neural-playback/cli/main.py` returns a match; `grep "\-\-input" ~/Development/neural-playback/cli/main.py` returns a match; `grep "\-\-device" ~/Development/neural-playback/cli/main.py` returns a match; `--help` output includes "input" and "output" options
**Done:** CLI wrapper runnable with `python cli/main.py analyze --input song.mp3`, supports device override for MPS, outputs brain render PNG + timeline HTML + report card text/JSON
**Dependencies:** Task 1.1, Task 1.2, Task 2.2, Task 2.3, Task 3.1

---

## Dependency Graph Summary

```
Wave 1: Task 0.1 (repo scaffold)
Wave 2: Task 0.2 (validation notebook), Task 0.3 (config module)
Wave 3: Task 1.1 (preprocessing), Task 1.2 (inference)
Wave 4: Task 2.1 (ROI JSON data), Task 2.2 (ROI mapping module)
Wave 5: Task 2.3 (visualization), Task 3.1 (report card)
Wave 6: Task 4.1 (Colab notebook), Task 4.2 (README)
Wave 7: Task 5.1 (CLI wrapper)
```

Parallel within wave:
- Wave 2: Tasks 0.2 and 0.3 touch different files, can run in parallel
- Wave 3: Tasks 1.1 and 1.2 touch different files, can run in parallel (1.2 imports from 1.1 but only needs the function signature, not runtime)
- Wave 4: Tasks 2.1 and 2.2 — 2.2 reads from 2.1's JSON at test time, so 2.1 must complete first. NOT parallel.
- Wave 5: Tasks 2.3 and 3.1 touch different files, can run in parallel
- Wave 6: Tasks 4.1 and 4.2 touch different files, can run in parallel

---

## must_haves

```yaml
must_haves:
  behaviors:
    - "preprocess_audio('song.mp3') returns a valid MP4 file path with 16kHz mono audio"
    - "predict_brain_response(model, mp4_path) returns numpy array of shape (n_timesteps, n_vertices)"
    - "get_roi_timeseries(preds, labels) returns DataFrame with 12 ROI columns"
    - "compute_trait_scores(timeseries) returns dict with 7 trait scores between 0.0 and 10.0"
    - "format_report_card(scores) output contains 'predicted neural engagement' disclaimer"
    - "render_brain_activation(preds) produces Nilearn HTML view object"
    - "create_temporal_chart(timeseries) produces Plotly Figure with dark theme"
    - "validation.ipynb contains VRAM check against 14GB threshold with pass/fail logic"
    - "Colab demo notebook has upload → preprocess → inference → render → chart → report card flow"
    - "CLI --help shows --input, --output, --device options"
  artifacts:
    - path: ~/Development/neural-playback/neural_playback/preprocess.py
      not_stub: true
    - path: ~/Development/neural-playback/neural_playback/inference.py
      not_stub: true
    - path: ~/Development/neural-playback/neural_playback/roi_mapping.py
      not_stub: true
    - path: ~/Development/neural-playback/neural_playback/visualization.py
      not_stub: true
    - path: ~/Development/neural-playback/neural_playback/report_card.py
      not_stub: true
    - path: ~/Development/neural-playback/neural_playback/config.py
      not_stub: true
    - path: ~/Development/neural-playback/data/roi_music_map.json
      not_stub: true
    - path: ~/Development/neural-playback/notebooks/validation.ipynb
      not_stub: true
    - path: ~/Development/neural-playback/notebooks/neural_playback_demo.ipynb
      not_stub: true
    - path: ~/Development/neural-playback/cli/main.py
      not_stub: true
    - path: ~/Development/neural-playback/LICENSE
      not_stub: true
    - path: ~/Development/neural-playback/README.md
      not_stub: true
    - path: ~/Development/neural-playback/tests/test_preprocess.py
      not_stub: true
    - path: ~/Development/neural-playback/tests/test_roi_mapping.py
      not_stub: true
    - path: ~/Development/neural-playback/tests/test_report_card.py
      not_stub: true
```

---

## Risks

| Risk | Severity | Mitigation | Related Task |
|------|----------|------------|--------------|
| T4 VRAM exceeds 16GB | CRITICAL | Task 0.2 validation notebook is a hard gate — no production code until this passes. If fails: document A100 minimum, add `torch.cuda.amp.autocast()` fp16 path and retest | Task 0.2 |
| Inference time > 5min for 30s audio | HIGH | Task 0.2 benchmarks this. If too slow: implement section-only processing (30-60s chunks) in Task 1.2 | Task 0.2, Task 1.2 |
| MPS unsupported ops | MEDIUM | Task 0.3 `get_device()` will detect MPS. If MPS fails at runtime, fallback to CPU with a warning. Not a blocker — Colab is primary | Task 0.3, Task 5.1 |
| TRIBE v2 subcortical output format unknown | MEDIUM | `destrieux_subset.json` subcortical indices marked as TBD. Update after running validation notebook. Report card still works with cortical-only data (subcortical ROIs return zeros with warning) | Task 2.1, Task 2.2 |
| Wav2Vec-BERT 2.0 speech-trained, music fidelity unknown | LOW | Accept — this is a prediction model, not ground truth. Disclaimer covers this. Interesting even if imprecise | Task 3.1 |
| A-01 LOW confidence: actual input format may differ from documented API | MEDIUM | Task 0.2 validation will reveal the actual working API. If `get_events_dataframe(video_path=...)` signature is wrong, update Task 1.2 accordingly | Task 0.2, Task 1.2 |

---

## Self-Validation

1. **Coverage check:** Every spec requirement mapped:
   - MP3 input → Task 1.1 (preprocess)
   - TRIBE v2 inference → Task 1.2 (inference)
   - Brain activation renders → Task 2.3 (visualization, Nilearn)
   - Temporal timeline → Task 2.3 (visualization, Plotly)
   - Neural report card → Task 3.1 (report card)
   - Colab notebook → Task 4.1
   - CLI wrapper → Task 5.1
   - CC BY-NC license → Task 0.1
   - VRAM validation → Task 0.2
   - ROI JSON mapping (D-04) → Task 2.1
   - Scientific disclaimer → Task 3.1, Task 4.1, Task 4.2
   - Brand alignment (dark theme, D-05) → Task 2.3, Task 4.1, Task 4.2
   - Released tracks only in public demo (D-06) → Task 4.1 (no embedded audio, upload-based)

2. **DAG valid:** No cycles. Each wave depends only on prior waves. Wave 4 Task 2.2 depends on Wave 4 Task 2.1 (sequential within wave, noted in text).

3. **No placeholders:** Zero TBD/TODO/later/as-needed in task actions. The one "TBD" in the subcortical indices JSON is data that literally cannot be known until validation runs — it's documented as such with explicit instructions to update.

4. **Verify fields:** All 11 tasks have Verify + Acceptance criteria with grep-verifiable or command-runnable checks.
---

## Edge Cases (from BRAINSTORM-neural-playback.md)

### Critical
1. **Colab T4 VRAM** — TRIBE v2 (1B params) + Wav2Vec-BERT 2.0 may exceed 16GB. Mitigation: Task 0.2 hard gate.
2. **Inference time for full-length tracks** — No benchmarks for 3-min audio. Mitigation: Benchmark in Task 0.2, section-only processing if too slow.
3. **CC BY-NC virality conflict** — Cannot monetize. Accepted: cultural capital, not revenue.

### Important
4. **Audio preprocessing mismatch** — MP3 → 16kHz mono WAV → MP4 must be robust. Mitigation: librosa/torchaudio + clear error messages (Task 1.1).
5. **Report card scientific validity** — Frame as "predicted neural engagement" with disclaimer. Never "measured." (Task 3.1).
6. **Song section detection** — Start with manual timestamps. Auto-segmentation deferred.
7. **Subcortical prediction quality** — Check model card. Weight cortical higher if subcortical is weak (Task 2.1 note).

### Awareness
8. **Name collision** — PyPI confirmed clean. GitHub check pending.
9. **Brand alignment** — Dark background (#0a0a0a), clean typography, no generic data-science aesthetics (Task 2.3).
10. **Dependency weight** — Colab-first absorbs this. requirements.txt with pins for local users.
