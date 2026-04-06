# neural-playback

Run your music through a brain.

[![CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wesleyfranklin/neural-playback/blob/main/notebooks/neural_playback_demo.ipynb)

Predict neural activation patterns from audio using Meta FAIR's [TRIBE v2](https://github.com/facebookresearch/tribev2) — a brain prediction foundation model. Upload an MP3, get back a labeled brain render showing which regions lit up and what that means for your music.

---

## What you get

- **Annotated brain render** — colored activation map with Jarvis-style callout lines to each active region, labeled with what it is and what it means at that moment in the song
- **Activation timeline** — per-region activation over time (interactive Plotly chart)
- **Neural Report Card** — 7 music traits scored 0-10: Groove, Emotional Intensity, Sonic Detail, Melodic Engagement, Sonic Complexity, Reward/Pleasure, Memory Resonance

![Example output](assets/example_output.png)

> *Scores reflect predicted neural engagement patterns — computational predictions, not clinical measurements.*

---

## Quick Start

### Colab (recommended — GPU required)

Click **Open in Colab** above. Change runtime to T4 GPU. Upload your MP3. Run all cells.

### Local (M4 Max / CUDA GPU)

```bash
git clone https://github.com/wesleyfranklin/neural-playback.git
cd neural-playback
pip install -r requirements.txt
python cli/main.py analyze --input song.mp3 --output ./results/
```

---

## Architecture

```
MP3 --> [ffmpeg: 16kHz mono MP4] --> [TRIBE v2: brain prediction] --> [preds: n_timesteps x n_vertices]
                                                                              |
                                                              [ROI mapping: Destrieux atlas]
                                                               |              |              |
                                                        [Brain render]  [Timeline]  [Report card]
                                                         (Nilearn)      (Plotly)    (JSON lookup)
```

---

## Neural Report Card traits

| Trait | Brain regions | What it means |
|-------|--------------|---------------|
| **Groove** | motor_cortex, premotor_cortex, supplementary_motor_area, cerebellum | Urge to move — motor system engagement with the rhythm |
| **Emotional Intensity** | amygdala, insula | Felt emotional arousal — tension, surprise, release |
| **Sonic Detail** | auditory_cortex | Timbral and harmonic complexity the ear is processing |
| **Melodic Engagement** | superior_temporal_gyrus | How hard the brain is tracking pitch and melody |
| **Sonic Complexity** | anterior_cingulate, prefrontal_cortex | Harmonic surprise, structural processing, arrangement depth |
| **Reward / Pleasure** | nucleus_accumbens | Dopamine response — chills, hook payoff, anticipation |
| **Memory Resonance** | hippocampus | Whether this section is likely to encode as a strong memory |

---

## Scientific disclaimer

Scores reflect predicted neural engagement patterns from Meta FAIR's TRIBE v2 model. These are computational predictions based on population-level fMRI training data, not clinical measurements of individual brain activity. See the [TRIBE v2 paper](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) for methodology.

---

## Hardware

| Environment | Requirement |
|-------------|------------|
| Google Colab | T4 GPU (free tier) — validated in `notebooks/validation.ipynb` |
| Local | CUDA GPU with 8GB+ VRAM, or Apple Silicon (MPS backend) |
| CPU | Supported, but inference will be slow |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — matching TRIBE v2 upstream license. Free for non-commercial use with attribution.

---

## Attribution

- [Meta FAIR TRIBE v2](https://github.com/facebookresearch/tribev2) — brain prediction model
- [Nilearn](https://nilearn.github.io/) — brain surface visualization
- Zatorre et al. 2007, Koelsch 2014, Trost et al. 2012, Grahn & Brett 2007 — ROI-trait literature

---

Built by [WESLEYFRANKLIN](https://wslyfrnkln.com) — artist, developer.
