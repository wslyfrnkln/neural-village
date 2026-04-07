"""
Local validation script — mirrors validation.ipynb but runs on MPS (Apple Silicon).
VRAM numbers won't apply to Colab T4, but confirms model loads + inference runs.
"""
import torch
import subprocess
import numpy as np
import tempfile
import os
import time

# Device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

# Create 30s test tone
print("\n--- Creating test audio ---")
import soundfile as sf

sr = 16000
duration = 30
t = np.linspace(0, duration, sr * duration, dtype=np.float32)
tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
wav_path = "/tmp/test_tone.wav"
sf.write(wav_path, tone, sr)

mp4_path = "/tmp/test_tone.mp4"
result = subprocess.run(
    ["ffmpeg", "-y", "-i", wav_path,
     "-f", "lavfi", "-i", "color=c=black:s=320x240:r=1",
     "-shortest", "-c:v", "libx264", "-c:a", "aac", mp4_path],
    capture_output=True
)
print(f"Test MP4 created: {os.path.getsize(mp4_path)} bytes")

# Load model
print("\n--- Loading TRIBE v2 ---")
from tribev2 import TribeModel

t0 = time.time()
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

# Inference
print("\n--- Running inference on 30s tone ---")
t0 = time.time()
df = model.get_events_dataframe(video_path=mp4_path)
preds, segments = model.predict(events=df)
inference_time = time.time() - t0

print(f"Inference time: {inference_time:.1f}s")
print(f"Output shape: {preds.shape}")
print(f"Segments: {len(segments)}")

# Results
print("\n--- Results ---")
TIME_LIMIT = 300
time_ok = inference_time < TIME_LIMIT

import json
results = {
    "device": device,
    "time_ok": time_ok,
    "inference_time_s": round(inference_time, 1),
    "output_shape": list(preds.shape),
    "load_time_s": round(load_time, 1),
}
print(json.dumps(results, indent=2))

if time_ok:
    print("\nLOCAL VALIDATION PASSED")
else:
    print(f"\nTOO SLOW: {inference_time:.1f}s for 30s audio")
