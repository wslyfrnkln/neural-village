import pytest
import numpy as np
import subprocess
from pathlib import Path


@pytest.fixture
def sample_wav(tmp_path):
    """2-second stereo 44100Hz sine wave WAV."""
    import soundfile as sf
    sr = 44100
    duration = 2
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    tone = np.stack([0.5 * np.sin(2 * np.pi * 440 * t),
                     0.5 * np.sin(2 * np.pi * 440 * t)], axis=1)
    wav_path = tmp_path / "sample.wav"
    sf.write(str(wav_path), tone, sr)
    return wav_path


@pytest.fixture
def sample_mp3(tmp_path, sample_wav):
    """MP3 converted from sample_wav via ffmpeg."""
    mp3_path = tmp_path / "sample.mp3"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(sample_wav), str(mp3_path)],
        capture_output=True
    )
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr.decode()}"
    return mp3_path
