import pytest
from pathlib import Path
from neural_playback.preprocess import preprocess_audio, validate_ffmpeg


def test_mp3_to_mp4_creates_output(sample_mp3, tmp_path):
    out = preprocess_audio(sample_mp3, output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".mp4"


def test_wav_to_mp4_creates_output(sample_wav, tmp_path):
    out = preprocess_audio(sample_wav, output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".mp4"


def test_invalid_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_audio(tmp_path / "nonexistent.mp3")


def test_non_audio_raises(tmp_path):
    txt = tmp_path / "not_audio.txt"
    txt.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        preprocess_audio(txt)


def test_validate_ffmpeg_returns_bool():
    result = validate_ffmpeg()
    assert isinstance(result, bool)
    assert result is True  # ffmpeg must be installed
