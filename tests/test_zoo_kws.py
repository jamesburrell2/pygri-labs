"""Tests for the keyword-spotting feature pipeline, on a tiny synthetic dataset."""

import sys
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zoo" / "kws"))
import features


def _write_wav(path: Path, freq: float, seconds: float = 0.8) -> None:
    t = np.arange(int(features.SAMPLE_RATE * seconds)) / features.SAMPLE_RATE
    pcm = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(features.SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def test_log_mel_shape_and_pitch():
    low = features.log_mel(np.sin(2 * np.pi * 200 * np.arange(16000) / 16000).astype("f4"))
    high = features.log_mel(np.sin(2 * np.pi * 2000 * np.arange(16000) / 16000).astype("f4"))
    assert low.shape == (49, features.N_MELS)
    assert low.mean(0).argmax() < high.mean(0).argmax()


def test_load_split_follows_official_lists(tmp_path):
    for i, word in enumerate(features.KEYWORDS):
        for j in range(3):
            _write_wav(tmp_path / word / f"spk{j}_nohash_0.wav", 200 + 150 * i)
    (tmp_path / "validation_list.txt").write_text("yes/spk1_nohash_0.wav\n")
    (tmp_path / "testing_list.txt").write_text("no/spk2_nohash_0.wav\ngo/spk2_nohash_0.wav\n")
    x_tr, _y_tr = features.load_split(tmp_path, "train")
    _x_te, y_te = features.load_split(tmp_path, "test")
    assert x_tr.shape == (27, 49 * features.N_MELS)
    assert sorted(y_te.tolist()) == [features.KEYWORDS.index("no"), features.KEYWORDS.index("go")]
    assert len(features.load_split(tmp_path, "validation")[1]) == 1
