"""Speech Commands loading and log-mel features, using only the standard library and NumPy.

Features: 1 s of 16 kHz audio (zero-padded or cut), 40 ms Hann window, 20 ms hop,
40 mel bands from 20 Hz to 4 kHz, log(energy + 1e-6). That gives 49 frames x 40 bands
= 1960 values per clip.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
WIN = 640  # 40 ms
HOP = 320  # 20 ms
N_FFT = 1024
N_MELS = 40
FMIN, FMAX = 20.0, 4000.0

KEYWORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1 or f.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit PCM")
        raw = f.readframes(f.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    out = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out[: min(len(audio), SAMPLE_RATE)] = audio[:SAMPLE_RATE]
    return out


def _mel_filterbank() -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * mel_to_hz(mels) / SAMPLE_RATE).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        lo, mid, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, mid):
            fb[m - 1, k] = (k - lo) / max(mid - lo, 1)
        for k in range(mid, hi):
            fb[m - 1, k] = (hi - k) / max(hi - mid, 1)
    return fb


_FB = _mel_filterbank()
_WINDOW = np.hanning(WIN).astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Return a (frames, N_MELS) log-mel spectrogram of one clip."""
    n_frames = 1 + (len(audio) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = audio[idx] * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2
    return np.log(power @ _FB.T + 1e-6).astype(np.float32)


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the 10-keyword subset for 'train', 'validation' or 'test' (official lists)."""
    root = Path(root)
    val = set((root / "validation_list.txt").read_text().split())
    test = set((root / "testing_list.txt").read_text().split())
    xs, ys = [], []
    for label, word in enumerate(KEYWORDS):
        for path in sorted((root / word).glob("*.wav")):
            rel = f"{word}/{path.name}"
            in_split = (
                rel in val
                if split == "validation"
                else rel in test
                if split == "test"
                else rel not in val and rel not in test
            )
            if in_split:
                xs.append(log_mel(read_wav(path)).reshape(-1))
                ys.append(label)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)
