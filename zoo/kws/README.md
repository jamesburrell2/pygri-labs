# Model card: keyword spotting, MLP (kws-mlp v0.1)

A small keyword-spotting network trained as a normal ANN, then converted to an
integrate-and-fire spiking network with `pygri.convert` and benchmarked on Pygri's
simulator targets. This is the first, deliberately simple zoo baseline; it is not
yet tuned for accuracy or energy.

## Task and data

- **Dataset:** Google Speech Commands v0.02 (CC BY 4.0), the official
  train / validation / test lists.
- **Classes (10):** yes, no, up, down, left, right, on, off, stop, go. No "unknown"
  or "silence" class yet, so this is not directly comparable with 12-class results.
- **Test set:** 4,074 clips.

## Model

- **Features:** log-mel spectrogram, 40 bands (20 Hz to 4 kHz), 40 ms window, 20 ms hop,
  49 frames, flattened to 1,960 values and standardised with training-set statistics
  (`features.py`, NumPy only).
- **Network:** Linear(1960, 256), ReLU, Linear(256, 256), ReLU, Linear(256, 10).
  About 570k weights.
- **Training:** AdamW (lr 1e-3, weight decay 1e-2), cosine schedule, 40 epochs,
  batch 128, label smoothing 0.1, Gaussian input noise (σ = 0.3). Best epoch by
  validation accuracy. Seed 0, CPU.
- **Conversion:** data-based weight normalisation (99.9th percentile, 2,000 training
  clips), IF neurons with threshold 1 and reset to zero, constant-current input,
  class = output neuron with most spikes.

## Results

All numbers are **measured in simulation** on CPU. None are hardware measurements.

| Model | Timesteps | Test clips | Accuracy | Spikes per inference |
| --- | --- | --- | --- | --- |
| ANN (PyTorch, non-spiking) | n/a | 4,074 | 0.850 | n/a |
| SNN on snnTorch | 10 | 4,074 | 0.790 | 254 |
| SNN on snnTorch | 20 | 4,074 | 0.824 | 555 |
| SNN on snnTorch | 50 | 4,074 | 0.836 | 1,450 |
| SNN on snnTorch | 100 | 4,074 | 0.841 | 2,965 |
| SNN on snnTorch | 200 | 4,074 | 0.843 | 5,956 |
| SNN on snnTorch | 50 | first 500 | 0.874 | 1,517 |
| SNN on Lava | 50 | first 500 | 0.874 | 1,517 |

- Conversion costs about 1 point of accuracy at 100 timesteps and 6 points at 10.
- Lava and snnTorch give the same predictions and spike counts on the 500-clip
  subset, as expected, since both run the same integrate-and-fire semantics.
- Simulator wall-clock (not a hardware latency): snnTorch about 0.16 ms per clip
  (batched, T = 50); Lava about 680 ms per clip. Lava ran on 500 clips only because
  of that cost.
- Train accuracy is 0.998 against 0.874 validation, so the model overfits; there is
  room to improve with augmentation or a convolutional front end once the converter
  supports convolutions.

## Limitations

- 10 keywords only, no unknown/silence handling, no background-noise augmentation.
- Rate coding needs many timesteps; energy and latency on real hardware are not yet
  estimated (Phase 2).
- Weights are not published yet. Reproduce with the commands below.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[snntorch]"
curl -LO https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands && tar -xzf speech_commands_v0.02.tar.gz -C data/speech_commands
python zoo/kws/train.py --data data/speech_commands --out build/kws        # ~10 min on 4 CPUs
python zoo/kws/benchmark.py --build build/kws --target snntorch --timesteps 10 20 50 100 200
# In a Python 3.10 environment with the lava extra:
python zoo/kws/benchmark.py --build build/kws --target lava --timesteps 50 --limit 500
```
