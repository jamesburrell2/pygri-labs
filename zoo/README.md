# Pygri model zoo

Trained models converted to spiking networks and benchmarked on Pygri targets. Each
model has a folder with its training script and a model card that labels every number
with how it was obtained.

| Model | Task | ANN accuracy | Best SNN accuracy (simulation) | Card |
| --- | --- | --- | --- | --- |
| kws-mlp v0.1 | Keyword spotting, 10 words | 0.850 | 0.843 (T = 200) | [kws](kws/README.md) |
