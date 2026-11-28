# Pygri

**Which neuromorphic chip runs your model best?** Pygri converts a PyTorch model into a
spiking neural network, runs it on several neuromorphic simulators and chips, and gives you
one side-by-side report of accuracy, latency and energy. Every number says whether it was
measured on hardware, measured in simulation, or estimated.

Status: **pre-alpha (Phase 1).** The pipeline runs end to end on synthetic data with one
simulator targets (snnTorch and Intel Lava).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[snntorch]"
pygri targets          # list targets and whether they are usable
pygri demo             # train, convert, benchmark, write reports/demo/report.html
```

The Lava target needs Python 3.10 (a limit of `lava-nc`): `pip install -e ".[lava]"` in a
3.10 environment.

## Planned targets

| Target | Kind | Phase |
| --- | --- | --- |
| snnTorch | simulator | 1 (done) |
| Lava (Intel) | simulator | 1 (done) |
| BrainChip Akida Cloud | cloud hardware | 2 |
| SynSense, SpiNNaker2, Loihi 2 | various | later |

Models move between tools through the open
[Neuromorphic Intermediate Representation (NIR)](https://github.com/neuromorphs/NIR).

## Model zoo (planned)

Ready, benchmarked spiking models for edge tasks, starting with keyword spotting.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

See [docs/architecture.md](docs/architecture.md) for the design and
[CLAUDE.md](CLAUDE.md) for how AI agents contribute.

## License

Apache-2.0. Copyright Pygri Labs.
