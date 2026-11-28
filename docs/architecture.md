# Architecture (v0.1, for review)

```
PyTorch model ─┐
               ├─> converter ─> NIR graph ─> targets (snnTorch, Lava, Akida Cloud, ...)
Model zoo ─────┘                                  │
                                                  v
                     report (HTML + JSON) <─ benchmark runner
```

## Modules

| Path | Owner agent | Purpose |
| --- | --- | --- |
| `src/pygri/results.py` | Architect | Result schema. Every `Metric` has a value, unit and `Provenance`. |
| `src/pygri/targets/base.py` | Architect | `Target` interface: `available()` and `run(workload)`. |
| `src/pygri/targets/__init__.py` | Architect | Registry: adapters register by name; nothing imports adapters directly. |
| `src/pygri/convert/` | Converter | PyTorch to spiking, NIR export (Phase 1). |
| `src/pygri/targets/<name>.py` | Target adapters | One file per target (Phase 1 to 2). |
| `src/pygri/energy/` | Benchmark and energy | Energy estimation model (Phase 2). |
| `src/pygri/report/` | Benchmark and energy | HTML and JSON rendering (Phase 1). |
| `zoo/` | Zoo trainer | Training scripts and model cards (Phase 1 to 3). |

## Key decisions

1. **NIR is the exchange format.** We build on it rather than inventing our own.
2. **Provenance is mandatory.** A `Metric` cannot be created without one. Reports must
   show it next to every number.
3. **Adapters fail softly.** `run()` returns `ok=False` with an error for expected
   problems (missing credentials, unsupported layers) so one bad target never breaks a report.
4. **No heavy dependencies in the core.** `torch`, `nir`, `lava` and vendor SDKs are
   optional extras, imported only inside the adapter or converter that needs them.

## Open questions for review

- Standard metric set: accuracy, latency, energy per inference. Add throughput or memory?
- Energy units: microjoules per inference proposed.

## Conversion and simulation semantics (Phase 1)

- **Converter** (`pygri.convert.convert_sequential`): supports `Flatten`, `Linear`, `ReLU`.
  Uses data-based weight normalisation (Diehl et al., 2015) and emits a NIR graph of
  `Affine` and `IF` nodes (r = 1, threshold = 1, reset to 0, `metadata["dt"] = 1`).
- **Inputs** are a constant current every timestep; the predicted class is the output
  neuron with the most spikes.
- **snnTorch target** loads NIR with Pygri's own loader instead of `snntorch.import_from_nir`,
  because snnTorch's importer maps `IF` to a leaky neuron (beta = 0.9), which changes results.
  A test checks the target against an independent NumPy simulation of the same graph.
- **Lava target** runs the same graph with Lava's floating-point CPU models: `Dense` for
  weights (biases through a separate `Dense` from a bias source) and `LIFReset` with
  `du = 1`, `dv = 0` as the IF neuron. Samples run back to back in one Lava run; each
  `Dense` adds one step of delay, so each layer resets at the start of its own window.
  A test checks spike counts match a NumPy reference exactly. On the demo, Lava and
  snnTorch give identical accuracy and spike counts.
- Unsupported layers or nodes are reported, never silently approximated.
