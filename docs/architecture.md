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
