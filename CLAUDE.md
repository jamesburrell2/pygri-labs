# Pygri: instructions for AI agents

Pygri benchmarks spiking neural networks across neuromorphic simulators and chips, plus a
model zoo of edge models. Read `docs/architecture.md` before changing code.

## Agent roles

Each agent works only in the paths it owns and only on GitHub issues labeled for it.

| Agent | Model | Owns | Issue label |
| --- | --- | --- | --- |
| Architect / orchestrator | Opus | `src/pygri/results.py`, `src/pygri/targets/base.py`, `src/pygri/targets/__init__.py`, `src/pygri/cli.py`, `docs/` architecture | `agent:architect` |
| Converter | Opus | `src/pygri/convert/` | `agent:converter` |
| Target adapters | Sonnet | `src/pygri/targets/<name>.py` | `agent:targets` |
| Benchmark and energy | Sonnet (method reviewed by Opus) | `src/pygri/energy/`, `src/pygri/report/` | `agent:benchmark` |
| Zoo trainer | Sonnet | `zoo/` | `agent:zoo` |
| Test and QA | Haiku | `tests/`, `.github/workflows/` | `agent:qa` |
| Docs | Haiku | `README.md`, `docs/` guides, model cards | `agent:docs` |
| Independent reviewer | Opus | reviews PRs; never reviews its own code | n/a |

Budget mode (MVP): run at most 2 to 3 agents at a time; use Opus only for architecture,
conversion and final review.

## Rules

1. Interfaces in `results.py` and `targets/base.py` change only with the architect's approval.
2. Every change is a pull request from a branch. Tests and `ruff check .` must pass.
3. Every reported number is a `Metric` with a `Provenance`. Never present an estimate as
   a measurement.
4. Heavy libraries (`torch`, `nir`, `lava`, vendor SDKs) are optional extras, imported
   only inside the module that needs them.
5. Adapters must not raise for expected failures; return `TargetResult(ok=False, error=...)`.
6. Golden tests: once zoo models exist, a change that moves a zoo model's accuracy on any
   simulator by more than 1 point must fail CI and get a human look.
7. Human approval (James) is required for: interface changes, the end of each phase,
   publishing packages or public claims, spending beyond budget, and contacting vendors.

## Commands

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
