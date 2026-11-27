"""A self-contained HTML report. Every number shows its provenance."""

from __future__ import annotations

from html import escape

from pygri.results import BenchmarkReport, Provenance

_BADGE = {
    Provenance.MEASURED_HARDWARE: ("measured on hardware", "#1b7f3b"),
    Provenance.MEASURED_SIMULATION: ("measured in simulation", "#2f5fb3"),
    Provenance.ESTIMATED: ("estimated", "#a15c00"),
}


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def render_html(report: BenchmarkReport) -> str:
    keys: list[str] = []
    for r in report.results:
        for k in r.metrics:
            if k not in keys:
                keys.append(k)

    head = "".join(f"<th>{escape(k.replace('_', ' '))}</th>" for k in keys)
    rows = []
    for r in report.results:
        if not r.ok:
            cells = f'<td colspan="{len(keys)}" class="err">failed: {escape(r.error)}</td>'
        else:
            cells = ""
            for k in keys:
                m = r.metrics.get(k)
                if m is None:
                    cells += "<td></td>"
                    continue
                label, color = _BADGE[m.provenance]
                note = f'<div class="note">{escape(m.note)}</div>' if m.note else ""
                cells += (
                    f"<td>{_fmt(m.value)} {escape(m.unit)}"
                    f'<div class="badge" style="color:{color}">{label}</div>{note}</td>'
                )
        rows.append(f'<tr><th scope="row">{escape(r.target)}</th>{cells}</tr>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pygri report: {escape(report.model)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a;background:#fff}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:8px 10px;text-align:left;vertical-align:top}}
.badge{{font-size:12px}} .note{{font-size:12px;color:#666}} .err{{color:#b00020}}
</style></head><body>
<h1>Pygri report: {escape(report.model)}</h1>
<p>Created {escape(report.created_at)} &middot; schema {escape(report.schema_version)}</p>
<table><thead><tr><th>Target</th>{head}</tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>
"""
