"""Render the report using separate HTML/CSS/JavaScript sources."""

import json
import os
from pathlib import Path
import tempfile

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def report_json(report: dict) -> str:
    """Strict JSON: undefined metrics must be null, never NaN/Infinity."""
    return json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def render_html(report: dict) -> str:
    """Embed data safely, including names containing HTML or script end tags."""
    data = (report_json(report).replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    template = (WEB_ROOT / "report.html").read_text(encoding="utf-8")
    # Data goes in last so data values can never be interpreted as template tokens.
    for token, filename in (("__REPORT_STYLES__", "report.css"), ("__REPORT_SCRIPT__", "report.js")):
        if template.count(token) != 1:
            raise ValueError(f"report template must contain exactly one {token}")
        template = template.replace(token, (WEB_ROOT / filename).read_text(encoding="utf-8"))
    if template.count("__REPORT_DATA__") != 1:
        raise ValueError("report template must contain exactly one __REPORT_DATA__")
    return template.replace("__REPORT_DATA__", data)


def _atomic_write(path: Path, content: str) -> None:
    """A reader sees either the previous complete file or the new complete file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_report(report: dict, output_dir: str | Path) -> tuple[Path, Path]:
    """Write a self-contained index.html and machine-readable report.json."""
    payload = report_json(report)
    html = render_html(report)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path, json_path = output_dir / "index.html", output_dir / "report.json"
    _atomic_write(json_path, payload)
    _atomic_write(html_path, html)
    return html_path, json_path
