"""Write portable machine-readable and human-readable analysis artifacts."""

import json
from pathlib import Path

from .config import SETTINGS, Settings


def write_reports(result: dict, settings: Settings = SETTINGS) -> list[Path]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = settings.output_dir / "analysis.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    markdown = [f"# {result['crime_type'].title()} analysis", "", f"Query: `{result['query']}`", "", f"- Records analyzed: {result['total_records']}", f"- Records with coordinates: {result['mapped_records']}", f"- Coverage: {result['date_range']['start']} to {result['date_range']['end']}", f"- Peak observed hour: {result['peak_hour']:02d}:00", f"- Arrest rate: {result['arrest_rate_percent']}%", f"- Domestic-related rate: {result['domestic_rate_percent']}%", ""]
    for key, heading in (("top_descriptions", "Top descriptions"), ("top_location_types", "Top location types"), ("top_communities", "Top community areas"), ("top_districts", "Top districts"), ("top_beats", "Top beats"), ("top_fbi_codes", "Top FBI codes")):
        markdown.extend([f"## {heading}", ""])
        markdown.extend(f"- {item['value']}: {item['incidents']}" for item in result[key])
        markdown.append("")
    markdown_path = settings.output_dir / "analysis.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    return [json_path, markdown_path]
