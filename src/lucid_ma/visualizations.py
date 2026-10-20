"""Create visual outputs from observed crime records, never simulated data."""

from pathlib import Path

import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import numpy as np

from .config import SETTINGS, Settings


def _bar_chart(items: list[dict], title: str, path: Path) -> Path | None:
    if not items:
        return None
    labels, counts = zip(*((item["value"], item["incidents"]) for item in items[:10]))
    figure, axis = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    axis.barh(labels[::-1], counts[::-1], color="#2563eb")
    axis.set(title=title, xlabel="Reported incidents")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def create_visualizations(result: dict, settings: Settings = SETTINGS) -> list[Path]:
    output = settings.output_dir
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    crime_name, hotspots = result["crime_type"].title(), result["hotspots"]
    if hotspots:
        map_path = output / "crime_hotspots.html"
        center = [float(np.mean([spot["latitude"] for spot in hotspots])), float(np.mean([spot["longitude"] for spot in hotspots]))]
        map_view = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
        HeatMap([[spot["latitude"], spot["longitude"], spot["incidents"]] for spot in hotspots], radius=20).add_to(map_view)
        for spot in hotspots:
            folium.Marker([spot["latitude"], spot["longitude"]], popup=f"{crime_name} grid<br>Incidents: {spot['incidents']}").add_to(map_view)
        map_view.save(str(map_path))
        generated.append(map_path)
    hours = np.arange(24)
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(hours, result["hourly_counts"], color=["#dc2626" if hour == result["peak_hour"] else "#94a3b8" for hour in hours])
    axis.set(title=f"Observed hourly incidents: {crime_name}", xlabel="Hour of day", ylabel="Reported incidents", xticks=hours)
    figure.tight_layout()
    hourly_path = output / "hourly_incidents.png"
    figure.savefig(hourly_path, dpi=150)
    plt.close(figure)
    generated.append(hourly_path)
    for key, title, filename in (("top_location_types", f"Top location types: {crime_name}", "location_types.png"), ("top_communities", f"Top community areas: {crime_name}", "community_areas.png"), ("top_districts", f"Top police districts: {crime_name}", "districts.png"), ("top_descriptions", f"Top descriptions: {crime_name}", "descriptions.png")):
        chart = _bar_chart(result[key], title, output / filename)
        if chart:
            generated.append(chart)
    return generated
