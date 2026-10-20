"""Semantic crime selection and evidence-based incident profiling."""

from typing import Any

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

from .config import SETTINGS, Settings


def _top_counts(frame: pd.DataFrame, column: str, limit: int) -> list[dict[str, Any]]:
    if column not in frame:
        return []
    counts = frame[column].dropna().replace("", pd.NA).dropna().value_counts().head(limit)
    return [{"value": str(value), "incidents": int(count)} for value, count in counts.items()]


class CrimeRetriever:
    def __init__(self, frame: pd.DataFrame, graph=None, settings: Settings = SETTINGS):
        self.frame, self.graph, self.settings = frame, graph, settings
        labels = frame[["primary_type", "description"]].drop_duplicates().fillna("")
        self.catalog = [f"{row.primary_type}: {row.description}" if row.description else row.primary_type for row in labels.itertuples(index=False)]
        self.catalog_types = labels["primary_type"].tolist()
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.embeddings = self.embedder.encode(self.catalog, convert_to_tensor=True)

    def _select_type(self, query: str) -> str:
        exact = self.frame.loc[self.frame["primary_type"] == query.upper().strip(), "primary_type"]
        if not exact.empty:
            return str(exact.iloc[0])
        scores = util.pytorch_cos_sim(self.embedder.encode(query, convert_to_tensor=True), self.embeddings)[0]
        return self.catalog_types[torch.argmax(scores).item()]

    def _graph_counts(self, crime_type: str, node_kind: str) -> list[dict[str, Any]]:
        """Return the strongest graph connections for one crime type and relation."""
        if self.graph is None:
            return []
        crime_node = f"crime:{crime_type}"
        if crime_node not in self.graph:
            return []
        matches = []
        for neighbor in self.graph.neighbors(crime_node):
            attributes = self.graph.nodes[neighbor]
            if attributes.get("kind") != node_kind:
                continue
            matches.append({
                "value": attributes["label"],
                "incidents": int(self.graph[crime_node][neighbor]["weight"]),
            })
        return sorted(matches, key=lambda item: (-item["incidents"], item["value"]))[: self.settings.top_results]

    def retrieve(self, user_query: str) -> dict[str, Any]:
        crime_type = self._select_type(user_query)
        subset = self.frame[self.frame["primary_type"] == crime_type].copy()
        graph_hotspots = self._graph_counts(crime_type, "hotspot")
        centroid_by_cell = subset[subset["has_coordinates"]].groupby("spatial_cell").agg(
            latitude=("latitude", "mean"), longitude=("longitude", "mean")
        )
        hotspots = [
            {
                "spatial_cell": item["value"],
                "incidents": item["incidents"],
                "latitude": float(centroid_by_cell.loc[item["value"], "latitude"]),
                "longitude": float(centroid_by_cell.loc[item["value"], "longitude"]),
            }
            for item in graph_hotspots
            if item["value"] in centroid_by_cell.index
        ]
        hourly = subset.groupby("hour").size().reindex(range(24), fill_value=0)
        return {
            "query": user_query, "crime_type": crime_type, "total_records": int(len(subset)), "mapped_records": int(subset["has_coordinates"].sum()),
            "date_range": {"start": subset["date"].min().isoformat(), "end": subset["date"].max().isoformat()},
            "peak_hour": int(hourly.idxmax()), "hourly_counts": [int(hourly.loc[hour]) for hour in range(24)],
            "arrest_rate_percent": round(float(subset["arrest"].mean() * 100), 1), "domestic_rate_percent": round(float(subset["domestic"].mean() * 100), 1),
            "weekend_incidents": int(subset["is_weekend"].sum()), "hotspots": hotspots,
            "top_descriptions": self._graph_counts(crime_type, "description"),
            "top_location_types": self._graph_counts(crime_type, "location"),
            "top_communities": self._graph_counts(crime_type, "community"),
            "top_districts": self._graph_counts(crime_type, "district"),
            "top_beats": self._graph_counts(crime_type, "beat"),
            "top_wards": self._graph_counts(crime_type, "ward"),
            "top_fbi_codes": self._graph_counts(crime_type, "fbi_code"),
            "weekday_counts": _top_counts(subset, "weekday", 7),
            "monthly_counts": _top_counts(subset, "month_name", 12),
        }

    @staticmethod
    def format_context(result: dict[str, Any]) -> str:
        peak = f"{result['peak_hour']:02d}:00"

        def entries(name: str) -> str:
            return "; ".join(f"{item['value']} ({item['incidents']})" for item in result[name][:5]) or "No data"
        hotspots = "; ".join(
            f"{spot['spatial_cell']} ({spot['incidents']})"
            for spot in result["hotspots"][:5]
        ) or "No coordinate data"
        return "\n".join([
            "--- VERIFIED CHICAGO CRIME PROFILE ---",
            f"Crime type: {result['crime_type']}",
            f"Records: {result['total_records']} ({result['mapped_records']} have coordinate data)",
            f"Coverage: {result['date_range']['start']} to {result['date_range']['end']}",
            f"Peak observed hour: {peak}",
            f"Arrest rate: {result['arrest_rate_percent']}% | Domestic-related rate: {result['domestic_rate_percent']}%",
            f"Top descriptions: {entries('top_descriptions')}",
            f"Top location types: {entries('top_location_types')}",
            f"Top community areas by reported count: {entries('top_communities')}",
            f"Top districts by reported count: {entries('top_districts')}",
            f"Top coarse hotspot cells (latitude/longitude grid, not addresses): {hotspots}",
        ])


GraphCrimeRetriever = CrimeRetriever
