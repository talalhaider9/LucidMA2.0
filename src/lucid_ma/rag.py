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
        self.frame, self.settings = frame, settings
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

    def retrieve(self, user_query: str) -> dict[str, Any]:
        crime_type = self._select_type(user_query)
        subset = self.frame[self.frame["primary_type"] == crime_type].copy()
        hotspots = subset[subset["has_coordinates"]].groupby("spatial_cell").agg(incidents=("id", "size"), latitude=("latitude", "mean"), longitude=("longitude", "mean")).sort_values("incidents", ascending=False).head(self.settings.top_results).reset_index()
        hourly = subset.groupby("hour").size().reindex(range(24), fill_value=0)
        return {
            "query": user_query, "crime_type": crime_type, "total_records": int(len(subset)), "mapped_records": int(subset["has_coordinates"].sum()),
            "date_range": {"start": subset["date"].min().isoformat(), "end": subset["date"].max().isoformat()},
            "peak_hour": int(hourly.idxmax()), "hourly_counts": [int(hourly.loc[hour]) for hour in range(24)],
            "arrest_rate_percent": round(float(subset["arrest"].mean() * 100), 1), "domestic_rate_percent": round(float(subset["domestic"].mean() * 100), 1),
            "weekend_incidents": int(subset["is_weekend"].sum()), "hotspots": hotspots.to_dict(orient="records"),
            "top_descriptions": _top_counts(subset, "description", self.settings.top_results), "top_location_types": _top_counts(subset, "location_description", self.settings.top_results),
            "top_communities": _top_counts(subset, "community_area", self.settings.top_results), "top_districts": _top_counts(subset, "district", self.settings.top_results),
            "top_beats": _top_counts(subset, "beat", self.settings.top_results), "top_wards": _top_counts(subset, "ward", self.settings.top_results),
            "top_fbi_codes": _top_counts(subset, "fbi_code", self.settings.top_results), "weekday_counts": _top_counts(subset, "weekday", 7), "monthly_counts": _top_counts(subset, "month_name", 12),
        }

    @staticmethod
    def format_context(result: dict[str, Any]) -> str:
        peak = f"{result['peak_hour']:02d}:00"
        def entries(name: str) -> str:
            return "; ".join(f"{item['value']} ({item['incidents']})" for item in result[name][:5]) or "No data"
        return "\n".join(["--- VERIFIED CHICAGO CRIME PROFILE ---", f"Crime type: {result['crime_type']}", f"Records: {result['total_records']} ({result['mapped_records']} mapped)", f"Peak hour: {peak}", f"Arrest rate: {result['arrest_rate_percent']}% | Domestic rate: {result['domestic_rate_percent']}%", f"Top descriptions: {entries('top_descriptions')}", f"Top location types: {entries('top_location_types')}", f"Top community areas: {entries('top_communities')}", f"Top districts: {entries('top_districts')}"])


GraphCrimeRetriever = CrimeRetriever
