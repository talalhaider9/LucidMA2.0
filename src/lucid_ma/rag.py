"""Semantic crime selection and evidence-based incident profiling."""

import re
from collections import Counter
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

    def _graph_profile(self, crime_type: str) -> dict[str, Counter]:
        """Traverse crime -> incident -> attribute paths and count retrieved evidence."""
        profile: dict[str, Counter] = {}
        if self.graph is None:
            return profile
        crime_node = f"crime:{crime_type}"
        if crime_node not in self.graph:
            return profile
        for incident_node in self.graph.neighbors(crime_node):
            if self.graph.nodes[incident_node].get("kind") != "incident":
                continue
            for attribute_node in self.graph.neighbors(incident_node):
                attributes = self.graph.nodes[attribute_node]
                node_kind = attributes.get("kind")
                if node_kind == "incident":
                    continue
                profile.setdefault(node_kind, Counter())[attributes["label"]] += 1
        return profile

    def _attribute_counts(
        self,
        profile: dict[str, Counter],
        subset: pd.DataFrame,
        node_kind: str,
        column: str,
    ) -> list[dict[str, Any]]:
        """Format graph path counts, with a table fallback for direct API callers."""
        counts = profile.get(node_kind)
        if counts is None and self.graph is None:
            if column in subset:
                counts = Counter(subset[column].dropna().astype(str).tolist())
        if not counts:
            return []
        return [
            {"value": str(value), "incidents": int(count)}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: self.settings.top_results]
        ]

    @staticmethod
    def _split_query(query: str) -> list[str]:
        """Split a query into distinct crime topics.

        Connectors such as 'and', 'also', 'as well as', commas, and semicolons separate
        distinct topics (e.g. 'child abuse and rape' -> ['child abuse', 'rape']).
        """
        parts = re.split(r"\b(?:and|also|as well as|plus)\b|[,;]", query, flags=re.IGNORECASE)
        topics = [part.strip(" .,;:!?") for part in parts if part.strip()]
        return topics or [query.strip()]

    def _profile(self, crime_type: str) -> dict[str, Any]:
        subset = self.frame[self.frame["primary_type"] == crime_type].copy()
        graph_profile = self._graph_profile(crime_type)
        graph_hotspots = self._attribute_counts(graph_profile, subset, "hotspot", "spatial_cell")
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
            "crime_type": crime_type, "total_records": int(len(subset)), "mapped_records": int(subset["has_coordinates"].sum()),
            "date_range": {"start": subset["date"].min().isoformat(), "end": subset["date"].max().isoformat()},
            "peak_hour": int(hourly.idxmax()), "hourly_counts": [int(hourly.loc[hour]) for hour in range(24)],
            "arrest_rate_percent": round(float(subset["arrest"].mean() * 100), 1), "domestic_rate_percent": round(float(subset["domestic"].mean() * 100), 1),
            "weekend_incidents": int(subset["is_weekend"].sum()), "hotspots": hotspots,
            "top_descriptions": self._attribute_counts(graph_profile, subset, "description", "description"),
            "top_location_types": self._attribute_counts(graph_profile, subset, "location", "location_description"),
            "top_communities": self._attribute_counts(graph_profile, subset, "community", "community_area"),
            "top_districts": self._attribute_counts(graph_profile, subset, "district", "district"),
            "top_beats": self._attribute_counts(graph_profile, subset, "beat", "beat"),
            "top_wards": self._attribute_counts(graph_profile, subset, "ward", "ward"),
            "top_fbi_codes": self._attribute_counts(graph_profile, subset, "fbi_code", "fbi_code"),
            "graph_retrieval": {
                "method": "crime_type -> incident -> attribute",
                "incident_nodes": int(self.graph.degree[f"crime:{crime_type}"]) if self.graph is not None and f"crime:{crime_type}" in self.graph else int(len(subset)),
                "attributes_by_relation": {
                    kind: int(sum(counts.values())) for kind, counts in graph_profile.items()
                },
            },
            "weekday_counts": _top_counts(subset, "weekday", 7),
            "monthly_counts": _top_counts(subset, "month_name", 12),
        }

    def retrieve(self, user_query: str) -> dict[str, Any]:
        """Profile every distinct crime topic in the query.

        A multi-part query ('child abuse and rape') returns a combined result carrying a
        list of per-crime-type profiles plus a summary of which types matched.
        """
        crime_type = self._select_type(user_query)
        base = self._profile(crime_type)
        topics = self._split_query(user_query)
        profiles: list[dict[str, Any]] = []
        seen: set[str] = set()
        for topic in topics:
            topic_type = self._select_type(topic)
            if topic_type in seen:
                continue
            seen.add(topic_type)
            profile = self._profile(topic_type)
            profile["query"] = topic
            profiles.append(profile)
        if len(profiles) > 1:
            base = dict(base)
            base["query"] = user_query
            base["crime_type"] = " ; ".join(profile["crime_type"] for profile in profiles)
            base["results"] = profiles
            base["total_records"] = sum(profile["total_records"] for profile in profiles)
        else:
            base["query"] = user_query
        return base

    @staticmethod
    def format_context(result: dict[str, Any]) -> str:
        profiles = result.get("results")
        if profiles:
            return "\n\n".join(
                CrimeRetriever.format_context(profile) for profile in profiles
            )

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
            f"GraphRAG paths traversed: {result['graph_retrieval']['method']} across {result['graph_retrieval']['incident_nodes']} matching incident nodes.",
        ])


GraphCrimeRetriever = CrimeRetriever
