"""Fetch, validate, and enrich Chicago crime records."""

import networkx as nx
import pandas as pd
from sodapy import Socrata

from .config import SETTINGS, Settings

REQUIRED_COLUMNS = {"id", "date", "primary_type"}
TEXT_COLUMNS = ("case_number", "block", "iucr", "primary_type", "description", "location_description", "beat", "district", "ward", "community_area", "fbi_code")
BOOLEAN_COLUMNS = ("arrest", "domestic")
NUMERIC_COLUMNS = ("latitude", "longitude", "x_coordinate", "y_coordinate")


def _as_boolean(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin({"true", "t", "1", "yes", "y"})


def fetch_crime_data(settings: Settings = SETTINGS) -> pd.DataFrame:
    """Download recent incidents without discarding records lacking coordinates."""
    client = Socrata(settings.dataset_domain, None, timeout=60)
    try:
        records = client.get(settings.dataset_id, limit=settings.data_limit, order="date DESC")
    finally:
        client.close()
    frame = pd.DataFrame.from_records(records)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    for column in TEXT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = frame[column].astype("string").str.strip()
    for column in BOOLEAN_COLUMNS:
        frame[column] = _as_boolean(frame[column]) if column in frame else False
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = float("nan")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["id", "date", "primary_type"]).drop_duplicates(subset="id").copy()
    frame["primary_type"] = frame["primary_type"].str.upper()
    for column in ("description", "location_description"):
        if column in frame:
            frame[column] = frame[column].str.upper()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["month_name"] = frame["date"].dt.month_name()
    frame["hour"] = frame["date"].dt.hour
    frame["weekday"] = frame["date"].dt.day_name()
    frame["is_weekend"] = frame["date"].dt.dayofweek >= 5
    frame["has_coordinates"] = frame[["latitude", "longitude"]].notna().all(axis=1)
    frame["spatial_cell"] = pd.NA
    located = frame["has_coordinates"]
    precision = settings.hotspot_precision
    frame.loc[located, "spatial_cell"] = (
        "Lat:" + frame.loc[located, "latitude"].round(precision).map(lambda value: f"{value:.{precision}f}")
        + "_Lon:" + frame.loc[located, "longitude"].round(precision).map(lambda value: f"{value:.{precision}f}")
    )
    return frame


def build_crime_graph(frame: pd.DataFrame) -> nx.Graph:
    """Build a weighted graph of crime types and their observed attributes."""
    graph = nx.Graph()
    dimensions = {
        "description": "description",
        "location_description": "location",
        "community_area": "community",
        "district": "district",
        "beat": "beat",
        "ward": "ward",
        "fbi_code": "fbi_code",
        "spatial_cell": "hotspot",
    }
    for column, node_kind in dimensions.items():
        if column not in frame:
            continue
        for (crime, value), count in frame.dropna(subset=[column]).groupby(["primary_type", column]).size().items():
            crime_node = f"crime:{crime}"
            attribute_node = f"{node_kind}:{value}"
            graph.add_node(crime_node, kind="crime_type", label=str(crime))
            graph.add_node(
                attribute_node,
                kind=node_kind,
                label=str(value),
                source_column=column,
            )
            graph.add_edge(
                crime_node,
                attribute_node,
                relation=node_kind,
                weight=int(count),
            )
    return graph
