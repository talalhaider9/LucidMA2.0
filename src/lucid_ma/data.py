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
    """Build an incident-level graph linking each crime to its observed attributes.

    An incident node connects its crime type to its own place, time, classification,
    and outcome values. Retrieval can therefore follow multi-hop paths through real
    incident records instead of joining unrelated category aggregates.
    """
    graph = nx.Graph()
    dimensions = {
        "iucr": "iucr",
        "description": "description",
        "location_description": "location",
        "community_area": "community",
        "district": "district",
        "beat": "beat",
        "ward": "ward",
        "fbi_code": "fbi_code",
        "spatial_cell": "hotspot",
        "hour": "hour",
        "weekday": "weekday",
        "month_name": "month",
        "year": "year",
        "arrest": "arrest",
        "domestic": "domestic",
    }
    available_dimensions = {column: kind for column, kind in dimensions.items() if column in frame}
    for row in frame.itertuples(index=False):
        values = row._asdict()
        crime = str(values["primary_type"])
        crime_node = f"crime:{crime}"
        incident_node = f"incident:{values['id']}"
        graph.add_node(crime_node, kind="crime_type", label=crime)
        graph.add_node(incident_node, kind="incident", label=str(values["id"]))
        graph.add_edge(crime_node, incident_node, relation="has_incident")
        for column, node_kind in available_dimensions.items():
            value = values[column]
            if pd.isna(value) or str(value).strip() == "":
                continue
            label = str(value)
            attribute_node = f"{node_kind}:{label}"
            graph.add_node(
                attribute_node,
                kind=node_kind,
                label=label,
                source_column=column,
            )
            graph.add_edge(incident_node, attribute_node, relation=node_kind)
    return graph
