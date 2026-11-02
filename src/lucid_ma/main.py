"""Run the LUCID-MA Chicago crime analysis pipeline."""

import argparse
import re
from dataclasses import replace
from pathlib import Path

from .config import SETTINGS


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Chicago crime incidents by semantic crime query.")
    parser.add_argument("--query", default=SETTINGS.crime_query, help="Crime type or description to analyze.")
    parser.add_argument("--limit", type=int, default=SETTINGS.data_limit, help="Recent records to request from the dataset.")
    parser.add_argument("--output-dir", default=str(SETTINGS.output_dir), help="Directory for reports and charts.")
    parser.add_argument("--agents", action="store_true", help="Run the optional AutoGen discussion after artifacts are created.")
    return parser.parse_args()


def main():
    args = _arguments()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    settings = replace(SETTINGS, crime_query=args.query, data_limit=args.limit, output_dir=Path(args.output_dir), run_agents=args.agents or SETTINGS.run_agents)
    from .data import build_crime_graph, fetch_crime_data
    from .rag import GraphCrimeRetriever
    from .reporting import write_reports
    from .visualizations import create_visualizations

    print(f"Analyzing query: {settings.crime_query}")
    print("Fetching Chicago crime data...")
    frame = fetch_crime_data(settings)
    print(f"Cleaned data: {len(frame):,} records")
    graph = build_crime_graph(frame)
    print(f"Knowledge graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    retriever = GraphCrimeRetriever(frame, graph, settings)
    result = retriever.retrieve(settings.crime_query)
    context = retriever.format_context(result)
    print(context)

    profiles = result.get("results") or [result]
    paths = []
    for index, profile in enumerate(profiles):
        profile_settings = settings
        if len(profiles) > 1:
            slug = re.sub(r"\W+", "_", profile["crime_type"].lower()).strip("_")
            profile_settings = replace(settings, output_dir=settings.output_dir / f"{index}_{slug}")
        paths += write_reports(profile, profile_settings) + create_visualizations(profile, profile_settings)
    if settings.run_agents:
        from .agents import run_agent_analysis
        run_agent_analysis(context, settings)
    for path in paths:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
