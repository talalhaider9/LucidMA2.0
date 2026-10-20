# LUCID-MA Chicago Crime Analysis

LUCID-MA analyzes the official City of Chicago Crimes dataset. The original notebook remains as an experiment record; the runnable application is under `src/lucid_ma`.

## Layout

```text
Project/
├── MLCS_Project.ipynb       Original notebook
├── pyproject.toml           Package metadata and CLI definition
├── requirements.txt         Editable package install
├── src/lucid_ma/
│   ├── config.py            Central settings
│   ├── data.py              Dataset download, validation, and enrichment
│   ├── rag.py               Semantic query matching and evidence profile
│   ├── reporting.py         JSON and Markdown report generation
│   ├── visualizations.py    Map and charts using observed counts
│   ├── agents.py            Optional AutoGen discussion
│   ├── main.py              Application orchestration
│   └── __main__.py          `python -m lucid_ma` entry point
└── outputs/                 Generated artifacts
```

## Setup and run

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m lucid_ma
```

For a different crime topic or output folder, for example:

```powershell
python -m lucid_ma --query "theft from motor vehicle" --limit 50000 --output-dir outputs/theft
```

The default workflow does not need Ollama. It downloads the configured data, maps the configured query to the best crime classification, and writes an analysis report plus charts under `outputs/`.

## Data coverage

The application uses incident details (ID, case number, date, block, IUCR, primary type, description, FBI code), location and jurisdiction fields (location description, beat, district, ward, community area, latitude, longitude), and outcomes (arrest and domestic flags). It derives hour, weekday, weekend flag, month, year, coordinate availability, and a coarse map grid.

Records without latitude or longitude remain in every non-map summary. The map uses only coordinate-bearing incidents.

## Output artifacts

- `analysis.json`: complete structured analysis for another program or notebook.
- `analysis.md`: concise report with top categories and rates.
- `crime_hotspots.html`: map of coordinate grid cells.
- PNG charts: observed hourly incidents, location types, community areas, districts, and descriptions.

## Optional agent discussion

Install `pip install -e .[agents]`, start Ollama and pull the configured model, then set `run_agents = True` in `src/lucid_ma/config.py`. Deterministic analysis and artifacts are created before the optional discussion runs.
