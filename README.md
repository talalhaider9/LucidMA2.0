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

The default workflow does not need Ollama. It maps the query to matching crime classifications, traverses the NetworkX graph from crime type to incident nodes and then to connected attributes, and assembles evidence from those retrieved paths. Each incident node links fields from the same source record, including description, location, community, district, beat, ward, FBI code, hour, weekday, month, year, arrest, domestic status, and coarse coordinate cell. Pandas calculates the hourly summary, outcome percentages, and map centroids. The reports and charts are written under `outputs/`.

## Data coverage

The application uses incident details (ID, case number, date, block, IUCR, primary type, description, FBI code), location and jurisdiction fields (location description, beat, district, ward, community area, latitude, longitude), and outcomes (arrest and domestic flags). It derives hour, weekday, weekend flag, month, year, coordinate availability, and a coarse map grid.

Records without latitude or longitude remain in every non-map summary. The map uses only coordinate-bearing incidents.

### CARTO basemap in Colab

The map uses OpenStreetMap by default. To use CARTO Voyager tiles, add a Colab Secret named `CARTO_API_KEY` containing your key, then run this before launching the project:

```python
from google.colab import userdata
import os

os.environ["CARTO_API_KEY"] = userdata.get("CARTO_API_KEY")
```

The key is read from the environment when the map is generated and is not stored in project files. Keep attribution visible on exported maps.

## Output artifacts

- `analysis.json`: complete structured analysis for another program or notebook.
- `analysis.md`: concise report with top categories and rates.
- `crime_hotspots.html`: map of coordinate grid cells.
- PNG charts: observed hourly incidents, location types, community areas, districts, and descriptions.

## Optional agent discussion

Install `pip install -e .[agents]`, start Ollama and pull the configured model, then set `run_agents = True` in `src/lucid_ma/config.py` or pass `--agents`. The Analyst, BriefingWriter, and EvidenceReviewer refine the briefing over multiple epochs. `epoch_limit` in `config.py` caps the process at 10 epochs; it may stop sooner when the reviewer replies `VALID`. Each epoch can use up to three model responses. The JSON and Markdown reports remain the deterministic source of truth.
