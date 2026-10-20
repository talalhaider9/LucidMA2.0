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

The default workflow does not need Ollama. It maps the configured query to the best crime classification, then uses the weighted NetworkX graph to retrieve the strongest observed links to descriptions, location types, community areas, districts, beats, wards, FBI codes, and coordinate cells. Pandas calculates time series, rates, and map centroids. The program writes an analysis report plus charts under `outputs/`.

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

Install `pip install -e .[agents]`, start Ollama and pull the configured model, then set `run_agents = True` in `src/lucid_ma/config.py`. The optional workflow performs one Analyst, BriefingWriter, and EvidenceReviewer cycle. It receives the verified summary, including coarse hotspot cells, and the JSON and Markdown reports remain the source of truth.
