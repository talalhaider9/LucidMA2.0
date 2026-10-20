"""Central configuration for the LUCID-MA analysis pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    dataset_domain: str = "data.cityofchicago.org"
    dataset_id: str = "ijzp-q8t2"
    data_limit: int = 30_000
    crime_query: str = "assault"
    top_results: int = 10
    hotspot_precision: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"
    output_dir: Path = Path("outputs")
    # Read at map generation time so notebook secrets added after import are honored.
    carto_api_key: str | None = None
    run_agents: bool = False
    model_name: str = "deepseek-r1:1.5b"
    model_base_url: str = "http://localhost:11434/v1"
    model_api_key: str = "ollama"
    model_temperature: float = 0.3
    max_rounds: int = 4

    @property
    def llm_config(self) -> dict:
        return {"config_list": [{"model": self.model_name, "base_url": self.model_base_url,
                "api_key": self.model_api_key, "price": [0, 0]}], "temperature": self.model_temperature}


SETTINGS = Settings()
