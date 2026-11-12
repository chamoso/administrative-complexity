"""Central configuration for the administrative complexity pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"
LOGS = OUTPUTS / "logs"

for _p in (RAW, INTERIM, PROCESSED, TABLES, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20260726
USER_AGENT = (
    "adminplex-research/0.1 (academic research on administrative complexity; "
    "contact: contact@example.org)"
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else ROOT / "config" / "default.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CONFIG = load_config()
