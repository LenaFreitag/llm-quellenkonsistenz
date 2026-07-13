# LLM-Quellenkonsistenz

Untersuchung der Konsistenz von Quellenangaben bei wiederholten Anfragen an ein KI-Suchsystem.

## Struktur

- **`collect_data.py`** – Datenerfassung über API
- **`prompts.csv`** – Testfragen
- **`data/raw/`** – Rohe API-Antworten
- **`data/processed/`** – Bereinigte Quellenlisten
- **`data/analysis/`** – Auswertungsergebnisse
- **`notebooks/`** – Analysen in Jupyter
- **`results/`** – Ergebnisdateien

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

API-Key in `.env` speichern.
