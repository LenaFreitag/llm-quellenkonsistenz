# LLM-Quellenkonsistenz

Analysen zur Konsistenz von Quellenangaben bei wiederholten Anfragen an ein KI-Suchsystem.

## Was ist das?

Dieses Projekt testet, wie stabil und konsistent ein KI-System Quellen angibt. Wenn ich dieselbe Frage mehrmals stelle, bekomme ich dann die gleichen Quellen? Diese Projekt sammelt und analysiert genau das.

## Wie ist es organisiert?

- **`collect_data.py`**: Sammelt Daten über die API
- **`data/raw/`**: Rohe Antworten von der API
- **`data/processed/`**: Bereinigte Quellenlisten zum Analysieren
- **`data/analysis/`**: Auswertungsergebnisse und Statistiken
- **`notebooks/`**: Jupyter Notebooks für die Datenanalyse
- **`prompts.csv`**: Die verwendeten Testfragen
- **`results/`**: Finale Ergebnisse

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # oder .venv\Scripts\activate auf Windows
pip install -r requirements.txt
```

Den API-Key in eine `.env` Datei speichern (nicht committen!).
