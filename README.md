# Konsistenz von LLM-Quellenangaben

Dieses Projekt enthält die Datenerhebung und Auswertung für ein Seminarpaper zur Stabilität von Quellenangaben bei wiederholten identischen Anfragen an ein generatives Suchsystem.

## Struktur

- `prompts.csv`: verwendete Prompts
- `collect_data.py`: Datenerhebung über die Perplexity API
- `data/raw/`: rohe API-Antworten
- `data/processed/`: aufbereitete Quellenlisten
- `notebooks/analysis.ipynb`: Auswertung
- `results/`: Ergebnisdateien und Abbildungen
- `paper/`: LaTeX-Dateien des Papers

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Der API-Key wird lokal in einer `.env` gespeichert und nicht versioniert.

## Hinweise

Wichtig: API-Key niemals committen. Nutze eine `.env` und schließe sie via `.gitignore` aus.
