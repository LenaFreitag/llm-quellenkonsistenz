import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================
# Konfiguration
# =========================

MODEL_NAME = "gemini-3.1-flash-lite"
RUNS_PER_PROMPT = 3
REQUEST_SLEEP_SECONDS = 7

PROMPTS_FILE = Path("prompts.csv")
RAW_OUTPUT_FILE = Path("data/raw/raw_runs.jsonl")
SOURCES_OUTPUT_FILE = Path("data/processed/sources.csv")

MAX_OUTPUT_TOKENS = 500
TEMPERATURE = 0.0


# =========================
# Hilfsfunktionen
# =========================

def ensure_directories() -> None:
    RAW_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt-Datei nicht gefunden: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"prompt_id", "category", "prompt"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "prompts.csv muss die Spalten prompt_id, category und prompt enthalten."
            )

        prompts = []
        for row in reader:
            prompts.append({
                "prompt_id": row["prompt_id"].strip(),
                "category": row["category"].strip(),
                "prompt": row["prompt"].strip(),
            })

    if not prompts:
        raise ValueError("prompts.csv enthält keine Prompts.")

    return prompts


def get_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def extract_grounding_sources(response) -> list[dict]:
    """
    Extrahiert Quellen aus Gemini Grounding Metadata.

    Erwartet werden Quellen ungefähr unter:
    response.candidates[*].grounding_metadata.grounding_chunks[*].web.uri
    """
    sources = []

    for candidate in response.candidates or []:
        grounding_metadata = getattr(candidate, "grounding_metadata", None)
        if not grounding_metadata:
            continue

        grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []

        for index, chunk in enumerate(grounding_chunks):
            web = getattr(chunk, "web", None)
            if not web:
                continue

            url = getattr(web, "uri", None)
            title = getattr(web, "title", None)

            if not url:
                continue

            sources.append({
                "source_index": index,
                "url": url,
                "domain": get_domain(url),
                "title": title,
            })

    # Duplikate pro Antwort entfernen
    seen_urls = set()
    unique_sources = []

    for source in sources:
        if source["url"] in seen_urls:
            continue

        seen_urls.add(source["url"])
        unique_sources.append(source)

    return unique_sources


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_sources_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "prompt_id",
        "category",
        "run_id",
        "timestamp",
        "model",
        "source_index",
        "url",
        "domain",
        "title",
    ]

    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)


def response_to_dict(response) -> dict:
    """
    Macht die vollständige Gemini-Antwort JSON-speicherbar.
    """
    try:
        return response.model_dump()
    except AttributeError:
        return json.loads(response.model_dump_json())


def call_gemini(client: genai.Client, prompt: str):
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    google_search=types.GoogleSearch()
                )
            ],
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )


# =========================
# Hauptprogramm
# =========================

def main() -> None:
    print("collect_data.py wurde gestartet")
    load_dotenv()
    ensure_directories()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY fehlt. Lege eine .env-Datei mit GEMINI_API_KEY=... an."
        )

    client = genai.Client(api_key=api_key)
    prompts = load_prompts(PROMPTS_FILE)

    print(f"Geladene Prompts: {len(prompts)}")
    print(f"Runs pro Prompt: {RUNS_PER_PROMPT}")
    print(f"Geplante Durchläufe insgesamt: {len(prompts) * RUNS_PER_PROMPT}")
    print(f"Modell: {MODEL_NAME}")
    print()

    for prompt_row in prompts:
        prompt_id = prompt_row["prompt_id"]
        category = prompt_row["category"]
        prompt = prompt_row["prompt"]

        for run_number in range(1, RUNS_PER_PROMPT + 1):
            run_id = f"{prompt_id}_R{run_number:02d}"
            timestamp = datetime.now(timezone.utc).isoformat()

            print(f"Starte {run_id}: {category}")

            try:
                response = call_gemini(client, prompt)
                sources = extract_grounding_sources(response)

                raw_record = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "system": "Gemini API mit Google Search Grounding",
                    "model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_text": response.text,
                    "sources": sources,
                    "raw_response": response_to_dict(response),
                    "error": None,
                }

                append_jsonl(RAW_OUTPUT_FILE, raw_record)

                source_rows = []
                for source in sources:
                    source_rows.append({
                        "prompt_id": prompt_id,
                        "category": category,
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "model": MODEL_NAME,
                        "source_index": source["source_index"],
                        "url": source["url"],
                        "domain": source["domain"],
                        "title": source["title"],
                    })

                append_sources_csv(SOURCES_OUTPUT_FILE, source_rows)

                print(f"  OK: {len(sources)} Quellen gefunden")

            except Exception as error:
                error_record = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "system": "Gemini API mit Google Search Grounding",
                    "model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_text": None,
                    "sources": [],
                    "raw_response": None,
                    "error": repr(error),
                }

                append_jsonl(RAW_OUTPUT_FILE, error_record)

                print(f"  FEHLER: {repr(error)}")

            time.sleep(REQUEST_SLEEP_SECONDS)

    print()
    print("Datenerhebung abgeschlossen.")
    print(f"Rohdaten: {RAW_OUTPUT_FILE}")
    print(f"Quellenliste: {SOURCES_OUTPUT_FILE}")


if __name__ == "__main__":
    main()