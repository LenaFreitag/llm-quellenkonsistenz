import csv
import json
import os
import re
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
RUNS_PER_PROMPT = 20
REQUEST_SLEEP_SECONDS = 5

PROMPTS_FILE = Path("prompts.csv")
RAW_OUTPUT_FILE = Path("data/raw/raw_runs_no_grounding_temp1.jsonl")
SOURCES_OUTPUT_FILE = Path("data/processed/sources_no_grounding_temp1.csv")

MAX_OUTPUT_TOKENS = 8165
TEMPERATURE = 1.0

MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 60


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


def load_completed_prompts(path: Path, runs_per_prompt: int) -> set[str]:
    """
    Gibt prompt_ids zurück, für die ALLE runs_per_prompt Runs erfolgreich sind.
    Für diesen Test gilt ein Run als erfolgreich, wenn error=None ist.
    Quellen sind hier optional, weil Search Grounding deaktiviert ist.
    """
    if not path.exists():
        return set()

    success_counts: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                has_no_error = record.get("error") is None

                if has_no_error:
                    pid = record["prompt_id"]
                    success_counts[pid] = success_counts.get(pid, 0) + 1

            except json.JSONDecodeError:
                continue

    return {pid for pid, count in success_counts.items() if count >= runs_per_prompt}


def get_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def clean_url(url: str) -> str:
    """
    Entfernt typische Satzzeichen, die beim Regex-Match am Ende einer URL hängen bleiben.
    """
    return url.strip().rstrip(").,;:]}>\"'")


def extract_urls_from_text(text: str) -> list[dict]:
    """
    Extrahiert URLs aus dem generierten Antworttext.
    Achtung: Das sind keine Grounding-Quellen, sondern nur im Text genannte URLs.
    """
    if not text:
        return []

    url_pattern = r"https?://[^\s\]\)\}\"'<>]+"
    matches = re.findall(url_pattern, text)

    sources = []
    seen_urls = set()

    for index, match in enumerate(matches):
        url = clean_url(match)

        if url in seen_urls:
            continue

        seen_urls.add(url)

        sources.append({
            "source_index": len(sources),
            "url": url,
            "domain": get_domain(url),
            "title": None,
            "source_extraction_method": "text_regex",
        })

    return sources


def extract_retry_delay(error_repr: str) -> int:
    match = re.search(r"retryDelay.*?(\d+)s", error_repr)
    if match:
        return int(match.group(1)) + 2
    return RETRY_SLEEP_SECONDS


def json_default(obj):
    """
    Macht nicht direkt JSON-serialisierbare Objekte speicherbar.
    Verhindert z.B. TypeError: Object of type bytes is not JSON serializable.
    """
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    return str(obj)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False, default=json_default) + "\n"
        )


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
        "source_extraction_method",
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
    Speichert die Rohantwort möglichst robust.
    Falls das SDK interne bytes enthält, fällt es auf Textrepräsentation zurück.
    """
    try:
        return json.loads(response.model_dump_json())
    except Exception:
        return {"raw_response_as_text": str(response)}


def call_gemini_with_retry(client: genai.Client, prompt: str):
    """
    Ruft Gemini OHNE Google Search Grounding auf.
    Das testet nur, ob das Textmodell funktioniert.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  API-Call ohne Search Grounding mit Modell: {MODEL_NAME}")

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )

        except Exception as e:
            error_str = repr(e)

            is_quota_limit = (
                "You exceeded your current quota" in error_str
                or "GenerateRequestsPerDay" in error_str
                or ("RESOURCE_EXHAUSTED" in error_str and "retryDelay" not in error_str)
            )

            is_retryable = (
                "503" in error_str
                or ("429" in error_str and not is_quota_limit)
            )

            print()
            print(f"  FEHLER bei Versuch {attempt}/{MAX_RETRIES}:")
            print(f"  Fehlertyp: {e.__class__.__name__}")
            print(f"  Fehlermeldung: {error_str}")

            if is_retryable and attempt < MAX_RETRIES:
                wait = extract_retry_delay(error_str)
                print(f"  Retry möglich, warte {wait}s...")
                print()
                time.sleep(wait)
            else:
                print("  Kein weiterer Retry. Fehler wird an main() weitergegeben.")
                print()
                raise


# =========================
# Hauptprogramm
# =========================

def main() -> None:
    print("collect_data.py wurde gestartet")
    print("Modus: OHNE Google Search Grounding")
    print()

    load_dotenv()
    ensure_directories()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY fehlt. Lege eine .env-Datei mit GEMINI_API_KEY=... an."
        )

    client = genai.Client(api_key=api_key)

    prompts = load_prompts(PROMPTS_FILE)
    completed_prompts = load_completed_prompts(RAW_OUTPUT_FILE, RUNS_PER_PROMPT)

    total = len(prompts) * RUNS_PER_PROMPT
    remaining = (len(prompts) - len(completed_prompts)) * RUNS_PER_PROMPT

    print(f"Modell:       {MODEL_NAME}")
    print(f"Prompts:      {len(prompts)}")
    print(f"Runs/Prompt:  {RUNS_PER_PROMPT}")
    print(
        f"Gesamt:       {total}  |  "
        f"Prompts fertig: {len(completed_prompts)}/{len(prompts)}  |  "
        f"offen: ~{remaining} Runs"
    )
    print(f"Rohdaten:     {RAW_OUTPUT_FILE}")
    print(f"Quellenliste: {SOURCES_OUTPUT_FILE}")
    print()

    for prompt_row in prompts:
        prompt_id = prompt_row["prompt_id"]
        category = prompt_row["category"]
        prompt = prompt_row["prompt"]

        if prompt_id in completed_prompts:
            print(f"Überspringe {prompt_id} (alle {RUNS_PER_PROMPT} Runs erfolgreich)")
            continue

        for run_number in range(1, RUNS_PER_PROMPT + 1):
            run_id = f"{prompt_id}_R{run_number:02d}"
            timestamp = datetime.now(timezone.utc).isoformat()

            print(f"Starte {run_id}: {category}")

            try:
                response = call_gemini_with_retry(client, prompt)
                response_text = response.text or ""

                sources = extract_urls_from_text(response_text)

                raw_record = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "system": "Gemini API ohne Google Search Grounding",
                    "model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_text": response_text,
                    "sources": sources,
                    "raw_response": response_to_dict(response),
                    "error": None,
                }

                append_jsonl(RAW_OUTPUT_FILE, raw_record)

                source_rows = [{
                    "prompt_id": prompt_id,
                    "category": category,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "model": MODEL_NAME,
                    "source_index": s["source_index"],
                    "url": s["url"],
                    "domain": s["domain"],
                    "title": s["title"],
                    "source_extraction_method": s["source_extraction_method"],
                } for s in sources]

                append_sources_csv(SOURCES_OUTPUT_FILE, source_rows)

                print("  OK: Antwort erhalten")

                if len(sources) == 0:
                    print("  Hinweis: Keine URLs im Antworttext gefunden.")
                else:
                    print(f"  OK: {len(sources)} URLs aus Antworttext extrahiert")

            except Exception as error:
                error_record = {
                    "prompt_id": prompt_id,
                    "category": category,
                    "prompt": prompt,
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "system": "Gemini API ohne Google Search Grounding",
                    "model": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_text": None,
                    "sources": [],
                    "raw_response": None,
                    "error": repr(error),
                }

                append_jsonl(RAW_OUTPUT_FILE, error_record)
                print(f"  FEHLER (wird nicht wiederholt): {repr(error)}")

            print()
            time.sleep(REQUEST_SLEEP_SECONDS)

    completed_after = load_completed_prompts(RAW_OUTPUT_FILE, RUNS_PER_PROMPT)

    print()
    print("Datenerhebung abgeschlossen.")
    print(f"Prompts vollständig: {len(completed_after)}/{len(prompts)}")
    print(f"Rohdaten:     {RAW_OUTPUT_FILE}")
    print(f"Quellenliste: {SOURCES_OUTPUT_FILE}")


if __name__ == "__main__":
    main()