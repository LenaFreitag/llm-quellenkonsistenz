from dotenv import load_dotenv
import os
import csv
import json
import time
from pathlib import Path
import requests

load_dotenv()

API_KEY = os.getenv('PERPLEXITY_API_KEY') or os.getenv('API_KEY')


def ensure_dirs():
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)


def load_prompts(path='prompts.csv'):
    prompts = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def fetch_answer(prompt_text):
    if not API_KEY:
        raise RuntimeError('API key not found. Set PERPLEXITY_API_KEY in .env')
    # NOTE: The following endpoint is illustrative. Replace with the real Perplexity API endpoint and payload.
    url = 'https://api.perplexity.ai/v1/answers'
    headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
    payload = {'prompt': prompt_text}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect(repeats=3, out_dir='data/raw'):
    ensure_dirs()
    prompts = load_prompts()
    for p in prompts:
        pid = p.get('prompt_id') or str(int(time.time()))
        text = p.get('prompt_text')
        for i in range(repeats):
            try:
                data = fetch_answer(text)
            except Exception as e:
                data = {'error': str(e)}
            fname = Path(out_dir) / f"prompt_{pid}_{i+1}.jsonl"
            with open(fname, 'w', encoding='utf-8') as w:
                json.dump({'prompt_id': pid, 'prompt': text, 'response': data, 'timestamp': time.time()}, w, ensure_ascii=False)
            time.sleep(1)


if __name__ == '__main__':
    # Simple CLI: python collect_data.py 5
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    collect(repeats=n)
