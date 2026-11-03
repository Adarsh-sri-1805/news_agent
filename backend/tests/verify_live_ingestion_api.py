"""Verification of live news ingestion via REST API with articles: []."""

import json
import sys
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=180.0)

payload = {
    "mock_mode": False,
    "articles": [],
    "lookback_hours": 1,
    "platforms": ["x_twitter", "linkedin"],
}

print("Executing POST /api/v1/news/run against live running server...")
print("Payload:", json.dumps(payload, indent=2))

resp = client.post("/api/v1/news/run", json=payload)
print(f"HTTP Status: {resp.status_code}")

data = resp.json()
print("\n--- INGESTION METADATA ---")
print(json.dumps(data.get("ingestion_metadata"), indent=2))

print(f"\nRetained Articles Count: {len(data.get('articles', []))}")
for idx, a in enumerate(data.get("articles", [])[:5], 1):
    print(f"  {idx}. [{a.get('publisher')}] {a.get('title')}")
    print(f"     URL: {a.get('source_url')}")
    print(f"     Published: {a.get('published_at')}")

print(f"\nExtracted Claims Count: {len(data.get('claims', []))}")
print(f"Verification Results Count: {len(data.get('verification_results', []))}")
print(f"Brief Title: {data.get('brief', {}).get('title') if data.get('brief') else None}")
print(f"Errors Count: {len(data.get('errors', []))}")
if data.get("errors"):
    print("Errors:", json.dumps(data.get("errors"), indent=2))
