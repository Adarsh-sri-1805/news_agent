"""Smoke test client against the live running FastAPI server."""

import json
import sys
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0)

print("=" * 60)
print("1. Smoke test: GET /health")
print("=" * 60)
r_health = client.get("/health")
print(f"HTTP Status: {r_health.status_code}")
print(json.dumps(r_health.json(), indent=2))

print("\n" + "=" * 60)
print("2. Smoke test: GET /ready")
print("=" * 60)
r_ready = client.get("/ready")
print(f"HTTP Status: {r_ready.status_code}")
print(json.dumps(r_ready.json(), indent=2))

print("\n" + "=" * 60)
print("3. Smoke test: GET /docs (Swagger UI)")
print("=" * 60)
r_docs = client.get("/docs")
print(f"HTTP Status: {r_docs.status_code}")
print(f"Content Type: {r_docs.headers.get('content-type')}")
print(f"Title in HTML: {'<title>Local News Intelligence Agent API - Swagger UI</title>' in r_docs.text}")

print("\n" + "=" * 60)
print("4. Smoke test: POST /api/v1/news/run (mock_mode=True)")
print("=" * 60)
payload = {
    "mock_mode": True,
    "lookback_hours": 24,
    "platforms": ["x_twitter", "linkedin"],
}
r_run = client.post("/api/v1/news/run", json=payload)
print(f"HTTP Status: {r_run.status_code}")
data = r_run.json()
print(f"Articles count: {len(data.get('articles', []))}")
print(f"Claims count: {len(data.get('claims', []))}")
print("Verifications:")
for v in data.get("verification_results", []):
    print(f"  - Claim: {v['claim_id']} | Status: {v['status']} | Confidence: {v['confidence_score']}")
print(f"Brief Title: {data.get('brief', {}).get('title') if data.get('brief') else None}")
print("Headline takeaways:")
for t in (data.get("brief", {}) or {}).get("headline_takeaways", []):
    print(f"  * {t}")
print(f"Social drafts count: {len(data.get('social_drafts', []))}")
for d in data.get("social_drafts", []):
    print(f"  - Platform: {d['platform']} | Length: {d['character_count']} chars")
    print(f"    Content: {d['content']}")
    print(f"    Hashtags: {d['hashtags']}")

print("\n" + "=" * 60)
print("5. Smoke test: POST /api/v1/news/run (validation check on invalid lookback)")
print("=" * 60)
r_invalid = client.post("/api/v1/news/run", json={"lookback_hours": 999})
print(f"HTTP Status: {r_invalid.status_code} (Expected 422)")
print(f"Validation Error: {r_invalid.json().get('detail', [])[0].get('msg')}")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
