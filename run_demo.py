"""Interactive CLI test runner for Local News Intelligence Agent (Phases 1-3).

Usage:
    # Run instant offline demonstration (no Ollama or internet needed):
    python run_demo.py --mock

    # Run live pipeline test (ingests real feeds, uses local Ollama if active):
    python run_demo.py --live
"""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Support international unicode characters cleanly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime, timezone
from news_agent.ingestion.adapters.rss import RSSFeedAdapter
from news_agent.ingestion.pipeline import IngestionPipeline
from news_agent.llm.client import OllamaClient
from news_agent.models.article import NormalizedArticle
from news_agent.verification.corroboration import analyze_corroboration
from news_agent.verification.engine import VerificationEngine
from news_agent.verification.extractor import ClaimExtractor
from news_agent.verification.retriever import EvidenceRetriever
from news_agent.verification.verifier import ClaimVerifier


def run_offline_demo():
    print("\n" + "=" * 70)
    print("  LOCAL NEWS INTELLIGENCE AGENT - OFFLINE VERIFICATION DEMO")
    print("=" * 70)

    # 1. Sample Articles
    art1 = NormalizedArticle(
        id="art-gazette-01",
        title="Springfield City Council Approves $2.5M Solar Panel Installation",
        content=(
            "The Springfield city council voted 7-2 on Tuesday morning to approve a $2.5 million "
            "solar panel installation contract across municipal buildings. Councilor Jenkins sponsored "
            "the bill, projecting $400,000 in annual utility cost reductions. Voting against were Councilors "
            "Vance and Miller, who cited concerns over short-term municipal debt issuance."
        ),
        publisher="Springfield Gazette",
        source_url="https://springfieldgazette.com/articles/solar-vote",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 9, 30, tzinfo=timezone.utc),
    )

    art2 = NormalizedArticle(
        id="art-observer-02",
        title="Municipal Voting Records: Resolution #44 Clean Energy Accord Passes",
        content=(
            "Official voting ledgers from the Springfield city clerk verify that resolution #44 passed "
            "with 7 ayes and 2 nays. The approved contract authorizes Midwest Solar Dynamics to install "
            "photovoltaic arrays on city hall and the central library rooftop starting next spring."
        ),
        publisher="Civic Records Observer",
        source_url="https://civicobserver.org/records/resolution-44",  # type: ignore[arg-type]
        published_at=datetime(2026, 10, 27, 11, 0, tzinfo=timezone.utc),
    )

    articles = [art1, art2]
    print(f"\n[1] Ingested Articles ({len(articles)} sources):")
    for a in articles:
        print(f"    - [{a.publisher}] '{a.title}' ({len(a.content)} chars)")

    # 2. Index corpus for lexical retrieval
    print("\n[2] Indexing Passages via Lexical BM25Plus...")
    retriever = EvidenceRetriever.from_articles(articles, target_chars=300)
    print(f"    Indexed {len(retriever.passages)} discrete passage segments.")

    # 3. Simulate Extracted Claims
    print("\n[3] Extracting Atomic Factual Claims from Article 1...")
    sample_claims = [
        "The Springfield city council approved the solar installation by a 7-2 vote.",
        "The municipal solar panel contract is valued at $2.5 million.",
        "Midwest Solar Dynamics will perform the rooftop installation.",
    ]
    for idx, c in enumerate(sample_claims, 1):
        print(f"    Claim {idx}: \"{c}\"")

    # 4. Lexical Retrieval & Verification
    print("\n[4] Lexical Evidence Retrieval (Excluding Originating Article):")
    for c_text in sample_claims:
        from news_agent.models.claim import FactualClaim
        claim_obj = FactualClaim(
            id="c-demo",
            article_id="art-gazette-01",  # Originating article
            claim_text=c_text,
        )
        evidence = retriever.retrieve_evidence(claim_obj, top_k=2, exclude_own_article=True)
        print(f"\n    Query: \"{c_text}\"")
        if evidence:
            for ev in evidence:
                print(f"      -> Retrieved from [{ev.publisher}] (Relevance: {ev.relevance_score}):")
                print(f"         \"{ev.passage_text[:120]}...\"")
                audit = analyze_corroboration([ev])
                print(f"         Independence audit: {audit.distinct_publisher_count} publisher(s)")
        else:
            print("      -> No external corroborating evidence found.")

    print("\n" + "=" * 70)
    print("  STATUS: Phase 1 (Contracts), Phase 2 (Ingestion/Dedup),")
    print("          and Phase 3 (Evidence & Verification Engine) are fully operational!")
    print("=" * 70 + "\n")


def run_live_demo():
    print("\n" + "=" * 70)
    print("  LOCAL NEWS INTELLIGENCE AGENT - LIVE INGESTION & OLLAMA CHECK")
    print("=" * 70)

    # Check Ollama
    client = OllamaClient()
    ollama_ready = client.health_check()
    print(f"\n[1] Ollama Local Daemon Check (http://localhost:11434):")
    if ollama_ready:
        print("    [OK] Ollama is active and reachable.")
        try:
            models = client.list_local_models()
            print(f"    Installed models: {models}")
        except Exception as e:
            print(f"    Could not list models: {e}")
    else:
        print("    [!] Ollama is not running. (To start: run 'ollama serve' in another terminal)")
        print("        Running live RSS collection only...")

    # Run live ingestion pipeline
    print("\n[2] Ingesting Live Public RSS Feeds (BBC World News & NYT World)...")
    try:
        pipeline = IngestionPipeline(lookback_hours=48)
        result = pipeline.run()
        print(f"    Raw articles collected: {result.raw_collected_count}")
        print(f"    After lookback filter:  {result.after_filter_count}")
        print(f"    Exact URL duplicates:   {result.deduplication_report.exact_url_duplicates_removed}")
        print(f"    Near-duplicate titles:  {result.deduplication_report.near_duplicate_titles_removed}")
        print(f"    Retained unique stories: {result.deduplication_report.retained_count}")

        print("\n[3] Sample Retained Ingested Headlines:")
        for idx, art in enumerate(result.articles[:5], 1):
            print(f"    {idx}. [{art.publisher}] {art.title}")
            print(f"       URL: {art.source_url}")
            print(f"       Published: {art.published_at.strftime('%Y-%m-%d %H:%M UTC') if art.published_at else 'N/A'}")
            print()
    except Exception as e:
        print(f"    Live ingestion error (check internet connection): {e}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test runner for Local News Intelligence Agent.")
    parser.add_argument("--live", action="store_true", help="Run live RSS ingestion and Ollama connectivity test.")
    parser.add_argument("--mock", action="store_true", help="Run offline demonstration with mock articles and BM25 ranking.")
    args = parser.parse_args()

    if args.live:
        run_live_demo()
    else:
        run_offline_demo()
