"""CLI runner for Phase 6 evaluation suite.

Usage:
    python run_evaluation.py
    python run_evaluation.py --json-out evaluation_results.json --md-out EVALUATION.md
"""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from news_agent.evaluation.runner import EvaluationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Evaluation Suite Runner")
    parser.add_argument("--json-out", default="evaluation_results.json", help="Path to save JSON metrics")
    parser.add_argument("--md-out", default="EVALUATION.md", help="Path to save Markdown summary")
    parser.add_argument("--live", action="store_true", help="Use live Ollama LLM instead of deterministic suite")
    parser.add_argument("--no-latency", action="store_true", help="Skip orchestrator latency benchmarking")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LOCAL NEWS INTELLIGENCE AGENT - PHASE 6 EVALUATION SUITE")
    print("=" * 70 + "\n")

    runner = EvaluationRunner(use_live_llm=args.live)
    print("[*] Running deterministic evaluation benchmarks...")
    metrics = runner.run_all(measure_latency=not args.no_latency)

    # Export machine-readable JSON
    json_path = Path(args.json_out)
    runner.export_json(metrics, json_path)
    print(f"[+] Saved machine-readable metrics: {json_path}")

    # Export markdown report
    md_report = runner.format_markdown_summary(metrics)
    if args.md_out:
        md_path = Path(args.md_out)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"[+] Saved Markdown summary report: {md_path}")

    print("\n" + md_report + "\n")

    # Benchmark assertions
    v = metrics.verification
    r = metrics.retrieval
    c = metrics.citations
    e = metrics.extraction

    passed = True
    if v.accuracy < 0.90:
        print(f"[!] Warning: Verification accuracy {v.accuracy:.2%} below 90% threshold.")
        passed = False
    if r.recall_at_3 < 0.90:
        print(f"[!] Warning: Retrieval Recall@3 {r.recall_at_3:.2%} below 90% threshold.")
        passed = False
    if e.fragment_rejection_rate < 1.0:
        print(f"[!] Warning: Fragment rejection rate {e.fragment_rejection_rate:.2%} below 100%.")
        passed = False

    if passed:
        print("[SUCCESS] All Phase 6 evaluation benchmarks achieved target thresholds.\n")
    else:
        print("[FAILURE] Some benchmarks did not meet target thresholds.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
