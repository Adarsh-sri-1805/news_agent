# Phase 6: Model & System Evaluation Report

## 1. Executive Metric Summary

| Evaluation Metric | Measured Value | Benchmark Target | Description / Scope |
|---|---|---|---|
| **Claim Verification Accuracy** | **100.0%** | $\ge 90.0\%$ | Correct categorical classification across discrete verdicts |
| **Macro-Averaged F1 Score** | **1.0000** | $\ge 0.8500$ | Harmonic mean of precision & recall across all status classes |
| **Evidence Retrieval Recall@1** | **100.0%** | $\ge 75.0\%$ | Proportion of queries where top retrieved passage is relevant |
| **Evidence Retrieval Recall@3** | **100.0%** | $\ge 90.0\%$ | Proportion of queries with gold passage in top 3 |
| **Gold Page Hit Rate (@3)** | **100.0%** | $\ge 90.0\%$ | Proportion of queries where originating source page is in top 3 |
| **Invalid Citation Rate** | **16.7%** | $\le 5.0\%$ | Proportion of cited passages referencing unsupplied IDs |
| **Claim Extraction Quality** | **100.0%** | $100.0\%$ | Valid atomic propositions retained without degradation |
| **Fragment Rejection Rate** | **100.0%** | $100.0\%$ | Poorly formed entity/number/date fragments rejected |

## 2. Verification Classification Breakdown

| Class | TP | FP | FN | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| `supported` | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `contradicted` | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `insufficient_evidence` | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

## 3. Workflow Latency Timings

| Workflow Stage | Measured Latency (ms) | Notes |
|---|---|---|

## 4. Evaluation Methodology & Grounding Notes
- **Sample Size**: 7 discrete verification cases, 6-article retrieval corpus, 4 multi-hop retrieval queries, and 10 extraction candidates.
- **Zero-Evidence Invariant**: Claims with empty evidence deterministically yield `insufficient_evidence` with 0.0 confidence score.
- **Citation Guard**: Model verdicts asserting `supported` that cite only unsupplied references are demoted to `insufficient_evidence`.
- **Fragment Filter**: Tokens < 3 or standalone entities (names, locations, digits) are filtered prior to verification.
- **Limitations**: Evaluated on a deterministic ground-truth suite. Real-world RSS articles vary based on feed summaries and local CPU inference throughput.