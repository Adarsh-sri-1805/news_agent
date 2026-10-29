# Local News Intelligence Agent

A backend-first intelligence pipeline designed to ingest local news, extract and verify factual claims, synthesize briefs, and generate draft distribution copy using **local LLM inference** via [Ollama](https://ollama.com).

> **Current Implementation State: Phase 7 (Final Integration, Minimal Frontend & Submission Complete)**  
> Full end-to-end integration featuring a local multi-agent intelligence pipeline, deterministic BM25 lexical corroboration engine, FastAPI REST API, stdio Model Context Protocol (MCP) server, reproducible evaluation suite, and a sleek, responsive React + Vite intelligence dashboard consuming backend REST endpoints with both live inference and offline demo modes.

---

## Architecture Intent & Constraints

- **Local Inference First**: Designed strictly for private, local execution using Ollama. No paid or external LLM APIs are permitted.
- **Hardware Profile**: Optimized for consumer hardware: **16 GB RAM, CPU-only execution**. The default model target is `llama3.2:3b`.
- **Atomic Factual Extraction**: Extracts verifiable assertions while explicitly rejecting opinions, speculation, and marketing hyperbole.
- **Deterministic Lexical Retrieval**: Uses BM25Plus indexing without vector databases or embedding models. Enforces originating article exclusion during corroboration searches.
- **Corroboration & Independence Safeguards**: Multiple passages from the same publisher are not counted as independent corroboration. Syndicated reprints are detected and deduplicated.
- **Separation of Verdict and Confidence**: Verification states are strictly categorized as `supported`, `contradicted`, or `insufficient_evidence`.
- **Defensive Structured Validation**: Raw model outputs and external feeds are never treated as trusted data. Hallucinated citations or unsupplied references are discarded deterministically in Python.

---

## Project Structure

```
news_agent/
├── .env.example              # Documented environment template (zero secrets)
├── .gitignore                # Git ignore rules for caches, venvs, and artifacts
├── pyproject.toml            # Packaging metadata, dependencies, and pytest configuration
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development and test dependencies
├── README.md                 # Project documentation and developer guide
├── EVALUATION.md             # Markdown export of deterministic evaluation results
├── evaluation_results.json   # Machine-readable evaluation benchmark metrics
├── run_api.py                # Convenience runner for FastAPI REST API server
├── run_demo.py               # Interactive CLI offline & live demonstration runner
├── run_evaluation.py         # Deterministic benchmark runner for Phase 6 evaluation
├── frontend/                 # React + Vite Intelligence Dashboard (Phase 7)
│   ├── package.json          # Frontend dependencies & scripts
│   ├── vite.config.js        # Vite dev proxy configuration (/api, /health, /ready -> :8000)
│   ├── index.html            # Clean HTML5 entrypoint with Google Fonts typography
│   └── src/
│       ├── App.jsx           # Main Intelligence Dashboard component
│       ├── main.jsx          # React DOM entrypoint
│       └── index.css         # Modern dark-mode glassmorphism design system
├── src/
│   └── news_agent/
│       ├── __init__.py       # Package entrypoint
│       ├── config.py         # Pydantic Settings configuration management
│       ├── api/              # FastAPI REST API Integration (Phase 5)
│       │   ├── __init__.py   # API exports
│       │   ├── __main__.py   # CLI entrypoint for python -m news_agent.api
│       │   ├── app.py        # FastAPI app factory and middleware configuration
│       │   ├── dependencies.py # Dependency injection providers
│       │   ├── mock_runner.py # Safe offline demonstration & mock workflow runner
│       │   ├── schemas.py    # Request and response Pydantic contracts
│       │   └── routes/       # API route modules
│       │       ├── __init__.py
│       │       ├── health.py # /health and /ready probe endpoints
│       │       └── news.py   # /api/v1/news/run execution endpoint
│       ├── models/           # Domain contracts and Pydantic schemas (Phase 1)
│       │   ├── __init__.py   # Model re-exports
│       │   ├── base.py       # UTC timestamp utilities & Base Pydantic configuration
│       │   ├── article.py    # NormalizedArticle schema
│       │   ├── claim.py      # FactualClaim schema
│       │   ├── evidence.py   # EvidencePassage schema
│       │   ├── verification.py # VerificationStatus enum & VerificationResult schema
│       │   ├── brief.py      # NewsBrief schema
│       │   └── social.py     # SocialMediaDraft schema
│       ├── ingestion/        # News ingestion & deduplication (Phase 2)
│       │   ├── __init__.py
│       │   ├── dedup.py      # URL canonicalization & title deduplication
│       │   ├── filters.py    # Lookback window filtering
│       │   ├── pipeline.py   # IngestionPipeline orchestrator & IngestionResult
│       │   └── adapters/     # Source adapters (RSS, Atom, GDELT)
│       ├── verification/     # Evidence & Verification Engine (Phase 3)
│       │   ├── __init__.py
│       │   ├── segmenter.py  # Passage segmentation with character offsets
│       │   ├── extractor.py  # Atomic claim extractor via Ollama
│       │   ├── retriever.py  # Deterministic lexical BM25 evidence retriever with coverage gating
│       │   ├── corroboration.py # Source independence & publisher diversity auditing
│       │   ├── verifier.py   # Evidence interpretation & discrete verdict generation
│       │   └── engine.py     # VerificationEngine orchestrator
│       ├── agents/           # Multi-Agent Architecture (Phase 4)
│       │   ├── __init__.py   # Agent re-exports
│       │   ├── researcher.py # NewsResearcherAgent (ingestion, filtering, dedup)
│       │   ├── fact_checker.py # FactCheckingAgent (claims, retrieval, verdicts)
│       │   ├── briefing.py   # NewsBriefingAgent (grounded executive briefs)
│       │   ├── social.py     # SocialContentAgent (platform character constraints)
│       │   └── orchestrator.py # NewsIntelligenceOrchestrator coordinator
│       ├── mcp/              # Model Context Protocol (MCP) Server (Phase 4)
│       │   ├── __init__.py   # MCP exports
│       │   ├── __main__.py   # Entrypoint for python -m news_agent.mcp
│       │   └── server.py     # FastMCP/MCPServer stdio server & tool handlers
│       └── llm/              # Local LLM integration (Phase 1)
│           ├── __init__.py   # LLM exports
│           ├── exceptions.py # Structured exception hierarchy
│           └── client.py     # Ollama client wrapper with structured output validation
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared test fixtures
    ├── test_api.py           # FastAPI REST API endpoints, readiness, and mock tests (Phase 5)
    ├── test_models.py        # Contract validation, edge cases, and boundary tests
    ├── test_llm_client.py    # Ollama client, network mock, and error-handling tests
    ├── test_ingestion_rss.py # RSS/Atom parsing, error handling, and date parsing
    ├── test_ingestion_dedup.py # Canonicalization, exact dedup, and distinct article preservation
    ├── test_ingestion_pipeline.py # Filtering, GDELT adapter, and IngestionPipeline
    ├── test_verification_segmenter.py # Passage segmentation and offset tests
    ├── test_verification_extractor.py # Atomic claim extraction and failure recovery
    ├── test_verification_retriever.py # BM25 retrieval, ranking, and origin exclusion
    ├── test_verification_corroboration.py # Multi-source auditing and syndication detection
    ├── test_verification_verifier.py  # Supported, contradicted, insufficient evidence tests
    ├── test_verification_engine.py    # End-to-end verification pipeline test
    ├── test_agent_researcher.py # NewsResearcherAgent tests
    ├── test_agent_fact_checker.py # FactCheckingAgent tests
    ├── test_agent_briefing.py # NewsBriefingAgent tests
    ├── test_agent_social.py   # SocialContentAgent tests
    ├── test_agent_orchestrator.py # Multi-agent orchestrator data flow tests
    ├── test_mcp_server.py    # Official MCP server tool registration & execution tests
    └── test_live_smoke.py    # Optional live-source internet smoke test (-m live)
```

---

## Domain Contracts (Phase 1)

| Contract | Purpose | Key Attributes |
| :--- | :--- | :--- |
| `NormalizedArticle` | Canonical representation of ingested news articles | `id`, `title`, `content`, `source_url`, `publisher`, `published_at`, `retrieved_at`, `raw_metadata` |
| `FactualClaim` | Atomic factual assertions extracted from articles | `id`, `article_id`, `claim_text`, `context`, `extracted_at`, `extraction_confidence` |
| `EvidencePassage` | Snippets retrieved to evaluate claims | `id`, `passage_text`, `source_url`, `publisher`, `retrieved_at`, `relevance_score` |
| `VerificationResult` | Verdict outcome of fact-checking a claim | `claim_id`, `status` (`supported`, `contradicted`, `insufficient_evidence`), `confidence_score`, `reasoning`, `evidence_passages` |
| `NewsBrief` | Synthesized multi-source executive brief | `id`, `title`, `headline_takeaways`, `synthesized_body`, `source_article_ids`, `verification_summary` |
| `SocialMediaDraft` | Channel-tailored post draft | `id`, `platform`, `content`, `character_count`, `source_article_ids`, `hashtags` |

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally

### 2. Environment Setup
Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 3. Configure Environment
Copy the example configuration:

```bash
cp .env.example .env
```

Configuration parameters (`src/news_agent/config.py`):
- `OLLAMA_BASE_URL`: Base URL of local Ollama server (default: `http://localhost:11434`).
- `OLLAMA_MODEL`: Target model (default: `llama3.2:3b`).
- `OLLAMA_TIMEOUT_SECONDS`: Request timeout in seconds (default: `90.0`).
- `OLLAMA_KEEP_ALIVE`: Model in-memory persistence duration (default: `5m`).

### 4. Pull Local Model (Manual Step)
Automatic model downloads are disabled by design. Pull your desired model manually:

```bash
ollama pull llama3.2:3b
```

---

## Running Tests

All unit tests use mock transports for network calls and require no active Ollama server to run:

```bash
# Run entire test suite
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with test coverage report
python -m pytest --cov=news_agent
```

---

## Phase 4: Multi-Agent Architecture & MCP Server

### Multi-Agent Coordination
Phase 4 introduces four modular specialist agents coordinated by `NewsIntelligenceOrchestrator`:
1. **News Researcher Agent (`NewsResearcherAgent`)**: Interfaces with the ingestion pipeline to fetch, lookback-filter, and deduplicate news. Tracks source-level errors without masking failures.
2. **Fact-Checking Agent (`FactCheckingAgent`)**: Extracts atomic verifiable claims, queries the lexical BM25 retrieval index, audits source independence, and verifies claims against evidence. Retains statuses `supported`, `contradicted`, and `insufficient_evidence`.
3. **News Briefing Agent (`NewsBriefingAgent`)**: Produces executive intelligence briefs with clear distinction between corroborated facts and unverified single-source reports. Enforces source provenance at the claim level.
4. **Social Content Agent (`SocialContentAgent`)**: Drafts channel-tailored copy for X (`x_twitter`, 280-char limit) and LinkedIn (3000-char limit). Uses deterministic Python validation to enforce length constraints and prevents contradicted or unsupported assertions from entering factual promotional copy. Generation only; zero automated posting.

### Evidence Quality Gate & Retrieval Precision
- **Query Term Coverage Gating**: Fixed lexical retrieval defect where passages matching only generic incidental tokens (e.g. "contract") were returned for specific numeric or valuation claims (e.g. "$2.5 million"). `EvidenceRetriever` now enforces a minimum query term coverage threshold (`min_query_coverage=0.35`).
- **Truth Gating**: Retrieval relevance scores represent lexical term overlap, never truth probabilities. Factual copy only asserts claims with `supported` verdicts; all other claims are omitted or explicitly qualified as unverified single-source reports.

### Launching the MCP Server
The project includes a Model Context Protocol (MCP) server built with the official Python MCP SDK (`mcp>=2.0.0`) communicating via standard I/O (`stdio`):

```bash
# Launch the MCP server locally over stdio:
python -m news_agent.mcp
```
*(All diagnostic logs and startup messages are sent strictly to `stderr` so `stdout` remains clean for MCP JSON-RPC protocol traffic).*

#### MCP Tools Exposed
| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `search_news` | `lookback_hours: int = 24` | Ingests and deduplicates recent news articles from configured sources. |
| `get_article` | `article_id: str` | Retrieves a cached article by ID. |
| `extract_claims` | `article_id: str | None`, `article_text: str | None` | Extracts atomic factual claims from an article in corpus or supplied text. |
| `retrieve_evidence` | `claim_text: str`, `top_k: int = 3`, `originating_article_id: str | None` | Retrieves ranked lexical evidence passages. |
| `verify_claim` | `claim_text: str`, `evidence_passage_texts: list[str] | None` | Evaluates a claim against evidence passages (`supported`, `contradicted`, `insufficient_evidence`). |
| `generate_news_brief` | `article_ids: list[str] | None` | Synthesizes an evidence-cited executive brief. |
| `generate_social_drafts`| `brief_id: str | None`, `platforms: list[str] | None` | Generates character-validated promotional copy (X, LinkedIn). |
| `run_news_intelligence` | `lookback_hours: int = 24`, `platforms: list[str] | None` | Executes complete coordinated intelligence workflow end-to-end. |

#### Connecting an MCP Client (e.g. Claude Desktop)
Add the server configuration to your MCP client config file (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "local-news-intelligence": {
      "command": "python",
      "args": ["-m", "news_agent.mcp"],
      "cwd": "/path/to/news_agent"
    }
  }
}
```
*(Note: As per project guidelines, external client integration has not been claimed unless physically connected; the server has been rigorously tested using in-process async protocol test fixtures).*

---

## Phase 5: FastAPI REST API Integration

Phase 5 exposes the multi-agent intelligence pipeline through a clean, asynchronous REST API powered by **FastAPI** and **Uvicorn**, adhering to the domain contracts established in Phases 1–4.

### 1. Installation & Environment

Install the API dependencies:

```bash
pip install -r requirements.txt
# or for development & testing:
pip install -r requirements-dev.txt
```

Key environment configuration variables (`.env`):
- `OLLAMA_BASE_URL`: Base URL for local Ollama daemon (default: `http://localhost:11434`).
- `OLLAMA_MODEL`: Target local model name (default: `llama3.2:3b`).
- `INGESTION_LOOKBACK_HOURS`: Default article lookback window in hours (default: `24`).
- `LOG_LEVEL`: Logging verbosity (default: `INFO`).

### 2. Launching the API Server

Start the server using either the convenience root runner or `uvicorn`:

```bash
# Option A: Convenience runner (recommended)
python run_api.py

# Option B: Direct uvicorn execution
python -m uvicorn news_agent.api.app:app --app-dir src --host 127.0.0.1 --port 8000

# Option C: With auto-reload for local development
python run_api.py --reload
```

To stop the server at any time, press `Ctrl + C` in the terminal.

### 3. Interactive Documentation (Swagger UI)

Once the server is running, navigate in your local web browser to:
- **Swagger UI (Interactive API documentation & testing)**: `http://127.0.0.1:8000/docs`
- **ReDoc UI**: `http://127.0.0.1:8000/redoc`
- **OpenAPI Schema (JSON)**: `http://127.0.0.1:8000/openapi.json`

---

### 4. REST Endpoints

#### `GET /health`
Liveness probe. Returns immediate service status. Does not execute model inference or news ingestion.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:8000/health
```

**Example Response (HTTP 200):**
```json
{
  "status": "healthy",
  "app": "local-news-intelligence-agent",
  "version": "0.1.0",
  "timestamp": "2026-10-28T15:06:30.341481Z"
}
```

---

#### `GET /ready`
Readiness probe. Checks external and internal dependencies without crashing the API process:
1. Checks Ollama daemon connectivity (`/api/version`).
2. Checks whether the configured model (`llama3.2:3b`) is installed locally (`/api/tags`).
3. Checks configured news ingestion feeds.

*Note: Automatic model downloads are strictly disabled.* If Ollama is offline or the model is missing, returns HTTP 503 with helpful remediation instructions.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:8000/ready
```

**Example Response (when Ollama is offline / HTTP 503):**
```json
{
  "status": "not_ready",
  "ready": false,
  "dependencies": {
    "ollama": {
      "status": "unavailable",
      "details": {
        "base_url": "http://localhost:11434",
        "model": "llama3.2:3b",
        "instruction": "Ensure the Ollama daemon is running locally."
      }
    },
    "ingestion_sources": {
      "status": "ready",
      "details": {
        "configured_rss_feeds": 2,
        "gdelt_enabled": false
      }
    }
  },
  "timestamp": "2026-10-28T15:06:35.136287Z"
}
```

---

#### `POST /api/v1/news/run`
Coordinates the end-to-end intelligence workflow across `NewsResearcherAgent`, `FactCheckingAgent`, `NewsBriefingAgent`, and `SocialContentAgent`.

**Request Parameters:**
- `lookback_hours` *(optional, int, 1–168)*: Freshness window for news collection (default: 24).
- `platforms` *(optional, list[str])*: Target platforms for copy generation (`['x_twitter', 'linkedin']`).
- `mock_mode` *(optional, bool)*: Safe offline demo mode enabling complete testing without requiring Ollama or live internet feeds.
- `articles` *(optional, list[NormalizedArticle])*: Pre-supplied article payload for direct verification and synthesis.

##### Safe Mock Mode (No Ollama or Internet Required)
Ideal for testing via Swagger UI immediately:

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/news/run \
  -H "Content-Type: application/json" \
  -d '{
    "mock_mode": true,
    "lookback_hours": 24,
    "platforms": ["x_twitter", "linkedin"]
  }'
```

**Response (HTTP 200):**
```json
{
  "ingestion_metadata": {
    "raw_collected_count": 2,
    "retained_count": 2,
    "exact_url_duplicates_removed": 0,
    "near_duplicate_titles_removed": 0,
    "source_errors_count": 0
  },
  "articles": [
    {
      "id": "art-gazette-solar-01",
      "title": "Springfield City Council Approves $2.5M Solar Panel Installation",
      "content": "The Springfield city council voted 7-2 on Tuesday morning...",
      "source_url": "https://springfieldgazette.com/articles/solar-vote",
      "publisher": "Springfield Gazette",
      "published_at": "2026-10-28T15:00:00Z"
    }
  ],
  "claims": [
    {
      "id": "c1",
      "article_id": "art-gazette-solar-01",
      "claim_text": "The Springfield city council voted 7-2 to approve a $2.5 million solar installation.",
      "extraction_confidence": 0.96
    }
  ],
  "verification_results": [
    {
      "claim_id": "c1",
      "status": "supported",
      "confidence_score": 0.95,
      "reasoning": "The evidence passage from Civic Records Observer directly confirms the 7-2 vote margin.",
      "evidence_passages": [
        {
          "id": "art-observer-records-02_p0",
          "publisher": "Civic Records Observer",
          "relevance_score": 0.94
        }
      ]
    }
  ],
  "brief": {
    "title": "Springfield Municipal Solar Installation Approved",
    "headline_takeaways": [
      "Springfield city council passed a $2.5M solar contract by a 7-2 vote.",
      "Midwest Solar Dynamics will perform rooftop installations starting in the spring."
    ],
    "verification_summary": {
      "supported": 1,
      "insufficient_evidence": 1
    }
  },
  "social_drafts": [
    {
      "platform": "x_twitter",
      "content": "Springfield City Council Approves $2.5M Solar Panel Installation\n\nDetails: https://springfieldgazette.com/articles/solar-vote\n#LocalNews #Springfield",
      "character_count": 149,
      "hashtags": ["LocalNews", "Springfield"]
    },
    {
      "platform": "linkedin",
      "content": "📰 Executive News Brief: Springfield City Council Approves $2.5M Solar Panel Installation...",
      "character_count": 549,
      "hashtags": ["LocalNews", "Springfield"]
    }
  ],
  "errors": [],
  "completed_at": "2026-10-28T15:06:39.123456Z"
}
```

##### Live Inference Mode (With Ollama & Live Feeds)
To run live news intelligence against configured RSS feeds and local Ollama:
1. Ensure Ollama is running: `ollama serve`
2. Verify model is installed: `ollama list` (or `ollama pull llama3.2:3b`)
3. Execute request with `mock_mode: false` (use `max_articles: 3` for responsive local CPU execution):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/news/run \
  -H "Content-Type: application/json" \
  -d '{
    "lookback_hours": 24,
    "max_articles": 3,
    "platforms": ["x_twitter", "linkedin"]
  }'
```

---

## Phase 6: Evaluation & Hardening Suite

Phase 6 introduces a fully reproducible, deterministic evaluation suite to measure factual reliability, retrieval precision, citation validity, and latency bounds under local CPU constraints.

### 1. Evaluation Methodology & Metric Formulas

The suite benchmarks the agent across four core tasks using a deterministic ground truth dataset:

1. **Claim Verification Accuracy**:
   $$\text{Accuracy} = \frac{\sum_{c} TP_c}{N_{\text{total}}}$$
2. **Precision, Recall, and F1 (Per Class & Macro-Averaged)**:
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   $$\text{Macro } F1 = \frac{1}{|C|} \sum_{c \in C} F1_c$$
3. **Evidence Retrieval Recall@k**:
   $$\text{Recall@k} = \frac{\text{queries where at least one gold passage is retrieved in top } k}{\text{total queries}}$$
4. **Gold Page Hit Rate (@k)**:
   $$\text{Gold Page Hit Rate} = \frac{\text{queries where originating article is in top } k}{\text{total queries}}$$
5. **Invalid Citation Rate**:
   $$\text{Invalid Citation Rate} = \frac{\text{cited passage IDs not matching supplied evidence}}{\text{total citations}}$$
6. **Claim Extraction Quality & Fragment Rejection**:
   $$\text{Quality Rate} = \frac{\text{valid propositional claims retained}}{\text{total valid propositional claims}}, \quad \text{Rejection Rate} = \frac{\text{fragments filtered}}{\text{total fragments}}$$

---

### 2. Measured Benchmark Results

Generated by running `python run_evaluation.py --no-latency` (persisted to [`evaluation_results.json`](file:///c:/Users/adars/OneDrive/Desktop/news_agent/evaluation_results.json) and [`EVALUATION.md`](file:///c:/Users/adars/OneDrive/Desktop/news_agent/EVALUATION.md)):

| Evaluation Metric | Measured Value | Benchmark Target | Status | Description / Scope |
|---|---|---|---|---|
| **Claim Verification Accuracy** | **100.0%** | $\ge 90.0\%$ | **PASS** | Correct classification across discrete verdicts |
| **Macro-Averaged F1 Score** | **1.0000** | $\ge 0.8500$ | **PASS** | Harmonic mean across `supported`, `contradicted`, `insufficient` |
| **Evidence Retrieval Recall@1** | **100.0%** | $\ge 75.0\%$ | **PASS** | Top retrieved passage is relevant |
| **Evidence Retrieval Recall@3** | **100.0%** | $\ge 90.0\%$ | **PASS** | Gold evidence passage present in top 3 |
| **Gold Page Hit Rate (@3)** | **100.0%** | $\ge 90.0\%$ | **PASS** | Originating publisher/article present in top 3 |
| **Invalid Citation Guard** | **100.0%** | $100.0\%$ | **PASS** | Hallucinated citations detected, discarded, and verdict demoted |
| **Claim Extraction Quality** | **100.0%** | $100.0\%$ | **PASS** | 100% of valid propositions retained |
| **Fragment Rejection Rate** | **100.0%** | $100.0\%$ | **PASS** | Standalone names, numbers, dates, and noun fragments filtered |

#### Detailed Verification Classification Breakdown
| Class | True Positives | False Positives | False Negatives | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|
| `supported` | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `contradicted` | 2 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `insufficient_evidence` | 3 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |

---

### 3. Critical Hardening Safeguards

1. **Zero-Evidence Invariant**: Missing or empty evidence immediately yields `insufficient_evidence` with `confidence_score=0.0`. Under no circumstances can a claim be classified as `supported` without evidence.
2. **Citation Integrity Audit**: The verifier cross-references every model-cited passage identifier against supplied passages. If a model outputs `supported` but cites only hallucinated or unsupplied references, the verdict is demoted to `insufficient_evidence`.
3. **Fragment Filtering**: Extracted claims under 10 characters, fewer than 3 tokens, or consisting of single entities/dates (e.g., `"Fernando Seco"`, `"Toledo"`, `"68"`, `"October 28, 2026"`) are deterministically rejected before verification.
4. **Empty-Input Truth Guard**: When zero articles are ingested or retained, briefs and social drafts output clear notices stating zero news articles or claims were processed, never falsely claiming verified intelligence.
5. **Stage Latency Telemetry**: Ingestion metadata records stage-level execution times (`research`, `fact_checking`, `briefing`, `social_drafting`, `total`) and logs completion times to diagnose bottlenecks.

---

### 4. Exact Demonstration & Testing Commands

#### Run Full Regression Test Suite
```bash
pytest -v
# Output: 132 passed, 1 deselected in ~120s
```

#### Run Deterministic Evaluation Suite
```bash
python run_evaluation.py --no-latency
# Saves machine-readable evaluation_results.json and EVALUATION.md
```

#### Run Interactive Mock Demo
```bash
python run_demo.py --mock
```

#### Launch REST API Server
```bash
python run_api.py --host 127.0.0.1 --port 8000
# Interactive Swagger UI available at http://127.0.0.1:8000/docs
```

---

## Phase 7: Frontend Intelligence Dashboard & Full Integration

Phase 7 delivers a modern, reactive web dashboard built with **React** and **Vite** that consumes the existing FastAPI backend (`http://127.0.0.1:8000`). It introduces no business logic duplication and zero paid API dependencies.

### 1. Dashboard Capabilities & Architecture

- **Live Backend Readiness Probe**: Interrogates `GET /ready` to display live Ollama connection state, installed model tag, and RSS feed availability.
- **Workflow Control & Execution**: Provides a single-click workflow execution trigger with selectable `lookback_hours` (1–72h) and `max_articles` (1–10, default: 3).
- **Offline Demo / Live Toggle**:
  - `⚡ OFFLINE DEMO MODE` (Default): Executes the deterministic mock pipeline instantaneously, demonstrating the complete verified intelligence output without requiring local Ollama downloads or risking feed network timeouts.
  - `🌐 LIVE PIPELINE`: Executes live multi-agent RSS ingestion, lexical BM25 retrieval, Ollama claim extraction, verification, briefing, and social drafting with honest stage progress.
- **Executive Intelligence Briefing**: Displays synthesized narrative summaries, headline key takeaways, and quantitative verdict distribution badges.
- **Factual Claim Verification Cards**: Clearly displays discrete verdict pills (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT EVIDENCE`), confidence percentages, explanatory reasoning, and clickable evidence passage citations. *Claims with insufficient evidence are never labeled as verified.*
- **Platform Distribution Drafts**: Previews channel-tailored copy for LinkedIn (3000-char limit) and X / Twitter (280-char limit) with live character counter badges and hashtag chips.
- **Ingested News Feed**: Displays publisher badges, publication timestamps, and direct clickable source links.
- **Telemetry Ribbon**: Shows real-time metrics for raw collected articles, retained articles, claims evaluated, verified facts, and total pipeline execution time in milliseconds.
- **Error & Loading States**: Transparent multi-stage loading animation and accessible error alerts with immediate retry triggers.

---

### 2. Frontend Setup & Launch Instructions

#### Step 1: Start the Backend REST API
In terminal 1:
```bash
python run_api.py --host 127.0.0.1 --port 8000
```
- **Backend API**: `http://127.0.0.1:8000`
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **Readiness Probe**: `http://127.0.0.1:8000/ready`

#### Step 2: Start the Frontend Development Server
In terminal 2:
```bash
cd frontend
npm install
npm run dev
```
- **Frontend Dashboard**: `http://localhost:5173/`

#### Step 3: Production Build (Optional)
To create an optimized production bundle:
```bash
cd frontend
npm run build
```

---

### 3. Model Context Protocol (MCP) Server Launch

The project includes a standard I/O (stdio) MCP server compatible with any MCP client (Claude Desktop, cursor, or command-line MCP inspector):

```bash
# Launch MCP server over stdio
python -m news_agent.mcp
```

#### Available MCP Tools:
| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `search_news` | `lookback_hours: int = 24` | Ingests and deduplicates recent news articles from configured sources. |
| `get_article` | `article_id: str` | Retrieves a cached article by ID. |
| `extract_claims` | `article_id: str \| None`, `article_text: str \| None` | Extracts atomic factual claims from an article in corpus or supplied text. |
| `retrieve_evidence` | `claim_text: str`, `top_k: int = 3`, `originating_article_id: str \| None` | Retrieves ranked lexical BM25 evidence passages. |
| `verify_claim` | `claim_text: str`, `evidence_passage_texts: list[str] \| None` | Evaluates a claim against evidence passages (`supported`, `contradicted`, `insufficient_evidence`). |
| `generate_news_brief` | `article_ids: list[str] \| None` | Synthesizes an evidence-cited executive brief. |
| `generate_social_drafts`| `brief_id: str \| None`, `platforms: list[str] \| None` | Generates character-validated promotional copy (X, LinkedIn). |
| `run_news_intelligence` | `lookback_hours: int = 24`, `platforms: list[str] \| None` | Executes complete coordinated intelligence workflow end-to-end. |

---

### 4. Verification & Submission Checklist

| Component | Target Verification | Measured Result |
| :--- | :--- | :--- |
| **Unit & Integration Tests** | `pytest -v` | **132 passed**, 1 deselected live test (0 failures) |
| **Evaluation Suite** | `python run_evaluation.py --no-latency` | **100% Verification Accuracy, 1.0 Macro F1, 100% Recall@3** |
| **CLI Demo** | `python run_demo.py --mock` | **Pass** (Clean end-to-end workflow execution) |
| **FastAPI REST Server** | `GET /health`, `GET /ready`, `POST /api/v1/news/run` | **HTTP 200 OK** |
| **Frontend Web App** | `http://localhost:5173/` | **Pass** (Vite build passed, interactive UI tested in browser) |
| **Zero Paid Dependencies** | Local Ollama + BM25 only | **Verified** (Zero external paid APIs, zero vector DBs) |

---

### 5. Hardware Profile & Local Inference Constraints

- **Hardware Target**: 16 GB RAM, CPU-only execution.
- **Model Target**: Local Ollama model (`llama3.2:3b` or `qwen3:4b`). No cloud or paid APIs are used.
- **CPU Throughput**: Ollama inference on CPU requires 10–25 seconds per LLM call.

---

### 6. Known Limitations

1. **Sequential CPU Inference Latency on Large Feeds**: Ingesting a full 24-hour window from major RSS feeds typically retains 60+ articles. Processing 60 articles sequentially through CPU Ollama inference requires 10–15+ minutes, which will exceed typical HTTP client and browser request timeouts.
   * **Mitigation**: The API and frontend provide a `max_articles` constraint (default: 3). Setting `max_articles: 3` limits downstream LLM processing to the top 3 freshest retained articles while keeping full ingestion counts visible in `ingestion_metadata`.
2. **Deterministic Offline Demo Mode**: When Ollama is offline or testing under time pressure, use `mock_mode: true` (or the `⚡ OFFLINE DEMO MODE` toggle in the frontend) for an instantaneous, reliable demonstration of all platform features.
3. **RSS Summary-Only Content**: Certain public RSS feeds (e.g. wire services) provide brief abstracts rather than full-length article bodies. If an abstract contains fewer than 30 characters, the extractor skips it to prevent hallucinated assertions.
4. **External Corroboration Scope**: An assertion can only be verified or contradicted if another independent publisher in the ingested corpus covers the exact same event. Uncorroborated single-source claims remain categorized as `insufficient_evidence` by design.



