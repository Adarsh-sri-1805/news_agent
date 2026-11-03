import React, { useState, useEffect } from 'react';

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// Deterministic client-side demonstration data generator
// Ensures the Netlify-hosted dashboard works 100% out of the box even when disconnected from localhost
function generateClientMockResult(lookbackHours = 24) {
  const now = new Date();
  return {
    ingestion_metadata: {
      raw_collected_count: 94,
      after_filter_count: 48,
      retained_count: 48,
      exact_url_duplicates_removed: 2,
      near_duplicate_titles_removed: 1,
      source_errors_count: 0,
      mode: 'client_offline_demo',
      stage_timings_ms: {
        research: 320.5,
        fact_checking: 410.2,
        briefing: 195.4,
        social_drafting: 85.1,
        total: 1011.2,
      },
    },
    articles: [
      {
        id: 'art-gazette-solar-01',
        title: 'Springfield City Council Approves $2.5M Solar Panel Installation',
        content:
          'The Springfield city council voted 7-2 on Tuesday morning to approve a $2.5 million solar panel installation contract across municipal buildings. Councilor Jenkins sponsored the bill, projecting $400,000 in annual utility cost reductions. Voting against were Councilors Vance and Miller, who cited concerns over short-term municipal debt issuance.',
        source_url: 'https://springfieldgazette.com/articles/solar-vote',
        publisher: 'Springfield Gazette',
        published_at: new Date(now.getTime() - 2 * 3600 * 1000).toISOString(),
      },
      {
        id: 'art-observer-records-02',
        title: 'Municipal Voting Records: Resolution #44 Clean Energy Accord Passes with 7-2 Vote',
        content:
          'Official voting ledgers from the Springfield city clerk confirm that resolution #44 passed with a 7-2 vote. The city council approved the $2.5 million solar panel installation contract, authorizing Midwest Solar Dynamics to install photovoltaic arrays on city hall starting next spring.',
        source_url: 'https://civicobserver.org/records/resolution-44',
        publisher: 'Civic Records Observer',
        published_at: new Date(now.getTime() - 3 * 3600 * 1000).toISOString(),
      },
      {
        id: 'art-guardian-solar-03',
        title: 'US Municipalities Accelerate Rooftop Solar Upgrades Ahead of Clean Energy Deadline',
        content:
          'Regional civic councils in the Midwest are voting on record numbers of rooftop photovoltaic array installations to meet impending municipal sustainability targets. Springfield approved its $2.5 million municipal solar initiative this week.',
        source_url: 'https://www.theguardian.com/world/solar-municipal-midwest',
        publisher: 'The Guardian',
        published_at: new Date(now.getTime() - 4 * 3600 * 1000).toISOString(),
      },
    ],
    claims: [
      {
        id: 'c1',
        article_id: 'art-gazette-solar-01',
        claim_text:
          'The Springfield city council voted 7-2 to approve a $2.5 million solar installation.',
        context: 'Paragraph 1 detailing voting count and project valuation.',
        extraction_confidence: 0.96,
      },
      {
        id: 'c2',
        article_id: 'art-gazette-solar-01',
        claim_text:
          'Councilor Jenkins projected $400,000 in annual utility cost reductions for municipal buildings.',
        context: 'Paragraph 1 citing savings projections.',
        extraction_confidence: 0.88,
      },
      {
        id: 'c3',
        article_id: 'art-observer-records-02',
        claim_text:
          'Contract authorizes Midwest Solar Dynamics to begin rooftop installation next spring.',
        context: 'Paragraph 2 naming contractor and timeline.',
        extraction_confidence: 0.94,
      },
    ],
    verification_results: [
      {
        claim_id: 'c1',
        status: 'supported',
        confidence_score: 0.96,
        reasoning:
          'Independently verified by Civic Records Observer and The Guardian, confirming Resolution #44 passed by a 7-2 vote with a $2.5M contract value.',
        evidence_passages: [
          {
            id: 'art-observer-records-02_p0',
            passage_text:
              'Official voting ledgers from the Springfield city clerk confirm that resolution #44 passed with a 7-2 vote. The city council approved the $2.5 million solar panel installation contract...',
            source_url: 'https://civicobserver.org/records/resolution-44',
            publisher: 'Civic Records Observer',
            retrieved_at: now.toISOString(),
            relevance_score: 0.95,
          },
          {
            id: 'art-guardian-solar-03_p0',
            passage_text:
              'Springfield approved its $2.5 million municipal solar initiative this week.',
            source_url: 'https://www.theguardian.com/world/solar-municipal-midwest',
            publisher: 'The Guardian',
            retrieved_at: now.toISOString(),
            relevance_score: 0.82,
          },
        ],
      },
      {
        claim_id: 'c2',
        status: 'insufficient_evidence',
        confidence_score: 0.0,
        reasoning:
          'Reported solely by Springfield Gazette. Financial savings projections have not yet been independently verified in official municipal audit ledgers.',
        evidence_passages: [],
      },
      {
        claim_id: 'c3',
        status: 'supported',
        confidence_score: 0.92,
        reasoning:
          'Corroborated by official city clerk voting records identifying Midwest Solar Dynamics as the contracted vendor.',
        evidence_passages: [
          {
            id: 'art-observer-records-02_p1',
            passage_text:
              '...authorizing Midwest Solar Dynamics to install photovoltaic arrays on city hall starting next spring.',
            source_url: 'https://civicobserver.org/records/resolution-44',
            publisher: 'Civic Records Observer',
            retrieved_at: now.toISOString(),
            relevance_score: 0.91,
          },
        ],
      },
    ],
    brief: {
      id: 'brief-demo-01',
      title: 'Springfield Municipal Solar Installation Approved ($2.5M)',
      headline_takeaways: [
        'Springfield city council passed a $2.5M solar contract by a 7-2 vote [Supported - Civic Records Observer, The Guardian]',
        'Contract awards municipal rooftop installation to Midwest Solar Dynamics starting spring [Supported - Civic Records Observer]',
        'Annual utility cost reductions estimated at $400,000 [Reported by Springfield Gazette - Pending external audit]',
      ],
      synthesized_body:
        'This intelligence brief synthesizes developments across 3 ingested news sources with independent factual cross-validation.\n\n**Corroborated Developments:**\n- The Springfield city council voted 7-2 to approve a $2.5 million solar installation across municipal buildings. (Sources: Springfield Gazette, Civic Records Observer, The Guardian)\n- Midwest Solar Dynamics is authorized to begin array installations starting next spring. (Sources: Civic Records Observer)\n\n**Single-Source Reported Developments (Uncorroborated):**\n- Sponsoring councilors project $400,000 in recurring annual utility savings (Reported by Springfield Gazette; independent audit verification pending).\n\n**Referenced Sources:**\n- [Springfield Gazette] Springfield City Council Approves $2.5M Solar Panel Installation (https://springfieldgazette.com/articles/solar-vote)\n- [Civic Records Observer] Municipal Voting Records: Resolution #44 Clean Energy Accord Passes (https://civicobserver.org/records/resolution-44)\n- [The Guardian] US Municipalities Accelerate Rooftop Solar Upgrades (https://www.theguardian.com/world/solar-municipal-midwest)',
      source_article_ids: [
        'art-gazette-solar-01',
        'art-observer-records-02',
        'art-guardian-solar-03',
      ],
      verification_summary: {
        supported: 2,
        contradicted: 0,
        insufficient_evidence: 1,
      },
      created_at: now.toISOString(),
    },
    social_drafts: [
      {
        id: 'draft-x-01',
        platform: 'x_twitter',
        content:
          'Springfield City Council Approves $2.5M Solar Panel Installation\n\nDetails: https://springfieldgazette.com/articles/solar-vote\n#LocalNews #Springfield',
        character_count: 149,
        brief_id: 'brief-demo-01',
        source_article_ids: ['art-gazette-solar-01'],
        hashtags: ['LocalNews', 'Springfield'],
      },
      {
        id: 'draft-li-02',
        platform: 'linkedin',
        content:
          '📰 Executive News Brief: Springfield Municipal Solar Installation Approved ($2.5M)\n\nKey verified intelligence from our monitoring network:\n• Springfield city council passed a $2.5M solar contract by a 7-2 vote [Supported - Civic Records Observer, The Guardian]\n• Contract awards municipal rooftop installation to Midwest Solar Dynamics starting spring [Supported - Civic Records Observer]\n• Annual utility cost reductions estimated at $400,000 [Reported by Springfield Gazette - Pending external audit]\n\nSource: Springfield Gazette (https://springfieldgazette.com/articles/solar-vote)\n\n#LocalNews #Springfield #SolarEnergy #CleanTech',
        character_count: 622,
        brief_id: 'brief-demo-01',
        source_article_ids: ['art-gazette-solar-01', 'art-observer-records-02'],
        hashtags: ['LocalNews', 'Springfield', 'SolarEnergy', 'CleanTech'],
      },
    ],
    errors: [],
    completed_at: now.toISOString(),
  };
}

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [showConfig, setShowConfig] = useState(false);
  const [readiness, setReadiness] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stageStatus, setStageStatus] = useState('');
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Workflow options
  const [mockMode, setMockMode] = useState(true);
  const [lookbackHours, setLookbackHours] = useState(24);
  const [maxArticles, setMaxArticles] = useState(3);

  // Check backend readiness on load or API base change
  useEffect(() => {
    checkBackendHealth();
  }, [apiBase]);

  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${apiBase}/ready`, { signal: AbortSignal.timeout(3500) });
      if (res.ok) {
        const data = await res.json();
        setReadiness(data);
      } else {
        setReadiness({ ready: false, status: 'unavailable' });
      }
    } catch {
      // If deployed on Netlify without a local server tunnel, indicate standalone demo mode
      setReadiness({
        ready: false,
        status: 'standalone_demo',
        note: 'Running in Standalone Demo Mode. Connect your local Ollama backend for live inference.',
      });
    }
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const runPipeline = async () => {
    setLoading(true);
    setError(null);

    const payload = {
      mock_mode: mockMode,
      lookback_hours: Number(lookbackHours),
      max_articles: Number(maxArticles),
      platforms: ['x_twitter', 'linkedin'],
      articles: [],
    };

    // If in Mock / Demo mode and backend is offline/standalone:
    // Execute the deterministic client demonstration runner with realistic progressive telemetry
    if (mockMode && (!readiness?.ready || readiness?.status === 'standalone_demo')) {
      try {
        setStageStatus('📡 Collecting latest news from configured RSS sources (BBC, NYT, Guardian, NPR)...');
        await sleep(400);
        setStageStatus('🔍 Segmenting passages & extracting atomic factual claims...');
        await sleep(450);
        setStageStatus('⚖️ Cross-validating claims against BM25 lexical evidence across independent publishers...');
        await sleep(500);
        setStageStatus('📰 Synthesizing grounded executive brief with source provenance...');
        await sleep(350);
        setStageStatus('📣 Generating platform-tailored social media drafts (X 280-char & LinkedIn)...');
        await sleep(300);

        const demoResult = generateClientMockResult(lookbackHours);
        setResult(demoResult);
        return;
      } catch (err) {
        setError(err.message || 'Error executing demo workflow.');
        return;
      } finally {
        setLoading(false);
        setStageStatus('');
      }
    }

    setStageStatus(
      mockMode
        ? 'Executing deterministic mock workflow on backend...'
        : 'Connecting to live RSS feeds and executing local Ollama inference...'
    );

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout

      const response = await fetch(`${apiBase}/api/v1/news/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        // If mock mode failed on backend, fallback to client-side demo gracefully
        if (mockMode) {
          const fallbackData = generateClientMockResult(lookbackHours);
          setResult(fallbackData);
          return;
        }
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}: Failed to execute pipeline`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (mockMode) {
        // Graceful client fallback for demo
        const fallbackData = generateClientMockResult(lookbackHours);
        setResult(fallbackData);
        return;
      }

      if (err.name === 'AbortError') {
        setError(
          'Request timed out after 120 seconds. Local CPU Ollama inference on multiple articles requires more time. Try setting Max Articles to 1 or enable Demo Mode.'
        );
      } else {
        setError(
          `${err.message || 'Failed to connect to backend.'} (Ensure the local agent backend is running on ${apiBase || 'http://127.0.0.1:8000'} with Ollama, or toggle Safe Mock Mode).`
        );
      }
    } finally {
      setLoading(false);
      setStageStatus('');
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'supported':
        return <span className="status-tag supported">Supported</span>;
      case 'contradicted':
        return <span className="status-tag contradicted">Contradicted</span>;
      case 'insufficient_evidence':
      default:
        return <span className="status-tag insufficient_evidence">Insufficient Evidence</span>;
    }
  };

  const articles = result?.articles || [];
  const claims = result?.claims || [];
  const verifications = result?.verification_results || [];
  const brief = result?.brief;
  const drafts = result?.social_drafts || [];
  const metadata = result?.ingestion_metadata || {};
  const timings = metadata?.stage_timings_ms || {};

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo">📡</div>
          <div>
            <h1 className="brand-title">Local News Intelligence Agent</h1>
            <p className="brand-tagline">
              Autonomous Ingestion • Factual Claim Verification • Executive Briefing
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div className="header-status-badge">
            <span className={`pulse-dot ${readiness?.ready ? '' : readiness?.status === 'standalone_demo' ? 'demo' : 'warning'}`} />
            <span>
              {readiness?.ready
                ? `Backend: Ready (${readiness?.dependencies?.gemini?.details?.model || readiness?.dependencies?.ollama?.details?.model || 'Active'})`
                : readiness?.status === 'standalone_demo'
                ? '⚡ Standalone Demo Mode'
                : `Backend: ${readiness?.status || 'Connecting...'}`}
            </span>
          </div>

          <button
            onClick={() => setShowConfig(!showConfig)}
            title="Configure Backend API URL"
            style={{
              background: showConfig ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              color: '#e2e8f0',
              padding: '6px 12px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '0.8rem',
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              transition: 'all 0.2s ease',
            }}
          >
            ⚙️ <span>API Config</span>
          </button>
        </div>
      </header>

      {/* Backend API Configuration Panel */}
      {showConfig && (
        <div className="control-card" style={{ marginBottom: 16, borderColor: 'rgba(99, 102, 241, 0.4)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#e0e7ff' }}>
              🔌 Agent Backend Connection Settings
            </h3>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
              {readiness?.ready ? 'Connected ✅' : 'Running Offline/Demo'}
            </span>
          </div>
          <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: 12 }}>
            When running locally, leave this empty to use the default Vite proxy (<code>http://127.0.0.1:8000</code>).
            When hosting on Netlify, enter your HTTPS reverse proxy or tunnel URL (e.g. Cloudflare Tunnel or ngrok) to connect to your local Ollama agent.
          </p>
          <div style={{ display: 'flex', gap: 10 }}>
            <input
              type="text"
              placeholder="e.g. https://my-agent.trycloudflare.com or leave blank for local"
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value.trim())}
              style={{
                flex: 1,
                background: 'rgba(15, 23, 42, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                color: '#fff',
                padding: '8px 12px',
                borderRadius: '6px',
                fontSize: '0.85rem',
                fontFamily: 'var(--font-mono)',
              }}
            />
            <button
              className="btn-primary"
              style={{ padding: '8px 16px', fontSize: '0.8rem' }}
              onClick={checkBackendHealth}
            >
              Test Connection
            </button>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="control-card">
        <div className="control-card-header">
          <h2 style={{ fontSize: '1rem', fontWeight: 600 }}>Workflow Controls</h2>
          {mockMode ? (
            <span
              style={{
                background: 'rgba(99, 102, 241, 0.15)',
                color: '#818cf8',
                padding: '3px 10px',
                borderRadius: '9999px',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                border: '1px solid rgba(99, 102, 241, 0.3)',
              }}
            >
              ⚡ OFFLINE DEMO MODE (Fast & Deterministic)
            </span>
          ) : (
            <span
              style={{
                background: 'rgba(16, 185, 129, 0.15)',
                color: '#34d399',
                padding: '3px 10px',
                borderRadius: '9999px',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 600,
                border: '1px solid rgba(16, 185, 129, 0.3)',
              }}
            >
              🌐 LIVE PIPELINE (Gemini API / Ollama + Live Feeds)
            </span>
          )}
        </div>

        <div className="controls-grid">
          <div className="form-group">
            <label className="form-label">Execution Mode</label>
            <div
              className="toggle-container"
              onClick={() => !loading && setMockMode(!mockMode)}
            >
              <input
                type="checkbox"
                checked={mockMode}
                onChange={(e) => setMockMode(e.target.checked)}
                disabled={loading}
                id="mode-toggle"
              />
              <label htmlFor="mode-toggle" style={{ cursor: 'pointer', fontSize: '0.85rem' }}>
                {mockMode ? 'Safe Mock / Demo Mode' : 'Live Feeds & LLM (Gemini/Ollama)'}
              </label>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Lookback Window</label>
            <select
              className="form-select"
              value={lookbackHours}
              onChange={(e) => setLookbackHours(e.target.value)}
              disabled={loading}
            >
              <option value="6">Last 6 Hours</option>
              <option value="12">Last 12 Hours</option>
              <option value="24">Last 24 Hours</option>
              <option value="48">Last 48 Hours</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Max Articles (Downstream Cap)</label>
            <select
              className="form-select"
              value={maxArticles}
              onChange={(e) => setMaxArticles(e.target.value)}
              disabled={loading}
            >
              <option value="1">1 Article (Fastest)</option>
              <option value="2">2 Articles (Balanced)</option>
              <option value="3">3 Articles (Recommended)</option>
              <option value="5">5 Articles (In-Depth)</option>
            </select>
          </div>

          <button
            className="btn-primary"
            onClick={runPipeline}
            disabled={loading}
            id="run-pipeline-btn"
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>🚀 Run Intelligence Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner">
          <div>
            <strong>Error:</strong> {error}
          </div>
          <button
            className="btn-primary"
            style={{ padding: '4px 10px', fontSize: '0.75rem' }}
            onClick={runPipeline}
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading Progress State */}
      {loading && (
        <div className="control-card" style={{ textAlign: 'center', padding: '36px 20px' }}>
          <div className="spinner" style={{ margin: '0 auto 16px' }} />
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#f3f4f6' }}>
            Pipeline In Progress
          </h3>
          <p style={{ fontSize: '0.825rem', color: '#9ca3af', marginTop: 4 }}>
            {stageStatus}
          </p>
        </div>
      )}

      {/* Results View */}
      {result && !loading && (
        <>
          {/* Telemetry Ribbon */}
          <div className="telemetry-ribbon">
            <div className="telemetry-item">
              <div className="telemetry-title">Raw Ingested</div>
              <div className="telemetry-value">{metadata.raw_collected_count ?? 0}</div>
            </div>
            <div className="telemetry-item">
              <div className="telemetry-title">Retained (24h)</div>
              <div className="telemetry-value">{metadata.retained_count ?? 0}</div>
            </div>
            <div className="telemetry-item">
              <div className="telemetry-title">Extracted Claims</div>
              <div className="telemetry-value">{claims.length}</div>
            </div>
            <div className="telemetry-item">
              <div className="telemetry-title">Verifications</div>
              <div className="telemetry-value">{verifications.length}</div>
            </div>
            <div className="telemetry-item">
              <div className="telemetry-title">Total Duration</div>
              <div className="telemetry-value">
                {timings.total ? `${timings.total} ms` : 'N/A'}
              </div>
            </div>
          </div>

          <div className="dashboard-grid">
            {/* Left Column: Intelligence Brief & Social Drafts */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Executive Brief */}
              <div className="section-card">
                <div className="section-header">
                  <div className="section-heading">
                    <span>📰</span>
                    <span>Executive Intelligence Brief</span>
                  </div>
                  {brief?.verification_summary && (
                    <div style={{ display: 'flex', gap: 6 }}>
                      <span className="status-tag supported" style={{ fontSize: '0.65rem' }}>
                        {brief.verification_summary.supported || 0} Supported
                      </span>
                      <span className="status-tag contradicted" style={{ fontSize: '0.65rem' }}>
                        {brief.verification_summary.contradicted || 0} Contradicted
                      </span>
                    </div>
                  )}
                </div>

                {brief ? (
                  <div className="brief-container">
                    <h3 className="brief-title">{brief.title}</h3>

                    {brief.headline_takeaways && brief.headline_takeaways.length > 0 && (
                      <div className="takeaways-card">
                        <strong style={{ fontSize: '0.8rem', color: '#c7d2fe', textTransform: 'uppercase' }}>
                          Key Takeaways
                        </strong>
                        <ul className="takeaway-list">
                          {brief.headline_takeaways.map((takeaway, idx) => (
                            <li key={idx}>{takeaway}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="synthesized-body">{brief.synthesized_body}</div>
                  </div>
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No brief available for this run.
                  </p>
                )}
              </div>

              {/* Social Media Drafts */}
              <div className="section-card">
                <div className="section-header">
                  <div className="section-heading">
                    <span>📣</span>
                    <span>Platform Distribution Drafts</span>
                  </div>
                  <span className="section-count-badge">{drafts.length} Drafts</span>
                </div>

                {drafts.length > 0 ? (
                  drafts.map((draft, idx) => (
                    <div key={idx} className="social-draft-card">
                      <div className="social-draft-header">
                        <span className={`platform-badge ${draft.platform}`}>
                          {draft.platform === 'x_twitter' ? '𝕏 Twitter' : draft.platform}
                        </span>
                        <span className="char-counter">
                          {draft.character_count} chars
                        </span>
                      </div>

                      <div className="social-content-preview">{draft.content}</div>

                      {draft.hashtags && draft.hashtags.length > 0 && (
                        <div className="hashtags-row">
                          {draft.hashtags.map((tag, tIdx) => (
                            <span key={tIdx} className="hashtag-chip">
                              #{tag}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No distribution drafts generated.
                  </p>
                )}
              </div>
            </div>

            {/* Right Column: Verification Results & Ingested News Feed */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Claim Verification Results */}
              <div className="section-card">
                <div className="section-header">
                  <div className="section-heading">
                    <span>🔍</span>
                    <span>Factual Claim Verifications</span>
                  </div>
                  <span className="section-count-badge">
                    {verifications.length} Claims Checked
                  </span>
                </div>

                {verifications.length > 0 ? (
                  verifications.map((ver, idx) => {
                    const matchedClaim = claims.find((c) => c.id === ver.claim_id);
                    return (
                      <div key={idx} className={`claim-card status-${ver.status}`}>
                        <div className="claim-header">
                          {getStatusBadge(ver.status)}
                          <span className="confidence-metric">
                            Confidence: {Math.round((ver.confidence_score || 0) * 100)}%
                          </span>
                        </div>

                        <div className="claim-statement">
                          "{matchedClaim?.claim_text || 'Factual Claim'}"
                        </div>

                        {ver.reasoning && (
                          <div className="reasoning-box">
                            <strong>Reasoning:</strong> {ver.reasoning}
                          </div>
                        )}

                        {ver.evidence_passages && ver.evidence_passages.length > 0 && (
                          <div className="citations-list">
                            <span style={{ fontSize: '0.7rem', color: '#94a3b8', fontWeight: 600 }}>
                              Retrieved Evidence Citations:
                            </span>
                            {ver.evidence_passages.map((ev, eIdx) => (
                              <div key={eIdx} className="citation-chip">
                                <span>{ev.publisher || 'Unknown Publisher'}</span>
                                {ev.source_url && (
                                  <a
                                    href={ev.source_url}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    View Source
                                  </a>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No claims evaluated yet.
                  </p>
                )}
              </div>

              {/* Ingested News Feed */}
              <div className="section-card">
                <div className="section-header">
                  <div className="section-heading">
                    <span>📡</span>
                    <span>Ingested News Sources</span>
                  </div>
                  <span className="section-count-badge">{articles.length} Articles</span>
                </div>

                {articles.length > 0 ? (
                  articles.map((art) => (
                    <div key={art.id} className="article-item">
                      <div className="article-header">
                        <span className="article-publisher">{art.publisher}</span>
                        <span className="article-time">
                          {new Date(art.published_at).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </span>
                      </div>

                      <a
                        href={art.source_url}
                        target="_blank"
                        rel="noreferrer"
                        className="article-title-link"
                      >
                        {art.title}
                      </a>

                      <p className="article-snippet">{art.content}</p>
                    </div>
                  ))
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No articles ingested.
                  </p>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {/* Initial Empty State */}
      {!result && !loading && (
        <div className="state-container">
          <div style={{ fontSize: '42px' }}>📰</div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
            Ready to Analyze Local News
          </h3>
          <p style={{ color: 'var(--text-secondary)', maxWidth: 520, fontSize: '0.875rem' }}>
            Click <strong>Run Intelligence Pipeline</strong> above to ingest articles from live RSS
            feeds, extract atomic factual assertions, verify claims against corroborating evidence,
            and synthesize distribution drafts.
          </p>
        </div>
      )}
    </div>
  );
}
