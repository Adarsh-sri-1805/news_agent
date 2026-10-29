import React, { useState, useEffect } from 'react';

const API_BASE = ''; // Uses Vite proxy configured for http://127.0.0.1:8000

export default function App() {
  const [readiness, setReadiness] = useState(null);
  const [loading, setLoading] = useState(false);
  const [stageStatus, setStageStatus] = useState('');
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Workflow options
  const [mockMode, setMockMode] = useState(true);
  const [lookbackHours, setLookbackHours] = useState(24);
  const [maxArticles, setMaxArticles] = useState(3);

  // Check backend readiness on load
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/ready`);
      if (res.ok) {
        const data = await res.json();
        setReadiness(data);
      } else {
        setReadiness({ ready: false, status: 'unavailable' });
      }
    } catch {
      setReadiness({ ready: false, status: 'offline' });
    }
  };

  const runPipeline = async () => {
    setLoading(true);
    setError(null);
    setStageStatus(
      mockMode
        ? 'Executing deterministic mock workflow...'
        : 'Connecting to live RSS feeds and running local Ollama inference...'
    );

    const payload = {
      mock_mode: mockMode,
      lookback_hours: Number(lookbackHours),
      max_articles: Number(maxArticles),
      platforms: ['x_twitter', 'linkedin'],
      articles: [],
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout

      const response = await fetch(`${API_BASE}/api/v1/news/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${response.status}: Failed to execute pipeline`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err.name === 'AbortError') {
        setError(
          'Request timed out after 120 seconds. Local CPU Ollama inference on multiple articles requires more time. Try enabling Demo Mode or setting Max Articles to 1.'
        );
      } else {
        setError(err.message || 'An unexpected error occurred executing the intelligence workflow.');
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

        <div className="header-status-badge">
          <span className={`pulse-dot ${readiness?.ready ? '' : 'warning'}`} />
          <span>
            Backend:{' '}
            {readiness?.ready
              ? `Ready (${readiness?.dependencies?.ollama?.details?.model || 'Ollama'})`
              : readiness?.status || 'Connecting...'}
          </span>
        </div>
      </header>

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
              🌐 LIVE PIPELINE (BBC & NYT Feeds + Ollama)
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
                {mockMode ? 'Safe Mock / Demo Mode' : 'Live Feeds & Ollama'}
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
