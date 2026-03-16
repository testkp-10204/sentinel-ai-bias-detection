/* ===================================================
   SentinelAI — app.js
   Wired to the Phase 3 Bias-Mitigated DistilBERT model
   trained on the Jigsaw Bias dataset.

   Expected backend API (FastAPI / Flask) exposes:
     POST /analyze          → single text analysis
     POST /analyze/batch    → batch analysis
     POST /analyze/compare  → side-by-side comparison
     GET  /stats            → aggregate stats
     GET  /history          → paginated history
     GET  /export           → CSV download
     DELETE /history/:id    → delete a record

   The backend loads the saved model from:
     Models/bert_mitigated_final/
   and uses the DistilBertForSequenceClassification
   weights + tokenizer saved in Phase 3 Cell 9.

   BERT output schema (what the backend should return):
   {
     id, text, label?,
     toxicity: "toxic" | "safe",
     toxicity_score: 0.0–1.0,          ← DistilBERT softmax prob
     sentiment: "POSITIVE" | "NEGATIVE",
     sentiment_score: 0.0–1.0,
     bias: "biased" | "neutral",
     bias_score: 0.0–1.0,
     overall_risk: "LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
     risk_score: 0.0–1.0,
     keywords: string[],
     highlighted_text: string,
     identity_mention: boolean,         ← identity-group flag from Phase 3
     created_at: string
   }
   =================================================== */

const API = 'http://localhost:8000';
let currentPage = 0;
const PAGE_SIZE = 15;
let totalItems = 0;

/* =============================================
   UTILITY HELPERS
   ============================================= */

function showToast(msg, duration = 3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), duration);
}

function switchPage(name, event) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  if (event && event.target) event.target.classList.add('active');
  if (name === 'dashboard') loadDashboard();
}

function setMode(btn) {
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
}

function updateCharCount() {
  const t = document.getElementById('main-input').value;
  document.getElementById('char-count').textContent = `${t.length} / 1000`;
}

function clearAll() {
  document.getElementById('main-input').value = '';
  document.getElementById('label-input').value = '';
  document.getElementById('results-area').innerHTML = '';
  document.getElementById('sidebar-results').innerHTML =
    '<div class="empty-state" id="empty-state"><div class="empty-icon">⚡</div><div class="empty-text">Results will appear here after analysis</div></div>';
  updateCharCount();
}

/* =============================================
   BADGE / COLOR HELPERS
   ============================================= */

function getBadgeClass(key, val) {
  if (key === 'sentiment') return val === 'POSITIVE' ? 'badge-positive' : 'badge-negative';
  if (key === 'toxicity')  return val === 'toxic'    ? 'badge-toxic'    : 'badge-safe';
  if (key === 'bias')      return (val === 'hate' || val === 'biased') ? 'badge-biased' : 'badge-safe';
  if (key === 'risk') {
    const m = { LOW: 'badge-low', MEDIUM: 'badge-medium', HIGH: 'badge-high', CRITICAL: 'badge-critical' };
    return m[val] || 'badge-low';
  }
  return '';
}

function getBarClass(key, label) {
  if (key === 'sentiment') return label === 'POSITIVE' ? 'bar-green' : 'bar-red';
  if (key === 'toxicity')  return label === 'toxic'    ? 'bar-red'   : 'bar-green';
  if (key === 'bias')      return (label === 'hate' || label === 'biased') ? 'bar-orange' : 'bar-green';
  if (key === 'risk') {
    const m = { LOW: 'bar-green', MEDIUM: 'bar-accent', HIGH: 'bar-orange', CRITICAL: 'bar-red' };
    return m[label] || 'bar-accent';
  }
  return 'bar-accent';
}

function getColor(key, label) {
  if (key === 'sentiment') return label === 'POSITIVE' ? 'var(--green)' : 'var(--red)';
  if (key === 'toxicity')  return label === 'toxic'    ? 'var(--red)'   : 'var(--green)';
  if (key === 'bias')      return (label === 'hate' || label === 'biased') ? 'var(--orange)' : 'var(--green)';
  if (key === 'risk') {
    const m = { LOW: 'var(--green)', MEDIUM: 'var(--yellow)', HIGH: 'var(--orange)', CRITICAL: 'var(--red)' };
    return m[label] || 'var(--accent)';
  }
  return 'var(--accent)';
}

/* =============================================
   SCORE CARD BUILDER
   ============================================= */

function scoreCard(title, label, score, key) {
  const bc   = getBadgeClass(key, label);
  const barc = getBarClass(key, label);
  const color = getColor(key, label);
  return `
    <div class="score-card">
      <div class="score-header">
        <span class="score-label">${title}</span>
        <span class="score-badge ${bc}">${label}</span>
      </div>
      <div class="score-value" style="color:${color}">${(score * 100).toFixed(1)}%</div>
      <div class="score-bar-wrap">
        <div class="score-bar ${barc}" id="bar-${key}" style="width:0"></div>
      </div>
    </div>`;
}

/* =============================================
   BERT INFO BANNER
   Shows a contextual note when an identity-group
   mention is detected (Phase 3 bias mitigation).
   ============================================= */

function bertInfoBanner(d) {
  if (!d.identity_mention) return '';
  return `
    <div class="bert-info-banner">
      <div class="bert-info-icon">🛡️</div>
      <div class="bert-info-text">
        <strong>Identity Group Detected</strong> — This text contains a reference to a protected
        identity group. The bias-mitigated DistilBERT model applied a
        <strong>3× fairness penalty</strong> during training to reduce false positives for
        identity-mentioning content. Scores reflect this correction.
      </div>
    </div>`;
}

/* =============================================
   MAIN ANALYZE — calls POST /analyze
   Backend loads bert_mitigated_final weights and
   runs DistilBertForSequenceClassification.
   ============================================= */

async function analyzeText() {
  const text  = document.getElementById('main-input').value.trim();
  const label = document.getElementById('label-input').value.trim();
  if (!text) { showToast('Please enter some text first'); return; }

  document.getElementById('btn-analyze').disabled = true;
  document.getElementById('loading-state').classList.add('visible');
  document.getElementById('results-area').innerHTML = '';
  document.getElementById('sidebar-results').innerHTML = '';

  // Cycle through BERT-specific loading messages
  const msgs = [
    'Tokenizing with DistilBERT…',
    'Running bias-mitigated model…',
    'Applying fairness corrections…',
    'Computing risk score…'
  ];
  let mi = 0;
  const lt = document.getElementById('loading-text');
  const interval = setInterval(() => { lt.textContent = msgs[mi++ % msgs.length]; }, 900);

  try {
    const params = new URLSearchParams({ text, ...(label && { label }) });
    const res = await fetch(`${API}/analyze?${params}`);
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();

    clearInterval(interval);
    document.getElementById('loading-state').classList.remove('visible');
    document.getElementById('btn-analyze').disabled = false;

    renderSingleResult(d);
  } catch (e) {
    clearInterval(interval);
    document.getElementById('loading-state').classList.remove('visible');
    document.getElementById('btn-analyze').disabled = false;
    showToast('Error: ' + (e.message.includes('fetch')
      ? 'API offline — start your FastAPI backend pointing to bert_mitigated_final/'
      : e.message));
  }
}

/* =============================================
   RENDER SINGLE RESULT
   ============================================= */

function renderSingleResult(d) {
  // --- Sidebar score cards ---
  const sidebar = document.getElementById('sidebar-results');
  sidebar.innerHTML = `
    ${bertInfoBanner(d)}
    ${scoreCard('Toxicity (DistilBERT)', d.toxicity, d.toxicity_score, 'toxicity')}
    ${scoreCard('Sentiment', d.sentiment, d.sentiment_score, 'sentiment')}
    ${scoreCard('Bias / Hate', d.bias, d.bias_score, 'bias')}
    <div class="risk-card">
      <div class="score-label">Overall Risk</div>
      <div class="risk-level" style="color:${getColor('risk', d.overall_risk)}">${d.overall_risk}</div>
      <div class="score-bar-wrap" style="margin:10px 0;">
        <div class="score-bar ${getBarClass('risk', d.overall_risk)}" id="bar-risk" style="width:0"></div>
      </div>
      <div class="risk-desc">
        Risk score: ${(d.risk_score * 100).toFixed(1)}% — composite score from the
        bias-mitigated DistilBERT model
        ${d.identity_mention
          ? '· <span class="identity-flag">⚖️ Identity-weighted</span>'
          : ''}
      </div>
    </div>
  `;

  // Animate bars after DOM update
  setTimeout(() => {
    [
      ['bar-toxicity',  d.toxicity_score],
      ['bar-sentiment', d.sentiment_score],
      ['bar-bias',      d.bias_score],
      ['bar-risk',      d.risk_score],
    ].forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el) el.style.width = (val * 100) + '%';
    });
  }, 50);

  // --- Main results area ---
  const keywords = d.keywords || [];
  document.getElementById('results-area').innerHTML = `
    <div class="input-panel">
      <div class="panel-header">
        <span class="panel-title">Analysis Results</span>
        <span class="tag-mono" style="color:var(--text3)">ID #${d.id} · DistilBERT</span>
      </div>
      <div style="padding:20px;">
        <div class="section-title">Keywords Detected</div>
        <div class="keywords-wrap">
          ${keywords.length
            ? keywords.map(k => `<span class="keyword-chip">${k}</span>`).join('')
            : '<span style="color:var(--text3);font-size:13px;">No significant keywords found</span>'}
        </div>
        <div style="margin-top:20px;">
          <div class="section-title">Text with Highlights</div>
          <div class="highlighted-text">${d.highlighted_text || d.text}</div>
        </div>
        <div style="margin-top:20px; padding-top:16px; border-top:1px solid var(--border);
                    display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
          <span style="font-size:12px;color:var(--text3);">Analyzed at ${new Date().toLocaleTimeString()}</span>
          <button class="btn-secondary" style="font-size:12px;padding:6px 14px;"
                  onclick="copyJSON(${JSON.stringify(JSON.stringify(d))})">Copy JSON</button>
        </div>
      </div>
    </div>`;
}

function copyJSON(str) {
  navigator.clipboard.writeText(str);
  showToast('Copied to clipboard ✓');
}

/* =============================================
   BATCH ANALYZE — calls POST /analyze/batch
   Backend runs DistilBERT on each text.
   ============================================= */

async function runBatch() {
  const raw   = document.getElementById('batch-input').value.trim();
  const texts = raw.split('\n').map(t => t.trim()).filter(Boolean);
  if (!texts.length) { showToast('Please enter at least one text'); return; }
  if (texts.length > 50) { showToast('Maximum 50 texts per batch'); return; }

  document.getElementById('batch-loading').style.display  = 'block';
  document.getElementById('batch-results').innerHTML      = '';
  document.getElementById('batch-summary').innerHTML      = '';

  try {
    const res = await fetch(`${API}/analyze/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();

    document.getElementById('batch-loading').style.display = 'none';

    const s = d.summary;
    document.getElementById('batch-summary').innerHTML = `
      <div class="summary-grid" style="margin-top:24px;">
        <div class="stat-card">
          <div class="stat-label">Total Analyzed</div>
          <div class="stat-value" style="font-size:32px;">${s.total}</div>
          <div class="stat-meta">${s.positive_count} positive · ${s.negative_count} negative</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Toxicity Rate</div>
          <div class="stat-value" style="font-size:32px; color:${s.toxic_count > 0 ? 'var(--red)' : 'var(--green)'}">
            ${s.toxic_count > 0 ? ((s.toxic_count / s.total) * 100).toFixed(0) + '%' : 'Clean'}
          </div>
          <div class="stat-meta">${s.toxic_count} toxic · ${s.safe_count} safe</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Avg Scores</div>
          <div class="stat-value" style="font-size:32px;">${(s.avg_sentiment_score * 100).toFixed(0)}%</div>
          <div class="stat-meta">Sentiment · Tox: ${(s.avg_toxicity_score * 100).toFixed(0)}% · Bias: ${(s.avg_bias_score * 100).toFixed(0)}%</div>
        </div>
      </div>`;

    const resultsEl = document.getElementById('batch-results');
    d.results.forEach((r, i) => {
      resultsEl.innerHTML += `
        <div class="batch-item">
          <div class="batch-item-text">${i + 1}. ${r.text}</div>
          <div class="batch-badges">
            <span class="score-badge ${getBadgeClass('sentiment', r.sentiment)}">${r.sentiment} ${(r.sentiment_score * 100).toFixed(0)}%</span>
            <span class="score-badge ${getBadgeClass('toxicity', r.toxicity)}">${r.toxicity}</span>
            <span class="score-badge ${getBadgeClass('risk', r.overall_risk)}">${r.overall_risk}</span>
            ${r.identity_mention ? '<span class="identity-flag">⚖️ ID-weighted</span>' : ''}
          </div>
        </div>`;
    });
  } catch (e) {
    document.getElementById('batch-loading').style.display = 'none';
    showToast('Error: ' + e.message);
  }
}

/* =============================================
   COMPARE — calls POST /analyze/compare
   ============================================= */

async function runCompare() {
  const a = document.getElementById('compare-a').value.trim();
  const b = document.getElementById('compare-b').value.trim();
  if (!a || !b) { showToast('Please enter both texts'); return; }

  document.getElementById('compare-loading').style.display = 'block';
  document.getElementById('compare-results').innerHTML     = '';

  try {
    const res = await fetch(`${API}/analyze/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_a: a, text_b: b })
    });
    if (!res.ok) throw new Error(await res.text());
    const d = await res.json();
    document.getElementById('compare-loading').style.display = 'none';
    renderCompareResults(d);
  } catch (e) {
    document.getElementById('compare-loading').style.display = 'none';
    showToast('Error: ' + e.message);
  }
}

function renderCompareResults(d) {
  const ra = d.text_a, rb = d.text_b, c = d.comparison;
  const col = (key, label) => getColor(key, label);

  document.getElementById('compare-results').innerHTML = `
    <div class="compare-grid" style="margin-bottom:24px;">
      ${['A', 'B'].map((label, i) => {
        const r = i === 0 ? ra : rb;
        return `
          <div class="score-card">
            <div class="section-title" style="margin-bottom:14px;">Text ${label}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text2);margin-bottom:16px;line-height:1.5;">
              "${r.text.substring(0, 120)}${r.text.length > 120 ? '…' : ''}"
            </div>
            <div class="batch-badges" style="gap:8px;">
              <span class="score-badge ${getBadgeClass('sentiment', r.sentiment)}">${r.sentiment}</span>
              <span class="score-badge ${getBadgeClass('toxicity', r.toxicity)}">${r.toxicity}</span>
              <span class="score-badge ${getBadgeClass('risk', r.overall_risk)}">${r.overall_risk} RISK</span>
              ${r.identity_mention ? '<span class="identity-flag">⚖️ ID-weighted</span>' : ''}
            </div>
            <div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
              ${[
                ['Toxicity', r.toxicity_score, 'toxicity', r.toxicity],
                ['Sentiment', r.sentiment_score, 'sentiment', r.sentiment],
                ['Bias', r.bias_score, 'bias', r.bias]
              ].map(([name, score, key, lbl]) => `
                <div style="text-align:center;">
                  <div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">${name}</div>
                  <div style="font-family:'DM Serif Display',serif;font-size:22px;color:${col(key, lbl)}">${(score * 100).toFixed(0)}%</div>
                </div>`).join('')}
            </div>
          </div>`;
      }).join('')}
    </div>
    <div class="score-card">
      <div class="section-title">Comparison Summary</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:16px;">
        ${[
          ['More Positive', c.more_positive, 'accent'],
          ['More Toxic',    c.more_toxic,    'red'],
          ['More Biased',   c.more_biased,   'orange'],
          ['Risk Diff',     Math.abs(c.risk_score_diff * 100).toFixed(1) + '%', 'text2'],
        ].map(([label, val, clr]) => `
          <div style="text-align:center;">
            <div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">${label}</div>
            <div style="font-family:'DM Serif Display',serif;font-size:28px;color:var(--${clr});">${val}</div>
          </div>`).join('')}
      </div>
    </div>`;
}

/* =============================================
   DASHBOARD
   ============================================= */

async function loadDashboard() {
  await Promise.all([loadStats(), loadHistory()]);
}

async function loadStats() {
  try {
    const res = await fetch(`${API}/stats`);
    const d   = await res.json();
    const positive_pct = d.total ? ((d.positive_count / d.total) * 100).toFixed(0) : 0;
    const toxic_pct    = d.total ? ((d.toxic_count    / d.total) * 100).toFixed(0) : 0;

    document.getElementById('dash-sub-text').textContent = `${d.total} total analyses in the database`;

    document.getElementById('stats-grid').innerHTML = `
      <div class="stat-card">
        <div class="stat-label">Total Analyses</div>
        <div class="stat-value" style="color:var(--accent)">${d.total}</div>
        <div class="stat-meta">All time</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Positive Rate</div>
        <div class="stat-value" style="color:var(--green)">${positive_pct}%</div>
        <div class="stat-meta">${d.positive_count} positive · ${d.negative_count} negative</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Toxicity Rate</div>
        <div class="stat-value" style="color:${toxic_pct > 20 ? 'var(--red)' : 'var(--green)'}">${toxic_pct}%</div>
        <div class="stat-meta">${d.toxic_count} toxic detected by DistilBERT</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg Sentiment</div>
        <div class="stat-value" style="color:var(--purple)">${d.total ? (d.avg_sentiment_score * 100).toFixed(0) + '%' : '—'}</div>
        <div class="stat-meta">Avg toxicity: ${d.total ? (d.avg_toxicity_score * 100).toFixed(0) + '%' : '—'}</div>
      </div>`;
  } catch (e) {
    document.getElementById('stats-grid').innerHTML =
      `<div style="color:var(--text3);padding:20px;grid-column:1/-1;">Could not load stats — is the API running?</div>`;
  }
}

async function loadHistory() {
  const risk   = document.getElementById('risk-filter')?.value || '';
  const offset = currentPage * PAGE_SIZE;

  try {
    const params = new URLSearchParams({ limit: PAGE_SIZE, offset, ...(risk && { risk }) });
    const res    = await fetch(`${API}/history?${params}`);
    const d      = await res.json();
    totalItems   = d.total;

    document.getElementById('page-info-text').textContent =
      `${offset + 1}–${Math.min(offset + PAGE_SIZE, totalItems)} of ${totalItems}`;
    document.getElementById('btn-prev').disabled = currentPage === 0;
    document.getElementById('btn-next').disabled = offset + PAGE_SIZE >= totalItems;

    if (!d.items.length) {
      document.getElementById('history-table').innerHTML =
        '<div class="empty-state"><div class="empty-icon">📂</div><div class="empty-text">No analyses found</div></div>';
      return;
    }

    document.getElementById('history-table').innerHTML = `
      <table>
        <thead><tr>
          <th>ID</th><th>Text</th><th>Toxicity</th><th>Sentiment</th><th>Risk</th><th>Identity</th><th>Date</th><th></th>
        </tr></thead>
        <tbody>
          ${d.items.map(item => `
            <tr>
              <td class="td-mono">#${item.id}</td>
              <td class="td-text" title="${item.text}">${item.text}</td>
              <td><span class="score-badge ${getBadgeClass('toxicity', item.toxicity)}">${item.toxicity}</span></td>
              <td><span class="score-badge ${getBadgeClass('sentiment', item.sentiment)}">${item.sentiment}</span></td>
              <td><span class="score-badge ${getBadgeClass('risk', item.overall_risk)}">${item.overall_risk}</span></td>
              <td>${item.identity_mention ? '<span class="identity-flag">⚖️ Yes</span>' : '<span style="color:var(--text3);font-size:11px;">—</span>'}</td>
              <td class="td-mono" style="font-size:11px;">${item.created_at?.slice(0, 16) || '—'}</td>
              <td><button class="btn-icon" onclick="deleteItem(${item.id})" title="Delete">✕</button></td>
            </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    document.getElementById('history-table').innerHTML =
      `<div style="padding:24px;color:var(--text3);">Failed to load history — is the API running on port 8000?</div>`;
  }
}

async function deleteItem(id) {
  try {
    await fetch(`${API}/history/${id}`, { method: 'DELETE' });
    showToast(`Analysis #${id} deleted`);
    loadHistory();
    loadStats();
  } catch (e) {
    showToast('Failed to delete');
  }
}

function changePage(dir) {
  currentPage += dir;
  loadHistory();
}

function exportCSV() {
  window.location.href = `${API}/export`;
  showToast('Downloading CSV…');
}

/* =============================================
   INIT — runs after DOM is fully loaded
   ============================================= */

document.addEventListener('DOMContentLoaded', () => {

  // Ctrl+Enter to analyze
  document.getElementById('main-input').addEventListener('keydown', e => {
    if (e.ctrlKey && e.key === 'Enter') analyzeText();
  });

  // Initial status hint
  document.getElementById('loading-text').textContent = 'DistilBERT ready…';

  // Wire nav tabs via JS (removes reliance on inline onclick passing event correctly)
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', function () {
      const page = this.textContent.trim().toLowerCase();
      const pageMap = { 'analyze': 'analyze', 'batch': 'batch', 'compare': 'compare', 'dashboard': 'dashboard' };
      const target = pageMap[page];
      if (!target) return;
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.getElementById('page-' + target).classList.add('active');
      this.classList.add('active');
      if (target === 'dashboard') loadDashboard();
    });
  });

});