/**
 * compare.js — Side-by-side user comparison overlay with charts.
 *
 * Exports (via window.deriveCompare):
 *   open(uidA)  – show compare prompt with current user pre-filled
 *   close()     – hide overlay
 */

(function () {
  const overlay = document.getElementById('compare-overlay');
  const content = document.getElementById('compare-content');
  let charts = [];
  let uidA = null;

  function open(currentUid) {
    uidA = currentUid;
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';

    content.innerHTML = `
      <div class="cmp-prompt">
        <div class="cmp-prompt-title">Compare User #${currentUid} with:</div>
        <div class="cmp-prompt-row">
          <input type="text" id="compare-uid-input" class="cmp-input" placeholder="Enter user ID" autocomplete="off" />
          <button class="cmp-go-btn" onclick="deriveCompare.runCompare()">Compare</button>
        </div>
        <div class="cmp-error" id="compare-error"></div>
      </div>
    `;

    setTimeout(() => {
      const input = document.getElementById('compare-uid-input');
      if (input) {
        input.focus();
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter') deriveCompare.runCompare();
        });
      }
    }, 100);
  }

  function close() {
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
    destroyCharts();
  }

  function destroyCharts() {
    charts.forEach(c => c.destroy());
    charts = [];
  }

  async function runCompare() {
    const input = document.getElementById('compare-uid-input');
    const errorEl = document.getElementById('compare-error');
    if (!input) return;

    const uidB = parseInt(input.value.trim());
    if (isNaN(uidB)) {
      errorEl.textContent = 'Enter a valid numeric user ID';
      return;
    }
    if (uidB === uidA) {
      errorEl.textContent = 'Choose a different user';
      return;
    }

    errorEl.textContent = '';
    content.innerHTML = '<div class="lb-loading">Loading comparison...</div>';

    try {
      const [resA, resB] = await Promise.all([
        fetch(`/api/report/${uidA}`),
        fetch(`/api/report/${uidB}`),
      ]);

      if (!resA.ok || !resB.ok) {
        content.innerHTML = '<div class="lb-loading">One or both users not found.</div>';
        return;
      }

      const dataA = await resA.json();
      const dataB = await resB.json();

      destroyCharts();
      renderComparison(dataA, dataB);
    } catch (e) {
      content.innerHTML = '<div class="lb-loading">Failed to load comparison data.</div>';
    }
  }

  function renderComparison(a, b) {
    content.innerHTML = `
      <div class="cmp-grid">
        <!-- User A header -->
        <div class="cmp-user-header cmp-user-a">
          <div class="cmp-user-id">#${a.uid}</div>
          <div class="cmp-user-label">${esc(a.label)}</div>
        </div>
        <div class="cmp-vs">VS</div>
        <!-- User B header -->
        <div class="cmp-user-header cmp-user-b">
          <div class="cmp-user-id">#${b.uid}</div>
          <div class="cmp-user-label">${esc(b.label)}</div>
        </div>
      </div>

      <!-- Stats comparison table -->
      <table class="cmp-table">
        <thead>
          <tr><th></th><th>#${a.uid}</th><th>#${b.uid}</th></tr>
        </thead>
        <tbody>
          <tr><td>History</td><td style="color:#60a5fa">${a.history_count}</td><td style="color:#60a5fa">${b.history_count}</td></tr>
          <tr><td>Test Set</td><td style="color:#f59e0b">${a.ground_truth_count}</td><td style="color:#f59e0b">${b.ground_truth_count}</td></tr>
          <tr><td>Predictions</td><td style="color:#10b981">${a.prediction_count}</td><td style="color:#10b981">${b.prediction_count}</td></tr>
          <tr><td>Hit Rate</td><td class="${a.hits > 0 ? 'cmp-hit' : ''}">${a.hit_rate}</td><td class="${b.hits > 0 ? 'cmp-hit' : ''}">${b.hit_rate}</td></tr>
          <tr><td>Spread</td><td>${a.spread}</td><td>${b.spread}</td></tr>
          <tr><td>Centroid</td><td>${a.centroid_lat}&deg;, ${a.centroid_lon}&deg;</td><td>${b.centroid_lat}&deg;, ${b.centroid_lon}&deg;</td></tr>
        </tbody>
      </table>

      <!-- Charts -->
      <div class="cmp-charts-row">
        <div class="cmp-chart-section">
          <div class="report-section-hdr">Activity Comparison</div>
          <div class="report-chart-card">
            <canvas id="cmp-chart-activity"></canvas>
          </div>
        </div>
        <div class="cmp-chart-section">
          <div class="report-section-hdr">Geographic Spread</div>
          <div class="report-chart-card">
            <canvas id="cmp-chart-scatter"></canvas>
          </div>
        </div>
      </div>

      <div class="cmp-actions">
        <button class="cmp-view-btn" onclick="deriveCompare.viewUser(${a.uid})">View #${a.uid} on Map</button>
        <button class="cmp-view-btn" onclick="deriveCompare.viewUser(${b.uid})">View #${b.uid} on Map</button>
      </div>
    `;

    requestAnimationFrame(() => {
      renderActivityChart(a, b);
      renderScatterComparison(a, b);
    });
  }

  function renderActivityChart(a, b) {
    const ctx = document.getElementById('cmp-chart-activity');
    if (!ctx) return;

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['History', 'Test Set', 'Predictions', 'Hits'],
        datasets: [
          {
            label: `#${a.uid}`,
            data: [a.history_count, a.ground_truth_count, a.prediction_count, a.hits],
            backgroundColor: 'rgba(96,165,250,0.6)',
            borderColor: 'rgba(96,165,250,1)',
            borderWidth: 1,
            borderRadius: 3,
          },
          {
            label: `#${b.uid}`,
            data: [b.history_count, b.ground_truth_count, b.prediction_count, b.hits],
            backgroundColor: 'rgba(167,139,250,0.6)',
            borderColor: 'rgba(167,139,250,1)',
            borderWidth: 1,
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top', labels: { boxWidth: 12, padding: 12 } } },
        scales: {
          x: { grid: { display: false } },
          y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { precision: 0 } },
        },
      },
    });
    charts.push(chart);
  }

  function renderScatterComparison(a, b) {
    const ctx = document.getElementById('cmp-chart-scatter');
    if (!ctx) return;

    const chart = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: `#${a.uid} History`,
            data: a.scatter.history,
            backgroundColor: 'rgba(96,165,250,0.4)',
            pointRadius: 3,
          },
          {
            label: `#${a.uid} Preds`,
            data: a.scatter.predictions,
            backgroundColor: 'rgba(96,165,250,0.9)',
            pointRadius: 5,
            pointStyle: 'triangle',
          },
          {
            label: `#${b.uid} History`,
            data: b.scatter.history,
            backgroundColor: 'rgba(167,139,250,0.4)',
            pointRadius: 3,
          },
          {
            label: `#${b.uid} Preds`,
            data: b.scatter.predictions,
            backgroundColor: 'rgba(167,139,250,0.9)',
            pointRadius: 5,
            pointStyle: 'triangle',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top', labels: { boxWidth: 10, padding: 8, font: { size: 10 } } } },
        scales: {
          x: { title: { display: true, text: 'Longitude', color: '#666' }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { title: { display: true, text: 'Latitude', color: '#666' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        },
      },
    });
    charts.push(chart);
  }

  function viewUser(uid) {
    close();
    if (window.deriveApp && window.deriveApp.loadAndSelectUser) {
      window.deriveApp.loadAndSelectUser(uid);
    }
  }

  function esc(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  window.deriveCompare = { open, close, runCompare, viewUser };
})();
