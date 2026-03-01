/**
 * report.js — Full-screen Data Analyst Report overlay with Chart.js charts & PDF export.
 *
 * Exports (via window.deriveReport):
 *   open(data)    – build report HTML, render charts, show overlay
 *   close()       – hide overlay, destroy chart instances
 *   exportPDF()   – html2canvas + jsPDF to download PDF
 */

(function () {
  let charts = [];
  const overlay = document.getElementById('report-overlay');
  const container = document.getElementById('report-content');

  // ── Chart.js global defaults for dark theme ────────────────────────────────
  Chart.defaults.color = '#999';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 11;

  // ── Open report ────────────────────────────────────────────────────────────
  function open(data) {
    destroyCharts();

    container.innerHTML = buildHTML(data);
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';

    // Render charts after DOM is ready
    requestAnimationFrame(() => {
      renderTimelineChart(data);
      renderPredDistanceChart(data);
      renderScatterChart(data);
    });
  }

  // ── Close report ───────────────────────────────────────────────────────────
  function close() {
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
    destroyCharts();
  }

  function destroyCharts() {
    charts.forEach(c => c.destroy());
    charts = [];
  }

  // ── Build report HTML ──────────────────────────────────────────────────────
  function buildHTML(d) {
    const analysisHtml = d.analysis
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    return `
      <div class="report-hero">
        <div class="report-user-id">#${d.uid} &middot; ${esc(d.label)}</div>
        <div class="report-meta">${esc(d.model_name)} &middot; ${d.centroid_lat}&deg;, ${d.centroid_lon}&deg; &middot; Spread: ${d.spread}</div>
      </div>

      <div class="report-stats-row">
        <div class="report-stat">
          <span class="report-stat-val" style="color:#60a5fa">${d.history_count}</span>
          <span class="report-stat-lbl">History</span>
        </div>
        <div class="report-stat">
          <span class="report-stat-val" style="color:#f59e0b">${d.ground_truth_count}</span>
          <span class="report-stat-lbl">Test Set</span>
        </div>
        <div class="report-stat">
          <span class="report-stat-val" style="color:#10b981">${d.prediction_count}</span>
          <span class="report-stat-lbl">Predictions</span>
        </div>
        <div class="report-stat">
          <span class="report-stat-val ${d.hits > 0 ? 'report-hit' : ''}">${d.hit_rate}</span>
          <span class="report-stat-lbl">Hit Rate</span>
        </div>
      </div>

      <div class="report-section">
        <div class="report-section-hdr">Analysis</div>
        <div class="report-analysis">${analysisHtml}</div>
      </div>

      <div class="report-section">
        <div class="report-section-hdr">Check-in Activity Over Time</div>
        <div class="report-chart-card report-chart-wide">
          <canvas id="report-chart-timeline"></canvas>
        </div>
      </div>

      <div class="report-charts-grid">
        <div class="report-section">
          <div class="report-section-hdr">Prediction Distance to Nearest GT</div>
          <div class="report-chart-card">
            <canvas id="report-chart-pred-dist"></canvas>
          </div>
        </div>
        <div class="report-section">
          <div class="report-section-hdr">Geographic Scatter</div>
          <div class="report-chart-card">
            <canvas id="report-chart-scatter"></canvas>
          </div>
        </div>
      </div>

      <div class="report-footer-text">
        Derive &middot; ${esc(d.model_name)} on Gowalla &middot; Mistral
      </div>
    `;
  }

  // ── Chart 1: Check-in Timeline (grouped bar) ──────────────────────────────
  function renderTimelineChart(data) {
    const ctx = document.getElementById('report-chart-timeline');
    if (!ctx) return;

    const tl = data.timeline;
    // Abbreviate month labels (2010-02 -> Feb '10)
    const labels = tl.labels.map(m => {
      const [y, mo] = m.split('-');
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      return months[parseInt(mo, 10) - 1] + " '" + y.slice(2);
    });

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'History',
            data: tl.history,
            backgroundColor: 'rgba(96,165,250,0.7)',
            borderColor: 'rgba(96,165,250,1)',
            borderWidth: 1,
            borderRadius: 2,
          },
          {
            label: 'Ground Truth',
            data: tl.ground_truth,
            backgroundColor: 'rgba(245,158,11,0.7)',
            borderColor: 'rgba(245,158,11,1)',
            borderWidth: 1,
            borderRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } },
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { maxRotation: 45, font: { size: 9 } },
          },
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { precision: 0 },
          },
        },
      },
    });
    charts.push(chart);
  }

  // ── Chart 2: Prediction Distance (horizontal bar) ─────────────────────────
  function renderPredDistanceChart(data) {
    const ctx = document.getElementById('report-chart-pred-dist');
    if (!ctx) return;

    const dists = data.pred_distances;
    if (!dists || dists.length === 0) {
      ctx.parentElement.innerHTML = '<div class="report-no-data">No predictions available</div>';
      return;
    }

    const labels = dists.map((d, i) => `Pred #${i + 1}`);
    const values = dists.map(d => d.distance_km != null ? d.distance_km : 0);
    const colors = dists.map(d => {
      if (d.is_hit) return 'rgba(16,185,129,0.8)';
      if (d.distance_km != null && d.distance_km < 5) return 'rgba(96,165,250,0.7)';
      if (d.distance_km != null && d.distance_km < 50) return 'rgba(245,158,11,0.7)';
      return 'rgba(239,68,68,0.6)';
    });

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Distance to nearest GT (km)',
          data: values,
          backgroundColor: colors,
          borderWidth: 0,
          borderRadius: 2,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const d = dists[ctx.dataIndex];
                let s = `${ctx.parsed.x.toFixed(1)} km`;
                if (d.is_hit) s += ' (HIT!)';
                return s;
              },
            },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: 'rgba(255,255,255,0.04)' },
            title: { display: true, text: 'Distance (km)', color: '#666' },
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 10 } },
          },
        },
      },
    });
    charts.push(chart);
  }

  // ── Chart 3: Geographic Scatter ────────────────────────────────────────────
  function renderScatterChart(data) {
    const ctx = document.getElementById('report-chart-scatter');
    if (!ctx) return;

    const sc = data.scatter;
    const chart = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'History',
            data: sc.history,
            backgroundColor: 'rgba(96,165,250,0.5)',
            pointRadius: 3,
            pointHoverRadius: 5,
          },
          {
            label: 'Ground Truth',
            data: sc.ground_truth,
            backgroundColor: 'rgba(245,158,11,0.8)',
            pointRadius: 5,
            pointStyle: 'rectRot',
            pointHoverRadius: 7,
          },
          {
            label: 'Predictions',
            data: sc.predictions,
            backgroundColor: 'rgba(16,185,129,0.8)',
            pointRadius: 5,
            pointStyle: 'triangle',
            pointHoverRadius: 7,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, padding: 16 } },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = ctx.parsed;
                return `${ctx.dataset.label}: (${p.y.toFixed(4)}, ${p.x.toFixed(4)})`;
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Longitude', color: '#666' },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
          y: {
            title: { display: true, text: 'Latitude', color: '#666' },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
        },
      },
    });
    charts.push(chart);
  }

  // ── Export PDF ──────────────────────────────────────────────────────────────
  async function exportPDF() {
    const btn = document.getElementById('report-export-btn');
    if (btn) {
      btn.textContent = 'Generating...';
      btn.disabled = true;
    }

    try {
      const target = document.querySelector('.report-container');
      const canvas = await html2canvas(target, {
        backgroundColor: '#0d0d14',
        scale: 2,
        useCORS: true,
        logging: false,
      });

      const imgData = canvas.toDataURL('image/png');
      const imgW = canvas.width;
      const imgH = canvas.height;

      // A4 landscape or portrait depending on aspect ratio
      const isLandscape = imgW > imgH;
      const pdf = new jspdf.jsPDF({
        orientation: isLandscape ? 'landscape' : 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      const pageW = pdf.internal.pageSize.getWidth();
      const pageH = pdf.internal.pageSize.getHeight();
      const margin = 8;
      const availW = pageW - margin * 2;
      const availH = pageH - margin * 2;

      const ratio = Math.min(availW / imgW, availH / imgH);
      const w = imgW * ratio;
      const h = imgH * ratio;
      const x = (pageW - w) / 2;
      const y = (pageH - h) / 2;

      pdf.addImage(imgData, 'PNG', x, y, w, h);
      pdf.save(`derive-report-user-${document.querySelector('.report-user-id')?.textContent?.match(/#(\d+)/)?.[1] || 'unknown'}.pdf`);
    } catch (err) {
      console.error('PDF export failed:', err);
      alert('PDF export failed. Check console for details.');
    } finally {
      if (btn) {
        btn.textContent = 'Export PDF';
        btn.disabled = false;
      }
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────
  function esc(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  window.deriveReport = { open, close, exportPDF };
})();
