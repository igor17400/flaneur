/**
 * timeline.js — Sidebar timeline and stats rendering.
 *
 * Exports (via window.deriveTimeline):
 *   renderTimeline(data)       – build the scrollable check-in list
 *   renderStats(data)          – update the stats grid
 *   renderRecentChips(recent, current, userData) – recent-user chips
 *   setHighlightCallbacks(onHighlight, onUnhighlight, onFlyTo)
 */

(function () {
  let _onHighlight = null;
  let _onUnhighlight = null;
  let _onFlyTo = null;
  let _onComparePair = null;
  let _lastCompareData = null; // { preds, gt, rows }

  function setHighlightCallbacks(onHighlight, onUnhighlight, onFlyTo, onComparePair) {
    _onHighlight = onHighlight;
    _onUnhighlight = onUnhighlight;
    _onFlyTo = onFlyTo;
    _onComparePair = onComparePair;
  }

  // ── Timeline ─────────────────────────────────────────────────────────

  function renderTimeline(data) {
    const tl = document.getElementById('timeline');
    const preds = data.predictions || [];
    let html = '<div class="tl-divider">History (train)</div>';

    data.history.forEach((pt, i) => {
      html += itemHTML(i, pt, 'hist', i);
    });

    html += '<div class="tl-divider">Ground Truth (test)</div>';
    const gtOffset = data.history.length;
    data.ground_truth.forEach((pt, i) => {
      html += itemHTML(gtOffset + i, pt, 'gt', i);
    });

    if (preds.length > 0) {
      const modelName = data.prediction_model || 'model';
      html += `<div class="tl-divider">Predictions (${modelName})</div>`;
      const predOffset = gtOffset + data.ground_truth.length;
      preds.forEach((pt, i) => {
        html += itemHTML(predOffset + i, pt, 'modelpred', i);
      });

      // ── Comparison table ──
      if (data.ground_truth.length > 0) {
        html += buildComparisonHTML(preds, data.ground_truth);
      }
    }

    tl.innerHTML = html;
  }

  // ── Haversine distance (km) ────────────────────────────────────────
  function haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const toRad = (d) => (d * Math.PI) / 180;
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  function formatDist(km) {
    if (km < 1) return `${Math.round(km * 1000)}m`;
    if (km < 100) return `${km.toFixed(1)}km`;
    return `${Math.round(km)}km`;
  }

  // ── Build comparison panel HTML ────────────────────────────────────
  function buildComparisonHTML(preds, gt) {
    const gtItemIds = new Set(gt.map((p) => p.item_id));
    const rows = preds.map((pred, i) => {
      const isHit = gtItemIds.has(pred.item_id);
      let nearestIdx = 0;
      let nearestDist = Infinity;
      gt.forEach((g, j) => {
        const d = haversineKm(pred.lat, pred.lon, g.lat, g.lon);
        if (d < nearestDist) {
          nearestDist = d;
          nearestIdx = j;
        }
      });
      return { predIdx: i, nearestGtIdx: nearestIdx, dist: nearestDist, isHit };
    });

    // Store for click handler
    _lastCompareData = { preds, gt, rows };

    const hits = rows.filter((r) => r.isHit).length;
    const avgDist = rows.reduce((s, r) => s + r.dist, 0) / rows.length;
    const minDist = Math.min(...rows.map((r) => r.dist));
    const maxDist = Math.max(...rows.map((r) => r.dist));

    let html = '<div class="tl-divider">Comparison <span class="tl-divider-hint">click a row to isolate</span></div>';
    html += `<div class="compare-summary">
      <div class="compare-stat">
        <span class="compare-val ${hits > 0 ? 'hit' : ''}">${hits}/${preds.length}</span>
        <span class="compare-lbl">Hits</span>
      </div>
      <div class="compare-stat">
        <span class="compare-val">${formatDist(avgDist)}</span>
        <span class="compare-lbl">Avg dist</span>
      </div>
      <div class="compare-stat">
        <span class="compare-val">${formatDist(minDist)}</span>
        <span class="compare-lbl">Closest</span>
      </div>
      <div class="compare-stat">
        <span class="compare-val">${formatDist(maxDist)}</span>
        <span class="compare-lbl">Farthest</span>
      </div>
    </div>`;

    // Sort by distance (closest first)
    const sorted = [...rows].sort((a, b) => a.dist - b.dist);

    html += '<div class="compare-table">';
    sorted.forEach((r) => {
      const distClass = r.dist < 50 ? 'near' : r.dist < 500 ? 'mid' : 'far';
      const predPt = preds[r.predIdx];
      const gtPt = gt[r.nearestGtIdx];
      html += `<div class="compare-row ${r.isHit ? 'hit' : ''} ${distClass}"
        onclick="deriveTimeline._comparePair(${r.predIdx}, ${r.nearestGtIdx})">
        <div class="compare-main">
          <div class="compare-pred">
            <span class="compare-dot pred"></span>
            P${r.predIdx + 1}
          </div>
          <div class="compare-arrow">&rarr;</div>
          <div class="compare-gt">
            <span class="compare-dot gt"></span>
            GT${r.nearestGtIdx + 1}
          </div>
          <div class="compare-dist">${formatDist(r.dist)}</div>
          ${r.isHit ? '<div class="compare-hit-badge">HIT</div>' : ''}
        </div>
        <div class="compare-coords">
          <span class="compare-coord-pred">${predPt.lat.toFixed(2)}, ${predPt.lon.toFixed(2)}</span>
          <span class="compare-coord-sep">&harr;</span>
          <span class="compare-coord-gt">${gtPt.lat.toFixed(2)}, ${gtPt.lon.toFixed(2)}</span>
        </div>
      </div>`;
    });
    html += '</div>';

    return html;
  }

  function _comparePair(predIdx, gtIdx) {
    if (!_lastCompareData || !_onComparePair) return;
    const { preds, gt, rows } = _lastCompareData;
    const predPt = preds[predIdx];
    const gtPt = gt[gtIdx];
    const dist = haversineKm(predPt.lat, predPt.lon, gtPt.lat, gtPt.lon);

    // Highlight the active row
    document.querySelectorAll('.compare-row').forEach((el) => el.classList.remove('active'));
    const clicked = document.querySelector(`.compare-row[onclick*="_comparePair(${predIdx}, ${gtIdx})"]`);
    if (clicked) clicked.classList.add('active');

    _onComparePair(predPt, gtPt, dist);
  }

  function itemHTML(globalIdx, pt, type, localIdx) {
    const date = pt.ts ? pt.ts.split('T')[0] : '\u2014';
    const coords = `${pt.lat.toFixed(4)}, ${pt.lon.toFixed(4)}`;
    return `<div class="tl-item ${type}"
      data-idx="${globalIdx}"
      onmouseenter="deriveTimeline._highlight(${globalIdx})"
      onmouseleave="deriveTimeline._unhighlight()"
      onclick="deriveTimeline._flyTo(${globalIdx})">
      <div class="tl-num ${type}">${localIdx + 1}</div>
      <div class="tl-info">
        <div class="tl-date">${date}</div>
        <div class="tl-coords">${coords}</div>
      </div>
      <div class="tl-item-id">item ${pt.item_id}</div>
    </div>`;
  }

  function highlightTimelineItem(idx) {
    document.querySelectorAll('.tl-item').forEach((el) => {
      el.classList.toggle('highlighted', parseInt(el.dataset.idx) === idx);
    });
  }

  function clearTimelineHighlight() {
    document.querySelectorAll('.tl-item').forEach((el) => el.classList.remove('highlighted'));
  }

  // Internal handlers called from inline HTML
  function _highlight(idx) {
    highlightTimelineItem(idx);
    if (_onHighlight) _onHighlight(idx);
  }

  function _unhighlight() {
    clearTimelineHighlight();
    if (_onUnhighlight) _onUnhighlight();
  }

  function _flyTo(idx) {
    if (_onFlyTo) _onFlyTo(idx);
  }

  // ── Stats ────────────────────────────────────────────────────────────

  function renderStats(data) {
    document.getElementById('stat-history').textContent = data.history.length;
    document.getElementById('stat-gt').textContent = data.ground_truth.length;
    const preds = data.predictions || [];
    document.getElementById('stat-pred').textContent = preds.length || '\u2014';
    const sp = data.spread;
    document.getElementById('stat-spread').textContent =
      sp > 100 ? 'Global' : sp > 10 ? 'Multi' : sp > 2 ? 'Regional' : sp > 0.5 ? 'City' : 'Local';
  }

  // ── Recent chips ─────────────────────────────────────────────────────

  function renderRecentChips(recentUsers, currentUser, userData) {
    const wrap = document.getElementById('recent-chips');
    wrap.innerHTML = recentUsers
      .map((uid) => {
        const d = userData[uid];
        const cls = uid == currentUser ? ' active' : '';
        const label = d ? d.label.split(' ')[0] : '';
        return `<span class="user-chip${cls}" onclick="deriveApp.selectUser(${uid})">#${uid} <span style="color:#666">${label}</span></span>`;
      })
      .join('');
  }

  // Public API
  window.deriveTimeline = {
    renderTimeline,
    renderStats,
    renderRecentChips,
    setHighlightCallbacks,
    highlightTimelineItem,
    clearTimelineHighlight,
    // internal (called from inline HTML)
    _highlight,
    _unhighlight,
    _flyTo,
    _comparePair,
  };
})();
