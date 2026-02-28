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

  function setHighlightCallbacks(onHighlight, onUnhighlight, onFlyTo) {
    _onHighlight = onHighlight;
    _onUnhighlight = onUnhighlight;
    _onFlyTo = onFlyTo;
  }

  // ── Timeline ─────────────────────────────────────────────────────────

  function renderTimeline(data) {
    const tl = document.getElementById('timeline');
    let html = '<div class="tl-divider">History (train)</div>';

    data.history.forEach((pt, i) => {
      html += itemHTML(i, pt, 'hist', i);
    });

    html += '<div class="tl-divider">Ground Truth (test)</div>';
    data.ground_truth.forEach((pt, i) => {
      html += itemHTML(data.history.length + i, pt, 'pred', i);
    });

    tl.innerHTML = html;
  }

  function itemHTML(globalIdx, pt, type, localIdx) {
    const date = pt.ts ? pt.ts.split('T')[0] : '\u2014';
    const coords = `${pt.lat.toFixed(4)}, ${pt.lon.toFixed(4)}`;
    return `<div class="tl-item ${type === 'pred' ? 'pred' : ''}"
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
    document.getElementById('stat-pred').textContent = data.ground_truth.length;
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
  };
})();
