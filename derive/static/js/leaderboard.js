/**
 * leaderboard.js — Leaderboard overlay showing ranked users.
 *
 * Exports (via window.deriveLeaderboard):
 *   open()   – fetch leaderboard data and show overlay
 *   close()  – hide overlay
 */

(function () {
  const overlay = document.getElementById('leaderboard-overlay');
  const content = document.getElementById('leaderboard-content');
  let activeTab = 'best_hit_rate';
  let cachedData = null;

  async function open() {
    overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';

    if (cachedData) {
      render(cachedData);
      return;
    }

    content.innerHTML = '<div class="lb-loading">Loading leaderboard...</div>';

    try {
      const res = await fetch('/api/leaderboard');
      cachedData = await res.json();
      render(cachedData);
    } catch (e) {
      content.innerHTML = '<div class="lb-loading">Failed to load leaderboard.</div>';
    }
  }

  function close() {
    overlay.classList.remove('visible');
    document.body.style.overflow = '';
  }

  function switchTab(tab) {
    activeTab = tab;
    if (cachedData) render(cachedData);
  }

  function render(data) {
    const tabs = [
      { key: 'best_hit_rate', label: 'Best Hit Rate', icon: '&#x1F3AF;' },
      { key: 'most_checkins', label: 'Most Check-ins', icon: '&#x1F4CD;' },
      { key: 'globetrotters', label: 'Globetrotters', icon: '&#x1F30D;' },
    ];

    const tabsHtml = tabs.map(t =>
      `<button class="lb-tab ${t.key === activeTab ? 'active' : ''}" onclick="deriveLeaderboard.switchTab('${t.key}')">
        <span>${t.icon}</span> ${t.label}
      </button>`
    ).join('');

    const users = data[activeTab] || [];
    const rowsHtml = users.map((u, i) => {
      const medal = i === 0 ? '&#x1F947;' : i === 1 ? '&#x1F948;' : i === 2 ? '&#x1F949;' : `<span class="lb-rank">${i + 1}</span>`;
      const statVal = activeTab === 'best_hit_rate'
        ? `<span style="color:#10b981">${u.hit_rate}</span> hits`
        : activeTab === 'most_checkins'
          ? `<span style="color:#60a5fa">${u.history_count}</span> check-ins`
          : `<span style="color:#a78bfa">${u.spread}</span> spread`;

      return `
        <div class="lb-row" onclick="deriveLeaderboard.selectUser(${u.uid})">
          <div class="lb-medal">${medal}</div>
          <div class="lb-user-info">
            <div class="lb-user-name">#${u.uid} &middot; ${esc(u.label)}</div>
            <div class="lb-user-stat">${statVal} &middot; ${u.centroid_lat}&deg;, ${u.centroid_lon}&deg;</div>
          </div>
          <div class="lb-user-counts">
            <span style="color:#60a5fa">${u.history_count}</span> /
            <span style="color:#f59e0b">${u.ground_truth_count}</span> /
            <span style="color:#10b981">${u.prediction_count}</span>
          </div>
        </div>
      `;
    }).join('');

    content.innerHTML = `
      <div class="lb-tabs">${tabsHtml}</div>
      <div class="lb-header-row">
        <span>User</span>
        <span>H / T / P</span>
      </div>
      <div class="lb-list">${rowsHtml}</div>
    `;
  }

  function selectUser(uid) {
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

  window.deriveLeaderboard = { open, close, switchTab, selectUser };
})();
