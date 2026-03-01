/**
 * app.js — Main application controller. Wires map, timeline, and animation together.
 * Supports switching between multiple prediction models and agentic chat.
 *
 * Exports (via window.deriveApp):
 *   selectUser(uid) – load and display a user
 *   loadAndSelectUser(uid) – fetch and display a user
 */

(function () {
  // ── State ──────────────────────────────────────────────────────────────
  let currentUser = null;
  let userData = {};
  let recentUsers = [];
  let highlightedIdx = -1;
  let currentModel = null;   // current model name (null = default)
  let availableModels = [];   // [{name, embed_dim, ...}]

  // ── Init ───────────────────────────────────────────────────────────────
  const mapInstance = deriveMap.initMap();

  // Wire up timeline hover ↔ map highlight
  deriveTimeline.setHighlightCallbacks(
    // onHighlight
    (idx) => {
      highlightedIdx = idx;
      rerenderMap();
    },
    // onUnhighlight
    () => {
      highlightedIdx = -1;
      rerenderMap();
    },
    // onFlyTo
    (idx) => {
      const data = userData[currentUser];
      if (!data) return;
      const all = [...data.history, ...data.ground_truth, ...(data.predictions || [])];
      const pt = all[idx];
      if (pt) deriveMap.flyTo(pt.lat, pt.lon);
    },
    // onComparePair
    (predPt, gtPt, distKm) => {
      deriveMap.renderComparePair(predPt, gtPt, distKm);
    }
  );

  // ── Model selector ──────────────────────────────────────────────────────
  const modelSelect = document.getElementById('model-select');
  const modelInfo = document.getElementById('model-info');

  async function loadModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      availableModels = data.models || [];
      currentModel = data.default || null;

      modelSelect.innerHTML = '';
      availableModels.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m.name;
        // Build a readable label
        const recall = m.val_recall_at_20 != null
          ? ` (R@20: ${(m.val_recall_at_20 * 100).toFixed(1)}%)`
          : ' (untrained)';
        opt.textContent = m.name + recall;
        if (m.name === currentModel) opt.selected = true;
        modelSelect.appendChild(opt);
      });

      updateModelInfo();
    } catch (e) {
      modelSelect.innerHTML = '<option value="">No models</option>';
    }
  }

  function updateModelInfo() {
    const meta = availableModels.find((m) => m.name === currentModel);
    if (!meta) {
      modelInfo.textContent = '';
      return;
    }
    const parts = [];
    if (meta.embed_dim) parts.push(`dim=${meta.embed_dim}`);
    if (meta.n_layers) parts.push(`layers=${meta.n_layers}`);
    if (meta.lr) parts.push(`lr=${meta.lr}`);
    if (meta.reg_weight) parts.push(`reg=${meta.reg_weight}`);
    modelInfo.textContent = parts.join(' · ');
  }

  modelSelect.addEventListener('change', () => {
    currentModel = modelSelect.value;
    updateModelInfo();
    // Clear cached user data (predictions depend on model)
    userData = {};
    // Reload current user with new model
    if (currentUser != null) {
      loadAndSelectUser(currentUser);
    }
  });

  // ── Search ─────────────────────────────────────────────────────────────
  const searchInput = document.getElementById('search-input');
  const searchError = document.getElementById('search-error');
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchUser();
  });

  function showError(msg) {
    searchError.textContent = msg;
    searchError.classList.toggle('visible', !!msg);
  }

  async function searchUser() {
    const raw = searchInput.value.trim();
    if (!raw) return;
    const uid = parseInt(raw);
    if (isNaN(uid)) {
      showError('Enter a numeric user ID');
      return;
    }
    await loadAndSelectUser(uid);
  }

  function _apiUrl(path) {
    if (currentModel) {
      const sep = path.includes('?') ? '&' : '?';
      return path + sep + 'model=' + encodeURIComponent(currentModel);
    }
    return path;
  }

  async function randomUser() {
    showError('');
    try {
      const res = await fetch(_apiUrl('/api/random'));
      const data = await res.json();
      searchInput.value = data.uid;
      userData[data.uid] = data;
      addToRecent(data.uid);
      selectUser(data.uid);
    } catch (e) {
      showError('Server not running? python3 derive/server.py');
    }
  }

  async function loadAndSelectUser(uid) {
    showError('');
    if (userData[uid]) {
      addToRecent(uid);
      selectUser(uid);
      return;
    }
    try {
      const res = await fetch(_apiUrl(`/api/user/${uid}`));
      if (!res.ok) {
        const err = await res.json();
        showError(err.error || 'Not found');
        return;
      }
      userData[uid] = await res.json();
      addToRecent(uid);
      selectUser(uid);
    } catch (e) {
      showError('Server not running? python3 derive/server.py');
    }
  }

  // ── Recent ─────────────────────────────────────────────────────────────
  function addToRecent(uid) {
    recentUsers = recentUsers.filter((u) => u !== uid);
    recentUsers.unshift(uid);
    if (recentUsers.length > 12) recentUsers.pop();
    deriveTimeline.renderRecentChips(recentUsers, currentUser, userData);
  }

  // ── Select user ────────────────────────────────────────────────────────
  function selectUser(uid) {
    if (deriveAnimation.isAnimating()) deriveAnimation.toggle(userData, currentUser, renderMapLayers);

    currentUser = uid;
    highlightedIdx = -1;
    const data = userData[uid];
    if (!data) return;

    // Update chat panel's user label if open (don't close it)
    const chatUser = document.getElementById('chat-user');
    if (chatUser) chatUser.textContent = `User #${uid}`;

    deriveTimeline.renderRecentChips(recentUsers, currentUser, userData);
    deriveTimeline.renderTimeline(data);
    deriveTimeline.renderStats(data);

    showAll();
    setTimeout(() => {
      const all = [...data.history, ...data.ground_truth, ...(data.predictions || [])];
      deriveMap.fitBounds(all);
    }, 100);
  }

  // ── Map rendering ──────────────────────────────────────────────────────
  function rerenderMap() {
    const data = userData[currentUser];
    if (!data) return;
    deriveMap.renderLayers(
      data.history, data.ground_truth, data.predictions || [],
      true, true, highlightedIdx
    );
  }

  function renderMapLayers(history, groundTruth, showGt, showPred) {
    const data = userData[currentUser];
    const preds = data ? (data.predictions || []) : [];
    deriveMap.renderLayers(
      history, groundTruth, preds,
      showGt !== false, showPred !== false, highlightedIdx
    );
  }

  function showAll() {
    if (deriveAnimation.isAnimating()) deriveAnimation.toggle(userData, currentUser, renderMapLayers);
    document.getElementById('progress').classList.remove('visible');
    // Clear any active comparison row highlight
    document.querySelectorAll('.compare-row.active').forEach((el) => el.classList.remove('active'));
    // Turn off all heatmaps if active
    if (deriveMap.isHeatmapActive()) {
      deriveMap.clearAllHeatmaps();
      Object.values(HEATMAP_BUTTON_IDS).forEach(id => {
        const b = document.getElementById(id);
        if (b) b.classList.remove('active');
      });
    }
    rerenderMap();
  }

  function fitBounds() {
    const data = userData[currentUser];
    if (!data) return;
    deriveMap.fitBounds([...data.history, ...data.ground_truth]);
  }

  function toggleAnimation() {
    deriveAnimation.toggle(userData, currentUser, renderMapLayers);
  }

  // ── Boot ───────────────────────────────────────────────────────────────
  mapInstance.on('load', async () => {
    await loadModels();
    randomUser();
  });

  // ── Chat panel ─────────────────────────────────────────────────────────
  function explainUser() {
    if (currentUser == null) return;
    deriveChat.open(currentUser);
  }

  function closeChatPanel() {
    deriveChat.close();
  }

  // ── Report (direct button) ────────────────────────────────────────────
  async function openReport() {
    if (currentUser == null) return;
    try {
      const res = await fetch(`/api/report/${currentUser}`);
      if (!res.ok) return;
      const data = await res.json();
      deriveReport.open(data);
    } catch (e) {
      console.error('Failed to load report:', e);
    }
  }

  // ── Leaderboard ───────────────────────────────────────────────────────
  function openLeaderboard() {
    deriveLeaderboard.open();
  }

  // ── Compare ───────────────────────────────────────────────────────────
  function openCompare() {
    if (currentUser == null) return;
    deriveCompare.open(currentUser);
  }

  // ── Per-category heatmaps ────────────────────────────────────────────
  const HEATMAP_BUTTON_IDS = {
    history: 'btn-heatmap-history',
    ground_truth: 'btn-heatmap-gt',
    predictions: 'btn-heatmap-pred',
  };

  function toggleCategoryHeatmap(category) {
    const btn = document.getElementById(HEATMAP_BUTTON_IDS[category]);
    if (!btn) return;
    const data = userData[currentUser];
    if (!data) return;

    // Toggle off
    if (btn.classList.contains('active')) {
      deriveMap.clearCategoryHeatmap(category);
      btn.classList.remove('active');
      return;
    }

    // Toggle on — get the right point array
    const points = data[category] || [];
    if (!points.length) return;

    deriveMap.renderCategoryHeatmap(category, points);
    btn.classList.add('active');
  }

  // Expose globals for HTML onclick handlers
  window.deriveApp = { selectUser, loadAndSelectUser };
  window.searchUser = searchUser;
  window.randomUser = randomUser;
  window.showAll = showAll;
  window.fitBounds = fitBounds;
  window.toggleAnimation = toggleAnimation;
  window.explainUser = explainUser;
  window.closeChatPanel = closeChatPanel;
  window.openReport = openReport;
  window.openLeaderboard = openLeaderboard;
  window.openCompare = openCompare;
  window.toggleCategoryHeatmap = toggleCategoryHeatmap;
})();
