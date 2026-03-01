/**
 * app.js — Main application controller. Wires map, timeline, and animation together.
 * Supports switching between multiple prediction models.
 *
 * Exports (via window.deriveApp):
 *   selectUser(uid) – load and display a user
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

    // Close explain panel when switching users
    if (deriveExplain.isOpen()) deriveExplain.close();

    currentUser = uid;
    highlightedIdx = -1;
    const data = userData[uid];
    if (!data) return;

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

  // ── Explain panel ──────────────────────────────────────────────────────
  function explainUser() {
    if (currentUser == null) return;
    const data = userData[currentUser];
    if (!data || !data.predictions || data.predictions.length === 0) {
      deriveExplain.showNoPredictions(currentUser);
      return;
    }
    deriveExplain.open(currentUser);
  }

  function closeExplainPanel() {
    deriveExplain.close();
  }

  // Expose globals for HTML onclick handlers
  window.deriveApp = { selectUser };
  window.searchUser = searchUser;
  window.randomUser = randomUser;
  window.showAll = showAll;
  window.fitBounds = fitBounds;
  window.toggleAnimation = toggleAnimation;
  window.explainUser = explainUser;
  window.closeExplainPanel = closeExplainPanel;
})();
