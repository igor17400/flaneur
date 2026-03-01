/**
 * chat.js — Agentic chat panel with Mistral function calling.
 *
 * Exports (via window.deriveChat):
 *   open(uid)   – slide panel in, start auto-analysis
 *   close()     – abort and close
 *   send()      – send user message
 *   isOpen()    – whether panel is visible
 */

(function () {
  let panelOpen = false;
  let abortController = null;
  let conversationHistory = []; // {role, content} for multi-turn
  let currentUid = null;
  const badgeTimestamps = {}; // call_id → creation time (for min display)

  // ── DOM refs ─────────────────────────────────────────────────────────────
  const panel = document.getElementById('chat-panel');
  const messagesEl = document.getElementById('chat-messages');
  const inputEl = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const userLabel = document.getElementById('chat-user');
  const statusEl = document.getElementById('chat-status');

  sendBtn.addEventListener('click', send);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // ── Open panel ───────────────────────────────────────────────────────────
  function open(uid) {
    if (uid == null) return;
    currentUid = uid;
    conversationHistory = [];

    messagesEl.innerHTML = '';
    userLabel.textContent = `User #${uid}`;
    statusEl.textContent = 'connecting';
    statusEl.className = 'chat-status connecting';

    panel.classList.add('open');
    panelOpen = true;

    const autoPrompt = `Analyze user ${uid}. Look up their data, show them on the map, and give a concise geographic analysis with a user_card widget.`;
    executeChat(autoPrompt);
  }

  // ── Close panel ──────────────────────────────────────────────────────────
  function close() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    panel.classList.remove('open');
    panelOpen = false;
  }

  function isOpen() {
    return panelOpen;
  }

  // ── Send message ─────────────────────────────────────────────────────────
  function send() {
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = '';

    appendMessage('user', text);
    executeChat(text);
  }

  // ── Execute chat (POST + SSE reader) ─────────────────────────────────────
  async function executeChat(userMessage) {
    conversationHistory.push({ role: 'user', content: userMessage });

    const assistantEl = appendMessage('assistant', '');
    const toolsContainer = document.createElement('div');
    toolsContainer.className = 'chat-tools';
    assistantEl.insertBefore(toolsContainer, assistantEl.firstChild);

    const textEl = assistantEl.querySelector('.chat-text');

    statusEl.textContent = 'thinking';
    statusEl.className = 'chat-status streaming';

    inputEl.disabled = true;
    sendBtn.disabled = true;

    if (abortController) abortController.abort();
    abortController = new AbortController();

    let accumulated = '';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: conversationHistory,
          current_user: currentUid,
        }),
        signal: abortController.signal,
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamDone = false;

      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let evt;
          try {
            evt = JSON.parse(raw);
          } catch {
            continue;
          }

          handleSSEEvent(evt, toolsContainer, assistantEl);

          if (evt.type === 'token') {
            accumulated += evt.content;
            renderMarkdown(textEl, accumulated);
            scrollToBottom();
          }

          if (evt.type === 'done') {
            if (accumulated) {
              conversationHistory.push({ role: 'assistant', content: accumulated });
            }
            streamDone = true;
            break;
          }

          if (evt.type === 'error') {
            textEl.innerHTML = `<p class="chat-error">Error: ${escapeHtml(evt.message || 'Unknown error')}</p>`;
            statusEl.textContent = 'error';
            statusEl.className = 'chat-status error';
            streamDone = true;
            break;
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        textEl.innerHTML = `<p class="chat-error">Connection lost. Is the server running?</p>`;
        statusEl.textContent = 'error';
        statusEl.className = 'chat-status error';
      }
    } finally {
      inputEl.disabled = false;
      sendBtn.disabled = false;
      inputEl.focus();
      statusEl.textContent = 'ready';
      statusEl.className = 'chat-status done';
      const cursor = textEl.querySelector('.chat-cursor');
      if (cursor) cursor.remove();
    }
  }

  // ── Handle SSE events ────────────────────────────────────────────────────
  function handleSSEEvent(evt, toolsContainer, bubbleEl) {
    if (evt.type === 'tool_call') {
      const badge = document.createElement('div');
      badge.className = 'tool-badge running';
      badge.id = `tool-${evt.call_id}`;
      badge.innerHTML = `<span class="tool-spinner"></span> <strong>${escapeHtml(evt.name)}</strong>`;
      toolsContainer.appendChild(badge);
      badgeTimestamps[evt.call_id] = Date.now();
      scrollToBottom();
    }

    if (evt.type === 'tool_result') {
      const callId = evt.call_id;
      const badge = document.getElementById(`tool-${callId}`);
      if (badge) {
        const minDelay = 600;
        const elapsed = Date.now() - (badgeTimestamps[callId] || 0);
        const remaining = Math.max(0, minDelay - elapsed);

        setTimeout(() => {
          const isErr = evt.error;
          badge.className = isErr ? 'tool-badge error' : 'tool-badge done';
          const icon = isErr ? '&#x2718;' : '&#x2714;';
          badge.innerHTML = `<span class="tool-icon">${icon}</span> <strong>${escapeHtml(evt.name)}</strong> ${escapeHtml(evt.summary || '')}`;
        }, remaining);
      }
      scrollToBottom();
    }

    if (evt.type === 'action') {
      dispatchMapAction(evt);
    }

    if (evt.type === 'widget') {
      renderWidget(evt, bubbleEl);
    }

    if (evt.type === 'report') {
      deriveReport.open(evt.data);
    }
  }

  // ── Widget rendering ─────────────────────────────────────────────────────
  function renderWidget(evt, bubbleEl) {
    const container = document.createElement('div');

    if (evt.widget_type === 'user_card') {
      const d = evt.data;
      container.className = 'widget widget-user-card';
      container.innerHTML = `
        <div class="widget-header">
          <span class="widget-emoji">&#x1F4CD;</span>
          User #${d.uid} &middot; ${escapeHtml(d.label)}
        </div>
        <div class="widget-grid">
          <div class="widget-stat">
            <span class="widget-val" style="color:#60a5fa">${d.history_count}</span>
            <span class="widget-lbl">History</span>
          </div>
          <div class="widget-stat">
            <span class="widget-val" style="color:#f59e0b">${d.ground_truth_count}</span>
            <span class="widget-lbl">Test</span>
          </div>
          <div class="widget-stat">
            <span class="widget-val" style="color:#10b981">${d.prediction_count}</span>
            <span class="widget-lbl">Preds</span>
          </div>
          <div class="widget-stat">
            <span class="widget-val ${d.hits > 0 ? 'hit' : ''}">${d.hit_rate}</span>
            <span class="widget-lbl">Hits</span>
          </div>
        </div>
        <div class="widget-detail">
          &#x1F4CC; ${d.centroid_lat}&deg;, ${d.centroid_lon}&deg; &middot; Spread: ${d.spread}
        </div>
      `;
    } else if (evt.widget_type === 'comparison') {
      const a = evt.data.user_a;
      const b = evt.data.user_b;
      container.className = 'widget widget-comparison';
      container.innerHTML = `
        <div class="widget-header">
          <span class="widget-emoji">&#x2696;</span>
          #${a.uid} vs #${b.uid}
        </div>
        <table class="widget-compare-table">
          <tr><th></th><th>#${a.uid}</th><th>#${b.uid}</th></tr>
          <tr><td>Label</td><td>${escapeHtml(a.label)}</td><td>${escapeHtml(b.label)}</td></tr>
          <tr><td>History</td><td>${a.history_count}</td><td>${b.history_count}</td></tr>
          <tr><td>Test</td><td>${a.ground_truth_count}</td><td>${b.ground_truth_count}</td></tr>
          <tr><td>Predictions</td><td>${a.prediction_count}</td><td>${b.prediction_count}</td></tr>
          <tr><td>Hits</td><td class="${a.hits > 0 ? 'hit' : ''}">${a.hit_rate}</td><td class="${b.hits > 0 ? 'hit' : ''}">${b.hit_rate}</td></tr>
          <tr><td>Spread</td><td>${a.spread}</td><td>${b.spread}</td></tr>
        </table>
      `;
    } else if (evt.widget_type === 'insight') {
      const d = evt.data;
      container.className = 'widget widget-insight';
      container.innerHTML = `
        <div class="widget-header">
          <span class="widget-emoji">&#x1F4A1;</span>
          ${escapeHtml(d.title)}
        </div>
        <div class="widget-insight-text">${escapeHtml(d.content)}</div>
      `;
    }

    if (container.className) {
      // Insert widget before the text element
      const textEl = bubbleEl.querySelector('.chat-text');
      bubbleEl.insertBefore(container, textEl);
      scrollToBottom();
    }
  }

  // ── Map action dispatch ──────────────────────────────────────────────────
  function dispatchMapAction(evt) {
    if (evt.name === 'fly_to' && evt.lat != null && evt.lon != null) {
      deriveMap.flyTo(evt.lat, evt.lon);
    } else if (evt.name === 'select_user' && evt.uid != null) {
      if (window.deriveApp && window.deriveApp.loadAndSelectUser) {
        window.deriveApp.loadAndSelectUser(evt.uid);
        currentUid = evt.uid;
        userLabel.textContent = `User #${evt.uid}`;
      }
    } else if (evt.name === 'fit_bounds') {
      if (window.fitBounds) window.fitBounds();
    }
  }

  // ── DOM helpers ──────────────────────────────────────────────────────────
  function appendMessage(role, text) {
    const wrap = document.createElement('div');
    wrap.className = `chat-msg chat-msg-${role}`;

    const bubble = document.createElement('div');
    bubble.className = `chat-bubble chat-bubble-${role}`;

    const textSpan = document.createElement('div');
    textSpan.className = 'chat-text';
    if (text) {
      renderMarkdown(textSpan, text);
    } else {
      textSpan.innerHTML = '<span class="chat-cursor"></span>';
    }
    bubble.appendChild(textSpan);
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return bubble;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ── Markdown rendering ───────────────────────────────────────────────────
  function renderMarkdown(container, text) {
    const paragraphs = text.split(/\n\n+/);
    let html = paragraphs
      .map((p) => {
        let s = escapeHtml(p.trim());
        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
        s = s.replace(/`(.+?)`/g, '<code>$1</code>');
        return s ? `<p>${s}</p>` : '';
      })
      .filter(Boolean)
      .join('');

    html += '<span class="chat-cursor"></span>';
    container.innerHTML = html;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ── Public API ───────────────────────────────────────────────────────────
  window.deriveChat = { open, close, send, isOpen };
})();
