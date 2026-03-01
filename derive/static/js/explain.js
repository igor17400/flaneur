/**
 * explain.js — Mistral streaming explanation panel.
 *
 * Exports (via window.deriveExplain):
 *   open(uid)   – open panel, start streaming explanation for user
 *   close()     – abort stream, close panel
 *   isOpen()    – whether panel is currently visible
 */

(function () {
  let eventSource = null;
  let panelOpen = false;
  let accumulated = '';

  function open(uid) {
    if (uid == null) return;

    const panel = document.getElementById('explain-panel');
    const body = document.getElementById('explain-body');
    const status = document.getElementById('explain-status');
    const userLabel = document.getElementById('explain-user');

    // Close any existing stream
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    accumulated = '';
    body.innerHTML = '<span class="explain-cursor"></span>';
    status.textContent = 'connecting...';
    status.className = 'explain-status connecting';
    userLabel.textContent = `User #${uid}`;

    // Slide panel in
    panel.classList.add('open');
    panelOpen = true;

    // Start SSE stream
    eventSource = new EventSource(`/api/explain/${uid}`);

    eventSource.onopen = function () {
      status.textContent = 'streaming';
      status.className = 'explain-status streaming';
    };

    eventSource.onmessage = function (e) {
      if (e.data === '[DONE]') {
        eventSource.close();
        eventSource = null;
        status.textContent = 'done';
        status.className = 'explain-status done';
        // Remove cursor
        const cursor = body.querySelector('.explain-cursor');
        if (cursor) cursor.remove();
        return;
      }

      try {
        const parsed = JSON.parse(e.data);

        if (parsed.error) {
          body.innerHTML = `<p class="explain-error">Error: ${escapeHtml(parsed.error)}</p>`;
          status.textContent = 'error';
          status.className = 'explain-status error';
          eventSource.close();
          eventSource = null;
          return;
        }

        if (parsed.token) {
          accumulated += parsed.token;
          renderMarkdown(body, accumulated);
        }
      } catch (err) {
        // Ignore parse errors for non-JSON lines
      }
    };

    eventSource.onerror = function () {
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      if (accumulated.length === 0) {
        body.innerHTML = '<p class="explain-error">Connection lost. Is the server running?</p>';
      }
      status.textContent = accumulated.length > 0 ? 'done' : 'error';
      status.className = accumulated.length > 0 ? 'explain-status done' : 'explain-status error';
      // Remove cursor on error too
      const cursor = body.querySelector('.explain-cursor');
      if (cursor) cursor.remove();
    };
  }

  function close() {
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    const panel = document.getElementById('explain-panel');
    panel.classList.remove('open');
    panelOpen = false;
  }

  function isOpen() {
    return panelOpen;
  }

  function showNoPredictions(uid) {
    const panel = document.getElementById('explain-panel');
    const body = document.getElementById('explain-body');
    const status = document.getElementById('explain-status');
    const userLabel = document.getElementById('explain-user');

    userLabel.textContent = `User #${uid}`;
    body.innerHTML = '<p class="explain-error">No model predictions available for this user. Run inference first to generate predictions, then try again.</p>';
    status.textContent = 'no data';
    status.className = 'explain-status error';

    panel.classList.add('open');
    panelOpen = true;
  }

  // ── Simple markdown rendering ─────────────────────────────────────────

  function renderMarkdown(container, text) {
    // Split into paragraphs, apply bold
    const paragraphs = text.split(/\n\n+/);
    let html = paragraphs
      .map((p) => {
        let s = escapeHtml(p.trim());
        // Bold: **text**
        s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        // Italic: *text*
        s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
        return s ? `<p>${s}</p>` : '';
      })
      .filter(Boolean)
      .join('');

    html += '<span class="explain-cursor"></span>';
    container.innerHTML = html;

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Public API
  window.deriveExplain = { open, close, isOpen, showNoPredictions };
})();
