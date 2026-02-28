/**
 * animation.js — Timeline animation controller.
 *
 * Exports (via window.deriveAnimation):
 *   toggle(userData, currentUser, renderFn) – start / stop animation
 *   isAnimating()                           – current state
 */

(function () {
  const DURATION_MS = 6000;
  let animating = false;
  let animFrame = null;

  function toggle(userData, currentUser, renderFn) {
    const btn = document.getElementById('btn-animate');
    const prog = document.getElementById('progress');

    if (animating) {
      animating = false;
      btn.textContent = 'Animate';
      btn.classList.remove('active');
      prog.classList.remove('visible');
      if (animFrame) cancelAnimationFrame(animFrame);
      return;
    }

    const data = userData[currentUser];
    if (!data) return;

    animating = true;
    btn.textContent = 'Stop';
    btn.classList.add('active');
    prog.classList.add('visible');

    const t0 = performance.now();
    const total = data.history.length;

    (function step(now) {
      if (!animating) return;

      const pct = Math.min((now - t0) / DURATION_MS, 1.0);
      const n = Math.floor(pct * total);
      const visible = data.history.slice(0, n + 1);
      const showPred = pct >= 1.0;

      renderFn(visible, showPred ? data.ground_truth : [], showPred);

      // Update progress UI
      document.getElementById('progress-fill').style.width = pct * 100 + '%';
      if (visible.length) {
        const last = visible[visible.length - 1];
        document.getElementById('progress-date').textContent = last.ts?.split('T')[0] || '';
        document.getElementById('progress-count').textContent = `${visible.length} / ${total}`;
      }

      if (pct < 1.0) {
        animFrame = requestAnimationFrame(step);
      } else {
        // Reveal predictions after a beat
        setTimeout(() => {
          renderFn(data.history, data.ground_truth, true);
          document.getElementById('progress-date').textContent = 'Predictions revealed';
          document.getElementById('progress-count').textContent = `${data.ground_truth.length} predicted`;
        }, 300);
        animating = false;
        btn.textContent = 'Replay';
        btn.classList.remove('active');
      }
    })(performance.now());
  }

  function isAnimating() {
    return animating;
  }

  window.deriveAnimation = { toggle, isAnimating };
})();
