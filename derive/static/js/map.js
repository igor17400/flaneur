/**
 * map.js — MapLibre + deck.gl initialization and layer rendering.
 *
 * Exports (via window.deriveMap):
 *   initMap()          – create MapLibre map
 *   renderLayers(...)  – update deck.gl overlay
 *   fitBounds(pts)     – fit map to array of {lat, lon}
 *   flyTo(lat, lon)    – animate to a point
 */

(function () {
  let map = null;
  let deckgl = null;

  function initMap() {
    map = new maplibregl.Map({
      container: 'map',
      style: {
        version: 8,
        sources: {
          'carto-dark': {
            type: 'raster',
            tiles: ['https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'],
            tileSize: 256,
            attribution: '&copy; CartoDB',
          },
        },
        layers: [
          {
            id: 'carto-dark-layer',
            type: 'raster',
            source: 'carto-dark',
            minzoom: 0,
            maxzoom: 20,
          },
        ],
      },
      center: [-40, 30],
      zoom: 2,
    });
    return map;
  }

  // ── Layer rendering ────────────────────────────────────────────────────

  function renderLayers(historyPoints, predPoints, showPredictions, highlightedIdx) {
    const allPts = [...historyPoints, ...(showPredictions ? predPoints : [])];

    // Path through history (clean line instead of arc spaghetti)
    const pathCoords = historyPoints.map((p) => [p.lon, p.lat]);

    // Highlighted point info
    const hlPt = highlightedIdx >= 0 ? allPts[highlightedIdx] : null;
    const hlIsHistory = highlightedIdx >= 0 && highlightedIdx < historyPoints.length;

    const layers = [
      // ── Path ──
      new deck.PathLayer({
        id: 'path',
        data: pathCoords.length > 1 ? [{ path: pathCoords }] : [],
        getPath: (d) => d.path,
        getColor: [96, 165, 250, 60],
        getWidth: 2,
        widthMinPixels: 1.5,
        widthMaxPixels: 3,
        jointRounded: true,
        capRounded: true,
      }),

      // ── History dots ──
      new deck.ScatterplotLayer({
        id: 'history',
        data: historyPoints,
        getPosition: (d) => [d.lon, d.lat],
        getFillColor: (d) => {
          const idx = historyPoints.indexOf(d);
          if (highlightedIdx >= 0 && idx === highlightedIdx) return [255, 255, 255, 255];
          return [96, 165, 250, 180];
        },
        getRadius: (d) => {
          const idx = historyPoints.indexOf(d);
          if (highlightedIdx >= 0 && idx === highlightedIdx) return 12;
          return 5;
        },
        radiusMinPixels: 3,
        radiusMaxPixels: 14,
        stroked: true,
        getLineColor: [96, 165, 250, 255],
        lineWidthMinPixels: 1,
        updateTriggers: { getFillColor: [highlightedIdx], getRadius: [highlightedIdx] },
      }),

      // ── History number labels ──
      new deck.TextLayer({
        id: 'history-labels',
        data: historyPoints,
        getPosition: (d) => [d.lon, d.lat],
        getText: (_d, { index }) => String(index + 1),
        getSize: 10,
        getColor: [255, 255, 255, 220],
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        fontFamily: 'JetBrains Mono, monospace',
        fontWeight: '600',
        outlineWidth: 3,
        outlineColor: [10, 10, 18, 200],
        getPixelOffset: [0, -14],
        sizeMinPixels: 8,
        sizeMaxPixels: 12,
      }),
    ];

    if (showPredictions && predPoints.length > 0) {
      layers.push(
        // Glow
        new deck.ScatterplotLayer({
          id: 'pred-glow',
          data: predPoints,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: [245, 158, 11, 50],
          getRadius: 18,
          radiusMinPixels: 8,
          radiusMaxPixels: 22,
        }),
        // Dots
        new deck.ScatterplotLayer({
          id: 'predictions',
          data: predPoints,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: (d) => {
            const gIdx = historyPoints.length + predPoints.indexOf(d);
            if (highlightedIdx >= 0 && gIdx === highlightedIdx) return [255, 255, 255, 255];
            return [245, 158, 11, 220];
          },
          getRadius: (d) => {
            const gIdx = historyPoints.length + predPoints.indexOf(d);
            if (highlightedIdx >= 0 && gIdx === highlightedIdx) return 12;
            return 6;
          },
          radiusMinPixels: 4,
          radiusMaxPixels: 14,
          stroked: true,
          getLineColor: [245, 158, 11, 255],
          lineWidthMinPixels: 1.5,
          updateTriggers: { getFillColor: [highlightedIdx], getRadius: [highlightedIdx] },
        }),
        // Labels
        new deck.TextLayer({
          id: 'pred-labels',
          data: predPoints,
          getPosition: (d) => [d.lon, d.lat],
          getText: (_d, { index }) => String(index + 1),
          getSize: 10,
          getColor: [245, 158, 11, 255],
          getTextAnchor: 'middle',
          getAlignmentBaseline: 'center',
          fontFamily: 'JetBrains Mono, monospace',
          fontWeight: '600',
          outlineWidth: 3,
          outlineColor: [10, 10, 18, 200],
          getPixelOffset: [0, -14],
          sizeMinPixels: 8,
          sizeMaxPixels: 12,
        })
      );

      // Arcs from last history point to each prediction
      if (historyPoints.length > 0) {
        const last = historyPoints[historyPoints.length - 1];
        layers.push(
          new deck.ArcLayer({
            id: 'pred-arcs',
            data: predPoints.map((p) => ({
              source: [last.lon, last.lat],
              target: [p.lon, p.lat],
            })),
            getSourcePosition: (d) => d.source,
            getTargetPosition: (d) => d.target,
            getSourceColor: [96, 165, 250, 30],
            getTargetColor: [245, 158, 11, 70],
            getWidth: 1,
            greatCircle: true,
          })
        );
      }
    }

    // Highlight ring
    if (hlPt) {
      layers.push(
        new deck.ScatterplotLayer({
          id: 'highlight-ring',
          data: [hlPt],
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: [0, 0, 0, 0],
          getRadius: 20,
          radiusMinPixels: 16,
          radiusMaxPixels: 28,
          stroked: true,
          getLineColor: hlIsHistory ? [96, 165, 250, 200] : [245, 158, 11, 200],
          lineWidthMinPixels: 2,
          updateTriggers: { getLineColor: [highlightedIdx] },
        })
      );
    }

    // Create or update overlay
    if (deckgl) {
      deckgl.setProps({ layers });
    } else {
      deckgl = new deck.MapboxOverlay({
        layers,
        getTooltip: ({ object }) => {
          if (!object || object.item_id === undefined) return null;
          return {
            html: `<div style="font-family:Inter,sans-serif;font-size:12px;padding:4px">
              <b>Item ${object.item_id}</b><br/>
              ${object.lat.toFixed(4)}, ${object.lon.toFixed(4)}<br/>
              <span style="color:#999">${object.ts ? object.ts.split('T')[0] : ''}</span>
            </div>`,
            style: {
              background: 'rgba(10,10,18,0.95)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '8px',
              color: '#e0e0e0',
            },
          };
        },
      });
      map.addControl(deckgl);
    }
  }

  function fitBounds(points) {
    if (!points || !points.length || !map) return;
    const lats = points.map((p) => p.lat);
    const lons = points.map((p) => p.lon);
    const pad = 0.05;
    map.fitBounds(
      [
        [Math.min(...lons) - pad, Math.min(...lats) - pad],
        [Math.max(...lons) + pad, Math.max(...lats) + pad],
      ],
      { padding: { top: 60, bottom: 60, left: 360, right: 60 }, duration: 1500, maxZoom: 14 }
    );
  }

  function flyTo(lat, lon) {
    if (!map) return;
    map.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 12), duration: 800 });
  }

  // Public API
  window.deriveMap = { initMap, renderLayers, fitBounds, flyTo };
})();
