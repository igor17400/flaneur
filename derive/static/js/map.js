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

  function renderLayers(historyPoints, gtPoints, predPoints, showGt, showPred, highlightedIdx) {
    const visibleGt = showGt ? gtPoints : [];
    const visiblePred = showPred ? predPoints : [];
    const allPts = [...historyPoints, ...visibleGt, ...visiblePred];

    // Path through history (clean line instead of arc spaghetti)
    const pathCoords = historyPoints.map((p) => [p.lon, p.lat]);

    // Highlighted point info
    const hlPt = highlightedIdx >= 0 ? allPts[highlightedIdx] : null;
    const hlIsHistory = highlightedIdx >= 0 && highlightedIdx < historyPoints.length;
    const hlIsGt = highlightedIdx >= historyPoints.length && highlightedIdx < historyPoints.length + visibleGt.length;

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

    // ── Ground truth (test set) — orange ──
    if (visibleGt.length > 0) {
      const gtOffset = historyPoints.length;
      layers.push(
        new deck.ScatterplotLayer({
          id: 'gt-glow',
          data: visibleGt,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: [245, 158, 11, 50],
          getRadius: 18,
          radiusMinPixels: 8,
          radiusMaxPixels: 22,
        }),
        new deck.ScatterplotLayer({
          id: 'gt-dots',
          data: visibleGt,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: (d) => {
            const gIdx = gtOffset + visibleGt.indexOf(d);
            if (highlightedIdx >= 0 && gIdx === highlightedIdx) return [255, 255, 255, 255];
            return [245, 158, 11, 220];
          },
          getRadius: (d) => {
            const gIdx = gtOffset + visibleGt.indexOf(d);
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
        new deck.TextLayer({
          id: 'gt-labels',
          data: visibleGt,
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

      if (historyPoints.length > 0) {
        const last = historyPoints[historyPoints.length - 1];
        layers.push(
          new deck.ArcLayer({
            id: 'gt-arcs',
            data: visibleGt.map((p) => ({
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

    // ── Model predictions — green ──
    if (visiblePred.length > 0) {
      const predOffset = historyPoints.length + visibleGt.length;
      layers.push(
        new deck.ScatterplotLayer({
          id: 'pred-glow',
          data: visiblePred,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: [16, 185, 129, 40],
          getRadius: 18,
          radiusMinPixels: 8,
          radiusMaxPixels: 22,
        }),
        new deck.ScatterplotLayer({
          id: 'pred-dots',
          data: visiblePred,
          getPosition: (d) => [d.lon, d.lat],
          getFillColor: (d) => {
            const gIdx = predOffset + visiblePred.indexOf(d);
            if (highlightedIdx >= 0 && gIdx === highlightedIdx) return [255, 255, 255, 255];
            return [16, 185, 129, 220];
          },
          getRadius: (d) => {
            const gIdx = predOffset + visiblePred.indexOf(d);
            if (highlightedIdx >= 0 && gIdx === highlightedIdx) return 12;
            return 6;
          },
          radiusMinPixels: 4,
          radiusMaxPixels: 14,
          stroked: true,
          getLineColor: [16, 185, 129, 255],
          lineWidthMinPixels: 1.5,
          updateTriggers: { getFillColor: [highlightedIdx], getRadius: [highlightedIdx] },
        }),
        new deck.TextLayer({
          id: 'pred-labels',
          data: visiblePred,
          getPosition: (d) => [d.lon, d.lat],
          getText: (_d, { index }) => String(index + 1),
          getSize: 10,
          getColor: [16, 185, 129, 255],
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

      if (historyPoints.length > 0) {
        const last = historyPoints[historyPoints.length - 1];
        layers.push(
          new deck.ArcLayer({
            id: 'pred-arcs',
            data: visiblePred.map((p) => ({
              source: [last.lon, last.lat],
              target: [p.lon, p.lat],
            })),
            getSourcePosition: (d) => d.source,
            getTargetPosition: (d) => d.target,
            getSourceColor: [96, 165, 250, 30],
            getTargetColor: [16, 185, 129, 70],
            getWidth: 1,
            greatCircle: true,
          })
        );
      }
    }

    // Highlight ring
    if (hlPt) {
      const hlColor = hlIsHistory
        ? [96, 165, 250, 200]
        : hlIsGt
          ? [245, 158, 11, 200]
          : [16, 185, 129, 200];
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
          getLineColor: hlColor,
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

  // ── Compare-pair view ──────────────────────────────────────────────────
  function renderComparePair(predPt, gtPt, distKm) {
    const layers = [
      // Connecting line
      new deck.ArcLayer({
        id: 'compare-arc',
        data: [{ source: [predPt.lon, predPt.lat], target: [gtPt.lon, gtPt.lat] }],
        getSourcePosition: (d) => d.source,
        getTargetPosition: (d) => d.target,
        getSourceColor: [16, 185, 129, 180],
        getTargetColor: [245, 158, 11, 180],
        getWidth: 3,
        greatCircle: true,
      }),

      // GT glow + dot
      new deck.ScatterplotLayer({
        id: 'compare-gt-glow',
        data: [gtPt],
        getPosition: (d) => [d.lon, d.lat],
        getFillColor: [245, 158, 11, 50],
        getRadius: 24,
        radiusMinPixels: 12,
        radiusMaxPixels: 30,
      }),
      new deck.ScatterplotLayer({
        id: 'compare-gt',
        data: [gtPt],
        getPosition: (d) => [d.lon, d.lat],
        getFillColor: [245, 158, 11, 240],
        getRadius: 8,
        radiusMinPixels: 6,
        radiusMaxPixels: 16,
        stroked: true,
        getLineColor: [245, 158, 11, 255],
        lineWidthMinPixels: 2,
      }),
      new deck.TextLayer({
        id: 'compare-gt-label',
        data: [gtPt],
        getPosition: (d) => [d.lon, d.lat],
        getText: () => `GT (item ${gtPt.item_id})`,
        getSize: 12,
        getColor: [245, 158, 11, 255],
        getTextAnchor: 'start',
        getAlignmentBaseline: 'center',
        fontFamily: 'JetBrains Mono, monospace',
        fontWeight: '600',
        outlineWidth: 4,
        outlineColor: [10, 10, 18, 220],
        getPixelOffset: [14, 0],
        sizeMinPixels: 10,
        sizeMaxPixels: 14,
      }),

      // Pred glow + dot
      new deck.ScatterplotLayer({
        id: 'compare-pred-glow',
        data: [predPt],
        getPosition: (d) => [d.lon, d.lat],
        getFillColor: [16, 185, 129, 50],
        getRadius: 24,
        radiusMinPixels: 12,
        radiusMaxPixels: 30,
      }),
      new deck.ScatterplotLayer({
        id: 'compare-pred',
        data: [predPt],
        getPosition: (d) => [d.lon, d.lat],
        getFillColor: [16, 185, 129, 240],
        getRadius: 8,
        radiusMinPixels: 6,
        radiusMaxPixels: 16,
        stroked: true,
        getLineColor: [16, 185, 129, 255],
        lineWidthMinPixels: 2,
      }),
      new deck.TextLayer({
        id: 'compare-pred-label',
        data: [predPt],
        getPosition: (d) => [d.lon, d.lat],
        getText: () => `Pred (item ${predPt.item_id})`,
        getSize: 12,
        getColor: [16, 185, 129, 255],
        getTextAnchor: 'start',
        getAlignmentBaseline: 'center',
        fontFamily: 'JetBrains Mono, monospace',
        fontWeight: '600',
        outlineWidth: 4,
        outlineColor: [10, 10, 18, 220],
        getPixelOffset: [14, 0],
        sizeMinPixels: 10,
        sizeMaxPixels: 14,
      }),

      // Distance label at midpoint
      new deck.TextLayer({
        id: 'compare-dist-label',
        data: [{
          lon: (predPt.lon + gtPt.lon) / 2,
          lat: (predPt.lat + gtPt.lat) / 2,
        }],
        getPosition: (d) => [d.lon, d.lat],
        getText: () => distKm < 1 ? `${Math.round(distKm * 1000)}m` : distKm < 100 ? `${distKm.toFixed(1)}km` : `${Math.round(distKm)}km`,
        getSize: 11,
        getColor: [255, 255, 255, 200],
        getTextAnchor: 'middle',
        getAlignmentBaseline: 'center',
        fontFamily: 'JetBrains Mono, monospace',
        fontWeight: '500',
        outlineWidth: 4,
        outlineColor: [10, 10, 18, 240],
        getPixelOffset: [0, -16],
        sizeMinPixels: 10,
        sizeMaxPixels: 13,
      }),
    ];

    if (deckgl) {
      deckgl.setProps({ layers });
    } else {
      deckgl = new deck.MapboxOverlay({ layers });
      map.addControl(deckgl);
    }

    // Fit both points
    fitBounds([predPt, gtPt]);
  }

  // Public API
  window.deriveMap = { initMap, renderLayers, renderComparePair, fitBounds, flyTo };
})();
