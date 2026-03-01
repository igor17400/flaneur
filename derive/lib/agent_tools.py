"""
agent_tools.py — Tool definitions and executors for the Derive chat agent.

Each tool executor returns (result_dict, actions_list).
result_dict is sent back to the model. The '_summary' key is shown in the UI badge.
actions_list contains map actions and widget events to dispatch to the browser.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

# ── Navigate sub-actions (for fallback parsing) ──────────────────────────────

_NAV_ACTIONS = {"fly_to", "select_user", "fit_bounds"}

# ── Tool JSON schemas for Mistral function calling ────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_user",
            "description": "Load summary stats for a user: label, check-in counts, geographic centroid, spread, and prediction hit count. Use this first to understand a user before diving deeper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "integer",
                        "description": "LightGCN user ID (0–29857)",
                    },
                },
                "required": ["uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_detail",
            "description": "Get actual coordinates for a user's history, ground truth, or predictions. Returns up to last_n lat/lon/timestamp points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "integer",
                        "description": "LightGCN user ID",
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["history", "ground_truth", "predictions"],
                        "description": "Which data to retrieve",
                    },
                    "last_n": {
                        "type": "integer",
                        "description": "Maximum number of points to return (default 30, max 30)",
                    },
                },
                "required": ["uid", "data_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_users",
            "description": "Search for interesting users by criteria. Can filter by user type, geographic bounds, or whether they have model predictions. Returns up to 10 user summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "string",
                        "enum": [
                            "has_predictions",
                            "most_checkins",
                            "globetrotter",
                            "city_dweller",
                        ],
                        "description": "Search filter",
                    },
                    "lat_min": {"type": "number", "description": "Southern latitude bound"},
                    "lat_max": {"type": "number", "description": "Northern latitude bound"},
                    "lon_min": {"type": "number", "description": "Western longitude bound"},
                    "lon_max": {"type": "number", "description": "Eastern longitude bound"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 5, max 10)",
                    },
                },
                "required": ["criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_stats",
            "description": "Get aggregate statistics about the LightGCN model's prediction performance across all users that have predictions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_behavior",
            "description": "Deep behavioral analysis for a single user: visit frequency, geographic clustering, revisit patterns, temporal trends, travel distances, movement style classification, and prediction accuracy breakdown. Use this to understand WHY the model succeeds or fails for a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "integer",
                        "description": "LightGCN user ID (0–29857)",
                    },
                },
                "required": ["uid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_users",
            "description": "Deep side-by-side behavioral comparison of two users: geographic overlap, movement similarity, temporal overlap, model performance comparison, and an overall behavioral similarity score (0–100). Use this to explain why the model performs differently for different users.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid_a": {
                        "type": "integer",
                        "description": "First user ID",
                    },
                    "uid_b": {
                        "type": "integer",
                        "description": "Second user ID",
                    },
                },
                "required": ["uid_a", "uid_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_map",
            "description": "Drive the browser map to show what you're discussing. ALWAYS use this to visually highlight relevant locations. Actions: fly_to (move camera), select_user (load and display a user's full data), fit_bounds (zoom to fit current data).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["fly_to", "select_user", "fit_bounds"],
                        "description": "Map action to perform",
                    },
                    "lat": {"type": "number", "description": "Latitude for fly_to"},
                    "lon": {"type": "number", "description": "Longitude for fly_to"},
                    "uid": {"type": "integer", "description": "User ID for select_user"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_widget",
            "description": "Display a rich visual card in the chat panel. Use to highlight key findings, show user stats, or compare users. This makes the analysis visually engaging.",
            "parameters": {
                "type": "object",
                "properties": {
                    "widget_type": {
                        "type": "string",
                        "enum": ["user_card", "comparison", "insight"],
                        "description": "Type of widget: user_card (stats for one user), comparison (two users side-by-side), insight (highlighted finding)",
                    },
                    "uid": {
                        "type": "integer",
                        "description": "User ID for user_card widget",
                    },
                    "uid_a": {
                        "type": "integer",
                        "description": "First user ID for comparison widget",
                    },
                    "uid_b": {
                        "type": "integer",
                        "description": "Second user ID for comparison widget",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for insight widget",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content for insight widget",
                    },
                },
                "required": ["widget_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "Generate a full-screen Data Analyst Report with charts and PDF export for a user. Use this when asked for a report, detailed analysis, deep dive, or comprehensive breakdown of a user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "integer",
                        "description": "LightGCN user ID to generate the report for",
                    },
                },
                "required": ["uid"],
            },
        },
    },
]


# ── Haversine helper ──────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _user_summary(uid: int, gd) -> dict | None:
    """Build a compact summary dict for a user."""
    data = gd.get_user_geo(uid)
    if data is None:
        return None
    history = data["history"]
    gt = data["ground_truth"]
    preds = data.get("predictions", [])
    all_pts = history + gt
    clat = sum(p["lat"] for p in all_pts) / len(all_pts) if all_pts else 0
    clon = sum(p["lon"] for p in all_pts) / len(all_pts) if all_pts else 0
    gt_ids = {p["item_id"] for p in gt}
    hits = sum(1 for p in preds if p["item_id"] in gt_ids)
    return {
        "uid": uid,
        "label": data["label"],
        "history_count": len(history),
        "ground_truth_count": len(gt),
        "prediction_count": len(preds),
        "hits": hits,
        "hit_rate": f"{hits}/{len(preds)}" if preds else "N/A",
        "centroid_lat": round(clat, 2),
        "centroid_lon": round(clon, 2),
        "spread": data["spread"],
    }


# ── Executors ─────────────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict[str, Any], gowalla_data) -> tuple[dict, list[dict]]:
    """Dispatch a tool call. Returns (result_dict, actions_list).

    Includes fallback parsing for when Mistral generates malformed tool calls
    (e.g. calling "fit_bounds" directly instead of navigate_map(action="fit_bounds")).
    """
    executors = {
        "lookup_user": _exec_lookup_user,
        "get_user_detail": _exec_get_user_detail,
        "find_users": _exec_find_users,
        "get_model_stats": _exec_get_model_stats,
        "navigate_map": _exec_navigate_map,
        "show_widget": _exec_show_widget,
        "generate_report": _exec_generate_report,
        "analyze_behavior": _exec_analyze_behavior,
        "compare_users": _exec_compare_users,
    }

    executor = executors.get(name)

    # Fallback 1: name is a navigate_map sub-action called directly
    if not executor and name in _NAV_ACTIONS:
        args = {**args, "action": name}
        name = "navigate_map"
        executor = _exec_navigate_map

    # Fallback 2: name is a JSON string (Mistral sometimes puts args as name)
    if not executor:
        try:
            parsed = json.loads(name)
            if isinstance(parsed, dict):
                if "action" in parsed and parsed["action"] in _NAV_ACTIONS:
                    merged = {**parsed, **args}
                    name = "navigate_map"
                    return _exec_navigate_map(merged, gowalla_data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not executor:
        return {"error": f"Unknown tool: {name}", "_summary": f"Error: unknown tool"}, []

    try:
        return executor(args, gowalla_data)
    except Exception as e:
        return {"error": str(e), "_summary": f"Error: {e}"}, []


def _exec_lookup_user(args: dict, gd) -> tuple[dict, list[dict]]:
    uid = args["uid"]
    summary = _user_summary(uid, gd)
    if summary is None:
        return {"error": f"User {uid} not found", "_summary": f"User #{uid} not found"}, []

    summary["_summary"] = (
        f"User #{uid}: {summary['label']}, "
        f"{summary['history_count']} history, {summary['hit_rate']} hits"
    )
    return summary, []


def _exec_get_user_detail(args: dict, gd) -> tuple[dict, list[dict]]:
    uid = args["uid"]
    data_type = args["data_type"]
    last_n = min(args.get("last_n", 30), 30)

    data = gd.get_user_geo(uid)
    if data is None:
        return {"error": f"User {uid} not found", "_summary": f"User #{uid} not found"}, []

    points = data.get(data_type, [])
    points = points[-last_n:]

    coords = [
        {"lat": p["lat"], "lon": p["lon"], "ts": p.get("ts"), "item_id": p["item_id"]}
        for p in points
    ]

    result = {
        "uid": uid,
        "data_type": data_type,
        "count": len(coords),
        "total_available": len(data.get(data_type, [])),
        "points": coords,
        "_summary": f"{len(coords)} {data_type} points",
    }
    return result, []


def _exec_find_users(args: dict, gd) -> tuple[dict, list[dict]]:
    criteria = args["criteria"]
    limit = min(args.get("limit", 5), 10)
    lat_min = args.get("lat_min")
    lat_max = args.get("lat_max")
    lon_min = args.get("lon_min")
    lon_max = args.get("lon_max")

    candidates = []

    for uid in gd.train_dict:
        data = gd.get_user_geo(uid)
        if data is None:
            continue

        history = data["history"]
        gt = data["ground_truth"]
        preds = data.get("predictions", [])

        if criteria == "has_predictions" and len(preds) == 0:
            continue
        if criteria == "globetrotter" and data["label"] != "Globetrotter":
            continue
        if criteria == "city_dweller" and data["label"] != "City Dweller":
            continue

        all_pts = history + gt
        if not all_pts:
            continue
        clat = sum(p["lat"] for p in all_pts) / len(all_pts)
        clon = sum(p["lon"] for p in all_pts) / len(all_pts)

        if lat_min is not None and clat < lat_min:
            continue
        if lat_max is not None and clat > lat_max:
            continue
        if lon_min is not None and clon < lon_min:
            continue
        if lon_max is not None and clon > lon_max:
            continue

        gt_ids = {p["item_id"] for p in gt}
        hits = sum(1 for p in preds if p["item_id"] in gt_ids)

        candidates.append({
            "uid": uid,
            "label": data["label"],
            "history_count": len(history),
            "ground_truth_count": len(gt),
            "prediction_count": len(preds),
            "hits": hits,
            "centroid_lat": round(clat, 4),
            "centroid_lon": round(clon, 4),
            "spread": data["spread"],
        })

        if len(candidates) >= 500:
            break

    if criteria == "most_checkins":
        candidates.sort(key=lambda u: u["history_count"], reverse=True)
    elif criteria == "has_predictions":
        candidates.sort(key=lambda u: u["hits"], reverse=True)

    results = candidates[:limit]
    summary = f"Found {len(results)} users ({criteria})"
    return {"users": results, "criteria": criteria, "_summary": summary}, []


def _exec_get_model_stats(args: dict, gd) -> tuple[dict, list[dict]]:
    total_users_with_preds = 0
    total_hits = 0
    total_preds = 0
    total_gt = 0

    # Multi-model: iterate the default model's predictions
    model_name = gd.default_model
    model_preds = gd.predictions.get(model_name, {}) if model_name else {}

    for uid in model_preds:
        data = gd.get_user_geo(uid)
        if data is None:
            continue
        total_users_with_preds += 1
        preds = data.get("predictions", [])
        gt = data["ground_truth"]
        gt_ids = {p["item_id"] for p in gt}
        hits = sum(1 for p in preds if p["item_id"] in gt_ids)
        total_hits += hits
        total_preds += len(preds)
        total_gt += len(gt)

    hit_rate = (total_hits / total_preds * 100) if total_preds > 0 else 0
    avg_preds = total_preds / total_users_with_preds if total_users_with_preds > 0 else 0

    # Multi-model: get model metadata from prediction_meta dict
    meta = gd.prediction_meta.get(model_name, {}) if model_name else {}

    result = {
        "users_with_predictions": total_users_with_preds,
        "total_predictions": total_preds,
        "total_ground_truth": total_gt,
        "total_hits": total_hits,
        "hit_rate_pct": round(hit_rate, 1),
        "avg_predictions_per_user": round(avg_preds, 1),
        "model_name": model_name or "unknown",
        "_summary": f"{hit_rate:.1f}% hit rate, {total_hits}/{total_preds} across {total_users_with_preds} users",
    }
    return result, []


def _exec_navigate_map(args: dict, gd) -> tuple[dict, list[dict]]:
    action = args.get("action", "")
    actions = []

    if action == "fly_to":
        lat = args.get("lat")
        lon = args.get("lon")
        if lat is None or lon is None:
            return {"error": "fly_to requires lat and lon", "_summary": "Missing coordinates"}, []
        actions.append({"name": "fly_to", "lat": lat, "lon": lon})
        return {"status": "ok", "_summary": f"Map: fly to ({lat:.2f}, {lon:.2f})"}, actions

    elif action == "select_user":
        uid = args.get("uid")
        if uid is None:
            return {"error": "select_user requires uid", "_summary": "Missing uid"}, []
        actions.append({"name": "select_user", "uid": uid})
        return {"status": "ok", "_summary": f"Map: select user #{uid}"}, actions

    elif action == "fit_bounds":
        actions.append({"name": "fit_bounds"})
        return {"status": "ok", "_summary": "Map: fit bounds"}, actions

    else:
        return {"error": f"Unknown action: {action}", "_summary": f"Unknown action"}, []


def _exec_analyze_behavior(args: dict, gd) -> tuple[dict, list[dict]]:
    uid = args["uid"]
    data = gd.get_user_geo(uid)
    if data is None:
        return {"error": f"User {uid} not found", "_summary": f"User #{uid} not found"}, []

    history = data["history"]
    gt = data["ground_truth"]
    preds = data.get("predictions", [])
    label = data["label"]
    spread = data["spread"]
    all_pts = history + gt

    # ── Visit frequency & density ────────────────────────────────────
    timestamps = [p["ts"] for p in all_pts if p.get("ts")]
    months_active = set()
    monthly_counts: dict[str, int] = defaultdict(int)
    for ts in timestamps:
        if ts and len(ts) >= 7:
            m = ts[:7]
            months_active.add(m)
            monthly_counts[m] += 1

    n_months = max(len(months_active), 1)
    checkins_per_month = round(len(all_pts) / n_months, 1)

    sorted_months = sorted(months_active) if months_active else []
    peak_month = max(monthly_counts, key=monthly_counts.get) if monthly_counts else None
    peak_count = monthly_counts.get(peak_month, 0) if peak_month else 0

    # Activity trend: compare first half vs second half
    if len(sorted_months) >= 4:
        mid = len(sorted_months) // 2
        first_half = sum(monthly_counts.get(m, 0) for m in sorted_months[:mid])
        second_half = sum(monthly_counts.get(m, 0) for m in sorted_months[mid:])
        if second_half > first_half * 1.3:
            trend = "increasing"
        elif first_half > second_half * 1.3:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # ── Geographic clustering (grid-based) ───────────────────────────
    GRID_SIZE = 0.01  # ~1km grid cells
    grid_cells: dict[tuple[int, int], int] = defaultdict(int)
    for p in all_pts:
        cell = (int(p["lat"] / GRID_SIZE), int(p["lon"] / GRID_SIZE))
        grid_cells[cell] += 1

    top_cells = sorted(grid_cells.values(), reverse=True)
    top3_count = sum(top_cells[:3]) if top_cells else 0
    concentration_pct = round(top3_count / len(all_pts) * 100, 1) if all_pts else 0

    n_hotspots = len(grid_cells)

    # ── Revisit rate ─────────────────────────────────────────────────
    item_visits: dict[int, int] = defaultdict(int)
    for p in all_pts:
        item_visits[p["item_id"]] += 1

    unique_locations = len(item_visits)
    revisited = sum(1 for c in item_visits.values() if c > 1)
    max_revisits = max(item_visits.values()) if item_visits else 0
    revisit_rate = round(revisited / unique_locations * 100, 1) if unique_locations else 0

    # ── Travel distances between consecutive check-ins ───────────────
    sorted_pts = sorted(all_pts, key=lambda p: p.get("ts", ""))
    step_distances = []
    for i in range(1, len(sorted_pts)):
        d = _haversine(
            sorted_pts[i - 1]["lat"], sorted_pts[i - 1]["lon"],
            sorted_pts[i]["lat"], sorted_pts[i]["lon"],
        )
        step_distances.append(d)

    if step_distances:
        avg_step_km = round(sum(step_distances) / len(step_distances), 2)
        sorted_dists = sorted(step_distances)
        median_step_km = round(sorted_dists[len(sorted_dists) // 2], 2)
        max_step_km = round(max(step_distances), 2)
    else:
        avg_step_km = median_step_km = max_step_km = 0.0

    # ── Movement style classification ────────────────────────────────
    if avg_step_km < 5 and spread < 2:
        movement_style = "routine_local"
    elif avg_step_km > 50 or spread > 50:
        movement_style = "long_range_traveler"
    else:
        movement_style = "mixed"

    # ── Prediction accuracy breakdown ────────────────────────────────
    gt_ids = {p["item_id"] for p in gt}
    exact_hits = 0
    near_misses = 0  # <5km
    far_misses = 0
    pred_distances_to_gt = []

    for p in preds:
        if p["item_id"] in gt_ids:
            exact_hits += 1
            pred_distances_to_gt.append(0.0)
        elif gt:
            min_dist = min(_haversine(p["lat"], p["lon"], g["lat"], g["lon"]) for g in gt)
            pred_distances_to_gt.append(min_dist)
            if min_dist < 5:
                near_misses += 1
            else:
                far_misses += 1
        else:
            far_misses += 1

    avg_pred_dist = round(sum(pred_distances_to_gt) / len(pred_distances_to_gt), 2) if pred_distances_to_gt else None

    if preds:
        if exact_hits / len(preds) > 0.3:
            model_verdict = "strong_match"
        elif (exact_hits + near_misses) / len(preds) > 0.3:
            model_verdict = "neighborhood_match"
        else:
            model_verdict = "weak_match"
    else:
        model_verdict = "no_predictions"

    # ── Centroid ─────────────────────────────────────────────────────
    clat = round(sum(p["lat"] for p in all_pts) / len(all_pts), 4) if all_pts else 0
    clon = round(sum(p["lon"] for p in all_pts) / len(all_pts), 4) if all_pts else 0

    result = {
        "uid": uid,
        "label": label,
        "spread": spread,
        "centroid": {"lat": clat, "lon": clon},
        "total_checkins": len(all_pts),
        "history_count": len(history),
        "ground_truth_count": len(gt),
        "unique_locations": unique_locations,
        "checkins_per_month": checkins_per_month,
        "months_active": n_months,
        "active_period": f"{sorted_months[0]} to {sorted_months[-1]}" if sorted_months else "unknown",
        "peak_month": peak_month,
        "peak_month_checkins": peak_count,
        "activity_trend": trend,
        "monthly_distribution": dict(monthly_counts),
        "geographic_concentration_pct": concentration_pct,
        "n_hotspot_cells": n_hotspots,
        "revisit_rate_pct": revisit_rate,
        "revisited_locations": revisited,
        "max_revisits_single_location": max_revisits,
        "avg_step_km": avg_step_km,
        "median_step_km": median_step_km,
        "max_step_km": max_step_km,
        "movement_style": movement_style,
        "prediction_count": len(preds),
        "exact_hits": exact_hits,
        "near_misses_lt5km": near_misses,
        "far_misses": far_misses,
        "avg_prediction_distance_km": avg_pred_dist,
        "model_verdict": model_verdict,
        "_summary": (
            f"User #{uid}: {movement_style}, {revisit_rate}% revisit, "
            f"{avg_step_km}km avg step, {model_verdict}"
        ),
    }
    return result, []


def _exec_compare_users(args: dict, gd) -> tuple[dict, list[dict]]:
    uid_a = args["uid_a"]
    uid_b = args["uid_b"]

    data_a = gd.get_user_geo(uid_a)
    data_b = gd.get_user_geo(uid_b)
    if data_a is None:
        return {"error": f"User {uid_a} not found", "_summary": f"User #{uid_a} not found"}, []
    if data_b is None:
        return {"error": f"User {uid_b} not found", "_summary": f"User #{uid_b} not found"}, []

    all_a = data_a["history"] + data_a["ground_truth"]
    all_b = data_b["history"] + data_b["ground_truth"]
    preds_a = data_a.get("predictions", [])
    preds_b = data_b.get("predictions", [])
    gt_a = data_a["ground_truth"]
    gt_b = data_b["ground_truth"]

    # ── Geographic overlap ───────────────────────────────────────────
    items_a = {p["item_id"] for p in all_a}
    items_b = {p["item_id"] for p in all_b}
    shared_exact = items_a & items_b

    # Nearby locations (within 1km)
    nearby_count = 0
    coords_a = [(p["lat"], p["lon"]) for p in all_a]
    coords_b = [(p["lat"], p["lon"]) for p in all_b]

    # Sample to avoid O(n*m) explosion on large histories
    sample_a = coords_a[:200]
    sample_b = coords_b[:200]
    for la, lo_a in sample_a:
        for lb, lo_b in sample_b:
            if _haversine(la, lo_a, lb, lo_b) < 1.0:
                nearby_count += 1
                break

    # Centroids
    clat_a = sum(p["lat"] for p in all_a) / len(all_a) if all_a else 0
    clon_a = sum(p["lon"] for p in all_a) / len(all_a) if all_a else 0
    clat_b = sum(p["lat"] for p in all_b) / len(all_b) if all_b else 0
    clon_b = sum(p["lon"] for p in all_b) / len(all_b) if all_b else 0
    centroid_distance_km = round(_haversine(clat_a, clon_a, clat_b, clon_b), 2)

    # ── Movement similarity ──────────────────────────────────────────
    spread_a = data_a["spread"]
    spread_b = data_b["spread"]
    max_spread = max(spread_a, spread_b, 0.01)
    spread_similarity = round(1 - abs(spread_a - spread_b) / max_spread, 3)

    # ── Temporal overlap ─────────────────────────────────────────────
    months_a = set()
    months_b = set()
    for p in all_a:
        ts = p.get("ts", "")
        if ts and len(ts) >= 7:
            months_a.add(ts[:7])
    for p in all_b:
        ts = p.get("ts", "")
        if ts and len(ts) >= 7:
            months_b.add(ts[:7])

    shared_months = months_a & months_b
    all_months = months_a | months_b
    temporal_overlap = round(len(shared_months) / len(all_months), 3) if all_months else 0

    peak_a = None
    peak_b = None
    if months_a:
        mc_a: dict[str, int] = defaultdict(int)
        for p in all_a:
            ts = p.get("ts", "")
            if ts and len(ts) >= 7:
                mc_a[ts[:7]] += 1
        peak_a = max(mc_a, key=mc_a.get)
    if months_b:
        mc_b: dict[str, int] = defaultdict(int)
        for p in all_b:
            ts = p.get("ts", "")
            if ts and len(ts) >= 7:
                mc_b[ts[:7]] += 1
        peak_b = max(mc_b, key=mc_b.get)

    # ── Model performance comparison ─────────────────────────────────
    gt_ids_a = {p["item_id"] for p in gt_a}
    gt_ids_b = {p["item_id"] for p in gt_b}
    hits_a = sum(1 for p in preds_a if p["item_id"] in gt_ids_a)
    hits_b = sum(1 for p in preds_b if p["item_id"] in gt_ids_b)
    hit_rate_a = round(hits_a / len(preds_a) * 100, 1) if preds_a else 0
    hit_rate_b = round(hits_b / len(preds_b) * 100, 1) if preds_b else 0

    # Avg prediction distance to nearest GT
    def avg_pred_dist(preds, gt_pts):
        if not preds or not gt_pts:
            return None
        total = 0
        for p in preds:
            min_d = min(_haversine(p["lat"], p["lon"], g["lat"], g["lon"]) for g in gt_pts)
            total += min_d
        return round(total / len(preds), 2)

    avg_dist_a = avg_pred_dist(preds_a, gt_a)
    avg_dist_b = avg_pred_dist(preds_b, gt_b)

    # ── Behavioral similarity score (0–100) ──────────────────────────
    # 30% geo overlap, 20% centroid proximity, 20% spread similarity,
    # 15% temporal overlap, 15% label match
    max_items = max(len(items_a), len(items_b), 1)
    geo_score = len(shared_exact) / max_items  # 0–1

    # Centroid proximity: 0 at >500km, 1 at 0km
    centroid_score = max(0, 1 - centroid_distance_km / 500)

    label_score = 1.0 if data_a["label"] == data_b["label"] else 0.0

    similarity = round(
        (geo_score * 30 + centroid_score * 20 + spread_similarity * 20
         + temporal_overlap * 15 + label_score * 15),
        1,
    )

    result = {
        "uid_a": uid_a,
        "uid_b": uid_b,
        "label_a": data_a["label"],
        "label_b": data_b["label"],
        "geographic_overlap": {
            "shared_exact_locations": len(shared_exact),
            "nearby_locations_lt1km": nearby_count,
            "centroid_distance_km": centroid_distance_km,
        },
        "movement_similarity": {
            "spread_a": spread_a,
            "spread_b": spread_b,
            "spread_similarity": spread_similarity,
            "history_count_a": len(data_a["history"]),
            "history_count_b": len(data_b["history"]),
        },
        "temporal_overlap": {
            "shared_active_months": len(shared_months),
            "total_unique_months": len(all_months),
            "overlap_ratio": temporal_overlap,
            "peak_month_a": peak_a,
            "peak_month_b": peak_b,
        },
        "model_performance": {
            "hit_rate_a_pct": hit_rate_a,
            "hit_rate_b_pct": hit_rate_b,
            "hits_a": hits_a,
            "hits_b": hits_b,
            "avg_pred_distance_a_km": avg_dist_a,
            "avg_pred_distance_b_km": avg_dist_b,
        },
        "behavioral_similarity_score": similarity,
        "_summary": (
            f"#{uid_a} ({data_a['label']}) vs #{uid_b} ({data_b['label']}): "
            f"similarity {similarity}/100, {len(shared_exact)} shared locations"
        ),
    }
    return result, []


def _exec_generate_report(args: dict, gd) -> tuple[dict, list[dict]]:
    uid = args["uid"]
    data = gd.get_user_geo(uid)
    if data is None:
        return {"error": f"User {uid} not found", "_summary": f"User #{uid} not found"}, []

    history = data["history"]
    gt = data["ground_truth"]
    preds = data.get("predictions", [])
    label = data["label"]
    spread = data["spread"]
    # Multi-model: get model name from the data or default
    model_name = data.get("prediction_model") or gd.default_model or "LightGCN"

    all_pts = history + gt
    clat = sum(p["lat"] for p in all_pts) / len(all_pts) if all_pts else 0
    clon = sum(p["lon"] for p in all_pts) / len(all_pts) if all_pts else 0

    gt_ids = {p["item_id"] for p in gt}
    hits = sum(1 for p in preds if p["item_id"] in gt_ids)
    hit_rate = f"{hits}/{len(preds)}" if preds else "N/A"

    # ── Monthly check-in buckets for timeline chart ──────────────────────
    monthly_history: dict[str, int] = defaultdict(int)
    monthly_gt: dict[str, int] = defaultdict(int)

    for p in history:
        ts = p.get("ts", "")
        if ts and len(ts) >= 7:
            monthly_history[ts[:7]] += 1
    for p in gt:
        ts = p.get("ts", "")
        if ts and len(ts) >= 7:
            monthly_gt[ts[:7]] += 1

    all_months = sorted(set(list(monthly_history.keys()) + list(monthly_gt.keys())))
    timeline_labels = all_months
    timeline_history = [monthly_history.get(m, 0) for m in all_months]
    timeline_gt = [monthly_gt.get(m, 0) for m in all_months]

    # ── Per-prediction distance to nearest GT ────────────────────────────
    pred_distances = []
    for p in preds:
        min_dist = float("inf")
        for g in gt:
            d = _haversine(p["lat"], p["lon"], g["lat"], g["lon"])
            if d < min_dist:
                min_dist = d
        is_hit = p["item_id"] in gt_ids
        pred_distances.append({
            "item_id": p["item_id"],
            "distance_km": round(min_dist, 2) if gt else None,
            "is_hit": is_hit,
            "lat": p["lat"],
            "lon": p["lon"],
        })

    # ── Scatter arrays ───────────────────────────────────────────────────
    scatter_history = [{"x": p["lon"], "y": p["lat"]} for p in history]
    scatter_gt = [{"x": p["lon"], "y": p["lat"]} for p in gt]
    scatter_preds = [{"x": p["lon"], "y": p["lat"]} for p in preds]

    # ── Data-driven analysis text ────────────────────────────────────────
    analysis_parts = []
    analysis_parts.append(
        f"User #{uid} is classified as a **{label}** with a geographic spread of "
        f"**{spread}** across **{len(history)}** training check-ins and "
        f"**{len(gt)}** test check-ins."
    )

    if preds:
        avg_dist = sum(
            d["distance_km"] for d in pred_distances if d["distance_km"] is not None
        ) / len(pred_distances) if pred_distances else 0
        analysis_parts.append(
            f"The {model_name} model generated **{len(preds)}** predictions with a hit rate of "
            f"**{hit_rate}** (exact location matches). The average distance from each prediction "
            f"to the nearest ground-truth location is **{avg_dist:.1f} km**."
        )
        if hits > 0:
            analysis_parts.append(
                f"The model successfully identified **{hits}** exact venue(s) the user visited, "
                f"demonstrating meaningful signal in the learned embeddings."
            )
        else:
            near_preds = sum(1 for d in pred_distances if d["distance_km"] is not None and d["distance_km"] < 5)
            if near_preds > 0:
                analysis_parts.append(
                    f"While no exact hits were recorded, **{near_preds}** prediction(s) fall within "
                    f"5 km of a ground-truth location, suggesting the model captures the user's "
                    f"geographic neighborhood."
                )
            else:
                analysis_parts.append(
                    "No predictions matched ground-truth locations. The model may struggle with "
                    "this user's check-in pattern."
                )
    else:
        analysis_parts.append(
            "No predictions were generated for this user."
        )

    if len(all_months) > 1:
        peak_month = max(all_months, key=lambda m: monthly_history.get(m, 0) + monthly_gt.get(m, 0))
        peak_count = monthly_history.get(peak_month, 0) + monthly_gt.get(peak_month, 0)
        analysis_parts.append(
            f"Activity peaked in **{peak_month}** with **{peak_count}** check-ins."
        )

    analysis_text = " ".join(analysis_parts)

    report_blob = {
        "uid": uid,
        "label": label,
        "model_name": model_name,
        "centroid_lat": round(clat, 2),
        "centroid_lon": round(clon, 2),
        "spread": spread,
        "history_count": len(history),
        "ground_truth_count": len(gt),
        "prediction_count": len(preds),
        "hits": hits,
        "hit_rate": hit_rate,
        "analysis": analysis_text,
        "timeline": {
            "labels": timeline_labels,
            "history": timeline_history,
            "ground_truth": timeline_gt,
        },
        "pred_distances": pred_distances,
        "scatter": {
            "history": scatter_history,
            "ground_truth": scatter_gt,
            "predictions": scatter_preds,
        },
    }

    summary = f"Report for user #{uid}: {label}, {len(history)} history, {hit_rate} hits"
    return {"status": "ok", "_summary": summary}, [{"name": "report", "data": report_blob}]


def _exec_show_widget(args: dict, gd) -> tuple[dict, list[dict]]:
    widget_type = args.get("widget_type", "")
    actions = []

    if widget_type == "user_card":
        uid = args.get("uid")
        if uid is None:
            return {"error": "user_card requires uid", "_summary": "Missing uid"}, []
        summary = _user_summary(uid, gd)
        if summary is None:
            return {"error": f"User {uid} not found", "_summary": f"User #{uid} not found"}, []
        actions.append({
            "name": "widget",
            "widget_type": "user_card",
            "data": summary,
        })
        return {"status": "ok", "_summary": f"Showing card for user #{uid}"}, actions

    elif widget_type == "comparison":
        uid_a = args.get("uid_a")
        uid_b = args.get("uid_b")
        if uid_a is None or uid_b is None:
            return {"error": "comparison requires uid_a and uid_b", "_summary": "Missing user IDs"}, []
        sa = _user_summary(uid_a, gd)
        sb = _user_summary(uid_b, gd)
        if sa is None or sb is None:
            return {"error": "One or both users not found", "_summary": "User not found"}, []
        actions.append({
            "name": "widget",
            "widget_type": "comparison",
            "data": {"user_a": sa, "user_b": sb},
        })
        return {"status": "ok", "_summary": f"Comparing #{uid_a} vs #{uid_b}"}, actions

    elif widget_type == "insight":
        title = args.get("title", "Key Finding")
        content = args.get("content", "")
        actions.append({
            "name": "widget",
            "widget_type": "insight",
            "data": {"title": title, "content": content},
        })
        return {"status": "ok", "_summary": f"Insight: {title}"}, actions

    else:
        return {"error": f"Unknown widget: {widget_type}", "_summary": "Unknown widget"}, []
