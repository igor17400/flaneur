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
