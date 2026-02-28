"""
gowalla.py — Loads and maps LightGCN ↔ SNAP Gowalla data.

Provides the bridge between LightGCN's remapped integer IDs and the original
SNAP dataset with lat/lon coordinates.

Usage:
    gowalla = GowallaData("path/to/data")
    user_geo = gowalla.get_user_geo(uid=42)
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GowallaData:
    """All in-memory data needed to serve the Derive visualization."""

    item_remap_to_org: dict[int, int]
    user_remap_to_org: dict[int, int]
    loc_coords: dict[int, tuple[float, float]]
    user_timelines: dict[int, dict[int, str]]
    train_dict: dict[int, list[int]]
    test_dict: dict[int, list[int]]
    predictions: dict[int, list[int]]  # user_id → [item_id, ...]
    prediction_meta: dict  # model info (name, embed_dim, etc.)

    @property
    def n_users(self) -> int:
        return len(self.train_dict)

    @property
    def n_items(self) -> int:
        return len(self.item_remap_to_org)

    def get_user_geo(self, uid: int) -> dict | None:
        """Build geographic data for a single LightGCN user ID."""
        org_uid = self.user_remap_to_org.get(uid)
        if org_uid is None:
            return None

        timeline = self.user_timelines.get(org_uid, {})

        def items_to_points(item_ids: list[int]) -> list[dict]:
            pts = []
            for iid in item_ids:
                org_loc = self.item_remap_to_org.get(iid)
                if org_loc and org_loc in self.loc_coords:
                    lat, lon = self.loc_coords[org_loc]
                    ts = timeline.get(org_loc, "2010-01-01T00:00:00Z")
                    pts.append({
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "item_id": iid,
                        "ts": ts,
                    })
            return sorted(pts, key=lambda p: p["ts"])

        history = items_to_points(self.train_dict.get(uid, []))
        ground_truth = items_to_points(self.test_dict.get(uid, []))

        if not history:
            return None

        all_pts = history + ground_truth
        lats = [p["lat"] for p in all_pts]
        lons = [p["lon"] for p in all_pts]
        spread = (max(lats) - min(lats)) + (max(lons) - min(lons))

        if spread > 100:
            label = "Globetrotter"
        elif spread > 10:
            label = "Explorer"
        elif spread > 2:
            label = "Regional"
        elif spread > 0.5:
            label = "City Dweller"
        else:
            label = "Neighborhood Local"

        # Model predictions (if available for this user)
        pred_items = self.predictions.get(uid, [])
        predictions = []
        for iid in pred_items:
            org_loc = self.item_remap_to_org.get(iid)
            if org_loc and org_loc in self.loc_coords:
                lat, lon = self.loc_coords[org_loc]
                predictions.append({
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "item_id": iid,
                    "ts": None,
                })

        return {
            "label": label,
            "org_uid": org_uid,
            "history": history,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "prediction_model": self.prediction_meta.get("model", None),
            "spread": round(spread, 2),
        }


def load(data_dir: str | Path) -> GowallaData:
    """Load all mapping + coordinate data from disk.

    Expected directory layout:
        data_dir/
        ├── gowalla/
        │   ├── train.txt
        │   ├── test.txt
        │   ├── item_list.txt
        │   └── user_list.txt
        └── gowalla_raw/
            └── loc-gowalla_totalCheckins.txt
    """
    data_dir = Path(data_dir)

    # Item mapping: remap_id → original SNAP location_id
    print("  Loading item mapping...")
    item_remap_to_org = {}
    with open(data_dir / "gowalla/item_list.txt") as f:
        next(f)  # skip header
        for line in f:
            org_id, remap_id = line.strip().split()
            item_remap_to_org[int(remap_id)] = int(org_id)

    # User mapping: remap_id → original SNAP user_id
    print("  Loading user mapping...")
    user_remap_to_org = {}
    with open(data_dir / "gowalla/user_list.txt") as f:
        next(f)
        for line in f:
            org_id, remap_id = line.strip().split()
            user_remap_to_org[int(remap_id)] = int(org_id)

    # SNAP check-in coordinates + timestamps
    print("  Loading SNAP coordinates (6.4M rows)...")
    loc_coords: dict[int, tuple[float, float]] = {}
    user_timelines: dict[int, dict[int, str]] = defaultdict(dict)
    with open(data_dir / "gowalla_raw/loc-gowalla_totalCheckins.txt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            uid_str, ts, lat_str, lon_str, loc_str = parts
            lat, lon = float(lat_str), float(lon_str)
            if (lat == 0.0 and lon == 0.0) or abs(lat) > 90 or abs(lon) > 180:
                continue
            loc_id = int(loc_str)
            loc_coords[loc_id] = (lat, lon)
            uid = int(uid_str)
            if loc_id not in user_timelines[uid] or ts < user_timelines[uid][loc_id]:
                user_timelines[uid][loc_id] = ts

    # LightGCN train/test splits
    print("  Loading LightGCN train/test splits...")
    train_dict = _parse_split(data_dir / "gowalla/train.txt")
    test_dict = _parse_split(data_dir / "gowalla/test.txt")

    # Model predictions (optional)
    predictions: dict[int, list[int]] = {}
    prediction_meta: dict = {}
    pred_path = data_dir / "predictions.json"
    if pred_path.exists():
        print("  Loading model predictions...")
        with open(pred_path) as f:
            pred_data = json.load(f)
        prediction_meta = {
            k: v for k, v in pred_data.items() if k != "users"
        }
        for uid_str, info in pred_data.get("users", {}).items():
            predictions[int(uid_str)] = info["items"]
        print(f"  Predictions loaded for {len(predictions)} users ({prediction_meta.get('model', '?')} model)")
    else:
        print("  No predictions.json found — run src/infer.py to generate")

    print(f"  Ready: {len(train_dict)} users, {len(item_remap_to_org)} items, {len(loc_coords)} locations")

    return GowallaData(
        item_remap_to_org=item_remap_to_org,
        user_remap_to_org=user_remap_to_org,
        loc_coords=loc_coords,
        user_timelines=dict(user_timelines),
        train_dict=train_dict,
        test_dict=test_dict,
        predictions=predictions,
        prediction_meta=prediction_meta,
    )


def _parse_split(path: Path) -> dict[int, list[int]]:
    result = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            result[int(parts[0])] = [int(x) for x in parts[1:]]
    return result
