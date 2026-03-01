"""
gowalla.py — Loads and maps LightGCN ↔ SNAP Gowalla data.

Provides the bridge between LightGCN's remapped integer IDs and the original
SNAP dataset with lat/lon coordinates. Supports multiple prediction models.

Usage:
    gowalla = GowallaData("path/to/data")
    user_geo = gowalla.get_user_geo(uid=42, model="dim256_layers4_...")
"""

import gzip
import json
import shutil
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SNAP_CHECKINS_URL = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"


@dataclass
class GowallaData:
    """All in-memory data needed to serve the Derive visualization."""

    item_remap_to_org: dict[int, int]
    user_remap_to_org: dict[int, int]
    loc_coords: dict[int, tuple[float, float]]
    user_timelines: dict[int, dict[int, str]]
    train_dict: dict[int, list[int]]
    test_dict: dict[int, list[int]]
    # model_name → {uid → [item_id, ...]}
    predictions: dict[str, dict[int, list[int]]]
    # model_name → metadata dict
    prediction_meta: dict[str, dict]
    # which model to use by default (best recall, or most recent)
    default_model: str | None = None

    @property
    def n_users(self) -> int:
        return len(self.train_dict)

    @property
    def n_items(self) -> int:
        return len(self.item_remap_to_org)

    @property
    def available_models(self) -> list[str]:
        return sorted(self.predictions.keys())

    def get_user_geo(self, uid: int, model: str | None = None) -> dict | None:
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

        # Resolve which model to use
        model_name = model or self.default_model
        model_preds = self.predictions.get(model_name, {}) if model_name else {}
        model_meta = self.prediction_meta.get(model_name, {}) if model_name else {}

        pred_items = model_preds.get(uid, [])
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
            "prediction_model": model_name,
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
    raw_dir = data_dir / "gowalla_raw"
    checkins_path = raw_dir / "loc-gowalla_totalCheckins.txt"
    if not checkins_path.exists():
        print("  Downloading SNAP Gowalla check-ins (~100 MB)...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        gz_path = raw_dir / "loc-gowalla_totalCheckins.txt.gz"
        urllib.request.urlretrieve(SNAP_CHECKINS_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(checkins_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
        print("  Download complete.")

    print("  Loading SNAP coordinates (6.4M rows)...")
    loc_coords: dict[int, tuple[float, float]] = {}
    user_timelines: dict[int, dict[int, str]] = defaultdict(dict)
    with open(checkins_path) as f:
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

    # Model predictions — load ALL files from predictions/ directory
    all_predictions: dict[str, dict[int, list[int]]] = {}
    all_prediction_meta: dict[str, dict] = {}
    default_model: str | None = None
    best_recall = -1.0

    pred_dir = data_dir.parent / "predictions"
    if pred_dir.is_dir():
        pred_files = sorted(pred_dir.glob("*.json"))
        if pred_files:
            print(f"  Loading {len(pred_files)} prediction file(s)...")
            for pred_path in pred_files:
                with open(pred_path) as f:
                    pred_data = json.load(f)
                model_name = pred_data.get("model", pred_path.stem)
                meta = {k: v for k, v in pred_data.items() if k != "users"}
                preds = {}
                for uid_str, info in pred_data.get("users", {}).items():
                    preds[int(uid_str)] = info["items"]
                all_predictions[model_name] = preds
                all_prediction_meta[model_name] = meta

                # Track best model by recall for default
                recall = meta.get("val_recall_at_20")
                if recall is not None and recall > best_recall:
                    best_recall = recall
                    default_model = model_name

                print(f"    {model_name}: {len(preds)} users, recall@20={recall}")

            # If no model had recall (e.g. all raw), use the first non-raw model
            if default_model is None:
                non_raw = [m for m in all_predictions if m != "raw_untrained"]
                default_model = non_raw[0] if non_raw else list(all_predictions.keys())[0]

            print(f"  Default model: {default_model}")
        else:
            print("  No prediction files found in predictions/ — run src/infer.py to generate")
    else:
        print("  No predictions/ directory found — run src/infer.py to generate")

    print(f"  Ready: {len(train_dict)} users, {len(item_remap_to_org)} items, {len(loc_coords)} locations")

    return GowallaData(
        item_remap_to_org=item_remap_to_org,
        user_remap_to_org=user_remap_to_org,
        loc_coords=loc_coords,
        user_timelines=dict(user_timelines),
        train_dict=train_dict,
        test_dict=test_dict,
        predictions=all_predictions,
        prediction_meta=all_prediction_meta,
        default_model=default_model,
    )


def _parse_split(path: Path) -> dict[int, list[int]]:
    result = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            result[int(parts[0])] = [int(x) for x in parts[1:]]
    return result
