"""Gowalla dataset: download, parse, build normalized adjacency matrix."""

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import jax.experimental.sparse as jsparse
import numpy as np
import scipy.sparse as sp

_BASE_URL = (
    "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/"
)


def _download_if_missing(data_dir: str) -> tuple[Path, Path]:
    """Download train.txt and test.txt if not already present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    train_file = data_path / "train.txt"
    test_file = data_path / "test.txt"
    for fname, fpath in [("train.txt", train_file), ("test.txt", test_file)]:
        if not fpath.exists():
            url = _BASE_URL + fname
            print(f"Downloading {url} ...")
            urlretrieve(url, fpath)
    return train_file, test_file


def _parse_interactions(filepath: Path) -> dict[int, list[int]]:
    """Parse LightGCN format: each line is `userID item1 item2 ...`."""
    interactions: dict[int, list[int]] = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            items = [int(x) for x in parts[1:]]
            interactions[user] = items
    return interactions


def _build_adj_norm(
    train_dict: dict[int, list[int]], n_users: int, n_items: int
) -> jsparse.BCOO:
    """Build D^{-1/2} A D^{-1/2} where A = [[0, R], [R^T, 0]]."""
    rows, cols = [], []
    for user, items in train_dict.items():
        for item in items:
            rows.append(user)
            cols.append(n_users + item)
    rows, cols = np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)

    n_nodes = n_users + n_items
    # Symmetric adjacency: add both directions
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    data = np.ones(len(sym_rows), dtype=np.float32)

    adj = sp.coo_matrix((data, (sym_rows, sym_cols)), shape=(n_nodes, n_nodes))
    adj = adj.tocsr()

    # D^{-1/2}
    degrees = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0.0)
    d_inv_sqrt_mat = sp.diags(d_inv_sqrt)

    # Normalized adjacency
    adj_norm = d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat
    adj_norm = adj_norm.tocoo()

    indices = np.stack(
        [adj_norm.row.astype(np.int32), adj_norm.col.astype(np.int32)], axis=1
    )
    return jsparse.BCOO(
        (adj_norm.data.astype(np.float32), indices),
        shape=adj_norm.shape,
    )


def sample_negatives(
    train_dict: dict[int, list[int]],
    n_items: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one negative sample per positive interaction (rejection sampling)."""
    users, pos_items, neg_items = [], [], []
    for user, items in train_dict.items():
        item_set = set(items)
        for item in items:
            users.append(user)
            pos_items.append(item)
            neg = rng.integers(0, n_items)
            while neg in item_set:
                neg = rng.integers(0, n_items)
            neg_items.append(neg)
    return (
        np.array(users, dtype=np.int32),
        np.array(pos_items, dtype=np.int32),
        np.array(neg_items, dtype=np.int32),
    )


@dataclass
class Dataset:
    n_users: int
    n_items: int
    adj_norm: jsparse.BCOO
    train_dict: dict[int, list[int]]
    val_dict: dict[int, list[int]]
    test_dict: dict[int, list[int]]
    n_train: int


def _split_train_val(
    train_dict: dict[int, list[int]],
    val_ratio: float = 0.1,
    seed: int = 2020,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Split train interactions per user into train + val."""
    rng = np.random.default_rng(seed)
    new_train: dict[int, list[int]] = {}
    val: dict[int, list[int]] = {}
    for user, items in train_dict.items():
        if len(items) < 2:
            # Too few items to split — keep all in train
            new_train[user] = items
            continue
        items_arr = np.array(items)
        rng.shuffle(items_arr)
        n_val = max(1, int(len(items_arr) * val_ratio))
        val[user] = items_arr[:n_val].tolist()
        new_train[user] = items_arr[n_val:].tolist()
    return new_train, val


def load_dataset(data_dir: str, val_ratio: float = 0.1, seed: int = 2020) -> Dataset:
    """Load and preprocess the Gowalla dataset with train/val/test split."""
    train_file, test_file = _download_if_missing(data_dir)
    raw_train_dict = _parse_interactions(train_file)
    test_dict = _parse_interactions(test_file)

    # Split raw train into train + val
    train_dict, val_dict = _split_train_val(raw_train_dict, val_ratio, seed)

    n_users = (
        max(
            max(train_dict.keys(), default=0),
            max(val_dict.keys(), default=0),
            max(test_dict.keys(), default=0),
        )
        + 1
    )
    all_items = set()
    for items in train_dict.values():
        all_items.update(items)
    for items in val_dict.values():
        all_items.update(items)
    for items in test_dict.values():
        all_items.update(items)
    n_items = max(all_items) + 1 if all_items else 0

    n_train = sum(len(v) for v in train_dict.values())
    n_val = sum(len(v) for v in val_dict.values())

    print(
        f"Dataset: {n_users} users, {n_items} items, "
        f"{n_train} train / {n_val} val / "
        f"{sum(len(v) for v in test_dict.values())} test interactions"
    )

    # Adjacency matrix built from train only (no val leakage)
    adj_norm = _build_adj_norm(train_dict, n_users, n_items)
    return Dataset(
        n_users=n_users,
        n_items=n_items,
        adj_norm=adj_norm,
        train_dict=train_dict,
        val_dict=val_dict,
        test_dict=test_dict,
        n_train=n_train,
    )
