"""Data loading and format conversion between MAMA-MIA (pandas) and CaPS (mbi)."""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Setup paths for CaPS mbi imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAPS_SRC = os.path.join(BASE_DIR, "refs", "CaPS", "private-pgm-master", "src")
CAPS_MECH = os.path.join(BASE_DIR, "refs", "CaPS", "private-pgm-master", "mechanisms")
MAMA_MIA_DIR = os.path.join(BASE_DIR, "refs", "MAMA-MIA")

for p in [CAPS_SRC, CAPS_MECH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mbi import Dataset, Domain

N_BINS = 20
ARTIFACT_DIR = os.path.join(BASE_DIR, "results", "artifacts")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def dump_artifact(artifact, name, artifact_dir=None):
    d = artifact_dir or ARTIFACT_DIR
    ensure_dir(d)
    with open(os.path.join(d, name), "wb") as f:
        pickle.dump(artifact, f)


def load_artifact(name, artifact_dir=None):
    d = artifact_dir or ARTIFACT_DIR
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snake_data(snake_dir=None):
    """Load SNAKE dataset.

    Returns:
        df: pandas DataFrame (201k records, 15 categorical/ordered features + HHID)
        columns: list of 15 feature column names
        meta: list of dicts with name/representation
        domain: mbi.Domain object
    """
    if snake_dir is None:
        snake_dir = os.path.join(MAMA_MIA_DIR, "SNAKE")

    meta_path = os.path.join(snake_dir, "meta.json")
    data_path = os.path.join(snake_dir, "base.parquet")

    with open(meta_path) as f:
        meta = json.load(f)

    df = pd.read_parquet(data_path)
    df["HHID"] = df.index
    df.index = range(df.shape[0])
    columns = df.columns[:15].tolist()

    # Build mbi Domain from meta
    domain_dict = {}
    for m in meta:
        domain_dict[m["name"]] = len(m["representation"])
    domain = Domain(list(domain_dict.keys()), list(domain_dict.values()))

    # Convert categorical columns to integer codes for mbi compatibility
    # mbi.Dataset.datavector() uses np.histogramdd which requires numeric data
    for m in meta:
        col = m["name"]
        if df[col].dtype.name == "category" or df[col].dtype == object:
            rep = m["representation"]
            mapping = {v: i for i, v in enumerate(rep)}
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    return df, columns, meta, domain


def load_california_data(n_bins=N_BINS):
    """Load California Housing dataset, discretized.

    Returns:
        df: pandas DataFrame with discretized features + HHID
        columns: list of 9 feature column names
        meta: list of dicts
        domain: mbi.Domain object
    """
    columns = [str(x) for x in range(9)]
    raw = fetch_california_housing(as_frame=True).frame
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(raw.sample(frac=1, random_state=42)),
        columns=columns,
    )

    # Equal-depth discretization
    thresholds = {}
    n_per_bin = scaled.shape[0] // n_bins
    for col in columns:
        vals = sorted(scaled[col].values)
        thresholds[col] = [vals[i] for i in range(0, scaled.shape[0], n_per_bin)]

    df = pd.DataFrame()
    for col in columns:
        df[col] = np.digitize(scaled[col].values, thresholds[col]) - 1  # 0-indexed
        df[col] = df[col].clip(0, n_bins - 1)

    # Assign household IDs (groups of 5 for set MI)
    df["HHID"] = np.hstack(
        [[i] * 5 for i in range(df.shape[0] // 5 + 1)]
    )[: df.shape[0]]

    meta = [{"name": col, "representation": list(range(n_bins))} for col in columns]
    domain_dict = {col: n_bins for col in columns}
    domain = Domain(list(domain_dict.keys()), list(domain_dict.values()))

    # Save thresholds for later use
    dump_artifact(thresholds, "cali_thresholds")

    return df, columns, meta, domain


def load_adult_data():
    """Load Adult dataset from CaPS repo.

    Returns:
        df, columns, meta, domain
    """
    data_dir = os.path.join(BASE_DIR, "refs", "CaPS", "private-pgm-master", "data")
    csv_path = os.path.join(data_dir, "adult.csv")
    domain_path = os.path.join(data_dir, "adult-domain.json")

    df = pd.read_csv(csv_path)
    with open(domain_path) as f:
        config = json.load(f)

    columns = list(config.keys())
    domain = Domain(columns, list(config.values()))
    meta = [{"name": col, "representation": list(range(config[col]))} for col in columns]
    df["HHID"] = df.index

    return df, columns, meta, domain


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def pandas_to_mbi(df, domain):
    """Convert a pandas DataFrame to mbi.Dataset.

    The DataFrame columns must be integer-encoded (0 to domain_size-1).
    """
    return Dataset(df[list(domain.attrs)].copy(), domain)


def mbi_to_pandas(mbi_dataset):
    """Convert mbi.Dataset to pandas DataFrame."""
    return mbi_dataset.df.copy()


# ---------------------------------------------------------------------------
# Experimental data sampling (adapted from MAMA-MIA util.py)
# ---------------------------------------------------------------------------

def sample_experimental_data(aux, columns, train_size, num_targets, num_members=None,
                             set_MI=False, household_min_size=5):
    """Sample training data, targets, and membership labels.

    Args:
        aux: full auxiliary DataFrame
        columns: feature columns
        train_size: number of training records
        num_targets: number of target records/sets
        num_members: number of targets that are actual members (default: num_targets // 2)
        set_MI: if True, use HHID-based set membership inference
        household_min_size: minimum household size for set MI

    Returns:
        target_ids: array of target identifiers
        targets: DataFrame of target records
        membership: binary array (1 = member)
        train: DataFrame of training records
    """
    if num_members is None:
        num_members = num_targets // 2

    if set_MI:
        hh_counts = aux["HHID"].value_counts()
        candidates = hh_counts[hh_counts >= household_min_size].index
        target_ids = pd.Series(candidates).sample(n=num_targets).values
        targets = aux[aux["HHID"].isin(target_ids)]
    else:
        target_ids = pd.Series(aux.index).sample(n=num_targets).values
        targets = aux[aux.index.isin(target_ids)]

    member_ids = pd.Series(target_ids).sample(n=num_members).values

    if set_MI:
        members = aux[aux["HHID"].isin(member_ids)]
    else:
        members = aux[aux.index.isin(member_ids)]

    # Sample non-target records for training
    target_mask = aux["HHID"].isin(target_ids) if set_MI else aux.index.isin(target_ids)
    non_targets = aux[~target_mask]
    rest = non_targets.sample(n=train_size - members.shape[0])
    train = pd.concat([rest, members]).sample(frac=1)  # shuffle

    membership = np.array([1 if c in member_ids else 0 for c in target_ids])

    return target_ids, targets, membership, train


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def default_workload(domain, degree=2, max_cells=10000):
    """Generate default workload: all k-way marginals within size limit.

    Args:
        domain: mbi.Domain
        degree: marginal degree (default 2 for pairs)
        max_cells: max domain size for a marginal

    Returns:
        list of attribute tuples
    """
    import itertools
    workload = list(itertools.combinations(domain.attrs, degree))
    workload = [cl for cl in workload if domain.size(cl) <= max_cells]
    return workload
