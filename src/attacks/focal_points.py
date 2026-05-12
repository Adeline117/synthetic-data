"""Focal-point determination via shadow modeling.

For each SDG algorithm, run it multiple times on auxiliary data and record
which marginals/cliques are selected. The aggregated frequencies form the
focal-point weights used in the MAMA-MIA density ratio attack.
"""

import numpy as np
from collections import Counter

import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from utils.data import pandas_to_mbi, dump_artifact, load_artifact


def _aggregate_fps(all_fps_lists):
    """Aggregate focal points across shadow runs into weighted dict."""
    counter = Counter()
    for fps in all_fps_lists:
        for fp in fps:
            key = tuple(sorted(fp))
            counter[key] += 1
    return dict(counter)


def determine_aim_focal_points(aux_df, domain, eps, n_size, n_shadow_runs=50,
                                delta=1e-9, max_model_size=80, seed=None):
    """Run shadow modeling to extract AIM focal points.

    Args:
        aux_df: auxiliary pandas DataFrame
        domain: mbi.Domain
        eps: privacy budget
        n_size: training size for each shadow run
        n_shadow_runs: number of shadow runs
        delta: privacy parameter
        max_model_size: AIM model size limit (MB)
        seed: random seed for reproducibility

    Returns:
        dict mapping (sorted) marginal tuple -> frequency count
    """
    from generators.aim_wrapper import AIMWithFocalPoints

    all_fps = []
    rng = np.random.RandomState(seed)

    for i in range(n_shadow_runs):
        sample = aux_df.sample(n=n_size, random_state=rng.randint(100000))
        mbi_data = pandas_to_mbi(sample, domain)

        aim = AIMWithFocalPoints(
            epsilon=eps, delta=delta, max_model_size=max_model_size,
            seed=rng.randint(100000),
        )
        # Terminate early — only need focal points, not high-quality synthetic data
        try:
            _, fps = aim.run(mbi_data)
        except Exception as e:
            print(f"Shadow run {i} failed: {e}")
            continue

        all_fps.append(fps)
        if (i + 1) % 10 == 0:
            print(f"  AIM shadow modeling: {i+1}/{n_shadow_runs} done")

    return _aggregate_fps(all_fps)


def determine_mwem_pgm_focal_points(aux_df, domain, eps, n_size, n_shadow_runs=50,
                                     delta=1e-9, max_model_size=25, seed=None):
    """Run shadow modeling to extract MWEM+PGM focal points.

    Args:
        aux_df: auxiliary pandas DataFrame
        domain: mbi.Domain
        eps: privacy budget
        n_size: training size for each shadow run
        n_shadow_runs: number of shadow runs

    Returns:
        dict mapping (sorted) marginal tuple -> frequency count
    """
    from generators.mwem_wrapper import MWEMPGMWithFocalPoints

    all_fps = []
    rng = np.random.RandomState(seed)

    for i in range(n_shadow_runs):
        sample = aux_df.sample(n=n_size, random_state=rng.randint(100000))
        mbi_data = pandas_to_mbi(sample, domain)

        mwem = MWEMPGMWithFocalPoints(
            epsilon=eps, delta=delta, max_model_size=max_model_size,
            seed=rng.randint(100000),
        )
        try:
            _, fps = mwem.run(mbi_data)
        except Exception as e:
            print(f"Shadow run {i} failed: {e}")
            continue

        all_fps.append(fps)
        if (i + 1) % 10 == 0:
            print(f"  MWEM+PGM shadow modeling: {i+1}/{n_shadow_runs} done")

    return _aggregate_fps(all_fps)


def save_focal_points(fps, sdg_name, eps, data_name, artifact_dir=None):
    """Save focal points to disk."""
    name = f"FP_{data_name}_{sdg_name}_{eps}"
    dump_artifact(fps, name, artifact_dir)
    print(f"Saved focal points: {name} ({len(fps)} unique FPs)")


def load_focal_points(sdg_name, eps, data_name, artifact_dir=None):
    """Load focal points from disk."""
    name = f"FP_{data_name}_{sdg_name}_{eps}"
    return load_artifact(name, artifact_dir)
