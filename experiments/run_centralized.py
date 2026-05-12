"""Run centralized MAMA-MIA experiments (Contribution 1).

Usage:
    python experiments/run_centralized.py --sdg aim --data adult --epsilon 10
    python experiments/run_centralized.py --sdg mwem_pgm --data adult --epsilon 1
"""

import sys
import os
import argparse
import time
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from utils.data import load_adult_data, load_snake_data, load_california_data
from utils.data import sample_experimental_data
from attacks.mama_mia import MaMAMIAAttack


DATA_LOADERS = {
    "adult": load_adult_data,
    "snake": load_snake_data,
    "cali": load_california_data,
}


def run_experiment(sdg_name, data_name, epsilon, n_runs=5, train_size=1000,
                   num_targets=32, n_shadow_runs=15, seed=42):
    """Run one experiment configuration."""
    print(f"\n{'='*60}")
    print(f"SDG={sdg_name} | Data={data_name} | eps={epsilon} | runs={n_runs}")
    print(f"{'='*60}")

    # Load data
    loader = DATA_LOADERS[data_name]
    df, columns, meta, domain = loader()
    print(f"Loaded {data_name}: {df.shape[0]} records, {len(columns)} features")
    print(f"Domain: {domain}")

    # Initialize attack
    attack = MaMAMIAAttack(sdg_name, data_name)

    # Shadow modeling (done once per epsilon)
    artifact_dir = os.path.join(BASE_DIR, "results", "artifacts")
    fps = attack.shadow_model(
        df, domain, epsilon, n_size=min(train_size, df.shape[0] // 2),
        n_shadow_runs=n_shadow_runs, seed=seed,
        cache=True, artifact_dir=artifact_dir,
    )
    print(f"Focal points: {len(fps)} unique, top-5 weights: {sorted(fps.values(), reverse=True)[:5]}")

    # Run attack trials
    results = {"ma": [], "auc": [], "time_generate": [], "time_attack": []}

    for run_i in range(n_runs):
        # Sample fresh data split
        target_ids, targets, membership, train = sample_experimental_data(
            df, columns, train_size=min(train_size, df.shape[0] // 2),
            num_targets=num_targets,
        )

        # Generate synthetic data
        t0 = time.time()
        synth_df, _ = attack.generate(train[columns], domain, epsilon, seed=run_i)
        t_gen = time.time() - t0

        # Run attack
        t0 = time.time()
        predictions, ma, auc_score, roc = attack.attack(
            synth_df, df[columns], targets[columns], membership, fps, epsilon,
        )
        t_atk = time.time() - t0

        results["ma"].append(ma)
        results["auc"].append(auc_score)
        results["time_generate"].append(t_gen)
        results["time_attack"].append(t_atk)

        print(f"  Run {run_i+1}/{n_runs}: MA={ma:.3f}, AUC={auc_score:.3f}, "
              f"gen={t_gen:.1f}s, atk={t_atk:.3f}s")

    # Summary
    print(f"\nResults ({n_runs} runs):")
    print(f"  MA:  {np.mean(results['ma']):.3f} +/- {np.std(results['ma']):.3f}")
    print(f"  AUC: {np.mean(results['auc']):.3f} +/- {np.std(results['auc']):.3f}")

    # Save results
    results_dir = os.path.join(BASE_DIR, "results", "centralized")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{sdg_name}_{data_name}_eps{epsilon}.json")
    with open(results_path, "w") as f:
        json.dump({
            "sdg": sdg_name,
            "data": data_name,
            "epsilon": epsilon,
            "n_runs": n_runs,
            "train_size": train_size,
            "num_targets": num_targets,
            "ma_mean": float(np.mean(results["ma"])),
            "ma_std": float(np.std(results["ma"])),
            "auc_mean": float(np.mean(results["auc"])),
            "auc_std": float(np.std(results["auc"])),
            "ma_all": [float(x) for x in results["ma"]],
            "auc_all": [float(x) if x is not None else None for x in results["auc"]],
        }, f, indent=2)
    print(f"  Saved: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run centralized MAMA-MIA experiment")
    parser.add_argument("--sdg", required=True, choices=["aim", "mwem_pgm"])
    parser.add_argument("--data", default="adult", choices=["adult", "snake", "cali"])
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--num_targets", type=int, default=32)
    parser.add_argument("--n_shadow_runs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        sdg_name=args.sdg,
        data_name=args.data,
        epsilon=args.epsilon,
        n_runs=args.n_runs,
        train_size=args.train_size,
        num_targets=args.num_targets,
        n_shadow_runs=args.n_shadow_runs,
        seed=args.seed,
    )
