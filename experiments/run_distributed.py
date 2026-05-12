"""Run distributed MAMA-MIA experiments (Contribution 2).

Compares MIA success across three threat models in CaPS distributed settings.

Usage:
    python experiments/run_distributed.py --sdg mwem_pgm --data adult --epsilon 10
    python experiments/run_distributed.py --sdg aim --data snake --epsilon 10 --n_holders 2
"""

import sys
import os
import argparse
import time
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from utils.data import (
    load_adult_data, load_snake_data, load_california_data,
    sample_experimental_data, mbi_to_pandas,
)
from distributed.partitioner import horizontal_partition
from distributed.caps_simulator import CaPSSimulator
from distributed.threat_models import ExternalAttacker, MaliciousDataHolder, ColludingServer


DATA_LOADERS = {
    "adult": load_adult_data,
    "snake": load_snake_data,
    "cali": load_california_data,
}


def run_experiment(sdg_name, data_name, epsilon, n_holders=2,
                   partition_type="horizontal", n_runs=3, train_size=1000,
                   num_targets=32, n_shadow_runs=10, seed=42):
    """Run distributed MIA experiment with all three threat models."""
    print(f"\n{'='*70}")
    print(f"DISTRIBUTED: SDG={sdg_name} | Data={data_name} | eps={epsilon} | "
          f"holders={n_holders} | partition={partition_type}")
    print(f"{'='*70}")

    # Load data
    loader = DATA_LOADERS[data_name]
    df, columns, meta, domain = loader()
    print(f"Loaded {data_name}: {df.shape[0]} records, {len(columns)} features")

    # Initialize threat models
    external = ExternalAttacker(sdg_name, data_name)
    malicious = MaliciousDataHolder(sdg_name, holder_index=0, data_name=data_name)
    colluding = ColludingServer(sdg_name, data_name)

    # Results storage
    results = {
        "external": {"ma": [], "auc": []},
        "malicious": {"ma": [], "auc": []},
        "colluding": {"ma": [], "auc": []},
    }

    # Shadow model for external attacker (done once)
    print("Running external attacker shadow model...")
    t0 = time.time()
    external_fps = external.shadow_model(
        df[columns], domain, epsilon,
        n_size=min(train_size, df.shape[0] // 2),
        n_shadow_runs=n_shadow_runs, seed=seed,
    )
    print(f"External shadow model: {len(external_fps)} FPs in {time.time()-t0:.1f}s")

    for run_i in range(n_runs):
        print(f"\n--- Run {run_i+1}/{n_runs} ---")

        # Sample data
        target_ids, targets, membership, train = sample_experimental_data(
            df, columns, train_size=min(train_size, df.shape[0] // 2),
            num_targets=num_targets,
        )

        # Partition training data
        partitions = horizontal_partition(train[columns], n_holders, seed=run_i)
        print(f"Partitioned: {[p.shape[0] for p in partitions]} records")

        # Generate synthetic data via CaPS
        sim = CaPSSimulator(domain, partition_type, epsilon, seed=run_i)
        try:
            t0 = time.time()
            if sdg_name == "mwem_pgm":
                synth_mbi, sim_fps = sim.run_mwem_pgm(partitions)
            else:
                synth_mbi, sim_fps = sim.run_aim(partitions)
            synth_df = mbi_to_pandas(synth_mbi)
            t_gen = time.time() - t0
            print(f"CaPS generation: {synth_df.shape[0]} records in {t_gen:.1f}s")
        except (ValueError, RuntimeError) as e:
            print(f"  Run {run_i+1}/{n_runs}: SKIPPED (generation failed: {e})")
            continue

        transcript = sim.get_transcript()

        # --- Threat Model A: External Attacker ---
        preds_a, ma_a, auc_a, _ = external.attack(
            synth_df, df[columns], targets[columns], membership,
            external_fps, epsilon,
        )
        results["external"]["ma"].append(ma_a)
        results["external"]["auc"].append(auc_a)
        print(f"  External:  MA={ma_a:.3f}, AUC={auc_a:.3f}")

        # --- Threat Model B: Malicious Data Holder ---
        # Simplified: use external FPs for now (full conditioned shadow model is slow)
        # In full experiment, would call malicious.shadow_model() per run
        preds_b, ma_b, auc_b, _ = malicious.attack(
            synth_df, partitions[0], df[columns], targets[columns],
            membership, external_fps, epsilon,
        )
        results["malicious"]["ma"].append(ma_b)
        results["malicious"]["auc"].append(auc_b)
        print(f"  Malicious: MA={ma_b:.3f}, AUC={auc_b:.3f}")

        # --- Threat Model C: Colluding Server ---
        preds_c, ma_c, auc_c, _ = colluding.attack(
            synth_df, df[columns], targets[columns], membership,
            transcript, epsilon,
        )
        results["colluding"]["ma"].append(ma_c)
        results["colluding"]["auc"].append(auc_c)
        print(f"  Colluding: MA={ma_c:.3f}, AUC={auc_c:.3f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS ({n_runs} runs):")
    for model_name in ["external", "malicious", "colluding"]:
        ma_vals = results[model_name]["ma"]
        auc_vals = [v for v in results[model_name]["auc"] if v is not None]
        print(f"  {model_name:12s}: MA={np.mean(ma_vals):.3f}+/-{np.std(ma_vals):.3f}, "
              f"AUC={np.mean(auc_vals):.3f}+/-{np.std(auc_vals):.3f}")

    # Save results
    results_dir = os.path.join(BASE_DIR, "results", "distributed")
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{sdg_name}_{data_name}_eps{epsilon}_{partition_type}_h{n_holders}.json"
    results_path = os.path.join(results_dir, fname)
    with open(results_path, "w") as f:
        json.dump({
            "sdg": sdg_name,
            "data": data_name,
            "epsilon": epsilon,
            "n_holders": n_holders,
            "partition_type": partition_type,
            "n_runs": n_runs,
            **{
                f"{m}_ma_mean": float(np.mean(results[m]["ma"]))
                for m in results
            },
            **{
                f"{m}_auc_mean": float(np.mean([v for v in results[m]["auc"] if v is not None]))
                for m in results
            },
            "raw": {
                m: {
                    "ma": [float(x) for x in results[m]["ma"]],
                    "auc": [float(x) if x is not None else None for x in results[m]["auc"]],
                }
                for m in results
            },
        }, f, indent=2)
    print(f"\nSaved: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run distributed MAMA-MIA experiment")
    parser.add_argument("--sdg", required=True, choices=["aim", "mwem_pgm"])
    parser.add_argument("--data", default="adult", choices=["adult", "snake", "cali"])
    parser.add_argument("--epsilon", type=float, default=10.0)
    parser.add_argument("--n_holders", type=int, default=2)
    parser.add_argument("--partition", default="horizontal", choices=["horizontal", "vertical"])
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--num_targets", type=int, default=32)
    parser.add_argument("--n_shadow_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        sdg_name=args.sdg,
        data_name=args.data,
        epsilon=args.epsilon,
        n_holders=args.n_holders,
        partition_type=args.partition,
        n_runs=args.n_runs,
        train_size=args.train_size,
        num_targets=args.num_targets,
        n_shadow_runs=args.n_shadow_runs,
        seed=args.seed,
    )
