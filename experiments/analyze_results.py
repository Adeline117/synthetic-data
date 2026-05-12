"""Analyze and visualize experiment results.

Generates plots and LaTeX tables for the paper.

Usage:
    python experiments/analyze_results.py
"""

import sys
import os
import json
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_results(results_dir):
    """Load all JSON result files from a directory."""
    results = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        key = os.path.basename(path).replace(".json", "")
        results[key] = data
    return results


def print_centralized_table(results_dir):
    """Print centralized results as a table."""
    results = load_results(results_dir)
    if not results:
        print("No centralized results found.")
        return

    print("\n" + "=" * 70)
    print("CENTRALIZED RESULTS (Contribution 1)")
    print("=" * 70)
    print(f"{'SDG':12s} {'Data':8s} {'Epsilon':>8s} {'MA':>12s} {'AUC':>12s}")
    print("-" * 70)

    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['sdg']:12s} {r['data']:8s} {r['epsilon']:8.1f} "
              f"{r['ma_mean']:6.3f}+/-{r['ma_std']:.3f} "
              f"{r['auc_mean']:6.3f}+/-{r['auc_std']:.3f}")


def print_distributed_table(results_dir):
    """Print distributed results comparing threat models."""
    results = load_results(results_dir)
    if not results:
        print("No distributed results found.")
        return

    print("\n" + "=" * 80)
    print("DISTRIBUTED RESULTS (Contribution 2)")
    print("=" * 80)
    print(f"{'SDG':10s} {'Data':6s} {'eps':>5s} {'Part':5s} {'H':>2s} "
          f"{'External':>10s} {'Malicious':>10s} {'Colluding':>10s}")
    print("-" * 80)

    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['sdg']:10s} {r['data']:6s} {r['epsilon']:5.1f} "
              f"{r['partition_type'][:5]:5s} {r['n_holders']:2d} "
              f"{r.get('external_auc_mean', 0):10.3f} "
              f"{r.get('malicious_auc_mean', 0):10.3f} "
              f"{r.get('colluding_auc_mean', 0):10.3f}")


def plot_epsilon_vs_auc(results_dir, output_dir):
    """Plot epsilon vs AUC for centralized experiments."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots")
        return

    results = load_results(results_dir)
    if not results:
        return

    # Group by (sdg, data)
    groups = {}
    for r in results.values():
        key = (r["sdg"], r["data"])
        if key not in groups:
            groups[key] = {"eps": [], "auc": [], "auc_std": []}
        groups[key]["eps"].append(r["epsilon"])
        groups[key]["auc"].append(r["auc_mean"])
        groups[key]["auc_std"].append(r["auc_std"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (sdg, data), vals in sorted(groups.items()):
        order = np.argsort(vals["eps"])
        eps = np.array(vals["eps"])[order]
        aucs = np.array(vals["auc"])[order]
        stds = np.array(vals["auc_std"])[order]
        ax.errorbar(eps, aucs, yerr=stds, marker="o", label=f"{sdg} ({data})")

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (epsilon)")
    ax.set_ylabel("Attack AUC")
    ax.set_title("MAMA-MIA Attack on MWEM+PGM and AIM")
    ax.axhline(y=0.5, color="gray", linestyle="--", label="Random Guess")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "epsilon_vs_auc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_threat_model_comparison(results_dir, output_dir):
    """Plot threat model comparison for distributed experiments."""
    if not HAS_MPL:
        return

    results = load_results(results_dir)
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["external", "malicious", "colluding"]
    colors = ["#2196F3", "#FF9800", "#F44336"]

    x_labels = []
    for i, (key, r) in enumerate(sorted(results.items())):
        label = f"{r['sdg']}\n{r['data']}\neps={r['epsilon']}"
        x_labels.append(label)
        for j, (model, color) in enumerate(zip(models, colors)):
            auc_key = f"{model}_auc_mean"
            if auc_key in r:
                offset = (j - 1) * 0.25
                ax.bar(i + offset, r[auc_key], 0.2, color=color,
                       label=model.capitalize() if i == 0 else "")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Attack AUC")
    ax.set_title("Threat Model Comparison in Distributed Setting")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "threat_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    centralized_dir = os.path.join(BASE_DIR, "results", "centralized")
    distributed_dir = os.path.join(BASE_DIR, "results", "distributed")
    plots_dir = os.path.join(BASE_DIR, "results", "plots")

    print_centralized_table(centralized_dir)
    print_distributed_table(distributed_dir)
    plot_epsilon_vs_auc(centralized_dir, plots_dir)
    plot_threat_model_comparison(distributed_dir, plots_dir)
