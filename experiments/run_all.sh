#!/bin/bash
# Run all experiments for the paper.
# Usage: bash experiments/run_all.sh

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "Phase 1: Centralized Experiments (C1)"
echo "========================================"

for SDG in mwem_pgm aim; do
  for DATA in snake; do
    for EPS in 0.1 1.0 10.0 100.0 1000.0; do
      echo ""
      echo ">>> $SDG / $DATA / eps=$EPS"
      python3 experiments/run_centralized.py \
        --sdg $SDG --data $DATA --epsilon $EPS \
        --n_runs 10 --train_size 1000 --num_targets 32 \
        --n_shadow_runs 15 --seed 42
    done
  done
done

echo ""
echo "========================================"
echo "Phase 2: Distributed Experiments (C2)"
echo "========================================"

for SDG in mwem_pgm aim; do
  for DATA in snake; do
    for EPS in 1.0 10.0 100.0; do
      echo ""
      echo ">>> Distributed: $SDG / $DATA / eps=$EPS"
      python3 experiments/run_distributed.py \
        --sdg $SDG --data $DATA --epsilon $EPS \
        --n_holders 2 --partition horizontal \
        --n_runs 5 --train_size 1000 --num_targets 32 \
        --n_shadow_runs 10 --seed 42
    done
  done
done

echo ""
echo "========================================"
echo "Analysis"
echo "========================================"
python3 experiments/analyze_results.py

echo "Done!"
