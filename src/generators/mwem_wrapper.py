"""MWEM+PGM algorithm wrapper with focal-point tracking.

Faithfully reproduces the original MWEM+PGM from CaPS (refs/CaPS/private-pgm-master/
mechanisms/mwem+pgm.py) with added focal-point recording for MAMA-MIA attacks.

Key behaviors preserved from original:
- Model size filtering of candidate cliques
- Penalty term (domain size subtracted from error)
- sigma passed as 1.0 to PGM (not actual sigma) for correct inference weighting
- Gaussian noise by default

Reference: McKenna et al., "Graphical-model based estimation and inference
for differential privacy", ICML 2019.
"""

import sys
import os
import itertools
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAPS_SRC = os.path.join(BASE_DIR, "refs", "CaPS", "private-pgm-master", "src")
CAPS_MECH = os.path.join(BASE_DIR, "refs", "CaPS", "private-pgm-master", "mechanisms")
for p in [CAPS_SRC, CAPS_MECH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mbi import Dataset, Domain, GraphicalModel, FactoredInference
from cdp2adp import cdp_rho
from scipy.special import softmax
from scipy import sparse


class MWEMPGMWithFocalPoints:
    """MWEM+PGM that records selected marginals as focal points."""

    def __init__(self, epsilon, delta=1e-9, rounds=None, max_model_size=25,
                 noise="gaussian", alpha=0.9, pgm_iters=1000, seed=None):
        self.epsilon = epsilon
        self.delta = delta
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.noise = noise
        self.alpha = alpha
        self.pgm_iters = pgm_iters
        self.seed = seed
        self.focal_points = []

    def run(self, data, workload=None, max_cells=10000, bounded=False):
        """Run MWEM+PGM and return (synthetic_data, focal_points).

        Args:
            data: mbi.Dataset
            workload: list of attribute tuples, or None for default 2-way
            max_cells: max domain size per marginal
            bounded: bounded vs unbounded DP

        Returns:
            synth: mbi.Dataset of synthetic records
            focal_points: list of selected marginal tuples
        """
        domain = data.domain
        self.focal_points = []

        if workload is None:
            workload = list(itertools.combinations(domain.attrs, 2))
            workload = [cl for cl in workload if domain.size(cl) <= max_cells]

        rounds = self.rounds or len(domain)
        total = data.records if bounded else None

        # Privacy budget allocation
        if self.noise == "laplace":
            eps_per_round = self.epsilon / rounds
            sigma = 1.0 / (self.alpha * eps_per_round)
            exp_eps = (1 - self.alpha) * eps_per_round
            marginal_sensitivity = 2.0 if bounded else 1.0
        else:
            rho = cdp_rho(self.epsilon, self.delta)
            rho_per_round = rho / rounds
            sigma = np.sqrt(0.5 / (self.alpha * rho_per_round))
            exp_eps = np.sqrt(8 * (1 - self.alpha) * rho_per_round)
            marginal_sensitivity = np.sqrt(2) if bounded else 1.0

        # Model size helper
        def model_size(cliques):
            return GraphicalModel(domain, cliques).size * 8 / 2**20

        # Pre-compute true answers
        workload_answers = {cl: data.project(cl).datavector() for cl in workload}

        # Set random state
        if self.seed is not None:
            np.random.seed(self.seed)

        engine = FactoredInference(domain, log=False, iters=self.pgm_iters, warm_start=True)
        measurements = []
        est = engine.estimate(measurements, total)
        cliques = []

        for i in range(1, rounds + 1):
            # Filter candidates by model size constraint
            candidates = [
                cl for cl in workload
                if model_size(cliques + [cl]) <= self.max_model_size * i / rounds
            ]
            if not candidates:
                continue

            # SELECT: exponential mechanism over approximation errors
            errors = np.array([])
            for cl in candidates:
                bias = domain.size(cl)
                x = workload_answers[cl]
                xest = est.project(cl).datavector()
                errors = np.append(errors, np.abs(x - xest).sum() - bias)

            sensitivity = 2.0 if bounded else 1.0
            prob = softmax(0.5 * exp_eps / sensitivity * (errors - errors.max()))
            key = np.random.choice(len(errors), p=prob)
            ax = candidates[key]

            # Record focal point
            self.focal_points.append(ax)

            # MEASURE: noisy marginal
            n = domain.size(ax)
            x = data.project(ax).datavector()
            if self.noise == "laplace":
                y = x + np.random.laplace(loc=0, scale=marginal_sensitivity * sigma, size=n)
            else:
                y = x + np.random.normal(loc=0, scale=marginal_sensitivity * sigma, size=n)

            Q = sparse.eye(n)
            # NOTE: original passes 1.0 as sigma to PGM, not actual sigma
            measurements.append((Q, y, 1.0, ax))
            est = engine.estimate(measurements, total)
            cliques.append(ax)

        # Generate synthetic data
        synth = est.synthetic_data(rows=data.df.shape[0])
        return synth, self.focal_points

    def get_focal_point_frequencies(self):
        """Return frequency count of each focal point (sorted tuples)."""
        return dict(Counter([tuple(sorted(fp)) for fp in self.focal_points]))
