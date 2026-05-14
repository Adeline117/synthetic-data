"""AIM algorithm wrapper with focal-point tracking.

Faithfully reproduces the original AIM from CaPS (refs/CaPS/private-pgm-master/
mechanisms/aim.py) with added focal-point recording for MAMA-MIA attacks.

Key behaviors preserved from original:
- Budget-based termination (not fixed rounds)
- Adaptive sigma halving when model improvement is small
- Model size filtering of candidate cliques
- Workload scoring via downward closure

Reference: McKenna et al., "AIM: An adaptive and iterative mechanism for
differentially private synthetic data", VLDB 2022.
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


def _powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def _downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(_powerset(proj))
    return list(sorted(ans, key=len))


def _compile_workload(workload):
    def score(cl):
        return sum(len(set(cl) & set(ax)) for ax in workload)
    return {cl: score(cl) for cl in _downward_closure(workload)}


def _hypothetical_model_size(domain, cliques):
    return GraphicalModel(domain, cliques).size * 8 / 2**20


def _filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = _downward_closure(model.cliques)
    for cl in candidates:
        cond1 = _hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


class AIMWithFocalPoints:
    """AIM that records selected cliques as focal points."""

    def __init__(self, epsilon, delta=1e-9, max_model_size=80, rounds=None, seed=None):
        self.epsilon = epsilon
        self.delta = delta
        self.max_model_size = max_model_size
        self.rounds = rounds
        self.seed = seed
        self.focal_points = []

    def _exponential_mechanism(self, errors, eps, sensitivity, prng):
        scores = np.array(list(errors.values()))
        keys = list(errors.keys())
        prob = softmax(0.5 * eps / sensitivity * (scores - scores.max()))
        idx = prng.choice(len(keys), p=prob)
        return keys[idx]

    def run(self, data, workload=None, max_cells=10000):
        """Run AIM and return (synthetic_data, focal_points).

        Args:
            data: mbi.Dataset
            workload: list of (clique, weight) tuples, or None for default 2-way
            max_cells: max domain size per marginal (for default workload)

        Returns:
            synth: mbi.Dataset of synthetic records
            focal_points: list of selected clique tuples
        """
        domain = data.domain
        prng = np.random.RandomState(self.seed)
        self.focal_points = []

        # Build workload
        if workload is None:
            cliques = list(itertools.combinations(domain.attrs, 2))
            cliques = [cl for cl in cliques if domain.size(cl) <= max_cells]
            workload = [(cl, 1.0) for cl in cliques]

        W = [cl for cl, _ in workload]
        candidates = _compile_workload(W)

        # Pre-compute true answers
        answers = {cl: data.project(cl).datavector() for cl in candidates}

        oneway = [cl for cl in candidates if len(cl) == 1]

        # Privacy budget
        rho = cdp_rho(self.epsilon, self.delta)
        rounds = self.rounds or 16 * len(domain)
        sigma = np.sqrt(rounds / (2 * 0.9 * rho))
        epsilon = np.sqrt(8 * 0.1 * rho / rounds)

        # Initial 1-way marginals
        measurements = []
        rho_used = len(oneway) * 0.5 / sigma**2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + prng.normal(0, sigma, x.size)
            I = sparse.eye(y.size)
            measurements.append((I, y, sigma, cl))

        engine = FactoredInference(domain, iters=1000, warm_start=True, log=False)
        model = engine.estimate(measurements)

        # Iterative measurement
        terminate = False
        max_iters = self.rounds or 16 * len(domain)
        n_iters = 0
        while not terminate and n_iters < max_iters:
            n_iters += 1
            # Budget check: can we afford another round?
            if rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                remaining = rho - rho_used
                if remaining <= 0:
                    break
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            size_limit = self.max_model_size * rho_used / rho

            # Filter candidates by model size
            small_candidates = _filter_candidates(candidates, model, size_limit)
            if not small_candidates:
                break

            # SELECT: worst-approximated clique via exponential mechanism
            errors = {}
            sensitivity_map = {}
            for cl in small_candidates:
                wgt = small_candidates[cl]
                x = answers[cl]
                bias = np.sqrt(2 / np.pi) * sigma * domain.size(cl)
                xest = model.project(cl).datavector()
                errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
                sensitivity_map[cl] = abs(wgt)

            max_sensitivity = max(sensitivity_map.values())
            cl = self._exponential_mechanism(errors, epsilon, max_sensitivity, prng)

            # Record focal point
            self.focal_points.append(cl)

            # MEASURE
            n = domain.size(cl)
            x = data.project(cl).datavector()
            y = x + prng.normal(0, sigma, n)
            Q = sparse.eye(n)
            measurements.append((Q, y, sigma, cl))

            z = model.project(cl).datavector()
            model = engine.estimate(measurements)
            w = model.project(cl).datavector()

            # Adaptive sigma halving
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                sigma /= 2
                epsilon *= 2

        # Final estimation with more iterations
        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data(rows=data.df.shape[0])
        return synth, self.focal_points

    def get_focal_point_frequencies(self):
        """Return frequency count of each focal point (sorted tuples)."""
        return dict(Counter([tuple(sorted(fp)) for fp in self.focal_points]))
