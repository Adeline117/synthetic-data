"""CaPS MPC simulation with focal-point tracking and protocol transcript logging.

Simulates the CaPS DP-in-MPC framework for MWEM+PGM and AIM in distributed
settings. Follows the original CaPS MPC.py and mwem+pgm+MPC_H/V.py patterns.

The key insight: MPC servers compute marginals via secret-shared addition of
local histograms. The OUTPUT (synthetic data) is in the same format as
centralized — only the generation process differs. We track:
1. Which focal points (marginals) are selected each round
2. The full protocol transcript (for threat model C)
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

from distributed.partitioner import horizontal_partition, vertical_partition


def _compute_local_histograms(partition_df, domain, workload, max_domain_size):
    """Compute local marginal histograms for a data holder's partition.

    Follows CaPS DataHolders.py HDataHolder.compute_answers().
    """
    answers = []
    for cl in workload:
        shape = domain.project(cl).shape
        bins = [range(n + 1) for n in shape]
        ans = np.histogramdd(partition_df[list(cl)].values, bins)[0]
        data_vector = ans.flatten()
        padded = np.pad(data_vector, (0, max_domain_size - len(data_vector)), "constant")
        answers.append(padded)
    return answers


class CaPSSimulator:
    """Simulate CaPS distributed SDG with focal-point and transcript tracking."""

    def __init__(self, domain, partition_type="horizontal", epsilon=1.0,
                 delta=1e-9, seed=None):
        self.domain = domain
        self.partition_type = partition_type
        self.epsilon = epsilon
        self.delta = delta
        self.seed = seed
        self.focal_points = []
        self.transcript = []  # Protocol transcript for threat model C

    def run_mwem_pgm(self, data_partitions, workload=None, rounds=None,
                     max_model_size=25, max_cells=10000, noise="gaussian",
                     alpha=0.9, pgm_iters=1000, bounded=False):
        """Run MWEM+PGM in distributed (CaPS) setting.

        Args:
            data_partitions: list of pandas DataFrames (one per holder)
            workload: list of attribute tuples
            rounds: number of MWEM rounds
            max_model_size: model size limit in MB
            noise: "gaussian" or "laplace"

        Returns:
            synth: mbi.Dataset of synthetic records
            focal_points: list of selected marginals
        """
        self.focal_points = []
        self.transcript = []
        n_holders = len(data_partitions)
        total_records = sum(p.shape[0] for p in data_partitions)

        if workload is None:
            workload = list(itertools.combinations(self.domain.attrs, 2))
            workload = [cl for cl in workload if self.domain.size(cl) <= max_cells]

        rounds = rounds or len(self.domain)
        rng = np.random.RandomState(self.seed)

        # Privacy budget
        if noise == "laplace":
            eps_per_round = self.epsilon / rounds
            sigma = 1.0 / (alpha * eps_per_round)
            exp_eps = (1 - alpha) * eps_per_round
            marginal_sensitivity = 2.0 if bounded else 1.0
        else:
            rho = cdp_rho(self.epsilon, self.delta)
            rho_per_round = rho / rounds
            sigma = np.sqrt(0.5 / (alpha * rho_per_round))
            exp_eps = np.sqrt(8 * (1 - alpha) * rho_per_round)
            marginal_sensitivity = np.sqrt(2) if bounded else 1.0

        total = total_records if bounded else None

        # Model size helper
        def model_size(cliques):
            return GraphicalModel(self.domain, cliques).size * 8 / 2**20

        # --- π_COMP: Compute marginals via MPC (secret-shared addition) ---
        max_domain_size = max(self.domain.size(cl) for cl in workload)

        # Each holder computes local histograms
        local_answers = []
        for p_df in data_partitions:
            local_answers.append(
                _compute_local_histograms(p_df, self.domain, workload, max_domain_size)
            )

        # MPC aggregation: sum local histograms
        aggregated_answers = {}
        for w_idx, cl in enumerate(workload):
            total_hist = sum(la[w_idx] for la in local_answers)
            aggregated_answers[cl] = total_hist[: self.domain.size(cl)]

        # Log to transcript
        self.transcript.append({
            "phase": "pi_COMP",
            "n_holders": n_holders,
            "workload_size": len(workload),
            "total_records": total_records,
        })

        # --- Iterative measurement (same as centralized, but using aggregated answers) ---
        engine = FactoredInference(self.domain, log=False, iters=pgm_iters, warm_start=True)
        measurements = []
        est = engine.estimate(measurements, total)
        cliques = []

        for i in range(1, rounds + 1):
            # Filter candidates by model size
            candidates = [
                cl for cl in workload
                if model_size(cliques + [cl]) <= max_model_size * i / rounds
            ]
            if not candidates:
                continue

            # --- π_SELECT: Exponential mechanism in MPC ---
            errors = np.array([])
            for cl in candidates:
                bias = self.domain.size(cl)
                x = aggregated_answers[cl]
                xest = est.project(cl).datavector()
                errors = np.append(errors, np.abs(x - xest).sum() - bias)

            sensitivity = 2.0 if bounded else 1.0
            prob = softmax(0.5 * exp_eps / sensitivity * (errors - errors.max()))
            key = rng.choice(len(errors), p=prob)
            ax = candidates[key]

            # Record focal point
            self.focal_points.append(ax)

            # --- π_MEASURE: Noisy marginal ---
            n = self.domain.size(ax)
            x = aggregated_answers[ax]
            if noise == "laplace":
                y = x + rng.laplace(loc=0, scale=marginal_sensitivity * sigma, size=n)
            else:
                y = x + rng.normal(loc=0, scale=marginal_sensitivity * sigma, size=n)

            Q = sparse.eye(n)
            measurements.append((Q, y, 1.0, ax))
            est = engine.estimate(measurements, total)
            cliques.append(ax)

            # Log to transcript
            self.transcript.append({
                "phase": "round",
                "round": i,
                "selected": ax,
                "noisy_measurement": y.copy(),
                "sigma": sigma,
                "errors": errors.copy(),
                "probabilities": prob.copy(),
            })

        # Generate synthetic data
        synth = est.synthetic_data(rows=total_records)
        self.transcript.append({
            "phase": "generate",
            "n_records": total_records,
        })

        return synth, self.focal_points

    def run_aim(self, data_partitions, workload=None, max_model_size=80,
                max_cells=10000):
        """Run AIM in distributed (CaPS) setting.

        Note: CaPS repo doesn't have aim+MPC, so we implement it following
        the same DP-in-MPC pattern as mwem+pgm+MPC.
        """
        self.focal_points = []
        self.transcript = []
        n_holders = len(data_partitions)
        total_records = sum(p.shape[0] for p in data_partitions)

        if workload is None:
            wl_tuples = list(itertools.combinations(self.domain.attrs, 2))
            wl_tuples = [cl for cl in wl_tuples if self.domain.size(cl) <= max_cells]
            workload = [(cl, 1.0) for cl in wl_tuples]

        W = [cl for cl, _ in workload]

        # Build candidates via downward closure
        def powerset(iterable):
            s = list(iterable)
            return itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(1, len(s) + 1)
            )

        def downward_closure(Ws):
            ans = set()
            for proj in Ws:
                ans.update(powerset(proj))
            return list(sorted(ans, key=len))

        def compile_workload(wl):
            def score(cl):
                return sum(len(set(cl) & set(ax)) for ax in wl)
            return {cl: score(cl) for cl in downward_closure(wl)}

        candidates = compile_workload(W)
        rng = np.random.RandomState(self.seed)

        # Privacy budget
        rho = cdp_rho(self.epsilon, self.delta)
        rounds_est = 16 * len(self.domain)
        sigma = np.sqrt(rounds_est / (2 * 0.9 * rho))
        epsilon = np.sqrt(8 * 0.1 * rho / rounds_est)

        # --- π_COMP: Aggregate local histograms ---
        max_domain_size = max(self.domain.size(cl) for cl in candidates)
        local_answers = []
        for p_df in data_partitions:
            holder_ans = {}
            for cl in candidates:
                shape = self.domain.project(cl).shape
                bins = [range(n + 1) for n in shape]
                try:
                    ans = np.histogramdd(p_df[list(cl)].values, bins)[0].flatten()
                except KeyError:
                    ans = np.zeros(self.domain.size(cl))
                holder_ans[cl] = ans
            local_answers.append(holder_ans)

        aggregated = {
            cl: sum(la[cl] for la in local_answers) for cl in candidates
        }

        self.transcript.append({
            "phase": "pi_COMP",
            "n_holders": n_holders,
            "n_candidates": len(candidates),
        })

        # Initial 1-way marginals
        oneway = [cl for cl in candidates if len(cl) == 1]
        measurements = []
        rho_used = len(oneway) * 0.5 / sigma**2

        for cl in oneway:
            x = aggregated[cl]
            y = x + rng.normal(0, sigma, x.size)
            I = sparse.eye(y.size)
            measurements.append((I, y, sigma, cl))

        engine = FactoredInference(self.domain, iters=1000, warm_start=True, log=False)
        model = engine.estimate(measurements)

        # Iterative rounds
        terminate = False
        while not terminate:
            if rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                remaining = rho - rho_used
                if remaining <= 0:
                    break
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            size_limit = max_model_size * rho_used / rho

            # Filter candidates
            free_cliques = downward_closure(model.cliques)
            small_candidates = {}
            for cl in candidates:
                cond1 = (
                    GraphicalModel(self.domain, model.cliques + [cl]).size * 8 / 2**20
                    <= size_limit
                )
                cond2 = cl in free_cliques
                if cond1 or cond2:
                    small_candidates[cl] = candidates[cl]

            if not small_candidates:
                break

            # π_SELECT: exponential mechanism
            errors = {}
            for cl in small_candidates:
                wgt = small_candidates[cl]
                x = aggregated[cl]
                bias = np.sqrt(2 / np.pi) * sigma * self.domain.size(cl)
                xest = model.project(cl).datavector()
                errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)

            scores = np.array(list(errors.values()))
            keys = list(errors.keys())
            max_sens = max(abs(small_candidates[cl]) for cl in small_candidates)
            prob = softmax(0.5 * epsilon / max_sens * (scores - scores.max()))
            idx = rng.choice(len(keys), p=prob)
            cl = keys[idx]

            self.focal_points.append(cl)

            # π_MEASURE
            n = self.domain.size(cl)
            x = aggregated[cl]
            y = x + rng.normal(0, sigma, n)
            Q = sparse.eye(n)
            measurements.append((Q, y, sigma, cl))

            z = model.project(cl).datavector()
            model = engine.estimate(measurements)
            w = model.project(cl).datavector()

            self.transcript.append({
                "phase": "round",
                "selected": cl,
                "noisy_measurement": y.copy(),
                "sigma": sigma,
            })

            # Adaptive sigma halving
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                sigma /= 2
                epsilon *= 2

        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data(rows=total_records)

        return synth, self.focal_points

    def get_focal_point_frequencies(self):
        return dict(Counter([tuple(sorted(fp)) for fp in self.focal_points]))

    def get_transcript(self):
        return self.transcript
