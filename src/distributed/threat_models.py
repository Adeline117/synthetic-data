"""Three threat models for MIA in the CaPS distributed setting.

Threat Model A (External Attacker):
    Sees only D_synth + knows which algorithm was used.
    Identical to standard MAMA-MIA — doesn't know data was distributed.

Threat Model B (Malicious Data Holder):
    One of N data holders. Has own D_i + D_synth + knows algorithm.
    Enhanced shadow model: conditions on own data partition, simulates
    only the other holders' data from auxiliary.

Threat Model C (Colluding MPC Server):
    Semi-honest MPC server that sees protocol transcripts.
    Knows EXACT focal points (no shadow modeling needed) and can use
    noisy measurements directly for better density estimation.
"""

import time
import numpy as np
from collections import Counter

from attacks.density import marginal_density_attack
from attacks.focal_points import _aggregate_fps
from distributed.caps_simulator import CaPSSimulator
from distributed.partitioner import horizontal_partition
from utils.data import pandas_to_mbi, mbi_to_pandas


class ExternalAttacker:
    """Threat Model A: External attacker, identical to centralized MAMA-MIA."""

    def __init__(self, sdg_name, data_name="snake"):
        self.sdg_name = sdg_name
        self.data_name = data_name

    def shadow_model(self, aux_df, domain, eps, n_size, n_shadow_runs=50,
                     n_holders=2, partition_type="horizontal", seed=None):
        """Shadow model without knowledge of distribution.

        The external attacker doesn't know data is distributed, so they
        run the standard centralized shadow model.
        """
        from attacks.focal_points import (
            determine_aim_focal_points,
            determine_mwem_pgm_focal_points,
        )

        fp_fn = {
            "aim": determine_aim_focal_points,
            "mwem_pgm": determine_mwem_pgm_focal_points,
        }[self.sdg_name]

        return fp_fn(aux_df, domain, eps, n_size, n_shadow_runs, seed=seed)

    def attack(self, synth_df, aux_df, targets_df, membership, fps, eps,
               target_ids=None):
        return marginal_density_attack(
            synth_df, aux_df, targets_df, membership, fps, eps,
            target_ids=target_ids,
        )


class MaliciousDataHolder:
    """Threat Model B: One data holder attacks other holders' members.

    The malicious holder knows their own data partition, so they can
    condition the shadow model on it — simulating only the unknown
    partitions from auxiliary data.
    """

    def __init__(self, sdg_name, holder_index=0, data_name="snake"):
        self.sdg_name = sdg_name
        self.holder_index = holder_index
        self.data_name = data_name

    def shadow_model(self, own_data_df, aux_df, domain, eps, n_size,
                     n_holders=2, partition_type="horizontal",
                     n_shadow_runs=50, seed=None):
        """Enhanced shadow model conditioned on own data.

        For horizontal partition:
        - Fix own partition to actual data
        - Sample other partitions from auxiliary
        - Run CaPS simulator
        - Record focal points

        This gives more accurate FP frequencies since one partition is exact.
        """
        all_fps = []
        rng = np.random.RandomState(seed)
        own_size = own_data_df.shape[0]
        other_size = n_size - own_size

        for i in range(n_shadow_runs):
            # Build partitions: own data is fixed, others are sampled
            partitions = []
            for h in range(n_holders):
                if h == self.holder_index:
                    partitions.append(own_data_df.copy())
                else:
                    sample_size = other_size // (n_holders - 1)
                    sample = aux_df.sample(n=sample_size, random_state=rng.randint(100000))
                    partitions.append(sample)

            # Run CaPS simulation
            sim = CaPSSimulator(
                domain, partition_type, eps,
                seed=rng.randint(100000),
            )
            try:
                if self.sdg_name == "mwem_pgm":
                    _, fps = sim.run_mwem_pgm(partitions)
                else:
                    _, fps = sim.run_aim(partitions)
            except Exception as e:
                print(f"Shadow run {i} failed: {e}")
                continue

            all_fps.append(fps)

        return _aggregate_fps(all_fps)

    def attack(self, synth_df, own_data_df, aux_df, targets_df, membership,
               fps, eps, target_ids=None):
        """Enhanced attack using own data for better baseline estimation.

        Instead of using full aux as denominator, use own data to better
        estimate the population distribution for columns the holder owns.
        """
        # For now, use standard attack. Enhancement: combine own_data + aux
        # for a more accurate D_aux estimate.
        combined_aux = aux_df
        return marginal_density_attack(
            synth_df, combined_aux, targets_df, membership, fps, eps,
            target_ids=target_ids,
        )


class ColludingServer:
    """Threat Model C: Semi-honest MPC server with protocol transcript.

    The strongest attacker: directly observes which marginals were selected
    and the noisy measurements. No shadow modeling needed.
    """

    def __init__(self, sdg_name, data_name="snake"):
        self.sdg_name = sdg_name
        self.data_name = data_name

    def extract_focal_points(self, transcript):
        """Extract exact focal points from protocol transcript.

        No shadow modeling — the server saw which marginals were selected.
        """
        fps = []
        for entry in transcript:
            if entry.get("phase") == "round" and "selected" in entry:
                fps.append(entry["selected"])
        # Weight = 1 for each (all were definitely used)
        freq = Counter([tuple(sorted(fp)) for fp in fps])
        return dict(freq)

    def extract_measurements(self, transcript):
        """Extract noisy measurements from transcript.

        These can be used for enhanced density estimation — the server
        knows the exact noisy marginals that shaped the synthetic data.
        """
        measurements = []
        for entry in transcript:
            if entry.get("phase") == "round":
                measurements.append({
                    "marginal": entry["selected"],
                    "noisy_y": entry["noisy_measurement"],
                    "sigma": entry["sigma"],
                })
        return measurements

    def attack(self, synth_df, aux_df, targets_df, membership, transcript,
               eps, target_ids=None):
        """Attack using exact knowledge from protocol transcript."""
        fps = self.extract_focal_points(transcript)
        return marginal_density_attack(
            synth_df, aux_df, targets_df, membership, fps, eps,
            target_ids=target_ids,
        )
