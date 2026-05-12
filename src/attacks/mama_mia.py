"""MAMA-MIA attack orchestration.

Ties together focal-point extraction (shadow modeling) and density ratio
attack for a complete end-to-end membership inference pipeline.
"""

import time
import numpy as np

from attacks.focal_points import (
    determine_aim_focal_points,
    determine_mwem_pgm_focal_points,
    save_focal_points,
    load_focal_points,
)
from attacks.density import marginal_density_attack
from generators.aim_wrapper import AIMWithFocalPoints
from generators.mwem_wrapper import MWEMPGMWithFocalPoints
from utils.data import pandas_to_mbi, mbi_to_pandas


# SDG name -> focal point function
_FP_FUNCTIONS = {
    "aim": determine_aim_focal_points,
    "mwem_pgm": determine_mwem_pgm_focal_points,
}

# SDG name -> generator class
_GENERATORS = {
    "aim": AIMWithFocalPoints,
    "mwem_pgm": MWEMPGMWithFocalPoints,
}


class MaMAMIAAttack:
    """End-to-end MAMA-MIA attack for marginals-based SDG algorithms."""

    def __init__(self, sdg_name, data_name="snake"):
        if sdg_name not in _FP_FUNCTIONS:
            raise ValueError(f"Unknown SDG: {sdg_name}. Supported: {list(_FP_FUNCTIONS.keys())}")
        self.sdg_name = sdg_name
        self.data_name = data_name

    def shadow_model(self, aux_df, domain, eps, n_size, n_shadow_runs=50,
                     seed=None, cache=True, artifact_dir=None):
        """Run shadow modeling to determine focal points.

        Args:
            aux_df: auxiliary DataFrame
            domain: mbi.Domain
            eps: privacy budget
            n_size: training size per shadow run
            n_shadow_runs: number of shadow runs
            seed: random seed
            cache: if True, load/save from disk

        Returns:
            dict of focal point weights
        """
        if cache:
            fps = load_focal_points(self.sdg_name, eps, self.data_name, artifact_dir)
            if fps is not None:
                print(f"Loaded cached focal points for {self.sdg_name} eps={eps}")
                return fps

        print(f"Running shadow modeling for {self.sdg_name} eps={eps}...")
        start = time.time()

        fp_fn = _FP_FUNCTIONS[self.sdg_name]
        fps = fp_fn(aux_df, domain, eps, n_size, n_shadow_runs, seed=seed)

        elapsed = time.time() - start
        print(f"Shadow modeling done in {elapsed:.1f}s ({len(fps)} unique FPs)")

        if cache:
            save_focal_points(fps, self.sdg_name, eps, self.data_name, artifact_dir)

        return fps

    def generate(self, train_df, domain, eps, delta=1e-9, seed=None):
        """Generate synthetic data using the target SDG algorithm.

        Returns:
            synth_df: synthetic pandas DataFrame
            fps: focal points from this run (for reference)
        """
        mbi_data = pandas_to_mbi(train_df, domain)
        gen_cls = _GENERATORS[self.sdg_name]
        gen = gen_cls(epsilon=eps, delta=delta, seed=seed)
        synth_mbi, fps = gen.run(mbi_data)
        return mbi_to_pandas(synth_mbi), fps

    def attack(self, synth_df, aux_df, targets_df, membership, fps, eps,
               set_MI=False, target_ids=None):
        """Run the MAMA-MIA density ratio attack.

        Args:
            synth_df: synthetic DataFrame
            aux_df: auxiliary DataFrame
            targets_df: target records
            membership: binary membership labels
            fps: focal point weights from shadow modeling
            eps: privacy budget
            set_MI: set membership inference
            target_ids: target identifiers

        Returns:
            predictions, ma, auc, roc
        """
        return marginal_density_attack(
            synth_df, aux_df, targets_df, membership, fps, eps,
            set_MI=set_MI, target_ids=target_ids,
        )

    def run_full_pipeline(self, train_df, aux_df, targets_df, membership,
                          domain, eps, n_shadow_size=None, n_shadow_runs=50,
                          set_MI=False, target_ids=None, seed=None,
                          cache_fps=True, artifact_dir=None):
        """Run complete attack: shadow model -> generate -> attack.

        Returns:
            dict with keys: predictions, ma, auc, roc, fps, synth_df, timings
        """
        timings = {}
        n_shadow_size = n_shadow_size or train_df.shape[0]

        # Phase 1: Shadow modeling
        t0 = time.time()
        fps = self.shadow_model(
            aux_df, domain, eps, n_shadow_size, n_shadow_runs,
            seed=seed, cache=cache_fps, artifact_dir=artifact_dir,
        )
        timings["shadow_model"] = time.time() - t0

        # Phase 2: Generate synthetic data
        t0 = time.time()
        synth_df, _ = self.generate(train_df, domain, eps, seed=seed)
        timings["generate"] = time.time() - t0

        # Phase 3: Attack
        t0 = time.time()
        predictions, ma, auc_score, roc = self.attack(
            synth_df, aux_df, targets_df, membership, fps, eps,
            set_MI=set_MI, target_ids=target_ids,
        )
        timings["attack"] = time.time() - t0

        return {
            "predictions": predictions,
            "ma": ma,
            "auc": auc_score,
            "roc": roc,
            "fps": fps,
            "synth_df": synth_df,
            "timings": timings,
        }
