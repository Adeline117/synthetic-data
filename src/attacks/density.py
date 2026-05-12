"""Density ratio attack for marginals-based SDG algorithms.

Computes the MAMA-MIA density estimation zeta: for each target record,
aggregate D_synth(FP) / D_aux(FP) across weighted focal points.

This implementation handles AIM, MWEM+PGM, and MST, all of which produce
k-way marginals as focal points. The density ratio logic is identical
for all three (adapted from MAMA-MIA conduct_attacks.py custom_mst_attack).
"""

import numpy as np
from utils.config import get_weight_threshold
from utils.metrics import score_attack


def marginal_density_attack(synth_df, aux_df, targets_df, membership,
                            fp_weights, eps, set_MI=False,
                            target_ids=None):
    """Run MAMA-MIA attack using marginal-based focal points.

    This works for any SDG that produces marginals as focal points:
    AIM, MWEM+PGM, MST.

    Args:
        synth_df: synthetic DataFrame
        aux_df: auxiliary DataFrame
        targets_df: target records DataFrame
        membership: binary array (1 = member)
        fp_weights: dict mapping marginal tuple -> weight
        eps: privacy budget (for threshold selection)
        set_MI: whether to use set membership inference
        target_ids: array of target IDs (for set MI)

    Returns:
        predictions: probability array
        ma: Membership Advantage
        auc_score: Area Under ROC Curve
        roc: (fpr, tpr, thresholds) tuple
    """
    n_targets = targets_df.shape[0]
    A = np.zeros(n_targets)
    num_queries_used = np.zeros(n_targets, dtype=int)

    threshold = _determine_weight_threshold(fp_weights, eps)

    for marginal, weight in fp_weights.items():
        if weight < threshold:
            continue

        marginal_list = list(marginal)

        # Check all columns exist in data
        if not all(c in synth_df.columns and c in aux_df.columns and c in targets_df.columns
                   for c in marginal_list):
            continue

        # Compute marginal distributions
        D_synth = synth_df[marginal_list].value_counts(normalize=True)
        D_aux = aux_df[marginal_list].value_counts(normalize=True)

        default_val = 1e-10
        target_vals = targets_df[marginal_list].values

        ratios = np.array([
            weight * D_synth.get(tuple(val), default=default_val)
            / max(D_aux.get(tuple(val), default=default_val), default_val)
            for val in target_vals
        ])

        A += ratios
        num_queries_used += 1

    return score_attack(
        A, num_queries_used, membership,
        set_MI=set_MI, targets=targets_df, target_ids=target_ids,
    )


def _determine_weight_threshold(fp_weights, eps):
    """Compute the minimum FP weight to include in attack.

    Low-weight FPs are noise-induced and hurt attack accuracy.
    Higher epsilon -> more deterministic FP selection -> use more FPs.
    """
    if not fp_weights:
        return 0
    max_weight = max(fp_weights.values())
    fraction = get_weight_threshold(eps)
    return max_weight * fraction
