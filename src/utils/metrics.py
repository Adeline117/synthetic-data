"""Evaluation metrics for membership inference attacks.

Adapted from MAMA-MIA util.py (Golob et al., 2025).
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix


def membership_advantage(y_true, scores):
    """Compute Membership Advantage (MA).

    MA = (TPR - FPR + 1) / 2, weighted by prediction confidence.
    """
    y_pred = scores > 0.5
    sample_weight = 2 * np.abs(0.5 - scores)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    return (tpr - fpr + 1) / 2


def area_under_curve(y_true, predictions):
    """Compute AUC from ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, predictions)
        return auc(fpr, tpr)
    except ValueError:
        return None


def get_roc(y_true, predictions):
    """Return (fpr, tpr, thresholds) for ROC curve."""
    try:
        return roc_curve(y_true, predictions)
    except ValueError:
        return None, None, None


# ---------------------------------------------------------------------------
# Activation functions: convert raw density ratios to probabilities
# ---------------------------------------------------------------------------

def activate_3(p_rel, confidence=1, center=True):
    """Log-zscore sigmoid activation (best performer in MAMA-MIA)."""
    logs = np.log(np.clip(p_rel, 1e-300, None))
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    return 1 / (1 + np.exp(-confidence * (zscores - median)))


def activate_1(p_rel, confidence=1, center=True):
    """Log sigmoid activation."""
    logs = np.log(np.clip(p_rel, 1e-300, None))
    median = np.median(logs) if center else 0
    return 1 / (1 + np.exp(-confidence * (logs - median)))


# ---------------------------------------------------------------------------
# Score attack: convert density ratios to MA/AUC
# ---------------------------------------------------------------------------

def score_attack(density_ratios, num_queries_used, membership, set_MI=False,
                 targets=None, target_ids=None, activation_fn=None):
    """Convert raw density ratios to predictions and evaluate.

    Args:
        density_ratios: array of aggregated D_synth/D_aux ratios per target
        num_queries_used: array of query counts per target
        membership: binary array (1 = member)
        set_MI: whether to aggregate by household
        targets: DataFrame (needed for set_MI)
        target_ids: array of target IDs (needed for set_MI)
        activation_fn: function to convert ratios to probabilities

    Returns:
        (predictions, ma, auc_score, roc)
    """
    if activation_fn is None:
        activation_fn = activate_3

    # Normalize by number of queries used
    safe_counts = np.maximum(num_queries_used, 1)
    normalized = density_ratios / safe_counts

    # Convert to probabilities
    predictions = activation_fn(normalized)

    if set_MI and targets is not None and target_ids is not None:
        # Aggregate predictions by household
        targets_copy = targets.copy()
        targets_copy["pred"] = predictions
        grouped = targets_copy.groupby("HHID")["pred"].mean()
        predictions = np.array([grouped.get(hid, 0.5) for hid in target_ids])

    ma = membership_advantage(membership, predictions)
    auc_score = area_under_curve(membership, predictions)
    roc = get_roc(membership, predictions)

    return predictions, ma, auc_score, roc
