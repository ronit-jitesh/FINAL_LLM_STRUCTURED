"""
utils/evaluate.py
==================
Common evaluation utilities for the NLI classification project.
All functions are typed and self-contained — no side effects on import.

Usage:
    from utils.evaluate import compute_metrics, per_class_report, mcnemar_pair
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

LABELS: List[str] = ["entailment", "neutral", "contradiction"]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: List[str] = LABELS,
) -> Dict[str, float]:
    """
    Compute accuracy and macro F1 for a single system.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Label order for F1 computation.

    Returns:
        Dict with keys: acc, macro_f1.
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    return {"acc": round(acc, 4), "macro_f1": round(f1, 4)}


def per_class_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    labels: List[str] = LABELS,
) -> List[Dict]:
    """
    Compute per-class Precision, Recall, F1 for a model.

    Args:
        y_true:     Ground-truth labels.
        y_pred:     Predicted labels.
        model_name: Label for this system (used in output rows).
        labels:     Label order for metric computation.

    Returns:
        List of dicts, one per class: {model, class, precision, recall, f1, support}.
    """
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return [
        {
            "model":     model_name,
            "class":     label,
            "precision": round(float(p[i]), 4),
            "recall":    round(float(r[i]), 4),
            "f1":        round(float(f[i]), 4),
            "support":   int(s[i]),
        }
        for i, label in enumerate(labels)
    ]


def confusion_matrix_df(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: List[str] = LABELS,
) -> pd.DataFrame:
    """
    Return a labelled confusion matrix as a DataFrame.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Label order for rows/columns.

    Returns:
        DataFrame with shape (len(labels), len(labels)).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def mcnemar_pair(
    y_true: pd.Series | np.ndarray,
    y_pred_a: pd.Series | np.ndarray,
    y_pred_b: pd.Series | np.ndarray,
) -> Dict[str, float | bool | str]:
    """
    Run McNemar's test comparing system A vs system B on the same test set.

    Uses exact binomial test when discordant pairs < 25, chi-squared otherwise.
    Applies continuity correction in both cases.

    Args:
        y_true:   Shared ground-truth labels (same for both systems).
        y_pred_a: Predictions from system A.
        y_pred_b: Predictions from system B.

    Returns:
        Dict with keys: n, acc_a, acc_b, diff_pp, discordant_pairs,
                        statistic, p_value, significant_p05, significant_p01, verdict.

    Reference:
        McNemar (1947). Psychometrika, 12(2), 153-157.
    """
    from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

    correct_a = np.asarray(y_true) == np.asarray(y_pred_a)
    correct_b = np.asarray(y_true) == np.asarray(y_pred_b)

    n10 = int((correct_a & ~correct_b).sum())   # A right, B wrong
    n01 = int((~correct_a & correct_b).sum())   # A wrong, B right
    n11 = int((correct_a & correct_b).sum())    # both right
    n00 = int((~correct_a & ~correct_b).sum())  # both wrong

    table = [[n11, n10], [n01, n00]]
    discordant = n01 + n10
    use_exact = discordant < 25

    result = _mcnemar(table, exact=use_exact, correction=True)
    p = float(result.pvalue)

    return {
        "n":                  len(y_true),
        "acc_a":              round(float(correct_a.mean()) * 100, 2),
        "acc_b":              round(float(correct_b.mean()) * 100, 2),
        "diff_pp":            round(float((correct_b.mean() - correct_a.mean()) * 100), 2),
        "discordant_pairs":   discordant,
        "test_type":          "exact_binomial" if use_exact else "chi_squared",
        "statistic":          round(float(result.statistic), 4),
        "p_value":            round(p, 4),
        "significant_p05":    p < 0.05,
        "significant_p01":    p < 0.01,
        "verdict": (
            "*** p<0.01" if p < 0.01 else
            "**  p<0.05" if p < 0.05 else
            "ns  (not significant)"
        ),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Splits confidence scores into n_bins equal-width bins and computes
    the weighted average of |accuracy - confidence| per bin.

    Args:
        confidences: Predicted confidence scores in [0, 1].
        correct:     Boolean array — True where prediction == ground truth.
        n_bins:      Number of equal-width bins (default 10).

    Returns:
        ECE as a float in [0, 1]. Lower is better.

    Reference:
        Naeini et al. (2015). AAAI.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = float(correct[mask].mean())
        bin_conf = float(confidences[mask].mean())
        bin_n    = int(mask.sum())
        ece += (bin_n / n) * abs(bin_acc - bin_conf)

    return round(ece, 4)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def cost_per_1k(cost_usd_series: pd.Series) -> float:
    """
    Compute cost per 1,000 queries from a series of per-query costs.

    Args:
        cost_usd_series: Pandas Series of per-query cost in USD.

    Returns:
        Cost per 1,000 queries, rounded to 4 decimal places.
    """
    return round(float(cost_usd_series.mean()) * 1000, 4)
