"""
utils/generate_tables.py
=========================
Table generation utilities for the NLI classification project.
All functions are typed and return DataFrames or formatted strings.

Usage:
    from utils.generate_tables import (
        encoder_summary_table,
        prompt_comparison_table,
        hybrid_summary_table,
        cost_pareto_table,
    )
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score

LABELS: List[str] = ["entailment", "neutral", "contradiction"]


def _safe_load(results_dir: str, filename: str) -> pd.DataFrame:
    """Load a CSV from the results directory, returning empty DataFrame if missing."""
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [SKIP] {filename} not found in {results_dir}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Section 2: Encoder baselines
# ---------------------------------------------------------------------------

def encoder_summary_table(
    results_dir: str,
    encoder_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Build a summary table of encoder accuracy on matched and mismatched sets.

    Args:
        results_dir:  Path to the results/ directory.
        encoder_cols: Map of {column_name: display_name}. Defaults to all 5 encoders.

    Returns:
        DataFrame with columns: model, matched_acc, mm_acc, matched_f1.
    """
    if encoder_cols is None:
        encoder_cols = {
            "bert_base_pred":        "BERT-base",
            "deberta_v3_small_pred": "DeBERTa-v3-small",
            "roberta_base_pred":     "RoBERTa-base",
            "deberta_v3_base_pred":  "DeBERTa-v3-base",
            "deberta_v3_large_pred": "DeBERTa-v3-large",
        }

    df_m  = _safe_load(results_dir, "encoder_predictions_matched.csv")
    df_mm = _safe_load(results_dir, "encoder_predictions_mm.csv")

    if df_m.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for col, name in encoder_cols.items():
        if col not in df_m.columns:
            continue

        acc_m = float(accuracy_score(df_m["label_text"], df_m[col]))
        f1_m  = float(f1_score(df_m["label_text"], df_m[col],
                               average="macro", labels=LABELS, zero_division=0))
        acc_mm = float(accuracy_score(df_mm["label_text"], df_mm[col])) \
                 if not df_mm.empty and col in df_mm.columns else None

        rows.append({
            "model":       name,
            "matched_acc": round(acc_m * 100, 2),
            "mm_acc":      round(acc_mm * 100, 2) if acc_mm is not None else None,
            "matched_f1":  round(f1_m, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 3: GPT-4o prompt comparison
# ---------------------------------------------------------------------------

def prompt_comparison_table(
    results_dir: str,
    prompts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a prompt comparison table for GPT-4o.

    Args:
        results_dir: Path to the results/ directory.
        prompts:     List of prompt names. Defaults to P1-P4.

    Returns:
        DataFrame with columns: prompt, matched_acc, mm_acc, avg_tokens, cost_per_1k.
    """
    if prompts is None:
        prompts = ["P1_zero_shot", "P2_zero_shot_def", "P3_few_shot", "P4_few_shot_cot"]

    df_m  = _safe_load(results_dir, "api_results_gpt4o.csv")
    df_mm = _safe_load(results_dir, "api_results_gpt4o_mm.csv")

    if df_m.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for prompt in prompts:
        sub_m  = df_m[df_m["prompt"] == prompt]
        sub_mm = df_mm[df_mm["prompt"] == prompt] if not df_mm.empty else pd.DataFrame()

        if sub_m.empty:
            continue

        acc_m  = float(accuracy_score(sub_m["label_true"], sub_m["predicted_label"]))
        acc_mm = float(accuracy_score(sub_mm["label_true"], sub_mm["predicted_label"])) \
                 if not sub_mm.empty else None
        avg_tokens = float(sub_m["total_tokens"].mean())
        cpt = float(sub_m["cost_usd"].mean()) * 1000

        rows.append({
            "prompt":      prompt,
            "matched_acc": round(acc_m * 100, 2),
            "mm_acc":      round(acc_mm * 100, 2) if acc_mm is not None else None,
            "avg_tokens":  round(avg_tokens, 0),
            "cost_per_1k": round(cpt, 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 5: Hybrid system summary
# ---------------------------------------------------------------------------

def hybrid_summary_table(results_dir: str) -> pd.DataFrame:
    """
    Build a unified summary table for all hybrid systems.

    Args:
        results_dir: Path to the results/ directory.

    Returns:
        DataFrame with columns: system, theta, matched_acc, mm_acc,
                                 api_pct, cost_per_1k, macro_f1.
    """
    rows: List[Dict] = []

    version_files: List[Tuple[str, str]] = [
        ("v1", "hybrid_v1_results.csv"),
        ("v2", "hybrid_v2_results.csv"),
        ("v3", "hybrid_v3_results.csv"),
        ("v4", "hybrid_v4_results.csv"),
    ]

    for version, fname in version_files:
        df = _safe_load(results_dir, fname)
        if df.empty or "threshold" not in df.columns:
            continue

        for theta in [0.85, 0.90, 0.95]:
            sub_m  = df[(df["set"] == "matched")    & (df["threshold"] == theta)]
            sub_mm = df[(df["set"] == "mismatched") & (df["threshold"] == theta)]

            if sub_m.empty:
                continue

            acc_m  = float(accuracy_score(sub_m["label_true"], sub_m["label_pred"]))
            f1_m   = float(f1_score(sub_m["label_true"], sub_m["label_pred"],
                                    average="macro", labels=LABELS, zero_division=0))
            api_pct = float((sub_m["source"] == "api").mean() * 100)
            cpt     = float(sub_m["cost_usd"].mean()) * 1000
            acc_mm  = float(accuracy_score(sub_mm["label_true"], sub_mm["label_pred"])) \
                      if not sub_mm.empty else None

            rows.append({
                "system":      f"Hybrid {version}",
                "theta":       theta,
                "matched_acc": round(acc_m * 100, 2),
                "mm_acc":      round(acc_mm * 100, 2) if acc_mm is not None else None,
                "api_pct":     round(api_pct, 1),
                "cost_per_1k": round(cpt, 4),
                "macro_f1":    round(f1_m, 4),
            })

    # v5 ensemble (no threshold column)
    df_v5 = _safe_load(results_dir, "hybrid_v5_results.csv")
    if not df_v5.empty:
        for set_name in ["matched", "mismatched"]:
            sub = df_v5[df_v5["set"] == set_name]
            if sub.empty:
                continue
            acc   = float(accuracy_score(sub["label_true"], sub["label_pred"]))
            f1    = float(f1_score(sub["label_true"], sub["label_pred"],
                                   average="macro", labels=LABELS, zero_division=0))
            api   = float((sub["source"] == "api").mean() * 100)
            cpt   = float(sub["cost_usd"].mean()) * 1000
            rows.append({
                "system":      "Hybrid v5 Ensemble",
                "theta":       "ensemble",
                "matched_acc": round(acc * 100, 2) if set_name == "matched" else None,
                "mm_acc":      round(acc * 100, 2) if set_name == "mismatched" else None,
                "api_pct":     round(api, 1),
                "cost_per_1k": round(cpt, 4),
                "macro_f1":    round(f1, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 7: Cost-accuracy Pareto table
# ---------------------------------------------------------------------------

def cost_pareto_table(results_dir: str) -> pd.DataFrame:
    """
    Build the full cost-accuracy table used in Section 7 of the report.

    Args:
        results_dir: Path to the results/ directory.

    Returns:
        DataFrame sorted by cost_per_1k ascending, with columns:
        system, matched_acc, cost_per_1k, on_pareto_frontier.
    """
    enc_m    = _safe_load(results_dir, "encoder_predictions_matched.csv")
    gpt4o    = _safe_load(results_dir, "api_results_gpt4o.csv")

    rows: List[Dict] = []

    # Encoders (free)
    if not enc_m.empty:
        for col, name in [
            ("deberta_v3_base_pred",  "DeBERTa-v3-base"),
            ("deberta_v3_large_pred", "DeBERTa-v3-large"),
        ]:
            if col in enc_m.columns:
                acc = float(accuracy_score(enc_m["label_text"], enc_m[col]))
                rows.append({"system": name, "matched_acc": round(acc * 100, 2),
                              "cost_per_1k": 0.0})

    # GPT-4o prompts
    if not gpt4o.empty:
        for p in gpt4o["prompt"].unique():
            sub = gpt4o[gpt4o["prompt"] == p]
            acc = float(accuracy_score(sub["label_true"], sub["predicted_label"]))
            cpt = float(sub["cost_usd"].mean()) * 1000
            rows.append({"system": f"GPT-4o {p.split('_')[0]}",
                         "matched_acc": round(acc * 100, 2),
                         "cost_per_1k": round(cpt, 4)})

    # Hybrids
    for fname, label, theta in [
        ("hybrid_v1_results.csv", "Hybrid v1 theta=0.90", 0.90),
        ("hybrid_v4_results.csv", "Hybrid v4 theta=0.90", 0.90),
    ]:
        df = _safe_load(results_dir, fname)
        if not df.empty:
            sub = df[(df["set"] == "matched") & (df["threshold"] == theta)]
            if not sub.empty:
                acc = float(accuracy_score(sub["label_true"], sub["label_pred"]))
                cpt = float(sub["cost_usd"].mean()) * 1000
                rows.append({"system": label,
                             "matched_acc": round(acc * 100, 2),
                             "cost_per_1k": round(cpt, 4)})

    df_v5 = _safe_load(results_dir, "hybrid_v5_results.csv")
    if not df_v5.empty:
        sub = df_v5[df_v5["set"] == "matched"]
        if not sub.empty:
            acc = float(accuracy_score(sub["label_true"], sub["label_pred"]))
            cpt = float(sub["cost_usd"].mean()) * 1000
            rows.append({"system": "Hybrid v5 Ensemble",
                         "matched_acc": round(acc * 100, 2),
                         "cost_per_1k": round(cpt, 4)})

    df_out = pd.DataFrame(rows).sort_values("cost_per_1k").reset_index(drop=True)

    # Mark Pareto frontier
    pareto: List[bool] = []
    for _, row in df_out.iterrows():
        dominated = any(
            (other["matched_acc"] >= row["matched_acc"] and
             other["cost_per_1k"] <= row["cost_per_1k"] and
             (other["matched_acc"] > row["matched_acc"] or
              other["cost_per_1k"] < row["cost_per_1k"]))
            for _, other in df_out.iterrows()
        )
        pareto.append(not dominated)

    df_out["on_pareto_frontier"] = pareto
    return df_out
