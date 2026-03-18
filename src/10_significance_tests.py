#!/usr/bin/env python3
"""
010_significance_tests.py
=========================
Performs McNemar's tests to evaluate statistical significance between key systems.
Addresses the requirement for quantitative significance testing of the hybrid gains.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluate import mcnemar_pair

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "significance_tests.csv")

def load_predictions():
    """Load all necessary predictions into a single aligned DataFrame."""
    # 1. Base Encoders
    df = pd.read_csv(os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv"))
    df = df[["label_text", "deberta_v3_base_pred", "deberta_v3_large_pred"]]
    df.columns = ["y_true", "base_pred", "large_pred"]
    
    # 2. Hybrid v1 (theta=0.9)
    v1 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v1_results.csv"))
    v1 = v1[(v1["set"] == "matched") & (v1["threshold"] == 0.9)]
    df["v1_pred"] = v1["label_pred"].values
    
    # 3. Hybrid v4 (theta=0.9)
    v4 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v4_results.csv"))
    v4 = v4[(v4["set"] == "matched") & (v4["threshold"] == 0.9)]
    df["v4_pred"] = v4["label_pred"].values
    
    # 4. Hybrid v5
    v5 = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v5_results.csv"))
    v5 = v5[v5["set"] == "matched"] # v5 usually only has one threshold/run
    df["v5_pred"] = v5["label_pred"].values
    
    # 5. GPT-4o P1 and P4
    gpt4o = pd.read_csv(os.path.join(RESULTS_DIR, "api_results_gpt4o.csv"))
    p1 = gpt4o[(gpt4o["prompt"] == "P1_zero_shot") & (gpt4o["set"] == "matched")]
    p4 = gpt4o[(gpt4o["prompt"] == "P4_few_shot_cot") & (gpt4o["set"] == "matched")]
    
    df["gpt4o_p1_pred"] = p1["predicted_label"].values
    df["gpt4o_p4_pred"] = p4["predicted_label"].values
    
    # 6. Hybrid v4 for final comparison (already loaded in step 3 as v4_pred)
    
    # Final check: ensure no NaNs and consistent length
    cols_to_check = ["y_true", "base_pred", "large_pred", "v1_pred", "v4_pred", "v5_pred", "gpt4o_p1_pred", "gpt4o_p4_pred"]
    for col in cols_to_check:
        if col not in df.columns:
            raise KeyError(f"Missing column {col} in aligned DataFrame")
        if len(df[col]) != 800:
            raise ValueError(f"Column {col} has length {len(df[col])}, expected 800")
            
    return df

def main():
    print("="*60)
    print("RUNNING SIGNIFICANCE TESTS")
    print("="*60)
    
    df = load_predictions()
    y_true = df["y_true"]
    
    comparisons = [
        ("Base Encoder vs Hybrid v1",   "base_pred",      "v1_pred"),
        ("Large Encoder vs Hybrid v4",  "large_pred",     "v4_pred"),
        ("Hybrid v4 vs Hybrid v5",      "v4_pred",        "v5_pred"),
        ("GPT-4o P1 vs P4",             "gpt4o_p1_pred",  "gpt4o_p4_pred"),
        ("GPT-4o P4 vs Hybrid v4",      "gpt4o_p4_pred",  "v4_pred"),
        ("Base Encoder vs Hybrid v5",   "base_pred",      "v5_pred"),
    ]
    
    results = []
    for label, col_a, col_b in comparisons:
        res = mcnemar_pair(y_true, df[col_a], df[col_b])
        res["comparison"] = label
        results.append(res)
        print(f"  {label:<30} | p = {res['p_value']:.4f} | {res['verdict']}")

    out_df = pd.DataFrame(results)
    # Reorder columns for readability
    cols = ["comparison", "acc_a", "acc_b", "diff_pp", "p_value", "verdict", "discordant_pairs", "test_type"]
    out_df[cols].to_csv(OUT_PATH, index=False)
    print(f"\n✅ Results saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
