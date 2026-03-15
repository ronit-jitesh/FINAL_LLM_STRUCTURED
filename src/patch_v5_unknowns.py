#!/usr/bin/env python3
"""
patch_v5_unknowns.py
====================
Fixes all label_pred == "unknown" rows in hybrid_v5_results.csv
WITHOUT making any API calls.

Strategy (mirrors what friend did):
  1. Read deb_preds column  →  "entailment|entailment|neutral" etc.
  2. Majority vote (2-of-3 or 3-of-3) → use that label
  3. Three-way tie (all different) → use DeBERTa-base prediction (middle field)

Run:
    python src/patch_v5_unknowns.py

Outputs:
    results/hybrid_v5_results.csv   (patched in-place, backup saved as hybrid_v5_results_backup.csv)
"""

import os
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CSV_PATH    = os.path.join(RESULTS_DIR, "hybrid_v5_results.csv")
BACKUP_PATH = os.path.join(RESULTS_DIR, "hybrid_v5_results_backup.csv")
LABELS      = ["entailment", "neutral", "contradiction"]


def majority_vote(deb_preds_str: str) -> str:
    """
    Given 'entailment|neutral|entailment', return majority label.
    deb_preds order: deberta_v3_small | deberta_v3_base | deberta_v3_large
    Tiebreaker (3-way tie): use deberta_v3_base (index 1).
    """
    parts = [p.strip().lower() for p in deb_preds_str.split("|")]
    if len(parts) != 3:
        return "unknown"

    counts = Counter(parts)
    most_common = counts.most_common()

    # 3-of-3 or 2-of-3 majority
    if most_common[0][1] >= 2:
        return most_common[0][0]

    # 3-way tie → use DeBERTa-base (index 1)
    return parts[1]


def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌  File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)

    # Backup before patching
    df.to_csv(BACKUP_PATH, index=False)
    print(f"✅  Backup saved → {BACKUP_PATH}")

    # Find unknown rows
    mask = df["label_pred"] == "unknown"
    n_unknown = mask.sum()

    if n_unknown == 0:
        print("✅  No unknowns found — nothing to patch.")
        return

    print(f"\n🔍  Found {n_unknown} unknown rows — applying majority vote fix...\n")

    # Apply fix
    fixed = 0
    for idx in df[mask].index:
        deb_str = str(df.at[idx, "deb_preds"])
        new_label = majority_vote(deb_str)
        row_set = df.at[idx, "set"]
        row_idx = df.at[idx, "idx"]
        print(f"  Row {idx:4d} | {row_set:12s} idx={row_idx:3d} | "
              f"deb_preds={deb_str:40s} → {new_label}")
        df.at[idx, "label_pred"] = new_label
        fixed += 1

    print(f"\n✅  Fixed {fixed}/{n_unknown} unknown rows via majority vote / DeBERTa-base fallback")

    # Save patched file
    df.to_csv(CSV_PATH, index=False)
    print(f"✅  Saved patched file → {CSV_PATH}")

    # Verify no unknowns remain
    remaining = (df["label_pred"] == "unknown").sum()
    if remaining > 0:
        print(f"⚠️   WARNING: {remaining} unknowns still remain after patching!")
    else:
        print(f"✅  0 unknowns remaining")

    # Print final metrics
    print("\n" + "=" * 60)
    print("FINAL METRICS AFTER PATCH")
    print("=" * 60)
    for set_name in ["matched", "mismatched"]:
        sub = df[df["set"] == set_name]
        n   = len(sub)
        acc = accuracy_score(sub["label_true"], sub["label_pred"])
        f1  = f1_score(sub["label_true"], sub["label_pred"],
                       average="macro", labels=LABELS)
        unk = (sub["label_pred"] == "unknown").sum()
        ens_pct = (sub["source"] == "ensemble").mean() * 100
        api_pct = (sub["source"] == "api").mean() * 100
        print(f"\n  {set_name} ({n}/{800 if set_name=='matched' else 400})")
        print(f"    Accuracy  : {acc*100:.2f}%")
        print(f"    Macro F1  : {f1:.4f}")
        print(f"    Unknowns  : {unk}")
        print(f"    Ensemble% : {ens_pct:.1f}%  |  API%: {api_pct:.1f}%")

    print("\n" + "=" * 60)
    print("PATCH COMPLETE ✅")
    print("=" * 60)
    print("\nNext steps:")
    print("  python src/06_cost_analysis.py")
    print("  python src/07a_figures_main.py")
    print("  python src/07b_figure2_pareto.py")
    print("  python src/09_genre_label_analysis.py")


if __name__ == "__main__":
    main()
